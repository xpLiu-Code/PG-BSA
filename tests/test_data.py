"""Signal-model and reproducibility checks for covariance synthesis."""
import copy
import json

import numpy as np
import pytest
import torch

from pgbsa.training.data import MODES, export_dataset, generate_batch, make_dataset_generator, make_generator


@pytest.mark.parametrize("mode", MODES)
def test_covariance_and_label_contract(mode):
    generator = make_generator(202610100101)
    x, angles, active, count, snapshots, snr = generate_batch(24, "cpu", generator, mode)
    assert x.shape == (24, 2, 16, 16)
    assert x.dtype == torch.float32
    covariance = torch.complex(x[:, 0], x[:, 1])
    torch.testing.assert_close(covariance, covariance.mH, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(covariance.diagonal(dim1=-2, dim2=-1).sum(-1),
                               torch.ones(24, dtype=torch.complex64), atol=2e-7, rtol=1e-6)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-7
    assert torch.equal(active.sum(1), count)
    assert torch.all(angles[~active] == 0)
    assert torch.all(torch.abs(angles[active]) <= 65)
    for row, number in zip(angles, count):
        assert torch.all(torch.diff(row[:number]) > 0)
    assert set(snapshots.tolist()) <= {16, 32, 64, 128, 256, 512, 1000}
    assert set(snr.tolist()) <= {-10., -5., 0., 5., 10., 20.}
    assert generator.samples_generated == 24
    if mode == "adapted":
        assert count.min() >= 11
    if mode == "close":
        assert (count == 1).sum() == 12
        assert (count == 2).sum() == 12
    if mode == "close_covered":
        assert (count == 3).sum() == 6


def test_random_stream_replay_and_separation():
    first = make_generator(123)
    replay = make_generator(123)
    different = make_generator(124, split="validation")
    for mode in ("ordinary", "adapted", "close_covered"):
        actual = generate_batch(8, "cpu", first, mode)
        expected = generate_batch(8, "cpu", replay, mode)
        for left, right in zip(actual, expected):
            assert torch.equal(left, right)
    assert not torch.equal(actual[0], generate_batch(8, "cpu", different, "close_covered")[0])


def small_config():
    return {
        "batch_size": 4, "samples_per_shard": 8,
        "mode_sequence": ["ordinary", "adapted", "close_covered"],
        "low_snapshot_fraction": .65, "low_snr_fraction": .15,
        "splits": {name: {"seed": seed, "samples": 11}
                   for name, seed in (("train", 1), ("validation", 2), ("test", 3))},
    }


def test_dataset_export_preserves_online_stream(tmp_path):
    config = small_config()
    output = tmp_path / "dataset"
    manifest = export_dataset(config, output)
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for split, options in config["splits"].items():
        shards = manifest["splits"][split]["files"]
        assert [shard["samples"] for shard in shards] == [8, 3]
        loaded = []
        for shard in shards:
            with np.load(output / shard["path"], allow_pickle=False) as data:
                loaded.append({key: data[key].copy() for key in data.files})
        data = {key: np.concatenate([part[key] for part in loaded]) for key in loaded[0]}
        generator = make_generator(options["seed"], split=split)
        arrays = []
        for size, mode in zip((4, 4, 3), config["mode_sequence"]):
            arrays.append(generate_batch(size, "cpu", generator, mode)[0].numpy())
        np.testing.assert_array_equal(data["in_mat"], np.concatenate(arrays))
        np.testing.assert_array_equal(data["source_power"], data["active"].astype(np.float32))
        np.testing.assert_allclose(data["noise_variance"], 10**(-data["snr_db"]/10))
    with pytest.raises(FileExistsError):
        export_dataset(config, output)


def test_invalid_dataset_configuration_creates_no_output(tmp_path):
    config = small_config()
    config["splits"]["test"]["seed"] = config["splits"]["train"]["seed"]
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="distinct seeds"):
        export_dataset(config, output)
    assert not output.exists()
    config = copy.deepcopy(small_config())
    config["samples_per_shard"] = 5
    with pytest.raises(ValueError, match="multiple"):
        export_dataset(config, output)


def test_saved_dataset_batches_and_rng_resume(tmp_path):
    config = small_config()
    output = tmp_path / "dataset"
    export_dataset(config, output)
    first = make_dataset_generator(output, seed=22)
    second = make_dataset_generator(output / "manifest.json", seed=22)
    for mode in config["mode_sequence"]:
        actual = generate_batch(5, "cpu", first, mode)
        expected = generate_batch(5, "cpu", second, mode)
        for left, right in zip(actual, expected):
            assert torch.equal(left, right)
    state = copy.deepcopy(first.rng.bit_generator.state)
    batch = generate_batch(5, "cpu", first, "ordinary")
    second.rng.bit_generator.state = state
    resumed = generate_batch(5, "cpu", second, "ordinary")
    assert torch.equal(batch[0], resumed[0])
    validation = make_dataset_generator(output, split="development", seed=22)
    assert validation.split == "development"
    assert validation.dataset_identity == first.dataset_identity
    with pytest.raises(ValueError, match="no scenes"):
        generate_batch(5, "cpu", first, "retrieval")
    shard = output / "train" / "part-00000.npz"
    shard.write_bytes(shard.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum"):
        make_dataset_generator(output)
