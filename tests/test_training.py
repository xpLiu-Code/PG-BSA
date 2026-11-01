"""Training-path checks for optimizer ownership, restart state, and checkpoint exports."""

import json
from pathlib import Path

import pytest
import torch

from pgbsa.training.losses import two_grid_labels
from pgbsa.training.train import main, parse_args


def arguments(output, *, steps=3, mode="refiner-scratch"):
    return [
        "--output",
        str(output),
        "--device",
        "cpu",
        "--mode",
        mode,
        "--steps",
        str(steps),
        "--batch-size",
        "4",
        "--calibration-batches",
        "1",
        "--validation-every",
        "0",
        "--validation-batches",
        "1",
        "--validation-batch-size",
        "4",
        "--snapshot-every",
        "0",
        "--log-every",
        "3",
    ]


def tensor_values(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_values(child, prefix + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for key, child in enumerate(value):
            yield from tensor_values(child, prefix + "/" + str(key))


def test_interpolated_targets_with_padding_and_ties():
    grid = torch.tensor([-1.0, 0.0, 1.0])
    angles = torch.tensor([[-0.5, 0.5], [0.0, 0.0]])
    active = torch.tensor([[True, True], [False, False]])
    target, offset, left, right = two_grid_labels(grid, angles, active)
    assert torch.equal(target[0], torch.ones(3))
    assert torch.equal(offset[0], torch.tensor([0.5, -0.5, -0.5]))
    assert not target[1].any()
    assert not offset[1].any()


def test_config_overrides_and_invalid_output(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"steps": 6000, "batch_size": 128}), encoding="utf-8")
    args = parse_args(["--config", str(config), "--output", "model.pth", "--steps", "3"])
    assert args.steps == 3 and args.batch_size == 128
    with pytest.raises(ValueError, match="extension"):
        main(["--output", str(tmp_path / "model.json")])
    assert not (tmp_path / "model.json").exists()


def test_refiner_resume_matches_uninterrupted_run(tmp_path):
    direct = tmp_path / "direct" / "model.pth"
    resumed = tmp_path / "resumed" / "model.pth"
    main(arguments(direct))
    main(arguments(resumed) + ["--stop-after", "1"])
    state = resumed.with_name("model_step000001.resume.pth")
    assert state.is_file() and not resumed.exists()
    main(arguments(resumed) + ["--resume", str(state)])
    expected = torch.load(direct, weights_only=True, map_location="cpu")
    actual = torch.load(resumed, weights_only=True, map_location="cpu")
    assert actual["metadata"]["compact_refinement_trained"] is True
    assert actual["metadata"]["training"]["from_scratch"] is False
    assert actual["metadata"]["training"]["refinement_from_scratch"] is True
    expected_tensors = dict(tensor_values(expected))
    actual_tensors = dict(tensor_values(actual))
    assert expected_tensors.keys() == actual_tensors.keys()
    assert all(torch.equal(expected_tensors[k], actual_tensors[k]) for k in expected_tensors)
    report = json.loads(resumed.with_suffix(".json").read_text())
    assert report["frozen_state_unchanged"]
    assert report["strict_state_roundtrip"] and report["strict_normalization_roundtrip"]
    assert report["trainable_parameters"] == 353337
    assert report["training"]["initialization"]["reset_parameters"] == 276823
    assert report["training"]["stage_visits"] == {
        "route_original": 1,
        "route_adapted": 1,
        "route_close": 1,
    }
    assert (
        report["training"]["last_training_rows"]
        == json.loads(direct.with_suffix(".json").read_text())["training"]["last_training_rows"]
    )


def test_full_scratch_visits_every_supervised_stage(tmp_path):
    output = tmp_path / "full" / "model.pth"
    main(arguments(output, steps=5, mode="full-scratch"))
    report = json.loads(output.with_suffix(".json").read_text())
    training = report["training"]
    assert training["initialization"]["all_network_weights_random"]
    assert training["initialization"]["reset_parameters"] == report["parameters_uncompacted"]
    assert report["trainable_parameters"] == report["parameters_uncompacted"]
    assert training["stage_visits"] == {
        "coarse_original": 1,
        "coarse_retrieval": 1,
        "route_original": 1,
        "route_adapted": 1,
        "route_close": 1,
    }
    assert report["strict_state_roundtrip"] and report["strict_normalization_roundtrip"]


def test_saved_dataset_training_and_validation(tmp_path):
    from pgbsa.training.data import export_dataset, MODES

    dataset = tmp_path / "dataset"
    config = {
        "batch_size": 4,
        "samples_per_shard": 20,
        "mode_sequence": list(MODES),
        "low_snapshot_fraction": 0.65,
        "low_snr_fraction": 0.15,
        "splits": {
            "train": {"seed": 31, "samples": 20},
            "validation": {"seed": 32, "samples": 20},
            "test": {"seed": 33, "samples": 20},
        },
    }
    export_dataset(config, dataset)
    output = tmp_path / "offline" / "model.pth"
    main(
        arguments(output)
        + [
            "--train-data",
            str(dataset),
            "--validation-data",
            str(dataset),
            "--validation-every",
            "3",
        ]
    )
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["training"]["validations"][0]["scenes"] == 4
    assert report["training"]["completed_steps"] == 3
    assert all(item["data_source"] == "dataset" and item["dataset_identity"]
               for item in report["training"]["calibration"])
    state = torch.load(output.with_name("model_step000003.resume.pth"), weights_only=True)
    assert state["generator"]["dataset_identity"]
    assert state["loop_state"]["training_config"]["train_data_sha256"]
