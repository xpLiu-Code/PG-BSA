"""Check the file input contract before model construction."""
import numpy as np
import pytest
import torch

from pgbsa import create_model
from pgbsa.cli import _load_input, main


def covariance_batch():
    value = np.zeros((1, 2, 16, 16), dtype=np.float32)
    value[0, 0] = np.eye(16, dtype=np.float32) / 16
    return value


@pytest.mark.parametrize("suffix", [".npy", ".npz", ".pt"])
def test_supported_input_files(tmp_path, suffix):
    expected = covariance_batch()
    path = tmp_path / ("input" + suffix)
    if suffix == ".npy":
        np.save(path, expected)
    elif suffix == ".npz":
        np.savez(path, in_mat=expected)
    else:
        torch.save({"in_mat": torch.from_numpy(expected)}, path)
    actual = _load_input(path)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("invalid", [
    np.zeros((1, 2, 16, 16), dtype=np.float64),
    np.zeros((0, 2, 16, 16), dtype=np.float32),
    np.zeros((1, 16, 16), dtype=np.float32),
    np.full((1, 2, 16, 16), np.nan, dtype=np.float32),
])
def test_invalid_input_rejected(tmp_path, invalid):
    path = tmp_path / "input.npy"
    np.save(path, invalid)
    with pytest.raises(ValueError):
        _load_input(path)


def test_npz_requires_covariance_key(tmp_path):
    path = tmp_path / "input.npz"
    np.savez(path, angles=np.array([0.0]))
    with pytest.raises(ValueError, match="in_mat"):
        _load_input(path)


def test_cli_preserves_existing_output(tmp_path):
    output = tmp_path / "predictions.json"
    output.write_text("existing result", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "input.npy"), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "existing result"


@pytest.mark.parametrize("options", [
    {"cpu_threads": 0}, {"search_workers": True}, {"graph_encoders": "yes"},
])
def test_invalid_runtime_options_rejected_before_loading(options):
    with pytest.raises(ValueError):
        create_model(**options)
