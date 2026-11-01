"""CPU portability and regression against the bundled predictions."""
from pathlib import Path
import json
import numpy as np
import torch
from pgbsa import create_model

ROOT = Path(__file__).resolve().parents[1]

def test_cpu_inference_from_another_working_directory(tmp_path, monkeypatch):
    with np.load(ROOT / "examples" / "sample_covariances.npz", allow_pickle=False) as data:
        x = torch.from_numpy(data["in_mat"].copy())
    expected = json.loads((ROOT / "examples" / "expected_predictions.json").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    with create_model("cpu") as model:
        result = model.predict(x)
        assert model.parameter_count() == 2367509
        assert model.metadata["refinement_training_steps"] == 6000
        options = model.metadata["runtime_options"]
        assert options["grid_kv_stride"] == 2
        assert options["rank2_pair_gradient"] is True
        assert options["angle_polish_steps"] == 3
    assert len(result) == len(expected["predictions"]) == 3
    for actual, reference in zip(result, expected["predictions"]):
        target = np.asarray(reference["angles_deg"], dtype=np.float64)
        assert len(actual) == len(target)
        np.testing.assert_allclose(np.sort(actual), np.sort(target), rtol=0, atol=expected["tolerance_deg"])
