"""Verify packaged sources, configuration, and checkpoints."""
from pathlib import Path
import hashlib
import json
import torch
from pgbsa import verify_release
import pgbsa

def test_release_integrity():
    report = verify_release()
    assert isinstance(report, dict)
    assets = Path(pgbsa.__file__).resolve().parent / "assets"
    manifest = json.loads((assets / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["deployed_parameters"] == 2367509
    assert manifest["checkpoint"]["sha256"] == "90d683e4a4cb496e0982c7961f935296f45aef3f7c43a558ccf09d57760c3934"
    assert any(k.startswith("pgbsa/_core/") for k in manifest["source_sha256"])
    assert hashlib.sha256((assets / "model.pth").read_bytes()).hexdigest() == manifest["checkpoint"]["sha256"]
    assert hashlib.sha256((assets / "training_initial.pth").read_bytes()).hexdigest() == manifest["training_initialization"]["sha256"]
    assert any(k.startswith("pgbsa/training/") for k in manifest["source_sha256"])
    metadata = torch.load(assets / "model.pth", map_location="cpu", weights_only=True)["metadata"]
    assert metadata["compact_refinement_trained"] is True
    assert metadata["training"]["from_scratch"] is False
    assert metadata["training"]["refinement_from_scratch"] is True
