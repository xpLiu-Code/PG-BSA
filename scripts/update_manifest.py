"""Refresh packaged source hashes after an intentional source change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pgbsa.api import TITLE, VERSION, _CHECKPOINT_SHA256, _INITIAL_SHA256, _FORMAT, _PARAMETERS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = ROOT / "src" / "pgbsa"
    assets = package / "assets"
    checkpoint_hash = digest(assets / "model.pth")
    if checkpoint_hash != _CHECKPOINT_SHA256:
        raise RuntimeError("The pretrained checkpoint has changed.")
    initial_hash = digest(assets / "training_initial.pth")
    if initial_hash != _INITIAL_SHA256:
        raise RuntimeError("The training initialization checkpoint has changed.")
    manifest = {
        "format": _FORMAT,
        "title": TITLE,
        "release_version": VERSION,
        "deployed_parameters": _PARAMETERS,
        "checkpoint": {"path": "model.pth", "sha256": checkpoint_hash},
        "training_initialization": {"path": "training_initial.pth", "sha256": initial_hash},
        "options": {"path": "inference.json", "sha256": digest(assets / "inference.json")},
        "source_sha256": {
            path.relative_to(package.parent).as_posix(): digest(path)
            for path in sorted(package.rglob("*.py"))
        },
    }
    path = assets / "release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
