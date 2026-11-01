"""Load PG-BSA with the bundled checkpoint and inference configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TITLE = (
    "PG-BSA: Physics-Guided Beam-Domain Sparse Attention for High-Resolution "
    "Underdetermined DOA Estimation With an Unknown Number of Sources"
)
VERSION = "1.1.0"
_FORMAT = "pgbsa_package_v1"
_PARAMETERS = 2_367_509
_CHECKPOINT_SHA256 = "90d683e4a4cb496e0982c7961f935296f45aef3f7c43a558ccf09d57760c3934"
_INITIAL_SHA256 = "e5fde9a6c880633f791c5094c658e0088c55944f0654efbb2f6704fd48edb5bc"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {Path(path).name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {Path(path).name}.")
    return value


def verify_release():
    """Verify source, inference configuration, and both checkpoint hashes.

    Returns the manifest without importing numerical dependencies.
    """
    package = Path(__file__).resolve().parent
    assets = package / "assets"
    manifest = _read_json(assets / "release_manifest.json")
    if (manifest.get("format") != _FORMAT
            or manifest.get("title") != TITLE
            or manifest.get("release_version") != VERSION
            or manifest.get("deployed_parameters") != _PARAMETERS):
        raise RuntimeError("The manifest does not match this PG-BSA package.")
    expected = manifest.get("source_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("The manifest contains no source hashes.")
    actual = {
        path.relative_to(package.parent).as_posix(): _sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(actual) & set(expected)
                         if actual[key] != expected[key])
        raise RuntimeError(
            f"Source verification failed: missing={missing}, extra={extra}, "
            f"changed={changed}. Regenerate the manifest after intentional source edits."
        )
    for key, filename in (("checkpoint", "model.pth"), ("options", "inference.json"),
                          ("training_initialization", "training_initial.pth")):
        entry = manifest.get(key, {})
        if entry.get("path") != filename or _sha256(assets / filename) != entry.get("sha256"):
            raise RuntimeError(f"Bundled {filename} does not match its manifest entry.")
    if manifest["checkpoint"]["sha256"] != _CHECKPOINT_SHA256:
        raise RuntimeError("The checkpoint differs from the pretrained PG-BSA model.")
    if manifest["training_initialization"]["sha256"] != _INITIAL_SHA256:
        raise RuntimeError("The training initialization checkpoint has changed.")
    return manifest


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def create_model(device="cpu", *, checkpoint=None, graph_encoders=True,
                 search_workers=4, cpu_threads=1):
    """Construct PG-BSA for CPU or CUDA inference.

    Pass trace-normalized float32 covariance tensors with shape
    ``[B, 2, 16, 16]`` to ``model.predict``. Channel 0 contains the real part
    and channel 1 the imaginary part. Predictions are arrays of angles in
    degrees, with one variable-length array per input scene.

    Use the model as a context manager, or call ``close()`` after inference.
    Set ``graph_encoders=False`` to disable CUDA graph capture.
    Pass ``checkpoint`` to load a checkpoint exported by ``pgbsa-train``;
    omitting it selects the bundled pretrained weights.
    """
    _positive_integer("search_workers", search_workers)
    _positive_integer("cpu_threads", cpu_threads)
    if not isinstance(graph_encoders, bool):
        raise ValueError("graph_encoders must be bool.")
    manifest = verify_release()
    assets = Path(__file__).resolve().parent / "assets"
    from ._core import PGBSA

    weights = assets / "model.pth" if checkpoint is None else Path(checkpoint)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    weights_digest = _sha256(weights)
    options = _read_json(assets / "inference.json")
    if not graph_encoders:
        options["runtime_options"]["graph_compact_refiner"] = False
        options["runtime_options"]["graph_expanded_candidates"] = False
    model = PGBSA(
        device, weights, graph_encoders=graph_encoders,
        search_workers=search_workers, cpu_threads=cpu_threads, **options,
    )
    try:
        if model.parameter_count() != _PARAMETERS:
            raise RuntimeError("The loaded parameter count does not match the checkpoint.")
        training = model.metadata.get("training", {})
        if checkpoint is None and (training.get("completed_steps") != 6000
                                   or training.get("mode") != "refiner-scratch"):
            raise RuntimeError("The checkpoint's refinement training metadata is inconsistent.")
        if _sha256(weights) != weights_digest:
            raise RuntimeError("Checkpoint changed while it was being loaded.")
        model.metadata = {
            "title": TITLE,
            "version": VERSION,
            "checkpoint_sha256": weights_digest,
            "bundled_checkpoint": checkpoint is None,
            "deployed_parameters": model.parameter_count(),
            "sensor_count": 16,
            "sensor_spacing_wavelengths": 0.5,
            "search_interval_deg": [-70.0, 70.0],
            "runtime_options": model.metadata["runtime_options"],
            "refinement_training_steps": training.get("completed_steps"),
            "training_mode": training.get("mode"),
            "device": str(model.device),
        }
    except Exception:
        model.close()
        raise
    return model
