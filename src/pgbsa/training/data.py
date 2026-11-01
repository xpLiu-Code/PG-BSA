"""Online covariance synthesis and sharded dataset export for PG-BSA."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

SNAPSHOTS = (16, 32, 64, 128, 256, 512, 1000)
SNR_VALUES = (-10., -5., 0., 5., 10., 20.)
MODES = ("retrieval", "ordinary", "adapted", "close", "close_covered")


@dataclass
class BatchGenerator:
    """Independent NumPy geometry and PyTorch snapshot random streams."""

    seed: int
    device: torch.device
    split: str = "train"
    low_snapshot_fraction: float = .65
    low_snr_fraction: float = .15
    rng: np.random.Generator = field(init=False, repr=False)
    torch_rng: torch.Generator = field(init=False, repr=False)
    samples_generated: int = field(default=0, init=False)
    last_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.device = torch.device(self.device)
        if self.split not in ("train", "development", "validation", "test"):
            raise ValueError("split must be train, development, validation, or test")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not 0 <= self.low_snapshot_fraction <= 1 or not 0 <= self.low_snr_fraction <= 1:
            raise ValueError("Sampling fractions must be in [0, 1]")
        self.rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device=self.device).manual_seed(self.seed)

    def provenance(self):
        return {
            "generator": "pgbsa_covariance_curriculum_v1",
            "seed": int(self.seed), "split": self.split,
            "samples_generated": self.samples_generated,
            "sensors": 16, "spacing_wavelengths": .5,
            "angle_range_deg": [-65., 65.], "maximum_sources": 20,
            "low_snapshot_fraction": self.low_snapshot_fraction,
            "low_snr_fraction": self.low_snr_fraction,
            "snapshots": list(SNAPSHOTS), "snr_values_db": list(SNR_VALUES),
            "snr_convention": "Per-source expected signal power divided by noise variance",
            "source_distribution": "Independent proper complex Gaussian with unit variance",
            "noise_distribution": "Spatially white proper complex Gaussian; variance 10**(-snr_db/10)",
            "steering_convention": "exp(-1j*pi*sensor_index*sin(angle_rad))",
            "covariance": "X @ X.conj().T / snapshots, divided by its trace; no mean subtraction",
            "noise_only_snr_db": "Stored as 0 for bookkeeping; no source SNR is defined for K=0",
            "retrieval_fractions": {
                "noise": .04, "single": .10, "pair_gap_1_to_4": .20,
                "counts_4_8_12_18_20": .36, "compact_count_3": .05, "general_counts_1_to_20": .25,
            },
            "ordinary_fractions": {
                "single": .15, "pair_gap_1_to_4": .25, "counts_4_8": .30,
                "compact_count_3": .10, "general_counts_3_to_10": .20,
            },
            "adapted_fractions": {"counts_12_18_20": .60, "general_counts_11_to_20": .40},
            "close_fractions": {"single": .50, "pair_gap_0p5_to_3p5": .50},
            "close_covered_fractions": {"compact_count_3": .25, "single": .375, "pair_gap_0p5_to_3p5": .375},
            "batch_quotas": "Fractions are rounded to integer reservations in order; the final group fills the batch",
            "unreserved_k2_families": {"spread": .25, "pair_recovery": .45, "close": .30},
            "unreserved_k3_to_20_families": {"pinned": .50, "spread": .25, "compact": .125, "localcluster": .125},
            "pinned_gap_range_deg": [2., 5.],
            "spread_minimum_gap_range_deg": [.5, 4.5],
            "compact_internal_gap_range_deg": [.5, 2.5],
            "localcluster_center_range_deg": [-50., 50.],
            "localcluster_gap_range_deg": [.5, 3.],
            "reproducibility": "Requires the same seed, software, device, mode sequence, and batch sizes",
        }


def make_generator(seed: int, device="cpu", split="train", *,
                   low_snapshot_fraction=.65, low_snr_fraction=.15):
    return BatchGenerator(seed, torch.device(device), split,
                          float(low_snapshot_fraction), float(low_snr_fraction))


class DatasetBatchGenerator(BatchGenerator):
    """Sample labeled shards by curriculum mode using an independent RNG.

    A batch samples without replacement when its mode pool is large enough;
    smaller pools use replacement. Shards are read on demand, so the complete
    dataset need not fit in memory. Dataset hashes are checked at construction.
    """

    def __init__(self, path, device="cpu", split="train", *, seed=None):
        path = Path(path)
        manifest_path = path / "manifest.json" if path.is_dir() else path
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("format") != "pgbsa_covariance_dataset_v1":
            raise ValueError("Unrecognized covariance dataset format")
        self.dataset_identity = hashlib.sha256(raw).hexdigest()
        self._dataset_root = manifest_path.parent.resolve()
        config = manifest["configuration"]
        _validate_config(config)
        stored_split = "validation" if split == "development" else split
        if stored_split not in manifest["splits"]:
            raise ValueError("Dataset does not contain split: " + stored_split)
        options = config["splits"][stored_split]
        super().__init__(options["seed"] if seed is None else seed, torch.device(device), split,
                         config["low_snapshot_fraction"], config["low_snr_fraction"])
        self._files = []
        self._mode_rows = {mode: [] for mode in MODES}
        total = 0
        for entry in manifest["splits"][stored_split]["files"]:
            shard = (self._dataset_root / entry["path"]).resolve()
            if not shard.is_relative_to(self._dataset_root):
                raise ValueError("Dataset shard path leaves the dataset directory")
            if hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Dataset shard checksum mismatch: " + shard.name)
            with np.load(shard, allow_pickle=False) as data:
                _validate_shard(data, entry["samples"])
                modes = data["mode"]
                for mode in MODES:
                    for row in np.flatnonzero(modes == mode):
                        self._mode_rows[mode].append((len(self._files), int(row)))
            self._files.append(shard)
            total += entry["samples"]
        if total != options["samples"]:
            raise ValueError("Dataset sample total differs from its configuration")
        self._mode_rows = {mode: np.asarray(rows, dtype=np.int64).reshape(-1, 2)
                           for mode, rows in self._mode_rows.items()}

    def provenance(self):
        return {**super().provenance(), "storage": "labeled_covariance_shards",
                "dataset_identity": self.dataset_identity,
                "mode_pool_sizes": {mode: len(rows) for mode, rows in self._mode_rows.items()},
                "sampling": "Without replacement within a batch; with replacement if the pool is smaller than the batch"}

    def generate_batch(self, batch_size, mode):
        if mode not in MODES:
            raise ValueError("Unsupported curriculum mode: " + str(mode))
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        rows = self._mode_rows[mode]
        if not len(rows):
            raise ValueError("Dataset has no scenes for curriculum mode: " + mode)
        n = int(batch_size)
        indices = self.rng.choice(len(rows), n, replace=n > len(rows))
        selected = rows[indices]
        fields = ("in_mat", "angles_deg", "active", "source_count", "snapshots", "snr_db")
        arrays = {}
        for shard_index in np.unique(selected[:, 0]):
            locations = np.flatnonzero(selected[:, 0] == shard_index)
            with np.load(self._files[int(shard_index)], allow_pickle=False) as data:
                for name in fields:
                    if name not in arrays:
                        arrays[name] = np.empty((n, *data[name].shape[1:]), dtype=data[name].dtype)
                    arrays[name][locations] = data[name][selected[locations, 1]]
        self.samples_generated += n
        self.last_metadata = {**self.provenance(), "mode": mode,
                              "sampled_rows": selected.tolist(),
                              "batch_count_histogram": {int(k): int((arrays["source_count"] == k).sum())
                                                        for k in np.unique(arrays["source_count"])}}
        return tuple(torch.from_numpy(arrays[name]).to(self.device) for name in fields)


def make_dataset_generator(path, device="cpu", split="train", *, seed=None):
    return DatasetBatchGenerator(path, device, split, seed=seed)


def _validate_shard(data, size):
    shapes = {"in_mat": (size, 2, 16, 16), "angles_deg": (size, 20),
              "active": (size, 20), "source_count": (size,), "snapshots": (size,),
              "snr_db": (size,), "mode": (size,)}
    dtypes = {"in_mat": np.float32, "angles_deg": np.float32, "active": np.bool_,
              "source_count": np.int64, "snapshots": np.int64, "snr_db": np.float32}
    for name, shape in shapes.items():
        if name not in data or data[name].shape != shape:
            raise ValueError("Invalid dataset field shape: " + name)
        if name in dtypes and data[name].dtype != dtypes[name]:
            raise ValueError("Invalid dataset field dtype: " + name)
        if name in dtypes and not np.isfinite(data[name]).all():
            raise ValueError("Non-finite dataset field: " + name)
    if not np.isin(data["mode"], MODES).all():
        raise ValueError("Dataset contains an unknown curriculum mode")
    counts = data["source_count"]
    if np.any((counts < 0) | (counts > 20)):
        raise ValueError("Dataset source counts must be in [0, 20]")
    if not np.array_equal(data["active"], np.arange(20)[None, :] < counts[:, None]):
        raise ValueError("Dataset active mask does not match source counts")


def _spread(k, rng, gap, lo=-65., hi=65.):
    if k == 1:
        return np.array([rng.uniform(lo, hi)], dtype=np.float32)
    slack = hi-lo-(k-1)*gap
    if slack < 0:
        raise ValueError("Infeasible minimum spacing")
    lengths = rng.dirichlet(np.ones(k+1))*slack
    return (lo+lengths[0]+np.r_[0., np.cumsum(gap+lengths[1:-1])]).astype(np.float32)


def _pinned_spread(k, rng, gap, lo=-65., hi=65.):
    """Fix one random adjacent gap; distribute the remaining slack randomly."""
    if k < 2:
        return _spread(k, rng, 0., lo, hi)
    slack = hi-lo-(k-1)*gap
    if slack < 0:
        raise ValueError("Infeasible pinned separation")
    lengths = rng.dirichlet(np.ones(k+1))
    lengths[int(rng.integers(1, k))] = 0.
    lengths = lengths/lengths.sum()*slack
    return (lo+lengths[0]+np.r_[0., np.cumsum(gap+lengths[1:-1])]).astype(np.float32)


def _angles(k, family, rng):
    if k == 0:
        return np.empty(0, np.float32)
    if k == 1:
        return _spread(k, rng, 0.)
    if family == "pinned":
        return _pinned_spread(k, rng, float(rng.uniform(2., 5.)))
    if family == "pair_recovery":
        gap = float(rng.uniform(1., 4.))
        center = float(rng.uniform(-65+gap/2, 65-gap/2))
        return np.array([center-gap/2, center+gap/2], np.float32)
    if family == "spread":
        return _spread(k, rng, float(rng.uniform(.5, 4.5)))
    if family == "multigap":
        return _spread(k, rng, float(rng.uniform(.75, 4.5)))
    if family in ("close", "compact"):
        gap = float(rng.uniform(.5, 3.5 if family == "close" else 2.5))
        width = gap*(k-1)
        center = rng.uniform(-65+width/2, 65-width/2)
        if family == "compact":
            spacing = rng.uniform(.5, 2.5, k-1)
            offsets = np.r_[0., np.cumsum(spacing)]
            center = rng.uniform(-65+offsets[-1]/2, 65-offsets[-1]/2)
            return (center+offsets-offsets[-1]/2).astype(np.float32)
        return (center+(np.arange(k)-(k-1)/2)*gap).astype(np.float32)
    center = rng.uniform(-50, 50)
    gap = rng.uniform(.5, 3.)
    cluster = center+np.array([-gap, 0., gap])
    background = np.arange(-64., 64.1, 3.75)+rng.uniform(-.1, .1)
    background = background[np.abs(background-center) > 7.]
    return np.sort(np.r_[cluster, rng.choice(background, k-3, replace=False)]).astype(np.float32)


@torch.no_grad()
def generate_batch(batch_size, device, generator: BatchGenerator, mode="retrieval"):
    """Return covariance, padded DOAs, active mask, count, snapshots, and SNR.

    Covariance has shape [B, 2, 16, 16]; DOAs and masks have shape [B, 20].
    The remaining outputs are scene labels of shape [B]. Only covariance is
    passed to the network. Simulation is grouped by snapshot count.
    """
    if not isinstance(generator, BatchGenerator):
        raise TypeError("Use make_generator for independent random streams")
    device = torch.device(device)
    if device != generator.device:
        raise ValueError("The requested device must equal the generator device")
    if isinstance(generator, DatasetBatchGenerator):
        return generator.generate_batch(batch_size, mode)
    if mode not in MODES:
        raise ValueError("Unsupported curriculum mode: " + str(mode))
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    n = int(batch_size)
    rng = generator.rng
    counts, forced = [], []

    def reserve(number, values, family=None):
        amount = min(max(int(number), 0), n-len(counts))
        counts.extend(rng.choice(values, amount).tolist())
        forced.extend([family]*amount)

    if mode == "retrieval":
        reserve(round(.04*n), [0], "noise")
        reserve(round(.10*n), [1], "single")
        reserve(round(.20*n), [2], "pair_recovery")
        reserve(round(.36*n), [4, 8, 12, 18, 20])
        reserve(round(.05*n), [3], "compact")
        reserve(n, np.arange(1, 21))
    elif mode == "ordinary":
        reserve(round(.15*n), [1], "single")
        reserve(round(.25*n), [2], "pair_recovery")
        reserve(round(.30*n), [4, 8])
        reserve(round(.10*n), [3], "compact")
        reserve(n, np.arange(3, 11))
    elif mode == "adapted":
        reserve(round(.60*n), [12, 18, 20])
        reserve(n, np.arange(11, 21))
    elif mode == "close_covered":
        reserve(round(.25*n), [3], "compact")
        reserve(round(.375*n), [1], "single")
        reserve(n, [2], "close")
    else:
        reserve(n//2, [1], "single")
        reserve(n, [2], "close")
    order = rng.permutation(n)
    counts = np.asarray(counts, dtype=np.int64)[order]
    forced = [forced[int(i)] for i in order]
    low_t = rng.random(n) < generator.low_snapshot_fraction
    snapshots = np.where(low_t, rng.choice(SNAPSHOTS[:5], n), rng.choice(SNAPSHOTS[5:], n))
    low_snr = rng.random(n) < generator.low_snr_fraction
    snrs = np.where(low_snr, rng.choice(SNR_VALUES[:2], n), rng.choice(SNR_VALUES[2:], n)).astype(np.float32)
    a = np.zeros((n, 20), np.float32)
    families = []
    for i, k0 in enumerate(counts):
        k = int(k0)
        if forced[i] is not None:
            family = forced[i]
        elif k == 0:
            family = "noise"
        elif k == 1:
            family = "single"
        elif k == 2:
            family = str(rng.choice(["spread", "pair_recovery", "close"], p=[.25, .45, .30]))
        else:
            family = str(rng.choice(["pinned", "spread", "compact", "localcluster"], p=[.50, .25, .125, .125]))
        if k == 0:
            snrs[i] = 0.
        families.append(family)
        a[i, :k] = _angles(k, family, rng)
    angles = torch.as_tensor(a, device=device)
    count = torch.as_tensor(counts, device=device, dtype=torch.long)
    active = torch.arange(20, device=device)[None, :] < count[:, None]
    ts = torch.as_tensor(snapshots, device=device, dtype=torch.long)
    snr = torch.as_tensor(snrs, device=device)
    output = torch.empty((n, 2, 16, 16), device=device, dtype=torch.float32)
    position = torch.arange(16, device=device, dtype=torch.float32)[None, :, None]
    for t0 in np.unique(snapshots):
        ix = torch.as_tensor(np.flatnonzero(snapshots == t0), device=device)
        t, m = int(t0), int(ix.numel())
        steering = torch.exp(-1j*math.pi*position*torch.sin(torch.deg2rad(angles[ix]))[:, None, :])
        source = torch.complex(torch.randn(m, 20, t, device=device, generator=generator.torch_rng),
                               torch.randn(m, 20, t, device=device, generator=generator.torch_rng))/math.sqrt(2)
        source *= active[ix, :, None]
        noise = torch.complex(torch.randn(m, 16, t, device=device, generator=generator.torch_rng),
                              torch.randn(m, 16, t, device=device, generator=generator.torch_rng))
        noise *= (torch.pow(10., -snr[ix]/10.)/2.).sqrt()[:, None, None]
        y = torch.bmm(steering, source)+noise
        r = torch.bmm(y, y.mH)/t
        r /= r.diagonal(dim1=-2, dim2=-1).real.sum(-1).clamp_min(1.e-12)[:, None, None]
        output[ix] = torch.stack((r.real, r.imag), dim=1)
    generator.samples_generated += n
    generator.last_metadata = {
        **generator.provenance(), "mode": mode, "families": families,
        "reserved_pair_scenes": sum(f == "pair_recovery" for f in forced),
        "reserved_compact_k3_scenes": sum(f == "compact" for f in forced),
        "batch_count_histogram": {int(k): int((counts == k).sum()) for k in np.unique(counts)},
    }
    return output, angles, active, count, ts, snr


def _validate_config(config):
    required = {"batch_size", "samples_per_shard", "mode_sequence", "splits",
                "low_snapshot_fraction", "low_snr_fraction"}
    if set(config) != required:
        raise ValueError("Dataset config fields must be: " + ", ".join(sorted(required)))
    for field_name in ("batch_size", "samples_per_shard"):
        value = config[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(field_name + " must be a positive integer")
    if config["samples_per_shard"] % config["batch_size"]:
        raise ValueError("samples_per_shard must be a multiple of batch_size")
    modes = config["mode_sequence"]
    if not isinstance(modes, list) or not modes or any(mode not in MODES for mode in modes):
        raise ValueError("mode_sequence must contain supported curriculum modes")
    if set(config["splits"]) != {"train", "validation", "test"}:
        raise ValueError("Provide train, validation, and test split configurations")
    seeds = []
    for split, options in config["splits"].items():
        if set(options) != {"seed", "samples"}:
            raise ValueError("Each split requires seed and samples")
        if isinstance(options["samples"], bool) or not isinstance(options["samples"], int) or options["samples"] < 1:
            raise ValueError("Split sample counts must be positive integers")
        make_generator(options["seed"], split=split,
                       low_snapshot_fraction=config["low_snapshot_fraction"],
                       low_snr_fraction=config["low_snr_fraction"])
        seeds.append(options["seed"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("Dataset splits must use distinct seeds")


def export_dataset(config: dict, output: str | Path, *, device="cpu") -> dict:
    """Write independent labeled NPZ splits with bounded memory use.

    Mode order continues across shard boundaries. A final partial batch has its
    own rounded quotas, as in online generation. Existing output is rejected.
    """
    _validate_config(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Dataset output already exists: " + str(output))
    # Validate the simulation device before creating the output directory.
    first = next(iter(config["splits"].values()))
    make_generator(first["seed"], device)
    output.mkdir(parents=True)
    manifest = {
        "format": "pgbsa_covariance_dataset_v1", "configuration": config,
        "environment": {"numpy": str(np.__version__), "torch": str(torch.__version__), "device": str(device)},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fields": {
            "in_mat": "float32 [N,2,16,16]; real then imaginary unit-trace covariance",
            "angles_deg": "float32 [N,20]; sorted active angles, zero padding",
            "active": "bool [N,20]; distinguishes labels from padding",
            "source_count": "int64 [N]", "snapshots": "int64 [N]",
            "snr_db": "float32 [N]; nominal per-source SNR, 0 placeholder for noise-only scenes",
            "source_power": "float32 [N,20]; expected source variances, zero for inactive slots",
            "noise_variance": "float32 [N]; expected noise variance per sensor",
            "geometry_family": "Unicode [N]", "mode": "Unicode [N]",
        },
        "splits": {},
    }
    for split, options in config["splits"].items():
        generator = make_generator(options["seed"], device, split,
                                   low_snapshot_fraction=config["low_snapshot_fraction"],
                                   low_snr_fraction=config["low_snr_fraction"])
        split_dir = output / split
        split_dir.mkdir()
        files = []
        batch_index = 0
        total = options["samples"]
        for shard_index, start in enumerate(range(0, total, config["samples_per_shard"])):
            size = min(config["samples_per_shard"], total-start)
            batches = []
            for offset in range(0, size, config["batch_size"]):
                count = min(config["batch_size"], size-offset)
                mode = config["mode_sequence"][batch_index % len(config["mode_sequence"])]
                tensors = generate_batch(count, device, generator, mode)
                keys = ("in_mat", "angles_deg", "active", "source_count", "snapshots", "snr_db")
                batch = {key: value.cpu().numpy() for key, value in zip(keys, tensors)}
                batch["source_power"] = batch["active"].astype(np.float32)
                batch["noise_variance"] = np.power(10., -batch["snr_db"]/10.).astype(np.float32)
                batch["geometry_family"] = np.asarray(generator.last_metadata["families"], dtype="U16")
                batch["mode"] = np.full(count, mode, dtype="U16")
                batches.append(batch)
                batch_index += 1
            arrays = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            shard = split_dir / f"part-{shard_index:05d}.npz"
            np.savez_compressed(shard, **arrays)
            files.append({"path": shard.relative_to(output).as_posix(), "samples": size,
                          "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()})
        manifest["splits"][split] = {"samples": total, "provenance": generator.provenance(), "files": files}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate labeled PG-BSA covariance datasets")
    parser.add_argument("--config", type=Path, required=True, help="Dataset JSON configuration")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument("--device", default="cpu", help="Simulation device, e.g. cpu or cuda:0")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = export_dataset(config, args.output, device=args.device)
    counts = {split: values["samples"] for split, values in manifest["splits"].items()}
    print(json.dumps({"output": str(args.output), "samples": counts}))


if __name__ == "__main__":
    main()
