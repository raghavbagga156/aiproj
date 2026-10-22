"""Design-level train/val/test splits.

official_seed42: identical to `VeriHGNDataset._split_designs` (official
  data/dataset.py:37-48): sorted design names, np.random.seed(42), shuffle,
  int(0.7n) / int(0.8n) cut points. Verified against the official method in tests.
pair_aware: same ratios, but designs with identical netlists (same
  `Netlist.signature()`) form one group and never cross split boundaries.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def official_split(designs) -> dict:
    np.random.seed(42)
    arr = np.array(sorted(designs))
    np.random.shuffle(arr)
    n = len(arr)
    return {"train": arr[:int(0.7 * n)].tolist(),
            "val": arr[int(0.7 * n):int(0.8 * n)].tolist(),
            "test": arr[int(0.8 * n):].tolist()}


def twin_groups(signatures: dict) -> list:
    by_sig = defaultdict(list)
    for d, s in signatures.items():
        by_sig[s].append(d)
    return sorted(sorted(g) for g in by_sig.values())


def pair_aware_split(signatures: dict, seed: int = 42, val_frac=0.1, test_frac=0.2) -> dict:
    """Shuffle twin groups (seed 42) and fill test, then val, up to the official
    design counts (int-cut on designs), keeping every group whole."""
    groups = twin_groups(signatures)
    n = sum(len(g) for g in groups)
    n_test = n - int((1 - test_frac) * n)      # same as official: n - int(0.8n)
    n_val = int((1 - test_frac) * n) - int((1 - test_frac - val_frac) * n)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    split = {"train": [], "val": [], "test": []}
    for gi in order:
        g = groups[gi]
        if len(split["test"]) + len(g) <= n_test and len(split["test"]) < n_test:
            split["test"] += g
        elif len(split["val"]) + len(g) <= n_val and len(split["val"]) < n_val:
            split["val"] += g
        else:
            split["train"] += g
    return {k: sorted(v) for k, v in split.items()}


def leakage_report(split: dict, signatures: dict) -> dict:
    sig_to_split = defaultdict(set)
    for part, ds in split.items():
        for d in ds:
            sig_to_split[signatures[d]].add(part)
    crossing = [sorted(d for d, s in signatures.items() if s == sig)
                for sig, parts in sig_to_split.items() if len(parts) > 1]
    all_designs = [d for ds in split.values() for d in ds]
    return {"duplicate_designs": len(all_designs) != len(set(all_designs)),
            "netlist_twins_across_splits": crossing,
            "leak_free": not crossing and len(all_designs) == len(set(all_designs))}


def build_split_manifest(name: str, split: dict, signatures: dict, samples_per_design: dict) -> dict:
    return {
        "name": name,
        "designs": split,
        "num_designs": {k: len(v) for k, v in split.items()},
        "num_samples": {k: sum(samples_per_design.get(d, 0) for d in v) for k, v in split.items()},
        "twin_groups": [g for g in twin_groups(signatures) if len(g) > 1],
        "leakage": leakage_report(split, signatures),
    }


def make_split_manifests(raw_root, scope: str, out_dir) -> dict:
    """Compute netlist signatures from graph_information and write both split manifests."""
    from .circuitnet import SAMPLES_PER_DESIGN, SCOPES, RawLayout
    from .netlist import load_netlist
    raw = RawLayout(Path(raw_root))
    sigs = {d: load_netlist(raw, d).signature() for d in SCOPES[scope]}
    out = {}
    for name, split in (("official_seed42", official_split(SCOPES[scope])), ("pair_aware", pair_aware_split(sigs))):
        m = build_split_manifest(name, split, sigs, SAMPLES_PER_DESIGN)
        m["scope"] = scope
        save_manifest(m, Path(out_dir) / f"{scope}__{name}.json")
        out[name] = m
    return out


def save_manifest(manifest: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=1))


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text())
