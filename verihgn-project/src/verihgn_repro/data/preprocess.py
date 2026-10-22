"""Raw CircuitNet-ISPD15 -> one compact `sample.npz` + `meta.json` per placement.

The official per-sample format (`cell_features.npy`, ..., `layout.csv`) cannot be
used as the storage format: its `layout.csv` requires `H_MacroMargin` /
`V_MacroMargin` columns that do not exist in CircuitNet, and its label columns
are pre-averaged. We therefore store every array the official builder needs plus
the H and V label maps separately, and assemble graphs in `graph.py` with the
official helper functions.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..utils.safe_npy import load_npy
from . import features as F
from .circuitnet import RawLayout, SCOPES, parse_sample_name
from .netlist import load_netlist

PREP_VERSION = 1
_SRC_FILES = ("features.py", "netlist.py", "preprocess.py")


# Unresolved research assumptions (docs/RESEARCH_TO_CODE_MAPPING.md, class U/A). They are part of the
# preprocessing hash, so changing any of them produces a new processed/<hash>/ directory instead of
# overwriting existing data.
ASSUMPTIONS = {
    "knn_k": "geometric cell-cell edges = symmetric k-nearest neighbours on cell centres (U)",
    "macro_pin_direction": "direction of macro pins; no LEF in the dataset (U)",
    "pin_offset_norm": "pin dx,dy scaling: bbox = divide by net bbox w/h (U)",
    "wirelength_reference": "HPWL / net bbox from cell centres; pin offsets absent from data (A, fixed)",
    "label_maps": "CircuitNet GR overflow H and V maps (A/D, fixed)",
    "macro_margin": "H/V MacroMargin not available in CircuitNet; omitted, not approximated (fixed)",
}


def prep_config(knn_k: int = 8, macro_pin_direction: str = "input", with_code_features: bool = True,
                pin_offset_norm: str = "bbox") -> dict:
    return {"version": PREP_VERSION, "knn_k": knn_k, "macro_pin_direction": macro_pin_direction,
            "with_code_features": with_code_features, "pin_offset_norm": pin_offset_norm,
            "wirelength_reference": "cell_center", "label_maps": "gr_overflow_h_v"}


def prep_hash(cfg: dict) -> str:
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode())
    here = Path(__file__).parent
    for name in _SRC_FILES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:12]


@lru_cache(maxsize=4)
def _netlist(raw_root: str, design: str, macro_pin_direction: str):
    return load_netlist(RawLayout(Path(raw_root)), design, macro_pin_direction)


def process_sample(raw_root, sample: str, out_root, cfg: dict, phash: str, overwrite=False) -> dict:
    t0 = time.time()
    info = parse_sample_name(sample)
    design = info["design"]
    raw = RawLayout(Path(raw_root))
    out_dir = Path(out_root) / design / info["sample"]
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text())
        if meta.get("prep_hash") == phash and (out_dir / "sample.npz").exists():
            return {"sample": info["sample"], "status": "cached"}

    nl = _netlist(str(raw_root), design, cfg["macro_pin_direction"])
    gcell = load_npy(raw.placement_gcell(info["sample"]))
    micron = load_npy(raw.placement_micron(info["sample"]))
    h = np.asarray(load_npy(raw.routability("gr_h_overflow", info["sample"])), dtype=np.float32)
    v = np.asarray(load_npy(raw.routability("gr_v_overflow", info["sample"])), dtype=np.float32)
    if h.shape != v.shape:
        raise ValueError(f"{sample}: H {h.shape} != V {v.shape}")
    M, N = h.shape

    code_maps = None
    if cfg["with_code_features"]:
        code_maps = {k: load_npy(raw.routability(k, info["sample"])) for k in F.GRID_FEATURES_CODE_AVAILABLE}

    feats = F.compute_sample_features(nl, micron, gcell, M, N, knn_k=cfg["knn_k"], code_maps=code_maps,
                                      pin_offset_norm=cfg.get("pin_offset_norm", "bbox"))

    checks = {}
    cd_path = raw.routability("cell_density", info["sample"])
    if cd_path.exists():
        cd = np.asarray(load_npy(cd_path), dtype=np.float32)
        checks["cell_density_exact_match"] = bool(cd.shape == (M, N) and
                                                  np.array_equal(cd, feats["cell_density_recomputed"]))
    gb = np.asarray([gcell[n] for n in nl.cell_names])
    checks["gcell_within_grid"] = bool(gb[:, 0].min() >= 0 and gb[:, 1].min() >= 0 and
                                       gb[:, 2].max() <= M and gb[:, 3].max() <= N)

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v_ for k, v_ in feats.items() if k != "cell_density_recomputed"}
    arrays["label_h"] = h
    arrays["label_v"] = v
    tmp = out_dir / "sample.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out_dir / "sample.npz")

    meta = {
        "sample": info["sample"], "design": design, "util": info["util"], "index": info["index"],
        "grid_M": int(M), "grid_N": int(N),
        "num_cells": nl.num_cells, "num_nets": nl.num_nets, "num_pins": nl.num_pins,
        "num_geom_edges": int(feats["geom_edge_index"].shape[1]),
        "num_macros": int(nl.is_macro.sum()),
        "netlist_signature": nl.signature(),
        "label": {"source": "CircuitNet GR overflow_based horizontal / vertical", "stored": ["label_h", "label_v"],
                  "h_max": float(h.max()), "v_max": float(v.max()),
                  "nonzero_frac_mean": float(((h + v) > 0).mean())},
        "feature_columns": {"cell_x": F.CELL_FEATURES, "net_x": F.NET_FEATURES, "pin_x": F.PIN_FEATURES,
                            "geom_x": F.GEOM_FEATURES, "grid_x_paper": F.GRID_FEATURES_PAPER,
                            "grid_x_code": F.GRID_FEATURES_CODE_AVAILABLE if code_maps else []},
        "provenance": F.FEATURE_PROVENANCE,
        "checks": checks,
        "prep_config": cfg, "prep_hash": phash,
        "seconds": round(time.time() - t0, 2),
    }
    tmpj = out_dir / "meta.tmp.json"
    tmpj.write_text(json.dumps(meta, indent=1))
    os.replace(tmpj, meta_path)
    return {"sample": info["sample"], "status": "written", "checks": checks, "seconds": meta["seconds"]}


def run(raw_root, out_root, scope: str = "official9", cfg: dict | None = None,
        workers: int = 4, limit_per_design: int | None = None, samples: list | None = None) -> dict:
    cfg = cfg or prep_config()
    phash = prep_hash(cfg)
    designs = set(SCOPES[scope])
    names = samples or [s for s in RawLayout(Path(raw_root)).samples() if parse_sample_name(s)["design"] in designs]
    if limit_per_design:
        per = {}
        keep = []
        for s in sorted(names, key=lambda x: parse_sample_name(x)["index"]):
            d = parse_sample_name(s)["design"]
            if per.get(d, 0) < limit_per_design:
                per[d] = per.get(d, 0) + 1
                keep.append(s)
        names = keep
    # group by design so each worker reuses its cached netlist
    names = sorted(names, key=lambda s: (parse_sample_name(s)["design"], parse_sample_name(s)["index"]))
    results, failures = [], []
    t0 = time.time()
    if workers <= 1:
        for s in names:
            try:
                results.append(process_sample(raw_root, s, out_root, cfg, phash))
            except Exception as e:  # noqa: BLE001 - reported, never hidden
                failures.append({"sample": s, "error": f"{type(e).__name__}: {e}"})
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_sample, raw_root, s, out_root, cfg, phash): s for s in names}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    results.append(f.result())
                except Exception as e:  # noqa: BLE001
                    failures.append({"sample": futs[f], "error": f"{type(e).__name__}: {e}"})
                if i % 25 == 0:
                    print(f"  {i}/{len(names)} samples ({time.time() - t0:.0f}s)")
    from collections import Counter
    status = Counter(r["status"] for r in results)
    manifest = {
        "scope": scope, "prep_config": cfg, "prep_hash": phash,
        "requested": len(names), "ok": len(results), "failed": failures,
        "written": status.get("written", 0), "skipped_already_processed": status.get("cached", 0),
        "failed_count": len(failures),
        "cell_density_mismatch": [r["sample"] for r in results
                                  if r.get("checks", {}).get("cell_density_exact_match") is False],
        "seconds": round(time.time() - t0, 1),
    }
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / f"preprocess_manifest_{scope}.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_sample(sample_dir) -> tuple[dict, dict]:
    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "sample.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads((sample_dir / "meta.json").read_text())
    return arrays, meta


def list_processed(out_root, designs=None) -> list:
    out = []
    for meta in sorted(Path(out_root).glob("*/*/meta.json")):
        d = meta.parent.parent.name
        if designs is None or d in designs:
            out.append(meta.parent)
    return out
