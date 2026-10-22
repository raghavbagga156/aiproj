"""Full-dataset audits, streaming validation, profiling and the stage report.

Every function here measures; none of them fills in a value it did not observe.
Missing measurements are reported as "not run".
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from . import official
from .data import features as F
from .data.circuitnet import SAMPLES_PER_DESIGN, SCOPES, RawLayout, parse_sample_name
from .data.graph import GRID_FEATURE_SETS, build_graph, compute_c_max
from .data.preprocess import list_processed, load_sample
from .data.validate import official_model_check, validate_graph


# ----------------------------------------------------------------- raw discovery
def discovery_report(raw_root, scope: str) -> dict:
    raw = RawLayout(Path(raw_root))
    found = Counter(parse_sample_name(s)["design"] for s in raw.samples())
    expected = {d: SAMPLES_PER_DESIGN[d] for d in SCOPES[scope]}
    return {"scope": scope, "expected_total": sum(expected.values()), "found_total": sum(found.values()),
            "per_design": {d: {"expected": n, "found": found.get(d, 0)} for d, n in expected.items()},
            "complete": all(found.get(d, 0) == n for d, n in expected.items()),
            "unexpected_designs": sorted(set(found) - set(expected))}


# ----------------------------------------------------------- processed audit
def audit_processed(processed_root, scope: str, expected_samples=None) -> dict:
    """Check every processed sample: files, feature column names/order, array
    shapes and index ranges, H/V label alignment with the placement grid,
    finiteness, and the preprocessing self-checks."""
    designs = set(SCOPES[scope])
    dirs = list_processed(processed_root, designs)
    per = []
    for d in dirs:
        errs = []
        try:
            a, m = load_sample(d)
            M, N = m["grid_M"], m["grid_N"]
            C, E, P = m["num_cells"], m["num_nets"], m["num_pins"]
            fc = m["feature_columns"]
            for key, cols in (("cell_x", F.CELL_FEATURES), ("net_x", F.NET_FEATURES), ("pin_x", F.PIN_FEATURES),
                              ("geom_x", F.GEOM_FEATURES), ("grid_x_paper", F.GRID_FEATURES_PAPER)):
                if fc.get(key) != cols:
                    errs.append(f"{key} column order differs")
            if "grid_x_code" in a and fc.get("grid_x_code") != F.GRID_FEATURES_CODE_AVAILABLE:
                errs.append("grid_x_code column order differs")
            shapes = {"cell_x": (C, 13), "net_x": (E, 6), "pin_x": (P, 3), "pin_edge_index": (2, P),
                      "cell_tile": (C,), "grid_x_paper": (M * N, 5), "label_h": (M, N), "label_v": (M, N)}
            if "grid_x_code" in a:
                shapes["grid_x_code"] = (M * N, 3)
            for k, shp in shapes.items():
                if a[k].shape != shp:
                    errs.append(f"{k} shape {a[k].shape} != {shp}")
            G = a["geom_edge_index"]
            if a["geom_x"].shape != (G.shape[1], 4):
                errs.append("geom_x rows != geom edges")
            pe = a["pin_edge_index"]
            if pe[0].max() >= C or pe[1].max() >= E or pe.min() < 0:
                errs.append("pin edge index out of range")
            if G.size and (G.max() >= C or G.min() < 0):
                errs.append("geom edge index out of range")
            if a["cell_tile"].max() >= M * N or a["cell_tile"].min() < 0:
                errs.append("cell tile out of range")
            for k in ("cell_x", "net_x", "pin_x", "geom_x", "grid_x_paper", "label_h", "label_v"):
                if not np.isfinite(a[k]).all():
                    errs.append(f"{k} non-finite")
            if (a["label_h"] < 0).any() or (a["label_v"] < 0).any():
                errs.append("negative label")
            ch = m.get("checks", {})
            if ch.get("cell_density_exact_match") is not True:
                errs.append(f"cell_density check: {ch.get('cell_density_exact_match')}")
            if ch.get("gcell_within_grid") is not True:
                errs.append("placement outside label grid")
            per.append({"sample": m["sample"], "design": m["design"], "grid": [M, N], "cells": C,
                        "errors": errs, "h_nonzero": float((a["label_h"] > 0).mean()),
                        "v_nonzero": float((a["label_v"] > 0).mean()),
                        "hv_corr": _safe_corr(a["label_h"], a["label_v"])})
        except Exception as e:  # noqa: BLE001 - reported, not hidden
            per.append({"sample": d.name, "design": d.parent.name, "errors": [f"{type(e).__name__}: {e}"]})
    found = {p["sample"] for p in per}
    missing = sorted(set(expected_samples) - found) if expected_samples is not None else None
    by_design = defaultdict(lambda: {"ok": 0, "failed": 0})
    for p in per:
        by_design[p["design"]]["failed" if p["errors"] else "ok"] += 1
    return {"scope": scope, "audited": len(per), "ok": sum(not p["errors"] for p in per),
            "failed": [p for p in per if p["errors"]], "missing": missing,
            "per_design": dict(by_design), "per_sample": per,
            "passed": all(not p["errors"] for p in per) and not missing}


def _safe_corr(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------- streaming validation
def validate_variants(processed_root, split_manifests: dict, device="cpu", model_checks_per_design=1) -> dict:
    """Validate every sample under every (grid feature set x normalisation) combination
    without keeping graphs in memory. per_dataset uses each split's train c_max."""
    root = Path(processed_root)
    cmax = {}
    for name, man in split_manifests.items():
        cmax[name] = compute_c_max(list_processed(root, set(man["designs"]["train"])))
    combos = [("paper_eq3", 5, "per_sample", None)]
    combos += [("paper_eq3", 5, "per_dataset", n) for n in split_manifests]
    combos += [("official_code_available", 3, "per_sample", None)]
    combos += [("official_code_available", 3, "per_dataset", n) for n in split_manifests]
    results = {f"{gf}|{norm}|{split or 'any'}": {"ok": 0, "failed": [], "skipped": []} for gf, _, norm, split in combos}
    checked_models = Counter()
    model_checks = []
    designs = set().union(*[set(sum(m["designs"].values(), [])) for m in split_manifests.values()])
    t0 = time.time()
    dirs = list_processed(root, designs)
    for i, d in enumerate(dirs, 1):
        a, m = load_sample(d)
        for gf, dim, norm, split in combos:
            key = f"{gf}|{norm}|{split or 'any'}"
            if GRID_FEATURE_SETS[gf] not in a:
                results[key]["skipped"].append({"sample": m["sample"], "reason": f"no {GRID_FEATURE_SETS[gf]}"})
                continue
            try:
                g = build_graph(a, m, grid_features=gf, label_norm=norm, c_max=cmax.get(split))
                r = validate_graph(g, dim)
                if r["errors"]:
                    results[key]["failed"].append({"sample": m["sample"], "errors": r["errors"]})
                else:
                    results[key]["ok"] += 1
                if norm == "per_sample" and checked_models[(gf, m["design"])] < model_checks_per_design:
                    c = official_model_check(g, dim, device=device)
                    c.update({"sample": m["sample"], "grid_features": gf})
                    model_checks.append(c)
                    checked_models[(gf, m["design"])] += 1
                del g
            except Exception as e:  # noqa: BLE001
                results[key]["failed"].append({"sample": m["sample"], "errors": [f"{type(e).__name__}: {e}"]})
        if i % 25 == 0:
            print(f"  validated {i}/{len(dirs)} samples ({time.time() - t0:.0f}s)")
    if device != "cpu":
        torch.cuda.empty_cache()
    passed = all(not v["failed"] and not v["skipped"] for v in results.values()) and all(c["ok"] for c in model_checks)
    return {"samples": len(dirs), "c_max": cmax, "variants": {k: {"ok": v["ok"], "failed_count": len(v["failed"]),
            "skipped_count": len(v["skipped"]), "failed": v["failed"][:20], "skipped": v["skipped"][:20]}
            for k, v in results.items()}, "official_model_checks": model_checks, "passed": passed,
            "seconds": round(time.time() - t0)}


def split_report(split_manifests: dict, processed_root, scope: str) -> dict:
    root = Path(processed_root)
    all_designs = set(SCOPES[scope])
    out = {}
    for name, man in split_manifests.items():
        assigned = sum(man["designs"].values(), [])
        counts = {p: len(list_processed(root, set(ds))) for p, ds in man["designs"].items()}
        out[name] = {"designs": man["designs"], "processed_samples": counts,
                     "all_designs_assigned_once": sorted(assigned) == sorted(all_designs),
                     "leakage": man["leakage"]}
    return out


# ------------------------------------------------------------- profiling
def graph_memory_profile(g) -> dict:
    rel = {}
    for et in g.edge_types:
        st = g[et]
        rel["__".join(et)] = sum(v.numel() * v.element_size() for v in st.values() if torch.is_tensor(v))
    nodes = {nt: sum(v.numel() * v.element_size() for v in g[nt].values() if torch.is_tensor(v)) for nt in g.node_types}
    total = sum(rel.values()) + sum(nodes.values())
    return {"total_mb": round(total / 1e6, 1), "nodes_mb": {k: round(v / 1e6, 2) for k, v in nodes.items()},
            "edges_mb": {k: round(v / 1e6, 2) for k, v in rel.items()}}


def data_loading_profile(processed_root, designs, device="cuda", per_design=2) -> list:
    """Per sample: Drive read + decompress, graph build, collate, host->device copy."""
    collate = official.dataset_module().collate_fn
    rows = []
    for design in sorted(designs):
        for d in list_processed(Path(processed_root), {design})[:per_design]:
            t = time.perf_counter(); a, m = load_sample(d); t_load = time.perf_counter() - t
            t = time.perf_counter(); g = build_graph(a, m); t_build = time.perf_counter() - t
            t = time.perf_counter(); b = collate([g]); t_coll = time.perf_counter() - t
            t_h2d = None
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize(); t = time.perf_counter()
                bg = b.to(device); torch.cuda.synchronize(); t_h2d = time.perf_counter() - t
                del bg
            rows.append({"design": design, "sample": m["sample"], "npz_mb": round((d / "sample.npz").stat().st_size / 1e6, 1),
                         "load_s": round(t_load, 3), "build_s": round(t_build, 3), "collate_s": round(t_coll, 3),
                         "h2d_s": None if t_h2d is None else round(t_h2d, 3), **graph_memory_profile(g)})
    return rows


# ---------------------------------------------------------- persistence
def persistence_check(paths: dict) -> dict:
    """List what is actually on Drive (count, bytes) under each given path."""
    out = {}
    for name, p in paths.items():
        p = Path(p)
        files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        out[name] = {"path": str(p), "exists": p.exists(), "files": len(files),
                     "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}
    return out


# ---------------------------------------------------------- stage report
def summarize_reports(reports_dir, out_path=None) -> str:
    """Collect every JSON report written by notebooks 01-05 into one markdown file.
    Sections whose report is missing say 'not run'."""
    R = Path(reports_dir)

    def load(name):
        p = R / name
        return json.loads(p.read_text()) if p.exists() else None

    lines = ["# Stage report (generated from MyDrive/VeriHGN/reports)", ""]
    env = load("01_environment.json")
    lines += ["## 1. Environment, Drive access, storage"]
    if env:
        hw = env["env"]["hardware"]
        lines += [f"- GPU: {hw['name']} · VRAM {hw['total_vram_gb']} GB · CC {hw['compute_capability']} · "
                  f"bf16 {hw['bf16_supported']} · CUDA runtime {hw['cuda_runtime']} · driver {hw['driver']}",
                  f"- torch {hw['torch_version']} · PyG {env['env'].get('torch_geometric')} · scatter {env['env'].get('scatter_impl')}",
                  f"- Drive access: ok={env['access']['ok']} via {env['access']['found'].get('method')}",
                  f"- Storage: {json.dumps(env['storage'])}"]
        for a in env["access"].get("archives", []):
            lines.append(f"  - {a['archive']}: size {a['size']} expected {a['expected']} ok={a['ok']}")
    else:
        lines.append("- not run")
    for title, name in (("2. Raw data discovery and audit", "02_audit_official9.json"),
                        ("3. Preprocessing", "03_preprocessing_official9.json"),
                        ("4. Processed-data audit", "03_processed_audit_official9.json"),
                        ("5. Splits", "03_splits_official9.json"),
                        ("6. Graph validation (all variants)", "04_validation_official9.json"),
                        ("7. Data loading and graph memory", "05_data_loading.json"),
                        ("8. Peak VRAM per design", "05_peak_memory.json"),
                        ("9. Smoke test", "05_smoke.json"),
                        ("10. Precision benchmark", "05_precision_benchmark.json"),
                        ("11. Drive persistence", "05_persistence.json")):
        rep = load(name)
        lines += ["", f"## {title}"]
        if rep is None:
            lines.append("- not run")
            continue
        lines.append("```json")
        lines.append(json.dumps(_trim(rep), indent=1)[:6000])
        lines.append("```")
    text = "\n".join(lines) + "\n"
    if out_path:
        Path(out_path).write_text(text)
    return text


def _trim(obj, depth=0):
    if isinstance(obj, dict):
        return {k: _trim(v, depth + 1) for k, v in obj.items() if k not in ("per_sample",)}
    if isinstance(obj, list) and len(obj) > 40:
        return [_trim(x, depth + 1) for x in obj[:40]] + [f"... {len(obj) - 40} more"]
    if isinstance(obj, list):
        return [_trim(x, depth + 1) for x in obj]
    return obj
