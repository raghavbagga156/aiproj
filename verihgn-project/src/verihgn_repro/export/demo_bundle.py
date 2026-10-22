"""Export a small, self-contained bundle of genuine run artefacts for the
offline dashboard (saved-artifact demo mode).

Everything in the bundle is copied or converted from files a real run wrote
(eval/test_metrics.json, history.csv, predictions/*.npz, checkpoints/best.pt)
or from preprocessed data; each item carries a provenance tag. Nothing is
synthesised. Arrays are stored as base64 little-endian float32 so the browser
can read them with a Float32Array.
"""
import base64
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

BUNDLE_VERSION = 1


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def export_run(run_dir, processed_root, out_dir, max_samples: int = 6, include_inputs: int = 2) -> dict:
    run_dir, processed_root, out_dir = Path(run_dir), Path(processed_root), Path(out_dir)
    test_metrics = run_dir / "eval" / "test_metrics.json"
    if not test_metrics.exists():
        raise FileNotFoundError(f"{test_metrics} missing: run the test evaluation before exporting")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    manifest = {"bundle_version": BUNDLE_VERSION, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": None, "items": []}

    shutil.copyfile(test_metrics, out_dir / "metrics" / "test_metrics.json")
    tm = json.loads(test_metrics.read_text())
    manifest["run_id"] = tm.get("run_id")
    manifest["items"].append({"path": "metrics/test_metrics.json", "provenance": "measured: test evaluation (FP32)"})
    if (run_dir / "history.csv").exists():
        with open(run_dir / "history.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        (out_dir / "metrics" / "history.json").write_text(json.dumps(rows))
        manifest["items"].append({"path": "metrics/history.json", "provenance": "measured: training log"})
    for name in ("config.yaml", "env.json", "status.json"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out_dir / name)
            manifest["items"].append({"path": name, "provenance": "run record"})

    preds = sorted((run_dir / "predictions").glob("*.npz"))[:max_samples]
    for p in preds:
        sample = p.stem
        z = np.load(p)
        design = sample.split("-", 1)[1].rsplit("-util_", 1)[0]
        arrays, meta = _load_processed(processed_root, design, sample)
        idx = z["cell_index"]
        rec = {
            "sample": sample, "design": design, "grid_M": int(z["grid_gt"].shape[0]),
            "grid_N": int(z["grid_gt"].shape[1]), "label_norm_note": "values are normalised model targets",
            "grid_gt": _b64(z["grid_gt"]), "grid_pred": _b64(z["grid_pred"]), "grid_abs_err": _b64(z["grid_abs_err"]),
            "raw_h_overflow": _b64(z["raw_h_overflow"]) if "raw_h_overflow" in z else None,
            "raw_v_overflow": _b64(z["raw_v_overflow"]) if "raw_v_overflow" in z else None,
            "cells": {"x_um": _b64(arrays["cell_x"][idx, 0]), "y_um": _b64(arrays["cell_x"][idx, 1]),
                      "gt": _b64(z["cell_gt"]), "pred": _b64(z["cell_pred"]), "count": int(len(idx)),
                      "total_cells": int(meta["num_cells"])},
            "provenance": f"model output: {manifest['run_id']} best checkpoint, FP32 inference",
        }
        (out_dir / "predictions" / f"{sample}.json").write_text(json.dumps(rec))
        manifest["items"].append({"path": f"predictions/{sample}.json", "provenance": rec["provenance"]})

    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        s = torch.load(best, map_location="cpu", weights_only=False)
        (out_dir / "model").mkdir(exist_ok=True)
        torch.save({"model": s["model"], "config": s["config"], "epoch": s["epoch"],
                    "best_avg_spearman": s["best_avg_spearman"]}, out_dir / "model" / "best_state_dict.pt")
        manifest["items"].append({"path": "model/best_state_dict.pt", "provenance": f"weights of epoch {s['epoch']}",
                                  "sha256_16": _sha(out_dir / "model" / "best_state_dict.pt")})
    for p in preds[:include_inputs]:
        sample = p.stem
        design = sample.split("-", 1)[1].rsplit("-util_", 1)[0]
        src = processed_root / design / sample
        dst = out_dir / "inputs" / design / sample
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("sample.npz", "meta.json"):
            shutil.copyfile(src / f, dst / f)
        manifest["items"].append({"path": f"inputs/{design}/{sample}",
                                  "provenance": "preprocessed model input (for local re-inference)"})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    return {"out_dir": str(out_dir), "items": len(manifest["items"]), "size_mb": round(size_mb, 1)}


def _load_processed(processed_root: Path, design: str, sample: str):
    d = processed_root / design / sample
    with np.load(d / "sample.npz", allow_pickle=False) as z:
        arrays = {"cell_x": z["cell_x"]}
    return arrays, json.loads((d / "meta.json").read_text())
