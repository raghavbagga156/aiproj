"""FP32 evaluation with the official metrics (utils/evaluation.py) and the
official pooling: metrics over the concatenation of all samples in a split,
plus per-design and per-sample breakdowns (as official test.py)."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import official


@torch.no_grad()
def predict(model, batches, device) -> list:
    """Run the model in FP32 eval mode. Returns one record per sample."""
    model.eval()
    recs = []
    for data in batches:
        data = data.to(device)
        out = model(data)
        recs.append({
            "sample": data.placement_name[0] if isinstance(data.placement_name, list) else data.placement_name,
            "design": data.design_name[0] if isinstance(data.design_name, list) else data.design_name,
            "cell_pred": out["cell_pred"].float().cpu(), "cell_gt": data["cell"].y.float().cpu(),
            "grid_pred": out["grid_pred"].float().cpu(), "grid_gt": data["grid_l0"].y.float().cpu(),
            "grid_M": int(data.grid_M if not isinstance(data.grid_M, list) else data.grid_M[0]),
            "grid_N": int(data.grid_N if not isinstance(data.grid_N, list) else data.grid_N[0]),
        })
    return recs


def loss_of(recs, alpha=10.0, lambda_var=0.5) -> float:
    lf = official.loss_module().verihgn_loss
    vals = [lf(r["cell_pred"], r["cell_gt"], r["grid_pred"], r["grid_gt"], alpha=alpha,
               lambda_var=lambda_var)["total"].item() for r in recs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def metrics_of(recs) -> dict:
    ev = official.evaluation_module().evaluate_predictions
    if not recs:
        return {}
    cell = ev(torch.cat([r["cell_pred"] for r in recs]), torch.cat([r["cell_gt"] for r in recs]))
    grid = ev(torch.cat([r["grid_pred"] for r in recs]), torch.cat([r["grid_gt"] for r in recs]))
    return {"cell": cell, "grid": grid, "avg_spearman": (cell["spearman"] + grid["spearman"]) / 2,
            "num_samples": len(recs)}


def prediction_health(recs) -> dict:
    """Spread of the pooled predictions and the fraction that is exactly 0 (the
    official heads clamp at 0, so 1.0 means every raw output was <= 0)."""
    out = {}
    for nt in ("cell", "grid"):
        if not recs:
            continue
        p = torch.cat([r[f"{nt}_pred"] for r in recs]).double()
        out[f"{nt}_pred_std"] = float(p.std()) if p.numel() > 1 else 0.0
        out[f"{nt}_pred_zero_frac"] = float((p == 0).double().mean())
    return out


def breakdown(recs) -> dict:
    by_design = defaultdict(list)
    for r in recs:
        by_design[r["design"]].append(r)
    return {
        "per_design": {d: metrics_of(rs) for d, rs in sorted(by_design.items())},
        "per_sample": {r["sample"]: metrics_of([r]) for r in recs},
    }


def save_predictions(recs, graphs_by_name: dict, out_dir, cell_subsample: int = 20000, seed: int = 0) -> list:
    """Per-sample compressed predictions for the dashboard: full L0 grid maps
    (gt, pred, |err|, raw H and V) and a fixed random subsample of cells."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    written = []
    for r in recs:
        g = graphs_by_name[r["sample"]]
        M, N = r["grid_M"], r["grid_N"]
        n_cells = r["cell_pred"].numel()
        idx = np.sort(rng.choice(n_cells, min(cell_subsample, n_cells), replace=False))
        payload = {
            "grid_gt": r["grid_gt"].numpy().reshape(M, N).astype(np.float32),
            "grid_pred": r["grid_pred"].numpy().reshape(M, N).astype(np.float32),
            "cell_index": idx.astype(np.int64),
            "cell_gt": r["cell_gt"].numpy()[idx].astype(np.float32),
            "cell_pred": r["cell_pred"].numpy()[idx].astype(np.float32),
            "cell_tile": g["cell", "in", "grid_l0"].edge_index[1].numpy()[idx],
        }
        payload["grid_abs_err"] = np.abs(payload["grid_pred"] - payload["grid_gt"])
        if hasattr(g["grid_l0"], "raw_hv"):
            raw = g["grid_l0"].raw_hv.numpy()
            payload["raw_h_overflow"] = raw[:, 0].reshape(M, N)
            payload["raw_v_overflow"] = raw[:, 1].reshape(M, N)
        path = out_dir / f"{r['sample']}.npz"
        np.savez_compressed(path, **payload)
        written.append(str(path))
    return written
