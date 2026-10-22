"""Per-head evaluation diagnostics for HEADPATH (docs/HEADPATH_PLAN.md). Additive only:
evaluate.predict / metrics_of / prediction_health and the official metric functions are used
unchanged. Three cell-prediction streams are kept apart:
  * official evaluation: the eval-mode forward with the official clamp (evaluate.predict)
  * eval-mode raw: the head output before the clamp, read by a forward hook (observation only)
  * training streams (raw and clamped) are logged per step by the Trainer, not here.
"""
import math
from collections import defaultdict

import numpy as np
import torch

from ..data.features import GRID_FEATURES_PAPER
from .evaluate import metrics_of, predict

HEADS = ("cell", "grid")
VAL_DESIGN = "mgc_matrix_mult_a"


# ------------------------------------------------------------ independent metrics (float64; NaN if undefined)
def _ranks_fast(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1-based); same as tools/diagnose_training._ranks_fast."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    start = np.r_[True, s[1:] != s[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    last = np.r_[first[1:], len(a)] - 1
    avg = (first + last) / 2.0 + 1.0
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = avg[gid]
    return ranks


def pearson_indep(x, y) -> float:
    """NaN when either input is constant: the correlation is undefined, not 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum() / den) if den > 0 else float("nan")


def spearman_indep(x, y) -> float:
    return pearson_indep(_ranks_fast(np.asarray(x, dtype=np.float64)), _ranks_fast(np.asarray(y, dtype=np.float64)))


# ------------------------------------------------------------ activity (kept separate from generalisation)
def activity_state(std: float, zero_frac: float, std_min: float = 1e-4, zero_max: float = 0.999) -> str:
    """DEAD: every prediction exactly 0. DEGENERATE: not dead, but std < std_min or zero
    fraction >= zero_max. HEALTHY: otherwise. Says nothing about correlation with the target."""
    if zero_frac >= 1.0:
        return "DEAD"
    if not (std >= std_min) or zero_frac >= zero_max:
        return "DEGENERATE"
    return "HEALTHY"


def stream_stats(p: torch.Tensor) -> dict:
    p = p.detach().double().flatten()
    return {"std": float(p.std()) if p.numel() > 1 else 0.0, "zero_frac": float((p == 0).double().mean()),
            "le0_frac": float((p <= 0).double().mean()), "mean": float(p.mean()), "max": float(p.max())}


# ------------------------------------------------------------ eval-mode prediction with raw head outputs
@torch.no_grad()
def predict_with_raw(model, batches, device) -> list:
    """evaluate.predict (official eval path, unchanged) plus the eval-mode raw head outputs,
    captured by read-only forward hooks on the heads' MLPs."""
    raw = {h: [] for h in HEADS}
    hooks = [getattr(model, f"{h}_head").mlp.register_forward_hook(
        lambda m, i, o, h=h: raw[h].append(o.detach().float().squeeze(-1).cpu())) for h in HEADS]
    try:
        recs = predict(model, batches, device)
    finally:
        for hk in hooks:
            hk.remove()
    if any(len(raw[h]) != len(recs) for h in HEADS):
        raise RuntimeError("raw-output capture does not match the predicted records")
    for r, c, g in zip(recs, raw["cell"], raw["grid"]):
        r["cell_raw"], r["grid_raw"] = c, g
    return recs


def cell_tiles(graph) -> torch.Tensor:
    """Index of each cell's L0 tile, in cell order (cell->grid_l0 'in' edges, graph.py)."""
    ei = graph["cell", "in", "grid_l0"].edge_index
    if not torch.equal(ei[0], torch.arange(ei.shape[1])):
        raise ValueError("cell->grid_l0 edges are not one per cell in cell order")
    return ei[1]


def tile_projection(recs, graphs_by_name) -> list:
    """Cell predictions read off the grid head: grid_pred[tile(cell)]. Cell labels are the tile's
    label (graph.py), so this shows how much cell-level ranking the grid branch already carries."""
    return [{**r, "cell_pred": r["grid_pred"][cell_tiles(graphs_by_name[r["sample"]])]} for r in recs]


def _cat(recs, key):
    return torch.cat([r[key] for r in recs])


def _head_block(recs, h, prefix, thresholds) -> dict:
    pred, gt, raw = _cat(recs, f"{h}_pred"), _cat(recs, f"{h}_gt"), _cat(recs, f"{h}_raw")
    st, rs = stream_stats(pred), stream_stats(raw)
    m = metrics_of(recs)[h]
    out = {f"{prefix}{h}_{k}": v for k, v in m.items()}                         # official metrics, pooled
    out.update({f"{prefix}{h}_spearman_indep": spearman_indep(pred.numpy(), gt.numpy()),
                f"{prefix}{h}_pearson_indep": pearson_indep(pred.numpy(), gt.numpy()),
                f"{prefix}{h}_pred_std": st["std"], f"{prefix}{h}_pred_zero_frac": st["zero_frac"],
                f"{prefix}{h}_pred_mean": st["mean"], f"{prefix}{h}_pred_max": st["max"],
                f"{prefix}{h}_activity": activity_state(st["std"], st["zero_frac"], thresholds["std_min"],
                                                        thresholds["zero_max"]),
                f"{prefix}{h}_evalraw_le0_frac": rs["le0_frac"], f"{prefix}{h}_evalraw_std": rs["std"],
                f"{prefix}{h}_evalraw_max": rs["max"]})
    return out


DEFAULT_ACTIVITY = {"std_min": 1e-4, "zero_max": 0.999}


def eval_summary(recs, graphs_by_name, thresholds=None) -> dict:
    """Flat per-head summary of one eval-mode pass over a validation set: pooled (official and
    independent metrics, activity, eval-mode raw stats), per design, matrix_mult_a only, and the
    tile-projection reference for the cell head."""
    th = thresholds or DEFAULT_ACTIVITY
    out = {"n_val": len(recs)}
    for h in HEADS:
        out.update(_head_block(recs, h, "val_", th))
    by = defaultdict(list)
    for r in recs:
        by[r["design"]].append(r)
    for d, rs in sorted(by.items()):
        for h in HEADS:
            pred, gt = _cat(rs, f"{h}_pred"), _cat(rs, f"{h}_gt")
            st = stream_stats(pred)
            out[f"design_{d}_{h}_spearman"] = metrics_of(rs)[h]["spearman"]
            out[f"design_{d}_{h}_spearman_indep"] = spearman_indep(pred.numpy(), gt.numpy())
            out[f"design_{d}_{h}_pred_zero_frac"] = st["zero_frac"]
            out[f"design_{d}_{h}_pred_std"] = st["std"]
    proj = tile_projection(recs, graphs_by_name)
    pc, gt = _cat(proj, "cell_pred"), _cat(proj, "cell_gt")
    out["tileproj_cell_spearman_indep"] = spearman_indep(pc.numpy(), gt.numpy())
    out["tileproj_cell_spearman"] = metrics_of(proj)["cell"]["spearman"]
    mm = [r for r in proj if r["design"] == VAL_DESIGN]
    if mm:
        out[f"tileproj_{VAL_DESIGN}_cell_spearman_indep"] = spearman_indep(_cat(mm, "cell_pred").numpy(),
                                                                            _cat(mm, "cell_gt").numpy())
    return out


BASELINE_NOTE = (
    "Each grid input feature is standardised per graph and per feature by the official builder, "
    "(x - mean) / max(std, 1e-6) (graph_builder.standardize). That map is strictly increasing within a graph, so a "
    "Spearman computed on ONE graph (sample_*) is identical to the one on the raw feature. Pooling several graphs "
    "(pooled_*, and design_* for designs with several validation graphs, e.g. 6 x matrix_mult_a) ranks values from "
    "different graphs on a common scale they do not share: a graph whose raw values are all high but spread out is "
    "mapped onto the same range as a graph with low values, so pooled baselines can differ from raw-feature "
    "baselines in either direction. design_*_mean_of_samples (mean of per-graph values) is standardisation-"
    "invariant. The model sees the same standardised features, so pooled baselines are comparable with the pooled "
    "cell-head Spearman as a 'what the inputs alone rank' reference, not as a raw-feature result. Descriptive only.")


def cell_feature_baseline(graphs) -> dict:
    """Non-learned cell baseline (pre-registered, descriptive only): for each paper-Eq.-3 grid input
    feature, as given to the model (grid_l0.x, standardised per graph), the Spearman between the
    feature of the cell's tile and the cell label: pooled over `graphs`, per design, per graph, and
    the per-design mean of per-graph values (standardisation-invariant; see BASELINE_NOTE)."""
    if graphs and graphs[0]["grid_l0"].x.shape[1] != len(GRID_FEATURES_PAPER):
        raise ValueError("cell_feature_baseline is defined for the paper_eq3 grid features")
    out = {"_note": BASELINE_NOTE}
    by = defaultdict(list)
    for g in graphs:
        d = g.design_name[0] if isinstance(g.design_name, list) else g.design_name
        by[d].append(g)
    for k, fname in enumerate(GRID_FEATURES_PAPER):
        def score(gs):
            f = torch.cat([g["grid_l0"].x[:, k][cell_tiles(g)] for g in gs]).numpy()
            y = torch.cat([g["cell"].y for g in gs]).numpy()
            return spearman_indep(f, y)
        out[f"pooled_{fname}"] = score(graphs)
        for d, gs in sorted(by.items()):
            out[f"design_{d}_{fname}"] = score(gs)
            per = []
            for g in gs:
                name = g.placement_name[0] if isinstance(g.placement_name, list) else g.placement_name
                out[f"sample_{name}_{fname}"] = v = score([g])
                per.append(v)
            out[f"design_{d}_{fname}_mean_of_samples"] = float(np.nanmean(per)) if np.isfinite(per).any() \
                else float("nan")
    return out
