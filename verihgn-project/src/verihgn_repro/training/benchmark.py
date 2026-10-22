"""Precision benchmark (decision I-3): FP32 vs official FP16-AMP vs BF16-AMP.

Each (precision, seed) pair trains from scratch for a fixed small number of
epochs with otherwise identical settings, then the table compares FP32 metrics
of the last epoch, stability (non-finite losses skipped, GradScaler scale),
seconds per epoch and peak VRAM. The table is a measurement; the choice of
default precision is left to the user.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np

from ..config import Config
from .trainer import Trainer


def measure_peak_memory(cfg: Config, graphs_by_split: dict, precisions=("fp32", "fp16", "bf16"), device="cuda") -> list:
    """One full training step (forward, loss, backward, optimiser step) on the
    largest graph of every design, per precision; reports peak allocated VRAM and
    step time. Used by the smoke test before any long training."""
    import time

    import torch

    from .. import official
    from .trainer import _autocast, build_model
    collate = official.dataset_module().collate_fn
    lf = official.loss_module().verihgn_loss
    largest = {}
    for gs in graphs_by_split.values():
        for g in gs:
            if g.design_name not in largest or g["cell"].num_nodes > largest[g.design_name]["cell"].num_nodes:
                largest[g.design_name] = g
    dev = torch.device(device)
    rows = []
    for prec in precisions:
        if prec == "bf16" and dev.type == "cuda" and not torch.cuda.is_bf16_supported():
            continue
        c = copy.deepcopy(cfg)
        c.training.precision = prec
        model = build_model(c).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=c.training.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16" and dev.type == "cuda"))
        for design, g in sorted(largest.items()):
            batch = collate([g]).to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            opt.zero_grad()
            with _autocast(prec, dev):
                out = model(batch)
                loss = lf(out["cell_pred"].float(), batch["cell"].y, out["grid_pred"].float(), batch["grid_l0"].y,
                          alpha=c.training.wmse_alpha, lambda_var=c.training.lambda_var)["total"]
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            scaler.step(opt) if scaler.is_enabled() else opt.step()
            if scaler.is_enabled():
                scaler.update()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            rows.append({"precision": prec, "design": design, "sample": g.placement_name,
                         "cells": int(g["cell"].num_nodes), "grid": [int(g.grid_M), int(g.grid_N)],
                         "step_seconds": round(time.time() - t0, 3), "loss_finite": bool(torch.isfinite(loss)),
                         "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                         if dev.type == "cuda" else None})
            del batch, out, loss
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _f(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def repeatability_check(base_cfg: Config, graphs_by_split: dict, out_root, precision: str, epochs: int = 3,
                        seed: int = 42, device=None) -> dict:
    """Train the same seed twice from scratch; report how far the two runs drift
    (GPU scatter-add is not bit-deterministic)."""
    import torch
    params, vals = [], []
    for rep in (1, 2):
        cfg = copy.deepcopy(base_cfg)
        cfg.training.precision, cfg.training.seed, cfg.training.num_epochs = precision, seed, epochs
        cfg.training.patience, cfg.training.max_epochs_this_session = 10 ** 9, None
        cfg.experiment.name = f"{base_cfg.experiment.name}_repeat{rep}"
        tr = Trainer(cfg, {k: v for k, v in graphs_by_split.items() if k in ("train", "val")},
                     Path(out_root) / f"repeat_{precision}_{rep}", device=device)
        tr.run()
        params.append(torch.cat([p.detach().float().flatten().cpu() for p in tr.model.parameters()]))
        with open(Path(out_root) / f"repeat_{precision}_{rep}" / "history.csv", newline="") as f:
            vals.append([_f(r, "val_avg_spearman") for r in csv.DictReader(f)])
    return {"precision": precision, "epochs": epochs, "seed": seed,
            "max_abs_param_diff": float((params[0] - params[1]).abs().max()),
            "val_avg_spearman_run1": vals[0], "val_avg_spearman_run2": vals[1],
            "max_abs_val_spearman_diff": float(np.nanmax(np.abs(np.array(vals[0]) - np.array(vals[1]))))}


def run_precision_benchmark(base_cfg: Config, graphs_by_split: dict, out_root, epochs: int = 20,
                            seeds=(42, 43), precisions=("fp32", "fp16", "bf16"), device=None,
                            repeat_epochs: int = 3) -> dict:
    """Train/val only: the test split is never evaluated while choosing a precision."""
    out_root = Path(out_root)
    graphs_by_split = {k: v for k, v in graphs_by_split.items() if k in ("train", "val")}
    rows = []
    for prec in precisions:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg.training.precision = prec
            cfg.training.seed = seed
            cfg.training.num_epochs = epochs
            cfg.training.patience = 10 ** 9            # fixed-length runs for comparability
            cfg.training.max_epochs_this_session = None
            cfg.experiment.name = f"{base_cfg.experiment.name}_precbench"
            run_dir = out_root / cfg.run_id()
            tr = Trainer(cfg, graphs_by_split, run_dir, device=device)
            tr.run()
            with open(run_dir / "history.csv", newline="") as f:
                hist = list(csv.DictReader(f))
            last = hist[-1]
            fl = lambda k: float(last[k]) if last.get(k) not in (None, "") else float("nan")  # noqa: E731
            rows.append({
                "precision": prec, "seed": seed, "epochs": len(hist),
                "val_avg_spearman_last": fl("val_avg_spearman"),
                "val_avg_spearman_best": max(float(r["val_avg_spearman"]) for r in hist),
                "val_cell_mae_last": fl("val_cell_mae"), "val_grid_mae_last": fl("val_grid_mae"),
                "skipped_nonfinite_total": int(sum(int(r["skipped_nonfinite"]) for r in hist)),
                "min_grad_scale": min((float(r["grad_scale"]) for r in hist if r.get("grad_scale")), default=None),
                "sec_per_epoch_mean": float(np.mean([float(r["train_seconds"]) for r in hist])),
                "peak_alloc_gb_max": max((float(r.get("train_peak_allocated_gb") or 0) for r in hist), default=0.0),
                "gpu_util_mean": float(np.nanmean([_f(r, "gpu_util_mean") for r in hist])),
                "grad_norm_mean": float(np.nanmean([_f(r, "grad_norm_mean") for r in hist])),
                "grad_norm_max": float(np.nanmax([_f(r, "grad_norm_max") for r in hist])),
                "grad_norm_nonfinite_total": int(np.nansum([_f(r, "grad_norm_nonfinite") for r in hist])),
                "grad_clipped_frac_mean": float(np.nanmean([_f(r, "grad_clipped_frac") for r in hist])),
                "train_loss_last": _f(last, "train_loss"), "val_loss_last": _f(last, "val_loss"),
                "val_curve_avg_spearman": [_f(r, "val_avg_spearman") for r in hist],
                "run_dir": str(run_dir),
            })
    summary = {}
    for prec in precisions:
        rs = [r for r in rows if r["precision"] == prec]
        summary[prec] = {k: float(np.mean([r[k] for r in rs])) for k in
                         ("val_avg_spearman_last", "val_avg_spearman_best", "val_cell_mae_last",
                          "val_grid_mae_last", "sec_per_epoch_mean", "peak_alloc_gb_max", "gpu_util_mean",
                          "grad_norm_mean", "grad_norm_max", "grad_clipped_frac_mean", "train_loss_last",
                          "val_loss_last")}
        summary[prec]["grad_norm_nonfinite_total"] = int(sum(r["grad_norm_nonfinite_total"] for r in rs))
        summary[prec]["skipped_nonfinite_total"] = int(sum(r["skipped_nonfinite_total"] for r in rs))
        summary[prec]["val_avg_spearman_last_std"] = float(np.std([r["val_avg_spearman_last"] for r in rs]))
    repeats = [repeatability_check(base_cfg, graphs_by_split, out_root, p, epochs=repeat_epochs, device=device)
               for p in precisions] if repeat_epochs else []
    result = {"epochs": epochs, "seeds": list(seeds), "runs": rows, "summary": summary, "repeatability": repeats,
              "evaluation": "validation metrics computed in FP32; test split not used",
              "note": "measurement only; default precision is chosen by the user"}
    (out_root / "precision_benchmark.json").write_text(json.dumps(result, indent=1))
    return result
