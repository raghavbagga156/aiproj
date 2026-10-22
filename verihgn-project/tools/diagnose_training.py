"""Step-level training diagnostic (smoke-test investigation, docs/SMOKE_TRAINING_DIAGNOSIS.md).

Runs the *unchanged* Trainer._train_epoch one graph at a time, so every step is
the exact production step (autocast, GradScaler, clipping, AdamW). The
instrumentation is observation only:
  * loss components (official verihgn_loss, wrapped)
  * gradient norm before clipping (clip_grad_norm_ return value) and after it
  * GradScaler scale before/after and whether optimizer.step() actually ran
  * L2 / max-abs parameter change per step
  * raw head outputs before the official clamp(min=0): fraction <= 0
  * eval-mode FP32 loss on the same graph before and after the update
  * prediction / target statistics, and every `--metric-every` steps the
    validation correlations from the official code and from an independent
    implementation (numpy, float64, own tie-averaged ranks)
Also reports the loss of the trivial all-zero predictor on every training graph.

Nothing is written outside --out. It never touches checkpoints or reports.

usage:
  python tools/diagnose_training.py --processed <processed/ispd15/<hash>> --split <split.json>
         --out diag.json [--precisions fp32 fp16] [--steps 60] [--train-samples 6] [--device cpu|cuda]
"""
import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verihgn_repro import official  # noqa: E402
from verihgn_repro.config import load_config  # noqa: E402
from verihgn_repro.data.graph import build_graph, compute_c_max  # noqa: E402
from verihgn_repro.data.preprocess import list_processed, load_sample  # noqa: E402
from verihgn_repro.evaluation.evaluate import loss_of, metrics_of, predict  # noqa: E402
from verihgn_repro.training import trainer as T  # noqa: E402


# ------------------------------------------------------------ independent metrics
def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1-based), written without scipy."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _ranks_fast(a: np.ndarray) -> np.ndarray:
    """Vectorised version of _ranks (same result; checked in tests)."""
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
    """nan when either input is constant (correlation undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum() / den) if den > 0 else float("nan")


def spearman_indep(x, y) -> float:
    return pearson_indep(_ranks_fast(np.asarray(x, dtype=np.float64)), _ranks_fast(np.asarray(y, dtype=np.float64)))


def stats(t) -> dict:
    a = t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()),
            "frac_zero": float((a == 0).mean()), "frac_pos": float((a > 0).mean()), "n": int(a.size)}


# ------------------------------------------------------------ instrumentation
class Probe:
    """Observes one production training step; restores everything on close()."""

    def __init__(self, tr):
        self.tr = tr
        self.rec = {}
        self._orig_clip = torch.nn.utils.clip_grad_norm_
        lm = official.loss_module()
        self._lm, self._orig_loss = lm, lm.verihgn_loss
        probe = self

        def clip(params, max_norm, *a, **k):
            params = list(params)
            probe.rec["scale_before_step"] = tr.scaler.get_scale() if tr.scaler.is_enabled() else None
            pre = probe._orig_clip(params, max_norm, *a, **k)
            grads = [p.grad.detach().float() for p in params if p.grad is not None]
            post = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])) if grads else 0.0
            probe.rec["grad_norm_pre_clip"] = float(pre)
            probe.rec["grad_norm_post_clip"] = float(post)
            probe.rec["grad_exactly_zero"] = bool(all((g == 0).all() for g in grads))
            probe.rec["clipped"] = bool(float(pre) > max_norm)
            return pre

        def loss(*a, **k):
            d = probe._orig_loss(*a, **k)
            probe.rec["loss_components"] = {"total": float(d["total"]), "cell_wmse": d["cell_wmse"],
                                            "grid_wmse": d["grid_wmse"], "var_reg": d["var_reg"]}
            return d

        torch.nn.utils.clip_grad_norm_ = clip
        lm.verihgn_loss = loss
        m = tr.model
        self.hooks = [
            m.cell_head.mlp.register_forward_hook(self._head_hook("cell")),
            m.grid_head.mlp.register_forward_hook(self._head_hook("grid")),
            tr.optimizer.register_step_pre_hook(self._pre_step),
            tr.optimizer.register_step_post_hook(self._post_step),
        ]

    def _head_hook(self, name):
        def h(mod, inp, out):
            if mod.training:
                o = out.detach().float().squeeze(-1)
                self.rec[f"{name}_raw_head"] = {"frac_le0": float((o <= 0).float().mean()),
                                                "mean": float(o.mean()), "max": float(o.max())}
        return h

    def _pre_step(self, opt, args, kwargs):
        self.rec["optimizer_step_ran"] = True
        self._before = [p.detach().clone() for p in self.tr.model.parameters()]

    def _post_step(self, opt, args, kwargs):
        d = [(p.detach() - b).float() for p, b in zip(self.tr.model.parameters(), self._before)]
        self.rec["param_delta_l2"] = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in d])))
        self.rec["param_delta_maxabs"] = float(max(x.abs().max() for x in d))
        self._before = None

    def start(self):
        self.rec = {"optimizer_step_ran": False}

    def close(self):
        torch.nn.utils.clip_grad_norm_ = self._orig_clip
        self._lm.verihgn_loss = self._orig_loss
        for h in self.hooks:
            h.remove()


def zero_predictor_loss(g, alpha, lambda_var) -> float:
    lf = official.loss_module().verihgn_loss
    cy, gy = g["cell"].y.float(), g["grid_l0"].y.float()
    return float(lf(torch.zeros_like(cy), cy, torch.zeros_like(gy), gy, alpha=alpha, lambda_var=lambda_var)["total"])


def val_metrics(tr, val_batches) -> dict:
    recs = predict(tr.model, val_batches, tr.device)
    m = metrics_of(recs)
    out = {"official": {"cell": m["cell"], "grid": m["grid"], "avg_spearman": m["avg_spearman"]}}
    for nt in ("cell", "grid"):
        p = torch.cat([r[f"{nt}_pred"] for r in recs]).numpy()
        y = torch.cat([r[f"{nt}_gt"] for r in recs]).numpy()
        out[f"{nt}_pred"] = stats(p)
        out[f"{nt}_target"] = stats(y)
        out[f"{nt}_independent"] = {"pearson": pearson_indep(p, y), "spearman": spearman_indep(p, y)}
    out["val_loss_fp32"] = loss_of(recs)
    return out


# ------------------------------------------------------------ main
def pick_samples(dirs, k):
    """Round-robin over designs so a small subset still covers every train design."""
    by = defaultdict(list)
    for d in dirs:
        by[d.name.split("-")[1]].append(d)
    picked, i = [], 0
    while len(picked) < min(k, len(dirs)):
        for ds in sorted(by):
            if i < len(by[ds]) and len(picked) < k:
                picked.append(by[ds][i])
        i += 1
    return picked


def run(args):
    t_start = time.time()
    device = torch.device(args.device)
    split = json.loads(Path(args.split).read_text())
    cfg0 = load_config(ROOT / "configs" / "base.yaml")
    cfg0.experiment.label_norm = args.label_norm
    cfg0.experiment.grid_features = args.grid_features
    tdirs = list_processed(args.processed, set(split["designs"]["train"]))
    vdirs = list_processed(args.processed, set(split["designs"]["val"]))
    c_max = compute_c_max(tdirs) if args.label_norm == "per_dataset" else None   # over the FULL train split
    train_dirs = pick_samples(tdirs, args.train_samples)
    val_dirs = pick_samples(vdirs, args.val_samples)

    def build(ds):
        out = []
        for d in ds:
            a, m = load_sample(d)
            out.append(build_graph(a, m, grid_features=args.grid_features, label_norm=args.label_norm, c_max=c_max))
        return out
    train_g, val_g = build(train_dirs), build(val_dirs)
    report = {"args": vars(args), "c_max": c_max, "train_samples": [d.name for d in train_dirs],
              "val_samples": [d.name for d in val_dirs], "torch": torch.__version__,
              "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "targets": {g.placement_name: {"cell": stats(g["cell"].y), "grid": stats(g["grid_l0"].y)} for g in train_g},
              "zero_predictor_loss": {g.placement_name: zero_predictor_loss(g, cfg0.training.wmse_alpha,
                                                                            cfg0.training.lambda_var) for g in train_g},
              "runs": {}}
    collate = official.dataset_module().collate_fn
    val_batches = [collate([g]) for g in val_g]
    for prec in args.precisions:
        if prec != "fp32" and device.type != "cuda":
            raise SystemExit(f"{prec} needs CUDA: on CPU the trainer runs FP32, the comparison would be meaningless")
        cfg = copy.deepcopy(cfg0)
        cfg.training.precision = prec
        cfg.training.seed = args.seed
        tr = T.Trainer(cfg, {"train": train_g, "val": val_g}, ROOT / "_unused_diag_run_dir", device=str(device))
        tr._init_state()                                   # official init, AdamW, StepLR, GradScaler; no files written
        order = torch.randperm(len(train_g), generator=torch.Generator().manual_seed(args.seed)).tolist()
        steps, probe = [], Probe(tr)
        metrics = [{"step": 0, **val_metrics(tr, val_batches)}]
        try:
            for step in range(1, args.steps + 1):
                gi = order[(step - 1) % len(order)]
                if (step - 1) % len(order) == len(order) - 1:
                    order = torch.randperm(len(train_g), generator=torch.Generator().manual_seed(args.seed + step)).tolist()
                g = train_g[gi]
                one = [collate([g])]
                tr.batches = {"train": one, "val": val_batches}
                before = loss_of(predict(tr.model, one, device))
                probe.start()
                res = tr._train_epoch()                    # exactly one production training step
                rec = dict(probe.rec)
                after_recs = predict(tr.model, one, device)
                rec.update({"step": step, "sample": g.placement_name, "train_loss": res["train_loss"],
                            "skipped_nonfinite": res["skipped_nonfinite"],
                            "scale_after_step": tr.scaler.get_scale() if tr.scaler.is_enabled() else None,
                            "eval_loss_before": before, "eval_loss_after": loss_of(after_recs),
                            "zero_predictor_loss": report["zero_predictor_loss"][g.placement_name],
                            "pred_cell": stats(after_recs[0]["cell_pred"]),
                            "pred_grid": stats(after_recs[0]["grid_pred"])})
                if not rec.get("optimizer_step_ran"):
                    rec["param_delta_l2"] = 0.0
                steps.append(rec)
                if args.verbose:
                    print(f"[{prec}] step {step:3d} loss {rec['train_loss']:.6f} (zero-pred {rec['zero_predictor_loss']:.6f}) "
                          f"g_pre {rec.get('grad_norm_pre_clip', float('nan')):.3e} step_ran {rec['optimizer_step_ran']} "
                          f"scale {rec.get('scale_after_step')} dP {rec['param_delta_l2']:.2e} "
                          f"raw<=0 cell {rec.get('cell_raw_head', {}).get('frac_le0', float('nan')):.3f} "
                          f"grid {rec.get('grid_raw_head', {}).get('frac_le0', float('nan')):.3f} "
                          f"pred_std grid {rec['pred_grid']['std']:.2e}", flush=True)
                if step % args.metric_every == 0 or step == args.steps:
                    metrics.append({"step": step, **val_metrics(tr, val_batches)})
        finally:
            probe.close()
        first_dead = next((s["step"] for s in steps if s.get("grad_exactly_zero")), None)
        report["runs"][prec] = {
            "steps": steps, "val_metrics": metrics,
            "summary": {"steps": len(steps),
                        "optimizer_steps_ran": sum(s["optimizer_step_ran"] for s in steps),
                        "steps_with_exactly_zero_grad": sum(bool(s.get("grad_exactly_zero")) for s in steps),
                        "first_step_with_zero_grad": first_dead,
                        "final_frac_raw_le0": {k: steps[-1].get(f"{k}_raw_head", {}).get("frac_le0") for k in ("cell", "grid")},
                        "final_pred_std": {k: steps[-1][f"pred_{k}"]["std"] for k in ("cell", "grid")},
                        "final_val_official_avg_spearman": metrics[-1]["official"]["avg_spearman"]}}
        print(f"[{prec}] summary:", json.dumps(report["runs"][prec]["summary"]))
    report["seconds"] = round(time.time() - t_start, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print("wrote", args.out)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp32"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--train-samples", type=int, default=6)
    ap.add_argument("--val-samples", type=int, default=2)
    ap.add_argument("--metric-every", type=int, default=20)
    ap.add_argument("--label-norm", default="per_dataset", choices=["per_dataset", "per_sample"])
    ap.add_argument("--grid-features", default="paper_eq3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
