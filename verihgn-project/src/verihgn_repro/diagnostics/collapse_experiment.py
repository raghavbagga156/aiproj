"""Controlled prediction-collapse experiment (docs/COLLAPSE_EXPERIMENT_PLAN.md).

Question: why did the 251-graph smoke run collapse (every raw head output <= 0,
gradient exactly 0) while the 6-graph Part B diagnostic kept learning?

Every run uses the normal training engine (Trainer.run): official model, loss,
optimizer, clipping, labels and features. Only the training subset, the seed,
the precision and the run length differ between runs. Diagnostic-only settings,
identical in all runs:
  * patience disabled (fixed-length runs, as in the precision benchmark)
  * periodic checkpoints off (keep_every = 0); last.pt / best.pt are still written
  * training.step_log = True (per-step steps.csv; observation only)
  * labels use the c_max of the FULL official training split for every arm, so the
    targets are identical whichever subset an arm trains on
The official test split is never loaded.

Selection rules are deterministic and fixed before any result is seen:
  * held out (validation only): the middle sample of each training design
  * validation: `val_from_val_design` evenly spaced samples of the official
    validation design + the held-out sample of every training design
  * FULL:      all non-held-out training samples, 1 epoch
  * DIVERSE6:  the first non-held-out sample of each training design (Part B's rule)
  * DIVERSE36: `per_design_mid` evenly spaced non-held-out samples per training design
  * SINGLE6:   6 evenly spaced samples of ONE training design, the design whose mean
               grid non-zero label fraction is the lower median of the training designs
  * arms other than FULL run ceil(|FULL| / n) epochs, so all arms take ~|FULL| steps
  * replicas:  the smoke configuration (full official training split incl. the held-out
               samples, 1 epoch) in FP16 and in FP32 for seeds 42, 43, 44 (seed 42 FP16 is
               the exact smoke run), validated on the official-validation-design samples only
"""
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..data.graph import compute_c_max, grid_label
from ..data.preprocess import list_processed

ARMS = ("FULL", "DIVERSE6", "DIVERSE36", "SINGLE6")


def _spread(items: list, k: int, interior: bool = False) -> list:
    """k evenly spaced items (deterministic). interior=True excludes both ends."""
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    if interior:
        idx = np.round(np.linspace(0, len(items) - 1, k + 2))[1:-1]
    else:
        idx = np.round(np.linspace(0, len(items) - 1, k))
    out = []
    for i in idx.astype(int):
        if items[i] not in out:
            out.append(items[i])
    return out


def _nonzero_fraction(sample_dir: Path) -> float:
    with np.load(sample_dir / "sample.npz", allow_pickle=False) as z:
        return float((grid_label(z["label_h"], z["label_v"]) > 0).mean())


def select_samples(names_by_design: dict, train_designs, val_designs, single_design: str,
                   holdout_per_design: int = 1, val_from_val_design: int = 6, per_design_mid: int = 6,
                   single_n: int = 6) -> dict:
    """The selection rules, on sample names only (deterministic; no data, no RNG).
    Names are sorted per design exactly as list_processed sorts them."""
    train_designs, val_designs = sorted(train_designs), sorted(val_designs)
    if single_design not in train_designs:
        raise ValueError(f"single_design {single_design} is not a training design")
    by = {d: sorted(names_by_design[d]) for d in train_designs + val_designs}
    for d, s in by.items():
        if not s:
            raise ValueError(f"no samples for {d}")
    holdout = {d: _spread(by[d], holdout_per_design, interior=True) for d in train_designs}
    pool = {d: [s for s in by[d] if s not in holdout[d]] for d in train_designs}
    val_main = [s for d in val_designs for s in _spread(by[d], val_from_val_design)]
    arms = {
        "FULL": [s for d in train_designs for s in pool[d]],
        "DIVERSE6": [pool[d][0] for d in train_designs],
        "DIVERSE36": [s for d in train_designs for s in _spread(pool[d], per_design_mid)],
        "SINGLE6": _spread(pool[single_design], single_n),
    }
    return {"train_designs": train_designs, "val_designs": val_designs, "single_design": single_design,
            "all_train": [s for d in train_designs for s in by[d]], "holdout": holdout, "val_main": val_main,
            "val": val_main + [s for d in train_designs for s in holdout[d]], "arms": arms}


def design_of(name: str) -> str:
    return name.split("-")[1]


def plan_experiment(processed_root, manifest: dict, seeds=(42, 43, 44), holdout_per_design: int = 1,
                    val_from_val_design: int = 6, per_design_mid: int = 6, single_n: int = 6,
                    single_design: str = "mgc_fft_a", replica_seeds=(42, 43, 44), arm_precision: str = "fp32",
                    expected: dict | None = None) -> dict:
    """Plan on the processed data. `expected` (configs/collapse_experiment_manifest.json) is the
    pre-registered selection; any difference raises before anything is trained."""
    processed_root = Path(processed_root)
    test_designs = sorted(manifest["designs"].get("test", []))
    if set(manifest["designs"]["train"]) & set(test_designs) or set(manifest["designs"]["val"]) & set(test_designs):
        raise ValueError("train/val and test designs overlap")
    dirs = {d: list_processed(processed_root, {d}) for d in manifest["designs"]["train"] + manifest["designs"]["val"]}
    for d, ds in dirs.items():
        if [p.name for p in ds] != sorted(p.name for p in ds):
            raise ValueError(f"{d}: list_processed order is not the sorted name order")
    sel = select_samples({d: [p.name for p in ds] for d, ds in dirs.items()}, manifest["designs"]["train"],
                         manifest["designs"]["val"], single_design, holdout_per_design, val_from_val_design,
                         per_design_mid, single_n)
    path_of = {p.name: p for ds in dirs.values() for p in ds}
    all_names = sel["all_train"] + sel["val_main"]
    if len(all_names) != len(set(all_names)):
        raise ValueError("duplicate sample names across designs")
    c_max = compute_c_max([path_of[n] for n in sel["all_train"]])     # full official training split (= smoke run)
    pool = {d: [n for n in sel["arms"]["FULL"] if design_of(n) == d] for d in sel["train_designs"]}
    sparsity = {d: float(np.mean([_nonzero_fraction(path_of[n]) for n in pool[d]])) for d in sel["train_designs"]}
    ranked = sorted(sel["train_designs"], key=lambda d: (sparsity[d], d))
    plan = {"c_max": c_max, **build_runs(sel, test_designs, seeds, replica_seeds, arm_precision),
            "grid_nonzero_fraction_by_design": sparsity,
            "lower_median_sparsity_design": ranked[(len(ranked) - 1) // 2],
            "sample_dirs": {n: str(path_of[n]) for n in sel["all_train"] + sel["val_main"]}}
    if expected is not None:
        diff = manifest_diff(plan, expected)
        if diff:
            raise ValueError(f"plan differs from the pre-registered manifest: {diff}")
    return plan


def build_runs(sel: dict, test_designs, seeds=(42, 43, 44), replica_seeds=(42, 43, 44),
               arm_precision: str = "fp32") -> dict:
    """Runs and step budgets from a selection: FULL = 1 epoch; every other arm runs
    ceil(|FULL| / n) epochs, i.e. the smallest whole number of epochs reaching |FULL| steps."""
    budget = len(sel["arms"]["FULL"])
    runs = []                               # replicas first: cheapest and most informative
    for prec in ("fp16", "fp32"):
        for seed in replica_seeds:
            runs.append({"name": f"REPLICA__{prec}__seed{seed}", "arm": "REPLICA", "seed": seed,
                         "precision": prec, "epochs": 1, "train": sel["all_train"], "val": sel["val_main"],
                         "steps": len(sel["all_train"])})
    for arm in ARMS:
        n = len(sel["arms"][arm])
        epochs = 1 if arm == "FULL" else math.ceil(budget / n)
        for seed in seeds:
            runs.append({"name": f"{arm}__{arm_precision}__seed{seed}", "arm": arm, "seed": seed,
                         "precision": arm_precision, "epochs": epochs, "train": sel["arms"][arm],
                         "val": sel["val"], "steps": epochs * n})
    for r in runs:
        if set(r["train"]) & set(r["val"]):
            raise ValueError(f"{r['name']}: train/val overlap")
        if {design_of(n) for n in r["train"] + r["val"]} & set(test_designs):
            raise ValueError(f"{r['name']}: uses a test design")
        if len(r["train"]) != len(set(r["train"])):
            raise ValueError(f"{r['name']}: duplicate training sample")
    return {"train_designs": sel["train_designs"], "val_designs": sel["val_designs"],
            "test_designs": sorted(test_designs), "holdout": sel["holdout"], "val": sel["val"],
            "val_main": sel["val_main"], "arms": sel["arms"], "single_design": sel["single_design"],
            "budget_steps": budget, "common_steps": min(r["steps"] for r in runs), "runs": runs}


def manifest_from_names(names_by_design: dict, split_manifest: dict, **kw) -> dict:
    """The pre-registered manifest from sample names alone (no data needed)."""
    sel_kw = {k: kw.pop(k) for k in ("holdout_per_design", "val_from_val_design", "per_design_mid", "single_n")
              if k in kw}
    sel = select_samples(names_by_design, split_manifest["designs"]["train"], split_manifest["designs"]["val"],
                         kw.pop("single_design", "mgc_fft_a"), **sel_kw)
    return manifest_of(build_runs(sel, split_manifest["designs"].get("test", []), **kw))


MANIFEST_KEYS = ("train_designs", "val_designs", "test_designs", "single_design", "holdout", "val_main", "val",
                 "arms", "budget_steps", "common_steps")


def manifest_of(plan: dict) -> dict:
    """The pre-registrable part of a plan (selections, budgets, runs; no data-derived numbers)."""
    out = {k: plan[k] for k in MANIFEST_KEYS}
    out["runs"] = [{k: r[k] for k in ("name", "arm", "seed", "precision", "epochs", "steps")}
                   | {"n_train": len(r["train"]), "n_val": len(r["val"])} for r in plan["runs"]]
    return out


def manifest_diff(plan: dict, expected: dict) -> list:
    """Keys of the pre-registered manifest whose value differs from the plan. Only the
    selection/budget keys are compared; informational keys (comment, split,
    samples_per_design) are ignored, but every compared key must be present."""
    got = manifest_of(plan)
    return [k for k in (*MANIFEST_KEYS, "runs") if got.get(k) != expected.get(k)]


# ------------------------------------------------------------------ execution
def run_config(base_cfg: Config, run: dict) -> Config:
    cfg = copy.deepcopy(base_cfg)
    t = cfg.training
    t.seed, t.precision, t.num_epochs = run["seed"], run["precision"], run["epochs"]
    t.patience = 10 ** 9
    t.keep_every = 0
    t.step_log = True
    t.max_epochs_this_session = None
    t.keep_dataset_on_gpu = "false"         # as in the smoke run (dataset_resident_on_gpu: false)
    cfg.experiment.name = f"collapsediag_{run['arm']}"
    return cfg


def _rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _collapse(steps: list, min_tail: int):
    zero = [r["grad_exactly_zero"] == "True" for r in steps]
    tail = 0
    for z in reversed(zero):
        if not z:
            break
        tail += 1
    step = int(steps[len(steps) - tail]["global_step"]) if tail else None
    return zero, tail, step, bool(step is not None and (tail >= min_tail or tail == len(zero)))


def summarize_run(run_dir: Path, min_tail: int = 20, common_steps: int | None = None) -> dict:
    """Collapse statistics from steps.csv / history.csv.
    collapse_step = first global step after which every logged step had an exactly-zero
    gradient; 'collapsed' requires that tail to be >= min_tail steps (or the whole run).
    The same rule applied to the first `common_steps` steps only gives a comparison at an
    identical step budget for every run."""
    steps = _rows(run_dir / "steps.csv")
    hist = _rows(run_dir / "history.csv")
    zero, tail, collapse_step, collapsed = _collapse(steps, min_tail)
    common = None
    if common_steps:
        _, ctail, cstep, ccol = _collapse([r for r in steps if int(r["global_step"]) <= common_steps], min_tail)
        common = {"common_steps": common_steps, "collapsed": ccol, "collapse_step": cstep, "zero_tail_len": ctail}
    gn = [_f(r["grad_norm"]) for r in steps]
    imax = int(np.nanargmax(gn)) if steps and not all(np.isnan(gn)) else None
    both = [max(_f(r.get("cell_raw_le0")), _f(r.get("grid_raw_le0"))) for r in steps]
    return {
        "steps_logged": len(steps), "zero_grad_steps": int(sum(zero)),
        "collapse_step": collapse_step, "zero_tail_len": tail, "collapsed": collapsed,
        "at_common_steps": common,
        "max_grad_norm": gn[imax] if imax is not None else None,
        "max_grad_step": int(steps[imax]["global_step"]) if imax is not None else None,
        "max_grad_sample": steps[imax]["sample"] if imax is not None else None,
        "min_cell_raw_le0": float(np.nanmin([_f(r.get("cell_raw_le0")) for r in steps])) if steps else None,
        "max_grid_raw_le0": float(np.nanmax([_f(r.get("grid_raw_le0")) for r in steps])) if steps else None,
        "steps_both_heads_ge_0999": int(sum(b >= 0.999 for b in both)),
        "first_samples": [r["sample"] for r in steps[:10]],
        "scaler_skipped_steps": int(sum(r.get("step_skipped_by_scaler") == "True" for r in steps)),
        "final_val_avg_spearman": _f(hist[-1].get("val_avg_spearman")) if hist else None,
        "best_val_avg_spearman": max((_f(r.get("val_avg_spearman")) for r in hist), default=None),
        "final_val_pred_zero_frac": {k: _f(hist[-1].get(f"val_{k}_pred_zero_frac")) for k in ("cell", "grid")} if hist else None,
    }


def run_plan(plan: dict, base_cfg: Config, graphs_by_name: dict, out_root, device=None, only=None,
             report_path=None) -> dict:
    """Run (or resume) every run in the plan with the normal Trainer. `graphs_by_name`
    must contain every sample named in the plan, built with plan['c_max']."""
    from ..evaluation.evaluate import breakdown, predict
    from ..training.trainer import Trainer
    out_root = Path(out_root)
    results = []
    for run in plan["runs"]:
        if only and run["name"] not in only and run["arm"] not in only:
            continue
        cfg = run_config(base_cfg, run)
        run_dir = out_root / run["name"]
        graphs = {"train": [graphs_by_name[n] for n in run["train"]], "val": [graphs_by_name[n] for n in run["val"]]}
        t0 = time.time()
        tr = Trainer(cfg, graphs, run_dir, device=device,
                     extra_meta={"collapse_experiment": {k: run[k] for k in ("arm", "seed", "precision", "epochs")}})
        tr.run()
        bd = breakdown(predict(tr.model, tr.batches["val"], tr.device))
        res = {**{k: run[k] for k in ("name", "arm", "seed", "precision", "epochs", "steps")},
               "n_train": len(run["train"]), "n_val": len(run["val"]),
               "seconds_this_session": round(time.time() - t0, 1),
               **summarize_run(run_dir, common_steps=plan.get("common_steps")),
               "final_val_per_design": {d: {"avg_spearman": m["avg_spearman"],
                                            "cell_spearman": m["cell"]["spearman"], "grid_spearman": m["grid"]["spearman"],
                                            "num_samples": m["num_samples"]} for d, m in bd["per_design"].items()}}
        results.append(res)
        print(f"[{run['name']}] collapsed={res['collapsed']} collapse_step={res['collapse_step']} "
              f"max_grad={res['max_grad_norm']} final_val_avgS={res['final_val_avg_spearman']}", flush=True)
        if report_path:
            _write(report_path, plan, results)
    return {"plan": {k: v for k, v in plan.items() if k != "sample_dirs"}, "results": results,
            "by_arm": aggregate(results)}


NOTE = ("Primary outcome: collapse from steps.csv (training diagnostics: exactly-zero gradient steps), "
        "compared at the common step budget. Validation metrics are secondary and comparable only between runs "
        "with the same validation set: REPLICA uses rows 1-6 (6 x matrix_mult_a), all other arms rows 1-12.")


def aggregate(results: list) -> dict:
    out = {"_note": NOTE}
    for r in results:
        a = out.setdefault(f"{r['arm']}__{r['precision']}", {"runs": 0, "collapsed": 0, "collapsed_at_common_steps": 0,
                                                             "collapse_steps": [], "final_val_avg_spearman": [],
                                                             "n_val": r.get("n_val")})
        a["runs"] += 1
        a["collapsed"] += int(r["collapsed"])
        a["collapsed_at_common_steps"] += int(bool((r.get("at_common_steps") or {}).get("collapsed")))
        a["collapse_steps"].append(r["collapse_step"])
        a["final_val_avg_spearman"].append(r["final_val_avg_spearman"])
    return out


def _write(path, plan, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"plan": {k: v for k, v in plan.items() if k != "sample_dirs"}, "results": results,
                               "by_arm": aggregate(results), "provenance": "measured (this run)"},
                              indent=1, default=str))
    tmp.replace(path)
