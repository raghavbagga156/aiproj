"""HEADPATH: does the official train-time clamp kill a learnable cell head? (docs/HEADPATH_PLAN.md)

Two arms, paired by seed, on the notebook-07 FULL training set and its 12-graph validation set:
  O = official model, unchanged
  U = diagnostic deviation: the cell head's clamp(min=0) is omitted in training forward passes only
      (config.HEADPATH_DEVIATION). Every evaluation uses the official clamped forward.
Everything else is identical: labels (per_dataset, c_max of all 251 official training samples),
paper_eq3 features, initial weights, sample order, RNG stream, AdamW, clipping, FP32, budget.

Pre-registered outcomes (research decisions R1-R9, 2026-10-20) are evaluated per run from
steps.csv / evals.csv / last.pt; the confirmatory analysis is a paired exact two-sided sign
test over seeds 42-47. A run is never dropped: an invalid run is reported as INVALID.
The run refuses to start while `pending_research_decisions` is not empty.
"""
import copy
import csv
import gc
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..config import HEADPATH_DEVIATION, Config, deviations
from ..data.graph import compute_c_max
from ..data.preprocess import list_processed
from ..evaluation import head_metrics as HM
from . import collapse_experiment as CE

ARMS = {"O": True, "U": False}                 # arm -> model.cell_head_train_clamp
SEEDS = (42, 43, 44, 45, 46, 47)               # R4: six paired seeds (two-sided exact sign test)
RETEST = (("O", 42), ("U", 42))                # R8: test-retest control (one repeat per arm)
EPOCHS = 2
EVAL_EVERY = 49                                # 49 divides 245: evaluations also fall on both epoch ends
VAL_DESIGN = "mgc_matrix_mult_a"

THRESHOLDS = {
    "activity_std_min": 1e-4,                  # R2: HEALTHY needs validation std >= 1e-4 ...
    "activity_zero_max": 0.999,                # ... and validation zero fraction < 0.999
    "p1_last_steps": 50,                       # P1 (training side): cell predictions all 0 on <= 10 %
    "p1_max_dead_frac": 0.10,                  #     of the last 50 steps
    "p2_last_steps": 98,                       # P2: clamped cell WMSE >= 10 % below the zero predictor
    "p2_min_improvement": 0.10,                #     over the last 98 steps (R1: clamped, both arms)
    "p2_borderline_low": 0.08,                 # R3 "borderline WMSE": 0.08 < improvement < 0.12 is BORDERLINE;
    "p2_borderline_high": 0.12,                #     PASS needs >= 0.12, FAIL is <= 0.08 (explicit bounds, no arithmetic)
    "p3_min_cell_spearman": 0.10,              # P3: pooled independent validation cell Spearman at the end
    "hsig_max_abs_wmse_change": 0.03,          # H-sig (decision 2): active, |clamped-WMSE change vs zero| <= 3 %,
    "hsig_max_cell_spearman": 0.05,            #     and pooled independent cell Spearman <= 0.05
    "sign_test_alpha": 0.05,
    "fresh_official_atol": 1e-6,               # decision 7: |a - b| <= atol + rtol * |b| elementwise
    "fresh_official_rtol": 1e-6,
}
STATISTICAL_TEST = {
    "name": "exact McNemar test for paired binary outcomes (= two-sided exact binomial sign test on discordant pairs)",
    "pairs": "the six independent paired seeds 42-47, arm U vs arm O; the two RETEST runs are excluded",
    "endpoint": "run category == LEARNS_AND_GENERALIZES (primary; the only confirmatory test)",
    "reported": "number of valid pairs, discordant pairs, U-only and O-only counts, effect direction, exact p",
    "ties": "concordant pairs carry no information and are not counted; pairs with an INVALID run are excluded "
            "and listed",
    "significant_if": "p <= 0.05; with 6 pairs only 6/6 discordant pairs in one direction (p = 0.03125) qualify",
    "secondary_not_tested": "H-sig (ACTIVE_WITHOUT_USEFUL_SIGNAL) is reported as counts of seeds per arm, "
                            "separately from the primary endpoint; no significance is claimed for it",
}
# 07 FULL end states (docs/POST_COLLAPSE_FORENSIC_REPORT.md section 7) at its last step, 245.
# The 07 data support DEAD vs not-DEAD only; compared with HEADPATH arm O at global step 245.
REPLICATION_REFERENCE = {"42": {"cell": "DEAD", "grid": "DEAD"}, "43": {"cell": "DEAD", "grid": "DEAD"},
                         "44": {"cell": "DEAD", "grid": "NOT_DEAD"}}
REPLICATION_STEP = 245
REPLICATION_POLICY = ("A failed replication of notebook 07 is flagged prominently and must be investigated; it does "
                      "not invalidate the experiment when the data, implementation, checkpoint and paired-run "
                      "validity checks pass (experiment_status and historical_replication are reported separately).")
PENDING_RESEARCH_DECISIONS = []                # all seven resolved 2026-10-20 (docs/HEADPATH_PLAN.md section 9)
CATEGORIES = ("LEARNS_AND_GENERALIZES", "LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION", "ACTIVE_WITHOUT_USEFUL_SIGNAL",
              "CELL_HEAD_FAILURE_PERSISTS", "INCONCLUSIVE", "INVALID")
MANIFEST_KEYS = ("train", "val", "val_main", "holdout", "epochs", "steps_per_run", "eval_every_steps", "eval_steps",
                 "thresholds", "statistical_test", "categories", "replication_reference", "replication_step",
                 "replication_policy", "pending_research_decisions")


# ------------------------------------------------------------ plan
def _selection(names_by_design, train_designs, val_designs, **kw):
    """Notebook-07 selection rules (FULL arm and its validation set), unchanged."""
    return CE.select_samples(names_by_design, train_designs, val_designs, kw.pop("single_design", "mgc_fft_a"), **kw)


def build_runs(train, seeds=SEEDS, retest=RETEST, epochs=EPOCHS) -> list:
    runs = []
    for seed in seeds:
        for arm, clamp in ARMS.items():
            runs.append({"name": f"HEADPATH_{arm}__seed{seed}", "arm": arm, "seed": seed, "retest": False,
                         "cell_head_train_clamp": clamp, "epochs": epochs, "steps": epochs * len(train)})
    for arm, seed in retest:
        runs.append({"name": f"HEADPATH_{arm}_RETEST__seed{seed}", "arm": arm, "seed": seed, "retest": True,
                     "cell_head_train_clamp": ARMS[arm], "epochs": epochs, "steps": epochs * len(train)})
    if len({r["name"] for r in runs}) != len(runs):
        raise ValueError("duplicate run names")
    return runs


def _plan_from_selection(sel, test_designs, seeds, retest, epochs, eval_every) -> dict:
    train, val = sel["arms"]["FULL"], sel["val"]
    if set(train) & set(val):
        raise ValueError("train/val overlap")
    if {CE.design_of(n) for n in train + val} & set(test_designs):
        raise ValueError("a test design is used")
    if len(train) != len(set(train)):
        raise ValueError("duplicate training sample")
    steps = epochs * len(train)
    return {"train": train, "val": val, "val_main": sel["val_main"], "holdout": sel["holdout"],
            "all_train": sel["all_train"], "epochs": epochs, "steps_per_run": steps, "eval_every_steps": eval_every,
            "eval_steps": list(range(0, steps + 1, eval_every)), "thresholds": dict(THRESHOLDS),
            "statistical_test": dict(STATISTICAL_TEST), "categories": list(CATEGORIES),
            "replication_reference": copy.deepcopy(REPLICATION_REFERENCE), "replication_step": REPLICATION_STEP,
            "replication_policy": REPLICATION_POLICY, "pending_research_decisions": list(PENDING_RESEARCH_DECISIONS),
            "runs": build_runs(train, seeds, retest, epochs)}


def manifest_of(plan: dict) -> dict:
    return json.loads(json.dumps({k: plan[k] for k in (*MANIFEST_KEYS, "runs")}))


def manifest_diff(plan: dict, expected: dict) -> list:
    got = manifest_of(plan)
    return [k for k in (*MANIFEST_KEYS, "runs") if got.get(k) != expected.get(k)]


def manifest_from_names(names_by_design: dict, split: dict, seeds=SEEDS, retest=RETEST, epochs=EPOCHS,
                        eval_every=EVAL_EVERY, **sel_kw) -> dict:
    sel = _selection(names_by_design, split["designs"]["train"], split["designs"]["val"], **sel_kw)
    return manifest_of(_plan_from_selection(sel, split["designs"].get("test", []), seeds, retest, epochs, eval_every))


def plan_headpath(processed_root, split_manifest: dict, expected: dict | None = None,
                  collapse_manifest: dict | None = None, seeds=SEEDS, retest=RETEST, epochs=EPOCHS,
                  eval_every=EVAL_EVERY, **sel_kw) -> dict:
    """Plan on the processed data. `expected` (configs/headpath_manifest.json) is the pre-registration;
    `collapse_manifest` (configs/collapse_experiment_manifest.json) pins FULL and the validation set to
    those of notebook 07. Any difference raises before anything is trained."""
    processed_root = Path(processed_root)
    test_designs = sorted(split_manifest["designs"].get("test", []))
    tr, va = split_manifest["designs"]["train"], split_manifest["designs"]["val"]
    if (set(tr) | set(va)) & set(test_designs):
        raise ValueError("train/val and test designs overlap")
    dirs = {d: list_processed(processed_root, {d}) for d in tr + va}
    for d, ds in dirs.items():
        if [p.name for p in ds] != sorted(p.name for p in ds):
            raise ValueError(f"{d}: list_processed order is not the sorted name order")
    sel = _selection({d: [p.name for p in ds] for d, ds in dirs.items()}, tr, va, **sel_kw)
    plan = _plan_from_selection(sel, test_designs, seeds, retest, epochs, eval_every)
    path_of = {p.name: p for ds in dirs.values() for p in ds}
    plan["c_max"] = compute_c_max([path_of[n] for n in plan["all_train"]])    # = notebook 07 / smoke run
    plan["sample_dirs"] = {n: str(path_of[n]) for n in plan["train"] + plan["val"]}
    if collapse_manifest is not None and (plan["train"] != collapse_manifest["arms"]["FULL"]
                                          or plan["val"] != collapse_manifest["val"]):
        raise ValueError("FULL / validation set differ from the notebook-07 manifest")
    if expected is not None:
        diff = manifest_diff(plan, expected)
        if diff:
            raise ValueError(f"plan differs from the pre-registered manifest: {diff}")
    return plan


# ------------------------------------------------------------ configs
def check_base_config(base_cfg: Config) -> None:
    e = base_cfg.experiment
    if (e.split, e.label_norm, e.grid_features) != ("official_seed42", "per_dataset", "paper_eq3"):
        raise ValueError("HEADPATH uses the primary variant official_seed42 / per_dataset / paper_eq3")
    if not base_cfg.model.cell_head_train_clamp:
        raise ValueError("the base config must be the official model (cell_head_train_clamp = True)")


def run_config(base_cfg: Config, run: dict, plan: dict) -> Config:
    """Identical for every run except seed, the arm's cell_head_train_clamp and the experiment name."""
    check_base_config(base_cfg)
    cfg = copy.deepcopy(base_cfg)
    t = cfg.training
    t.seed, t.num_epochs = run["seed"], plan["epochs"]
    t.precision, t.allow_tf32 = "fp32", False      # removes the GradScaler confound (forensic report 6.3)
    t.patience, t.keep_every, t.max_epochs_this_session = 10 ** 9, 0, None
    t.step_log, t.step_log_heads, t.eval_every_steps = True, True, plan["eval_every_steps"]
    t.keep_dataset_on_gpu = "false"                # as notebook 07 and the smoke run
    cfg.model.cell_head_train_clamp = run["cell_head_train_clamp"]
    cfg.experiment.name = f"headpath_{run['arm']}" + ("_retest" if run["retest"] else "")
    return cfg


# ------------------------------------------------------------ per-run summary
def _rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def p2_outcome(improvement: float, th: dict) -> str:
    if not math.isfinite(improvement) or improvement <= th["p2_borderline_low"]:
        return "FAIL"
    return "PASS" if improvement >= th["p2_borderline_high"] else "BORDERLINE"


def is_hsig(p1: bool, improvement: float, rho: float, th: dict) -> bool:
    """H-sig (decision 2): active cell head, clamped WMSE within +-3 % of the zero predictor, and
    pooled independent cell Spearman <= 0.05. Undefined (NaN) values never satisfy it."""
    return bool(p1 and math.isfinite(improvement) and abs(improvement) <= th["hsig_max_abs_wmse_change"]
                and math.isfinite(rho) and rho <= th["hsig_max_cell_spearman"])


def categorize(valid: bool, p1: bool, p2: str, p3: bool, hsig: bool = False) -> str:
    """R3 + decision 2, pre-registered. The categories are mutually exclusive: H-sig needs
    |improvement| <= 0.03, i.e. P2 = FAIL, so it can never overlap a P2 = PASS category."""
    if not valid:
        return "INVALID"
    if not p1:
        return "CELL_HEAD_FAILURE_PERSISTS"
    if p2 == "PASS":
        return "LEARNS_AND_GENERALIZES" if p3 else "LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION"
    if hsig:
        return "ACTIVE_WITHOUT_USEFUL_SIGNAL"
    return "INCONCLUSIVE"


def _improvement(rows, key, zero_key) -> float:
    v = np.array([_f(r.get(key)) for r in rows])
    z = np.array([_f(r.get(zero_key)) for r in rows])
    ok = np.isfinite(v) & np.isfinite(z)
    return float(1.0 - v[ok].sum() / z[ok].sum()) if ok.any() and z[ok].sum() > 0 else float("nan")


def _activity(row, head) -> str:
    return row.get(f"val_{head}_activity", "MISSING") if row else "MISSING"


def checkpoint_checks(run_dir: Path) -> dict:
    from ..training.checkpointing import load_checkpoint
    p = run_dir / "checkpoints" / "last.pt"
    if not p.exists():
        return {"present": False}
    s = load_checkpoint(p)
    params_finite = all(bool(torch.isfinite(v).all()) for v in s["model"].values()
                        if torch.is_tensor(v) and v.is_floating_point())
    steps, moments_finite = set(), True
    for st in s["optimizer"].get("state", {}).values():
        steps.add(int(float(st["step"])) if "step" in st else -1)
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and not bool(torch.isfinite(st[k]).all()):
                moments_finite = False
    return {"present": True, "epoch": s["epoch"], "run_id": s["run_id"], "params_finite": params_finite,
            "adam_moments_finite": moments_finite, "adam_step_counts": sorted(steps)}


def summarize_run(run_dir, plan: dict, run: dict, fresh_check: dict | None = None) -> dict:
    run_dir, th = Path(run_dir), plan["thresholds"]
    steps, evals, hist = _rows(run_dir / "steps.csv"), _rows(run_dir / "evals.csv"), _rows(run_dir / "history.csv")
    ev = {int(r["global_step"]): r for r in evals}
    final = ev.get(plan["steps_per_run"])
    ck = checkpoint_checks(run_dir)
    opt_calls = sum(1 for r in steps if r.get("grad_norm") not in ("", "None")) - \
        sum(r.get("step_skipped_by_scaler") == "True" for r in steps)
    reasons = []
    if len(steps) != plan["steps_per_run"]:
        reasons.append(f"{len(steps)} steps logged, expected {plan['steps_per_run']}")
    if not all(math.isfinite(_f(r.get("loss"))) for r in steps):
        reasons.append("non-finite training loss")
    if not all(math.isfinite(_f(r.get("grad_norm"))) for r in steps):
        reasons.append("non-finite gradient norm")
    if sorted(ev) != plan["eval_steps"]:
        reasons.append(f"evaluations at {sorted(ev)}, expected {plan['eval_steps']}")
    if not ck.get("present"):
        reasons.append("last.pt missing")
    else:
        if ck["epoch"] != plan["epochs"]:
            reasons.append(f"last.pt epoch {ck['epoch']} != {plan['epochs']}")
        if ck["adam_step_counts"] != [opt_calls]:
            reasons.append(f"Adam steps {ck['adam_step_counts']} != optimizer.step() calls {opt_calls}")
        if not (ck["params_finite"] and ck["adam_moments_finite"]):
            reasons.append("non-finite parameters or Adam moments in last.pt")
    if fresh_check is not None and not fresh_check.get("equivalent"):
        reasons.append("final evaluation differs from a fresh official model with the same weights "
                       f"(max |diff| {fresh_check.get('max_abs_diff_vs_fresh_official')})")
    valid = not reasons
    lastp1 = steps[-th["p1_last_steps"]:]
    dead_frac = float(np.mean([_f(r.get("cell_clamped_zero_frac")) >= 1.0 for r in lastp1])) if lastp1 else float("nan")
    cell_state = _activity(final, "cell")
    p1 = cell_state == "HEALTHY" and dead_frac <= th["p1_max_dead_frac"]
    lastp2 = steps[-th["p2_last_steps"]:]
    imp_cell = _improvement(lastp2, "cell_wmse_clamped", "cell_wmse_zero")
    p2 = p2_outcome(imp_cell, th)
    rho = _f(final.get("val_cell_spearman_indep")) if final else float("nan")
    p3 = bool(math.isfinite(rho) and rho >= th["p3_min_cell_spearman"])
    hsig = is_hsig(p1, imp_cell, rho, th)
    zero, tail, cstep, collapsed = CE._collapse(steps, 20) if steps else ([], 0, None, False)
    rep = ev.get(plan["replication_step"])
    keys = [k for k in (final or {}) if k.startswith("design_") or k.startswith("tileproj_")]
    # decision 1: the zero-predictor reference depends only on the graphs of the last steps; the two arms
    # of a pair must have exactly the same one (checked in analyse)
    zero_ref = {"samples_sha256": hashlib.sha256("\n".join(r.get("sample", "") for r in lastp2).encode()).hexdigest(),
                "sum_cell_wmse_zero": float(np.nansum([_f(r.get("cell_wmse_zero")) for r in lastp2]))}
    return {
        "name": run["name"], "arm": run["arm"], "seed": run["seed"], "retest": run["retest"],
        "cell_head_train_clamp": run["cell_head_train_clamp"], "valid": valid, "invalid_reasons": reasons,
        "category": categorize(valid, p1, p2, p3, hsig),
        "hsig_criteria_met": hsig, "p2_zero_reference": zero_ref,
        "P1": {"pass": p1, "final_cell_activity": cell_state,
               "final_cell_val_std": _f(final.get("val_cell_pred_std")) if final else None,
               "final_cell_val_zero_frac": _f(final.get("val_cell_pred_zero_frac")) if final else None,
               "dead_frac_last_steps": dead_frac},
        "P2": {"outcome": p2, "cell_wmse_clamped_improvement_vs_zero": imp_cell,
               "cell_wmse_raw_improvement_vs_zero": _improvement(lastp2, "cell_wmse_raw", "cell_wmse_zero"),
               "cell_wmse_loss_improvement_vs_zero": _improvement(lastp2, "cell_wmse_loss", "cell_wmse_zero")},
        "P3": {"pass": p3, "final_val_cell_spearman_indep": rho,
               "final_val_cell_spearman_official": _f(final.get("val_cell_spearman")) if final else None},
        "grid": {"final_activity": _activity(final, "grid"),
                 "wmse_improvement_vs_zero": _improvement(lastp2, "grid_wmse_loss", "grid_wmse_zero"),
                 "final_val_spearman_indep": _f(final.get("val_grid_spearman_indep")) if final else None,
                 "final_val_spearman_official": _f(final.get("val_grid_spearman")) if final else None},
        "per_design_and_tileproj_final": {k: _f(final[k]) for k in keys} if final else {},
        "state_at_replication_step": {"cell": _activity(rep, "cell"), "grid": _activity(rep, "grid")},
        "both_head_collapse_07_rule": {"collapsed": collapsed, "collapse_step": cstep, "zero_grad_steps": sum(zero)},
        "optimizer_step_calls": opt_calls, "checkpoint": ck, "fresh_official_check": fresh_check,
        "history_last": {k: hist[-1].get(k) for k in ("train_loss", "train_cell_spearman", "train_cell_raw_spearman",
                                                       "val_avg_spearman")} if hist else None,
    }


# ------------------------------------------------------------ analysis
def sign_test_two_sided(n_pos: int, n_neg: int) -> float:
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = max(n_pos, n_neg)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _not_dead(state: str) -> str:
    return "DEAD" if state == "DEAD" else "NOT_DEAD"


def _completed(r) -> bool:
    return r is not None and r.get("run_status", "COMPLETED") == "COMPLETED"


def analyse(results: list, plan: dict) -> dict:
    """Primary: exact McNemar test (= exact two-sided sign test on discordant pairs) on
    LEARNS_AND_GENERALIZES over the six independent paired seeds; RETEST runs never enter it.
    H-sig: counts of seeds per arm, reported separately (no test). Experiment validity and the
    historical notebook-07 replication are separate verdicts (decision 5). Only COMPLETED runs carry
    results; any of the 14 planned runs that FAILED, is INVALID, INCOMPLETE, NOT_STARTED or missing
    prevents a VALID verdict."""
    by = {r["name"]: r for r in results}
    th, ref = plan["thresholds"], plan["replication_reference"]
    primary_runs = [r for r in plan["runs"] if not r["retest"]]
    main = [by[r["name"]] for r in primary_runs if _completed(by.get(r["name"]))]
    statuses = {r["name"]: (by[r["name"]].get("run_status", "COMPLETED") if r["name"] in by else "MISSING")
                for r in plan["runs"]}
    failed = [n for n, s in statuses.items() if s == "FAILED"]
    not_done = [n for n, s in statuses.items() if s in ("NOT_STARTED", "INCOMPLETE", "MISSING")]
    invalid = [n for n, s in statuses.items() if s == "COMPLETED" and not by[n]["valid"]]
    counts = {arm: {c: sum(r["category"] == c for r in main if r["arm"] == arm) for c in CATEGORIES} for arm in ARMS}
    activity = {arm: {h: [r["P1"]["final_cell_activity"] if h == "cell" else r["grid"]["final_activity"]
                          for r in main if r["arm"] == arm] for h in ("cell", "grid")} for arm in ARMS}
    pairs, excluded, pair_problems = [], [], []
    u_only = o_only = 0
    for s in sorted({r["seed"] for r in primary_runs}):
        o, u = by.get(f"HEADPATH_O__seed{s}"), by.get(f"HEADPATH_U__seed{s}")
        if not (_completed(o) and _completed(u)):
            excluded.append({"seed": s, "reason": f"run status O={statuses[f'HEADPATH_O__seed{s}']}, "
                                                  f"U={statuses[f'HEADPATH_U__seed{s}']}"})
            continue
        if not (o["valid"] and u["valid"]):
            excluded.append({"seed": s, "reason": "invalid run: " + "; ".join(
                f"{x['name']}: {x['invalid_reasons']}" for x in (o, u) if not x["valid"])})
            continue
        if o["p2_zero_reference"] != u["p2_zero_reference"]:       # decision 1: same zero-predictor reference
            excluded.append({"seed": s, "reason": "zero-predictor reference differs between the arms"})
            pair_problems.append(s)
            continue
        xo, xu = o["category"] == "LEARNS_AND_GENERALIZES", u["category"] == "LEARNS_AND_GENERALIZES"
        u_only += xu and not xo
        o_only += xo and not xu
        pairs.append({"seed": s, "O": o["category"], "U": u["category"], "discordant": xo != xu})
    p = sign_test_two_sided(u_only, o_only)
    significant = bool(u_only + o_only > 0 and p <= th["sign_test_alpha"])
    direction = "U > O" if u_only > o_only else "O > U" if o_only > u_only else "none"
    hsig = {arm: {"n_seeds_meeting_hsig": sum(r["hsig_criteria_met"] for r in main if r["arm"] == arm and r["valid"]),
                  "seeds": [r["seed"] for r in main if r["arm"] == arm and r["valid"] and r["hsig_criteria_met"]],
                  "n_valid_runs": sum(r["valid"] for r in main if r["arm"] == arm)} for arm in ARMS}
    repl = {}
    for s, exp in ref.items():
        r = by.get(f"HEADPATH_O__seed{s}")
        got = {h: _not_dead(r["state_at_replication_step"][h]) for h in ("cell", "grid")} if _completed(r) else None
        repl[s] = {"expected": exp, "observed": got, "match": got == exp}
    historical = ("NOT_EVALUATED" if any(v["observed"] is None for v in repl.values())
                  else "PASSED" if all(v["match"] for v in repl.values()) else "FAILED")
    status_reasons = ([f"FAILED runs: {failed}"] if failed else []) + \
                     ([f"INVALID runs: {invalid}"] if invalid else []) + \
                     ([f"zero-predictor reference differs for seeds {pair_problems}"] if pair_problems else []) + \
                     ([f"runs not completed: {not_done}"] if not_done else [])
    status = ("FAILED" if failed else "INVALID" if invalid or pair_problems else "INCOMPLETE" if not_done
              else "VALID")
    retest = {}
    for arm, s in RETEST:
        a, b = by.get(f"HEADPATH_{arm}__seed{s}"), by.get(f"HEADPATH_{arm}_RETEST__seed{s}")
        if _completed(a) and _completed(b):
            retest[arm] = {"excluded_from_inference": True, "retest_run_valid": b["valid"],
                           "same_category": a["category"] == b["category"],
                           "abs_diff_cell_wmse_improvement": abs(a["P2"]["cell_wmse_clamped_improvement_vs_zero"]
                                                                 - b["P2"]["cell_wmse_clamped_improvement_vs_zero"]),
                           "abs_diff_final_cell_spearman": abs(a["P3"]["final_val_cell_spearman_indep"]
                                                               - b["P3"]["final_val_cell_spearman_indep"]),
                           "abs_diff_final_grid_spearman": abs(_f(a["grid"]["final_val_spearman_indep"])
                                                               - _f(b["grid"]["final_val_spearman_indep"]))}
    stat = (f"exact McNemar test on LEARNS_AND_GENERALIZES: {len(pairs)} valid pairs, {u_only + o_only} discordant "
            f"(U-only {u_only}, O-only {o_only}), direction {direction}, exact two-sided p = {p:.5f}")
    if status != "VALID":
        conclusion = f"{status} EXPERIMENT ({'; '.join(status_reasons)}); no conclusion is drawn. {stat}."
    elif significant and u_only > o_only:
        conclusion = f"Supports H-dyn: {stat}."
    elif significant:
        conclusion = f"Significant in the opposite direction to H-dyn: {stat}."
    else:
        conclusion = f"Not statistically conclusive: {stat}. Descriptive results only."
    conclusion += (f" H-sig criteria met in {hsig['U']['n_seeds_meeting_hsig']}/{hsig['U']['n_valid_runs']} arm-U "
                   f"and {hsig['O']['n_seeds_meeting_hsig']}/{hsig['O']['n_valid_runs']} arm-O seeds (descriptive).")
    if historical == "FAILED":
        conclusion = "WARNING - HISTORICAL REPLICATION OF NOTEBOOK 07 FAILED: investigate before interpreting. " + conclusion
    return {"experiment_status": status, "status_reasons": status_reasons, "historical_replication": historical,
            "verdict": f"{status}" + (" - HISTORICAL REPLICATION FAILED (investigate)" if historical == "FAILED" else ""),
            "run_statuses": statuses, "category_counts": counts, "final_activity": activity,
            "primary_test": {"name": plan["statistical_test"]["name"], "endpoint": plan["statistical_test"]["endpoint"],
                             "pairs": pairs, "excluded": excluded, "n_valid_pairs": len(pairs),
                             "n_discordant": u_only + o_only, "U_only": u_only, "O_only": o_only,
                             "direction": direction, "p_two_sided_exact": p, "alpha": th["sign_test_alpha"],
                             "significant": significant},
            "hsig": hsig, "replication": repl, "replication_policy": plan["replication_policy"],
            "test_retest": retest, "missing_runs": failed + not_done, "conclusion": conclusion,
            "deviation_notice": HEADPATH_DEVIATION}


# ------------------------------------------------------------ execution
def fresh_official_check(trainer, atol: float, rtol: float) -> dict:
    """Arm-independent: the final weights loaded into a freshly built OFFICIAL model must give the
    same eval-mode predictions as the run's own model: |a - b| <= atol + rtol * |b| for every element.
    The run's own model is evaluated twice so GPU scatter non-determinism is recorded alongside."""
    from ..evaluation.evaluate import predict
    from ..training.trainer import build_model
    cfg = copy.deepcopy(trainer.cfg)
    cfg.model.cell_head_train_clamp = True
    ref = build_model(cfg).to(trainer.device)
    ref.load_state_dict(trainer.model.state_dict())
    a = predict(trainer.model, trainer.batches["val"], trainer.device)
    a2 = predict(trainer.model, trainer.batches["val"], trainer.device)
    b = predict(ref, trainer.batches["val"], trainer.device)

    def compare(x, y):
        mx_abs = mx_rel = worst = 0.0
        for r, q in zip(x, y):
            for k in ("cell_pred", "grid_pred"):
                u, v = r[k].double(), q[k].double()
                d = (u - v).abs()
                mx_abs = max(mx_abs, float(d.max()))
                nz = v.abs() > 0
                if bool(nz.any()):
                    mx_rel = max(mx_rel, float((d[nz] / v.abs()[nz]).max()))
                tol = atol + rtol * v.abs()
                ratio = torch.where(d == 0, torch.zeros_like(d), d / tol)       # equal elements pass at any tolerance
                worst = max(worst, float(ratio.max()))
        return mx_abs, mx_rel, worst
    fa, fr, fw = compare(a, b)
    sa, _, _ = compare(a, a2)
    return {"max_abs_diff_vs_fresh_official": fa, "max_rel_diff_vs_fresh_official": fr,
            "max_tolerance_ratio": fw, "max_abs_diff_self_repeat": sa, "atol": atol, "rtol": rtol,
            "equivalent": fw <= 1.0, "bit_identical": fa == 0.0, "recs": a}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# Earlier results HEADPATH must never touch: (DriveLayout attribute, glob below it). Every file below a matched
# directory is included. STAGE_REPORT.md is notebook 05's stage report.
PROTECTED_PATTERNS = (("runs", "verihgn_ispd15_smoke__*"), ("runs", "precision_benchmark"), ("runs", "collapse_diag"),
                      ("reports", "05_*"), ("reports", "06_*"), ("reports", "07_*"), ("reports", "STAGE_REPORT.md"))


def protected_artifacts(layout, patterns=PROTECTED_PATTERNS) -> dict:
    """'<runs|reports>/<relative path>' -> {size, sha256, mtime_ns} for every protected file. The sha256 is of the
    full content of every file, checkpoints included. Read-only."""
    out = {}
    for base, pat in patterns:
        root = Path(getattr(layout, base))
        if not root.exists():
            continue
        for top in sorted(root.glob(pat)):
            files = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file())
            for p in files:
                st = p.stat()
                out[f"{base}/{p.relative_to(root).as_posix()}"] = {"size": st.st_size, "sha256": _sha256(p),
                                                                    "mtime_ns": st.st_mtime_ns}
    return out


def compare_fingerprints(before: dict, after: dict) -> dict:
    """Unchanged means: the same files, each with the same size and sha256. A change of the modification time
    alone is reported but is not a modification (content is what is protected)."""
    common = set(before) & set(after)
    modified = sorted(k for k in common if (before[k]["size"], before[k]["sha256"]) != (after[k]["size"], after[k]["sha256"]))
    added, removed = sorted(set(after) - set(before)), sorted(set(before) - set(after))
    return {"unchanged": not (added or removed or modified), "n_files": len(before),
            "bytes": sum(e["size"] for e in before.values()), "added": added, "removed": removed,
            "modified": modified,
            "mtime_only_changes": sorted(k for k in common if k not in modified
                                         and before[k]["mtime_ns"] != after[k]["mtime_ns"])}


RUN_STATUSES = ("COMPLETED", "INCOMPLETE", "FAILED", "NOT_STARTED")
# Messages of errors after which the CUDA context (and so every later run in this process) is unusable.
RUNTIME_FATAL_MARKERS = ("CUDA error", "CUBLAS_STATUS", "CUDNN_STATUS", "cuDNN error", "device-side assert",
                         "illegal memory access", "unspecified launch failure", "CUDA driver", "NVML", "ECC error")
PREDICTIONS_DIR = "predictions_val_official"
PREDICTIONS_MARKER = "_COMPLETE.json"


class ProtectedArtifactError(RuntimeError):
    """A protected notebook 05-07 artifact changed during the run."""


def device_usable(device=None) -> bool:
    """A tiny CUDA round trip; False if the device no longer works. The CPU is always usable."""
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        return True
    try:
        torch.cuda.synchronize(dev)
        ok = float((torch.ones(8, device=dev) * 2).sum()) == 16.0
        torch.cuda.synchronize(dev)
        return ok
    except Exception:  # noqa: BLE001 - any failure means unusable
        return False


def runtime_fatal(exc: BaseException, device=None) -> bool:
    """True if no later run can safely start in this process after `exc`: an interrupt, host memory
    exhaustion, a CUDA/accelerator error that corrupts the context, or a device that fails a probe.
    A CUDA out-of-memory error is not fatal by itself (memory is released and the probe decides)."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return True
    acc = getattr(torch, "AcceleratorError", None)
    if acc is not None and isinstance(exc, acc):
        return True
    if any(m in str(exc) for m in RUNTIME_FATAL_MARKERS):
        return True
    return not device_usable(device)


def _base_record(run: dict) -> dict:
    return {k: run[k] for k in ("name", "arm", "seed", "retest", "cell_head_train_clamp")}


def not_started_record(run: dict, reason: str) -> dict:
    return {**_base_record(run), "run_status": "NOT_STARTED", "reason": reason, "valid": False, "category": None}


def failed_record(run: dict, exc: BaseException, fatal: bool, run_dir: Path, seconds: float) -> dict:
    trainer_status = None
    try:
        p = Path(run_dir) / "status.json"
        trainer_status = json.loads(p.read_text()) if p.exists() else None
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        trainer_status = {"unreadable": repr(e)}
    return {**_base_record(run), "run_status": "FAILED", "valid": False, "category": None,
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                      "runtime_fatal": fatal},
            "run_dir": str(run_dir), "trainer_status": trainer_status, "seconds_this_session": round(seconds, 1)}


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - a broken context is detected by device_usable instead
            pass


# ------------------------------------------------------------ prediction export (atomic)
def predictions_complete(d) -> bool:
    """True only for a directory finalised by export_predictions_atomic: its marker lists every file with
    its sha256 and every listed file is present and unchanged. Anything else is partial."""
    d = Path(d)
    m = d / PREDICTIONS_MARKER
    if not m.exists():
        return False
    try:
        files = json.loads(m.read_text())["files"]
    except Exception:  # noqa: BLE001
        return False
    present = {p.name for p in d.iterdir() if p.name != PREDICTIONS_MARKER}
    return bool(files) and present == set(files) and all(_sha256(d / f) == h for f, h in files.items())


def unique_path(p, append: bool = False) -> Path:
    """`p` if it does not exist, else the first free variant with _1, _2, ... inserted before the file suffix
    (append=True: added at the very end, for directories). Never returns an existing path."""
    p = Path(p)

    def variant(k):
        return p.with_name(f"{p.name}_{k}") if append else p.with_name(f"{p.stem}_{k}{p.suffix}")
    if not p.exists():
        return p
    k = 1
    while variant(k).exists():
        k += 1
    return variant(k)


def new_report_path(reports_dir, name) -> Path:
    """reports/<name>, or reports/<stem>__<YYYYmmdd_HHMMSS>[_k]<suffix> if taken: never an existing file."""
    p = Path(reports_dir) / name
    return p if not p.exists() else unique_path(p.with_name(f"{p.stem}__{time.strftime('%Y%m%d_%H%M%S')}{p.suffix}"))


def _rename(src, dst, attempts: int = 8, delay_s: float = 0.25) -> None:
    """os.replace with a bounded retry: right after files are written, a directory rename can be refused
    transiently (open handles from virus scanners on Windows, the Drive FUSE upload). Raises after the last try."""
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(delay_s * (k + 1))


def export_predictions_atomic(recs, graphs_by_name, final_dir) -> dict:
    """Write into <final>.tmp_<pid>/, add the marker, then rename to <final>. A complete existing export is
    kept as is. A partial final directory (no valid marker, e.g. from an interrupted export) or a leftover
    temporary directory is moved aside to <final>.partial_<ts> (never deleted, never used) and the export
    is written again."""
    from ..evaluation.evaluate import save_predictions
    final_dir = Path(final_dir)
    if predictions_complete(final_dir):
        return {"status": "existing_complete", "dir": str(final_dir)}
    moved = []
    stale = ([final_dir] if final_dir.exists() else []) + sorted(final_dir.parent.glob(final_dir.name + ".tmp_*"))
    for p in stale:
        aside = unique_path(final_dir.with_name(f"{final_dir.name}.partial_{time.strftime('%Y%m%d_%H%M%S')}"),
                            append=True)
        _rename(p, aside)
        moved.append(str(aside))
    tmp = final_dir.with_name(f"{final_dir.name}.tmp_{os.getpid()}")
    written = save_predictions(recs, graphs_by_name, tmp)
    from ..training.checkpointing import atomic_write_json
    atomic_write_json({"files": {Path(w).name: _sha256(w) for w in written}, "n_files": len(written),
                       "created_at": time.time()}, tmp / PREDICTIONS_MARKER)
    _rename(tmp, final_dir)
    if not predictions_complete(final_dir):
        raise RuntimeError(f"prediction export {final_dir} failed its completeness check")
    return {"status": "written", "dir": str(final_dir), "n_files": len(written), "moved_aside": moved}


def run_plan(plan: dict, base_cfg: Config, graphs_by_name: dict, out_root, device=None, only=None,
             report_path=None, allow_pending: bool = False) -> dict:
    """Run (or resume) every run of the plan with the normal Trainer. Each run is isolated: an exception is
    recorded as FAILED (type, message, traceback) and the next run starts, unless the runtime is no longer
    usable (runtime_fatal), in which case the remaining runs are recorded as NOT_STARTED. Every planned run is
    always in the report, which is written atomically before the first run and after every run.
    Refuses while research decisions are pending (`allow_pending` exists for tests only)."""
    from ..training.trainer import Trainer
    if plan["pending_research_decisions"] and not allow_pending:
        raise RuntimeError("HEADPATH has pending research decisions; it may not run:\n- "
                           + "\n- ".join(plan["pending_research_decisions"]))
    check_base_config(base_cfg)
    out_root = Path(out_root)
    if "collapse_diag" in out_root.parts:
        raise ValueError("HEADPATH must never write into the notebook-07 results (runs/collapse_diag)")
    records = {r["name"]: not_started_record(r, "not yet run in this session") for r in plan["runs"]}
    selected = [r for r in plan["runs"] if not only or r["name"] in only or r["arm"] in only]
    for r in plan["runs"]:
        if r not in selected:
            records[r["name"]]["reason"] = f"not selected in this session (only={only})"
    try:
        baseline = HM.cell_feature_baseline([graphs_by_name[n] for n in plan["val"]])
    except Exception as e:  # noqa: BLE001 - descriptive only; recorded
        baseline = {"error": f"{type(e).__name__}: {e}"}

    def persist():
        if report_path:
            _write(report_path, plan, [records[r["name"]] for r in plan["runs"]], baseline)
    persist()
    stop, interrupt = None, None
    for run in selected:
        if stop:
            records[run["name"]]["reason"] = stop
            continue
        run_dir, t0, tr = out_root / run["name"], time.time(), None
        try:
            cfg = run_config(base_cfg, run, plan)
            graphs = {"train": [graphs_by_name[n] for n in plan["train"]],
                      "val": [graphs_by_name[n] for n in plan["val"]]}
            tr = Trainer(cfg, graphs, run_dir, device=device,
                         extra_meta={"headpath": {k: run[k] for k in ("arm", "seed", "retest", "cell_head_train_clamp")},
                                     "deviations": deviations(cfg)})
            st = tr.run()          # a completed run is only re-loaded (last.pt), never re-trained
            fc = fresh_official_check(tr, plan["thresholds"]["fresh_official_atol"],
                                      plan["thresholds"]["fresh_official_rtol"])
            export = export_predictions_atomic(fc.pop("recs"), graphs_by_name, run_dir / PREDICTIONS_DIR)
            res = summarize_run(run_dir, plan, run, fc)
            res["run_status"] = "COMPLETED" if (st or {}).get("state") == "completed" else "INCOMPLETE"
            res["trainer_state"] = (st or {}).get("state")
            res["predictions_export"] = export
            res["seconds_this_session"] = round(time.time() - t0, 1)
            records[run["name"]] = res
            print(f"[{run['name']}] {res['run_status']} valid={res['valid']} category={res['category']} "
                  f"P2={res['P2']['outcome']} rho_cell={res['P3']['final_val_cell_spearman_indep']}", flush=True)
        except BaseException as e:  # noqa: BLE001 - isolate the run; interrupts are re-raised below
            del tr
            tr = None
            _free_memory()
            fatal = runtime_fatal(e, device)
            records[run["name"]] = failed_record(run, e, fatal, run_dir, time.time() - t0)
            print(f"[{run['name']}] FAILED {type(e).__name__}: {e} (runtime_fatal={fatal})", flush=True)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                interrupt = e
                stop = f"not started: the session was interrupted during {run['name']}"
            elif fatal:
                stop = f"not started: the runtime was unusable after {run['name']} failed ({type(e).__name__})"
        finally:
            tr = None
            _free_memory()
            persist()
    for run in selected:                                   # runs skipped after an interrupt / fatal failure
        if records[run["name"]]["run_status"] == "NOT_STARTED" and stop:
            records[run["name"]]["reason"] = stop
    persist()
    if interrupt is not None:
        raise interrupt
    results = [records[r["name"]] for r in plan["runs"]]
    return {"plan": _public(plan), "results": results, "analysis": analyse(results, plan),
            "cell_feature_baseline": baseline}


def _public(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if k != "sample_dirs"}


def run_status_counts(results: list) -> dict:
    return {s: sum(r.get("run_status", "COMPLETED") == s for r in results) for s in RUN_STATUSES}


def _write(path, plan, results, baseline):
    """Atomic (temporary file, fsync, rename): an interrupted write never replaces a good report."""
    from ..training.checkpointing import atomic_write_json
    atomic_write_json({"plan": _public(plan), "results": results, "run_status_counts": run_status_counts(results),
                       "analysis": analyse(results, plan), "cell_feature_baseline": baseline,
                       "provenance": "measured (this run)"}, path)


def update_report(path, section: dict) -> None:
    """Add top-level keys to a report (atomic). Creates the file if the run sequence never wrote one."""
    from ..training.checkpointing import atomic_write_json
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(section)
    atomic_write_json(data, path)


def execute(plan: dict, base_cfg: Config, layout, report_path, build_graphs, verify_paths=(), flush=None,
            device=None, only=None) -> dict:
    """The notebook-08 run sequence:
      1. fingerprint the protected notebook 05-07 artifacts (content hashes);
      2. build the graphs, then run_plan (every run isolated, report written after every run);
      3. always (finally): fingerprint again and write the comparison into the report;
      4. always, last: flush(paths) - e.g. colab.flush_drive (flush_and_unmount); nothing is written after it.
    Afterwards, in this order of priority: re-raises the run sequence's error, raises ProtectedArtifactError if
    a protected file changed, raises the flush error. Nothing is converted into success."""
    report_path = Path(report_path)
    before = protected_artifacts(layout)
    print(f"protected files: {len(before)} ({sum(e['size'] for e in before.values()) / 1e6:.1f} MB, all "
          "content-hashed)", flush=True)
    rep, run_error, protected, persistence, flush_error = None, None, None, None, None
    try:
        try:
            graphs = build_graphs()
        except BaseException as e:
            _write(report_path, plan, [not_started_record(r, f"not started: building the graphs failed "
                                                            f"({type(e).__name__}: {e})") for r in plan["runs"]], None)
            raise
        rep = run_plan(plan, base_cfg, graphs, layout.runs / "headpath", device=device, only=only,
                       report_path=report_path)
    except BaseException as e:  # noqa: BLE001 - re-raised after the protected check and the flush
        run_error = e
        try:                                    # e.g. refused before the first run: still list all 14 runs
            has_runs = report_path.exists() and len(json.loads(report_path.read_text()).get("results", [])) == len(plan["runs"])
            if not has_runs:
                _write(report_path, plan, [not_started_record(r, f"not started: {type(e).__name__}: {e}")
                                           for r in plan["runs"]], None)
        except Exception as w:  # noqa: BLE001 - recorded on the raised error
            e.add_note(f"could not write the not-started report: {w}")
    finally:
        try:
            after = protected_artifacts(layout)
            protected = compare_fingerprints(before, after)
        except Exception as e:  # noqa: BLE001
            protected = {"unchanged": None, "error": f"{type(e).__name__}: {e}"}
        try:
            update_report(report_path, {"protected_artifacts": protected})
        except Exception as e:  # noqa: BLE001
            protected["report_write_error"] = f"{type(e).__name__}: {e}"
        if flush is not None:                                   # LAST: after every report write
            try:
                persistence = flush([report_path, *verify_paths])
            except Exception as e:  # noqa: BLE001
                flush_error = e
    print("protected artifacts unchanged:", protected.get("unchanged"), flush=True)
    print("drive persistence:", persistence if flush_error is None else f"FAILED: {flush_error}", flush=True)
    if run_error is not None:
        if protected.get("unchanged") is not True:
            run_error.add_note(f"protected artifact check: {protected}")
        if flush_error is not None:
            run_error.add_note(f"drive flush failed: {flush_error}")
        raise run_error
    if protected.get("unchanged") is not True:
        raise ProtectedArtifactError(f"protected notebook 05-07 artifacts changed or could not be checked: {protected}")
    if flush_error is not None:
        raise flush_error
    return {"report": rep, "protected_artifacts": protected, "persistence": persistence}
