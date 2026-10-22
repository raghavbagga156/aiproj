"""Results analysis for HEADPATH (docs/DASHBOARD_PLAN.md, Stage 1). Pure functions; no I/O except CSV parsing.

Everything here READS finished experiment records and re-derives, as a consistency check, what the
pre-registered code (diagnostics/headpath.py) stored. Nothing replaces a stored result: the stored report
stays the authoritative record, and any disagreement is reported as a failed check.

Integrity rules implemented here:
  * non-finite numbers (NaN, +-Inf) are never turned into 0: `sanitize` replaces them with None and
    records where they were (the browser cannot parse NaN, and an undefined correlation is not 0);
  * the primary statistic is the pre-registered exact McNemar test over seeds 42-47; RETEST runs are
    excluded from it;
  * per-seed paired differences are EXPLORATORY: labelled so, no inferential statistic attached;
  * the recorded threshold `p2_min_improvement` (0.10) is reported as inconsistent with the implemented
    rule (FAIL <= 0.08 < BORDERLINE < 0.12 <= PASS); the historical record is not modified.
"""
import csv
import hashlib
import math
from pathlib import Path

import numpy as np

ARMS = ("O", "U")
PRIMARY_SEEDS = (42, 43, 44, 45, 46, 47)
ENDPOINT = "LEARNS_AND_GENERALIZES"
EXPLORATORY_LABEL = ("EXPLORATORY - not pre-registered. Descriptive per-seed paired differences (U - O) over the six "
                     "primary seeds; no inferential statistic is computed and none may be inferred.")
UNDEFINED_KINDS = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}


# ------------------------------------------------------------ JSON sanitising
def _kind(x: float) -> str:
    return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")


def sanitize(obj, path: str = ""):
    """Return (clean, undefined): `clean` equals `obj` with every non-finite float replaced by None;
    `undefined` lists {"path": JSON pointer, "kind": nan|inf|-inf} for each replacement."""
    undefined = []

    def walk(o, p):
        if isinstance(o, float) and not math.isfinite(o):
            undefined.append({"path": p, "kind": _kind(o)})
            return None
        if isinstance(o, dict):
            return {k: walk(v, f"{p}/{_esc(k)}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}/{i}") for i, v in enumerate(o)]
        return o
    return walk(obj, path), undefined


def _esc(k) -> str:
    return str(k).replace("~", "~0").replace("/", "~1")


# ------------------------------------------------------------ CSV
def parse_cell(s: str):
    """CSV text -> value. '' and 'None' -> None; 'True'/'False' -> bool; numbers -> int or float
    (non-finite floats are kept as float here and handled by `columns`); anything else -> str."""
    if s is None or s == "" or s == "None":
        return None
    if s in ("True", "False"):
        return s == "True"
    try:
        if s.lstrip("-").isdigit():
            return int(s)
        return float(s)
    except ValueError:
        return s


def read_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def columns(rows: list) -> dict:
    """Column-oriented form for the dashboard: {"columns": [...], "data": {col: [values]},
    "undefined": {col: {row_index: kind}}}. Non-finite floats become None and are listed in `undefined`."""
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    data, undefined = {c: [] for c in cols}, {}
    for i, r in enumerate(rows):
        for c in cols:
            v = parse_cell(r.get(c))
            if isinstance(v, float) and not math.isfinite(v):
                undefined.setdefault(c, {})[str(i)] = _kind(v)
                v = None
            data[c].append(v)
    return {"columns": cols, "n_rows": len(rows), "data": data, "undefined": undefined}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------ independent re-derivation of the stored outcomes
def p2_outcome_effective(improvement: float, th: dict) -> str:
    """The implemented rule (headpath.p2_outcome), re-written from docs/HEADPATH_PLAN.md section 5."""
    if not math.isfinite(improvement) or improvement <= th["p2_borderline_low"]:
        return "FAIL"
    return "PASS" if improvement >= th["p2_borderline_high"] else "BORDERLINE"


def improvement(rows, key, zero_key) -> float:
    v = np.array([_f(r.get(key)) for r in rows])
    z = np.array([_f(r.get(zero_key)) for r in rows])
    ok = np.isfinite(v) & np.isfinite(z)
    return float(1.0 - v[ok].sum() / z[ok].sum()) if ok.any() and z[ok].sum() > 0 else float("nan")


def category_from(valid: bool, p1: bool, p2: str, p3: bool, hsig: bool) -> str:
    if not valid:
        return "INVALID"
    if not p1:
        return "CELL_HEAD_FAILURE_PERSISTS"
    if p2 == "PASS":
        return "LEARNS_AND_GENERALIZES" if p3 else "LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION"
    if hsig:
        return "ACTIVE_WITHOUT_USEFUL_SIGNAL"
    return "INCONCLUSIVE"


def rederive_run(steps: list, evals: list, plan: dict) -> dict:
    """Recompute, from steps.csv / evals.csv rows only, every per-run quantity the report derives from them.
    (Checkpoint and fresh-official checks are not CSV-derived; they are verified separately.)"""
    th = plan["thresholds"]
    ev = {int(_f(r["global_step"])): r for r in evals}
    final = ev.get(plan["steps_per_run"]) or {}
    last1, last2 = steps[-th["p1_last_steps"]:], steps[-th["p2_last_steps"]:]
    dead = float(np.mean([_f(r.get("cell_clamped_zero_frac")) >= 1.0 for r in last1])) if last1 else float("nan")
    cell_state = final.get("val_cell_activity", "MISSING")
    p1 = cell_state == "HEALTHY" and dead <= th["p1_max_dead_frac"]
    imp = improvement(last2, "cell_wmse_clamped", "cell_wmse_zero")
    p2 = p2_outcome_effective(imp, th)
    rho = _f(final.get("val_cell_spearman_indep"))
    p3 = bool(math.isfinite(rho) and rho >= th["p3_min_cell_spearman"])
    hsig = bool(p1 and math.isfinite(imp) and abs(imp) <= th["hsig_max_abs_wmse_change"]
                and math.isfinite(rho) and rho <= th["hsig_max_cell_spearman"])
    rep = ev.get(plan["replication_step"]) or {}
    return {
        "n_steps": len(steps), "eval_steps": sorted(ev),
        "losses_finite": all(math.isfinite(_f(r.get("loss"))) for r in steps),
        "grad_norms_finite": all(math.isfinite(_f(r.get("grad_norm"))) for r in steps),
        "P1": {"pass": p1, "final_cell_activity": cell_state, "dead_frac_last_steps": dead,
               "final_cell_val_std": _f(final.get("val_cell_pred_std")),
               "final_cell_val_zero_frac": _f(final.get("val_cell_pred_zero_frac"))},
        "P2": {"outcome": p2, "cell_wmse_clamped_improvement_vs_zero": imp,
               "cell_wmse_raw_improvement_vs_zero": improvement(last2, "cell_wmse_raw", "cell_wmse_zero"),
               "cell_wmse_loss_improvement_vs_zero": improvement(last2, "cell_wmse_loss", "cell_wmse_zero")},
        "P3": {"pass": p3, "final_val_cell_spearman_indep": rho,
               "final_val_cell_spearman_official": _f(final.get("val_cell_spearman"))},
        "hsig_criteria_met": hsig,
        "category_given_valid": category_from(True, p1, p2, p3, hsig),
        "grid": {"final_activity": final.get("val_grid_activity", "MISSING"),
                 "wmse_improvement_vs_zero": improvement(last2, "grid_wmse_loss", "grid_wmse_zero"),
                 "final_val_spearman_indep": _f(final.get("val_grid_spearman_indep")),
                 "final_val_spearman_official": _f(final.get("val_grid_spearman"))},
        "per_design_and_tileproj_final": {k: _f(v) for k, v in final.items()
                                          if k.startswith("design_") or k.startswith("tileproj_")},
        "state_at_replication_step": {"cell": rep.get("val_cell_activity", "MISSING"),
                                      "grid": rep.get("val_grid_activity", "MISSING")},
        "p2_zero_reference": {
            "samples_sha256": hashlib.sha256("\n".join(r.get("sample", "") for r in last2).encode()).hexdigest(),
            "sum_cell_wmse_zero": float(np.nansum([_f(r.get("cell_wmse_zero")) for r in last2]))},
        "optimizer_step_calls": sum(1 for r in steps if r.get("grad_norm") not in ("", "None", None))
        - sum(r.get("step_skipped_by_scaler") == "True" for r in steps),
    }


def same(a, b, rtol: float = 1e-12, atol: float = 1e-12) -> bool:
    """Equality for re-derived values: floats within tolerance, NaN == NaN (both undefined), None == NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], rtol, atol) for k in a)
    na = a is None or (isinstance(a, float) and math.isnan(a))
    nb = b is None or (isinstance(b, float) and math.isnan(b))
    if na or nb:
        return na and nb
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def compare_run(stored: dict, derived: dict) -> list:
    """Field-by-field comparison of a stored report result with its CSV re-derivation. Returns mismatches."""
    out = []

    def chk(name, a, b, **kw):
        if not same(a, b, **kw):
            out.append({"field": name, "stored": a, "rederived": b})
    for blk in ("P1", "P2", "P3", "grid"):
        for k, v in derived[blk].items():
            chk(f"{blk}.{k}", stored.get(blk, {}).get(k), v)
    chk("hsig_criteria_met", stored.get("hsig_criteria_met"), derived["hsig_criteria_met"])
    chk("state_at_replication_step", stored.get("state_at_replication_step"), derived["state_at_replication_step"])
    chk("p2_zero_reference", stored.get("p2_zero_reference"), derived["p2_zero_reference"])
    chk("optimizer_step_calls", stored.get("optimizer_step_calls"), derived["optimizer_step_calls"])
    sp, dp = stored.get("per_design_and_tileproj_final", {}), derived["per_design_and_tileproj_final"]
    chk("per_design_and_tileproj_final.keys", sorted(sp), sorted(dp))
    for k in dp:
        chk(f"per_design_and_tileproj_final.{k}", sp.get(k), dp[k])
    expected_cat = derived["category_given_valid"] if stored.get("valid") else "INVALID"
    chk("category", stored.get("category"), expected_cat)
    return out


# ------------------------------------------------------------ primary statistic (pre-registered)
def sign_test_exact(n_pos: int, n_neg: int) -> float:
    """Exact two-sided binomial sign test on discordant pairs (= exact McNemar), from math.comb only."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = max(n_pos, n_neg)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def primary_test_from_results(results: list, alpha: float) -> dict:
    """Recompute the McNemar summary from the per-run categories (seeds 42-47; RETEST excluded)."""
    by = {(r["arm"], r["seed"]): r for r in results if not r.get("retest") and r.get("run_status") == "COMPLETED"}
    pairs, excluded, u_only, o_only = [], [], 0, 0
    for s in PRIMARY_SEEDS:
        o, u = by.get(("O", s)), by.get(("U", s))
        if not (o and u) or not (o.get("valid") and u.get("valid")) or o["p2_zero_reference"] != u["p2_zero_reference"]:
            excluded.append(s)
            continue
        xo, xu = o["category"] == ENDPOINT, u["category"] == ENDPOINT
        u_only += xu and not xo
        o_only += xo and not xu
        pairs.append({"seed": s, "O": o["category"], "U": u["category"], "discordant": xo != xu})
    p = sign_test_exact(u_only, o_only)
    return {"pairs": pairs, "excluded_seeds": excluded, "n_valid_pairs": len(pairs), "n_discordant": u_only + o_only,
            "U_only": u_only, "O_only": o_only,
            "direction": "U > O" if u_only > o_only else "O > U" if o_only > u_only else "none",
            "p_two_sided_exact": p, "alpha": alpha, "significant": bool(u_only + o_only > 0 and p <= alpha),
            "minimum_p_achievable_with_these_pairs": sign_test_exact(len(pairs), 0) if pairs else None}


# ------------------------------------------------------------ threshold record inconsistency
def threshold_note(plan: dict, results: list) -> dict:
    """Flags (never fixes) that the report records p2_min_improvement while the implemented rule uses the
    borderline bounds. Lists every run whose stored improvement lies where the two readings could differ."""
    th = plan["thresholds"]
    rec = th.get("p2_min_improvement")
    lo, hi = th["p2_borderline_low"], th["p2_borderline_high"]
    affected = []
    for r in results:
        imp = _f(r.get("P2", {}).get("cell_wmse_clamped_improvement_vs_zero"))
        if rec is not None and math.isfinite(imp) and min(lo, rec) < imp < max(hi, rec):
            affected.append({"name": r["name"], "improvement": imp, "implemented_outcome": p2_outcome_effective(imp, th),
                             "single_threshold_reading": "PASS" if imp >= rec else "FAIL"})
    return {"recorded_field": "plan.thresholds.p2_min_improvement", "recorded_value": rec,
            "implemented_rule": f"FAIL if improvement <= {lo}; BORDERLINE if {lo} < improvement < {hi}; PASS if >= {hi} "
                                "(diagnostics/headpath.py: p2_outcome)",
            "recorded_value_used_by_code": False,
            "runs_in_disputed_interval": affected,
            "note": "Documentation inconsistency in the historical record. The report is not modified; every stored "
                    "category was re-derived with the implemented rule (see the per-run checks)."}


# ------------------------------------------------------------ tables for the dashboard
def paired_table(results: list) -> list:
    """Seeds 42-47, one row per seed with both arms (primary). RETEST runs are not included."""
    by = {(r["arm"], r["seed"]): r for r in results if not r.get("retest")}
    rows = []
    for s in PRIMARY_SEEDS:
        row = {"seed": s}
        for a in ARMS:
            r = by.get((a, s)) or {}
            row[a] = {"name": r.get("name"), "run_status": r.get("run_status"), "valid": r.get("valid"),
                      "category": r.get("category"), "P1_pass": r.get("P1", {}).get("pass"),
                      "P2_outcome": r.get("P2", {}).get("outcome"),
                      "P2_improvement": r.get("P2", {}).get("cell_wmse_clamped_improvement_vs_zero"),
                      "P3_pass": r.get("P3", {}).get("pass"),
                      "final_cell_spearman_indep": r.get("P3", {}).get("final_val_cell_spearman_indep"),
                      "final_cell_activity": r.get("P1", {}).get("final_cell_activity"),
                      "final_grid_activity": r.get("grid", {}).get("final_activity"),
                      "final_grid_spearman_indep": r.get("grid", {}).get("final_val_spearman_indep")}
        rows.append(row)
    return rows


def exploratory_paired_differences(results: list) -> dict:
    """EXPLORATORY: per-seed U - O differences. An undefined value on either side gives None (not 0)."""
    by = {(r["arm"], r["seed"]): r for r in results if not r.get("retest") and r.get("run_status") == "COMPLETED"}
    metrics = {"P2_cell_wmse_clamped_improvement": lambda r: r["P2"]["cell_wmse_clamped_improvement_vs_zero"],
               "final_val_cell_spearman_indep": lambda r: r["P3"]["final_val_cell_spearman_indep"],
               "final_val_grid_spearman_indep": lambda r: r["grid"]["final_val_spearman_indep"],
               "grid_wmse_improvement": lambda r: r["grid"]["wmse_improvement_vs_zero"]}
    out = {"label": EXPLORATORY_LABEL, "metrics": {}}
    for name, get in metrics.items():
        per_seed = []
        for s in PRIMARY_SEEDS:
            o, u = by.get(("O", s)), by.get(("U", s))
            vo, vu = (_f(get(o)) if o else float("nan")), (_f(get(u)) if u else float("nan"))
            d = vu - vo if math.isfinite(vo) and math.isfinite(vu) else None
            per_seed.append({"seed": s, "O": vo if math.isfinite(vo) else None, "U": vu if math.isfinite(vu) else None,
                             "U_minus_O": d})
        ds = [x["U_minus_O"] for x in per_seed if x["U_minus_O"] is not None]
        out["metrics"][name] = {"per_seed": per_seed, "n_defined_pairs": len(ds),
                                "n_U_greater": sum(d > 0 for d in ds), "n_O_greater": sum(d < 0 for d in ds),
                                "n_equal": sum(d == 0 for d in ds)}
    return out


def retest_table(results: list) -> list:
    by = {r["name"]: r for r in results}
    rows = []
    for a in ARMS:
        base, re = by.get(f"HEADPATH_{a}__seed42"), by.get(f"HEADPATH_{a}_RETEST__seed42")
        rows.append({"arm": a, "excluded_from_inference": True,
                     "base": {"name": base and base["name"], "category": base and base.get("category")},
                     "retest": {"name": re and re["name"], "category": re and re.get("category")}})
    return rows
