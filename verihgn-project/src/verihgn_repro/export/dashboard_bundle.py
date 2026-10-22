"""Read-only export of the HEADPATH results (and the notebook-07 context) into an offline dashboard bundle
(docs/DASHBOARD_PLAN.md, Stage 1).

Sources (Google Drive, never written):
  reports/08_headpath_experiment.json (+ the 08 plan), runs/headpath/<run>/{steps,evals,history}.csv,
  config.yaml, status.json, env.json, checkpoints/last.pt (read for the validity checks only, not exported),
  predictions_val_official/*.npz + _COMPLETE.json; the 12 processed validation samples (label provenance);
  reports/07_collapse_*.json, 07_forensic_audit.json, runs/collapse_diag/<run>/{steps,history}.csv.

Output: a new directory (refused below runs/, reports/, processed/, splits/ or code/) and a zip of it.
Every source file is fingerprinted (size + sha256 of the full content) before and after; any change fails
the export. Every exported array is stored losslessly (little-endian, 4-byte shuffled, gzip) and decoded
again and compared bit for bit before the zip is accepted. Numbers are never invented: every value in the
bundle is copied from a source file or computed from one, and each check is recorded in the integrity
report. JSON files in the bundle never contain NaN/Infinity (browser-safe): non-finite values become null
and their locations are listed, so the dashboard can show "undefined" instead of 0.
"""
import gzip
import hashlib
import io
import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..analysis import headpath_results as A

SCHEMA_VERSION = 1
EXPORTER_VERSION = "1.0.0"
ENCODING = "le+byteshuffle+gzip"
PROTECTED = (("runs", "headpath"), ("runs", "collapse_diag"), ("reports", "07_*"), ("reports", "08_*"),
             ("runs", "verihgn_ispd15_smoke__*"), ("runs", "precision_benchmark"), ("reports", "05_*"),
             ("reports", "06_*"), ("reports", "STAGE_REPORT.md"))
FORBIDDEN_OUTPUT_PARENTS = ("runs", "reports", "processed", "splits", "code")
SHARED_ARRAYS = ("grid_gt", "raw_h_overflow", "raw_v_overflow", "cell_index", "cell_gt", "cell_tile")
RUN_ARRAYS = ("grid_pred", "cell_pred")
NPZ_KEYS = {"grid_gt", "grid_pred", "grid_abs_err", "cell_index", "cell_gt", "cell_pred", "cell_tile",
            "raw_h_overflow", "raw_v_overflow"}
NPZ_VS_EVAL_ATOL = 1e-4      # NPZ predictions come from a second eval-mode pass (GPU scatter non-determinism)


class ExportValidationError(RuntimeError):
    pass


# ------------------------------------------------------------ hashing / encoding
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _js_dtype(a: np.ndarray) -> np.dtype:
    if a.dtype.kind == "f":
        if a.dtype != np.float32:
            raise ExportValidationError(f"unexpected float dtype {a.dtype} (only float32 is exported losslessly)")
        return np.dtype("<f4")
    if a.dtype.kind in "iu":
        if a.size and (a.min() < -2 ** 31 or a.max() >= 2 ** 31):
            raise ExportValidationError("integer array does not fit int32 (JavaScript Int32Array)")
        return np.dtype("<i4")
    raise ExportValidationError(f"unsupported dtype {a.dtype}")


def encode_array(a: np.ndarray) -> tuple[bytes, dict]:
    """Lossless: cast to the little-endian JS dtype (float32 stays bit-identical; ints must fit int32), split
    the bytes into itemsize planes (byte shuffle), gzip (mtime 0, deterministic)."""
    dt = _js_dtype(a)
    c = np.ascontiguousarray(a.astype(dt, copy=False))
    if a.dtype.kind in "iu" and not np.array_equal(c.astype(np.int64), a.astype(np.int64)):
        raise ExportValidationError("integer conversion to int32 is not exact")
    raw = c.tobytes()
    shuffled = np.frombuffer(raw, np.uint8).reshape(-1, dt.itemsize).T.tobytes() if raw else b""
    blob = gzip.compress(shuffled, compresslevel=9, mtime=0)
    return blob, {"dtype": {"<f4": "float32", "<i4": "int32"}[dt.str], "shape": list(a.shape), "encoding": ENCODING,
                  "itemsize": dt.itemsize, "sha256_raw": sha256_bytes(raw), "nbytes_raw": len(raw),
                  "nbytes_file": len(blob)}


def decode_array(blob: bytes, meta: dict) -> np.ndarray:
    dt = np.dtype({"float32": "<f4", "int32": "<i4"}[meta["dtype"]])
    sh = gzip.decompress(blob)
    n = len(sh) // dt.itemsize
    raw = np.frombuffer(sh, np.uint8).reshape(dt.itemsize, n).T.tobytes() if sh else b""
    if sha256_bytes(raw) != meta["sha256_raw"]:
        raise ExportValidationError("decoded bytes do not match sha256_raw")
    return np.frombuffer(raw, dt).reshape(meta["shape"])


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bit-for-bit equality (NaN-safe, distinguishes -0.0 from +0.0)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype.itemsize == b.dtype.itemsize and a.tobytes() == b.tobytes()


def _reject_constant(c):
    raise ValueError(f"non-strict JSON constant {c}")


def dumps_strict(obj) -> str:
    """JSON for the browser: raises on NaN/Infinity (they must have been sanitised first)."""
    return json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False)


# ------------------------------------------------------------ integrity log
class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, ok, message, status=None, **details):
        st = status or ("PASS" if ok else "FAIL")
        self.items.append({"id": cid, "status": st, "message": message, **({"details": details} if details else {})})
        return ok

    @property
    def failed(self):
        return [c for c in self.items if c["status"] == "FAIL"]

    def summary(self):
        out = {s: 0 for s in ("PASS", "WARN", "INFO", "FAIL")}
        for c in self.items:
            out[c["status"]] += 1
        return out


# ------------------------------------------------------------ fingerprints
def fingerprint(root: Path, extra_files=()) -> dict:
    from ..diagnostics.headpath import protected_artifacts
    fp = protected_artifacts(SimpleNamespace(runs=root / "runs", reports=root / "reports"), PROTECTED)
    for p in extra_files:
        p = Path(p)
        st = p.stat()
        fp[f"extra/{p.as_posix()}"] = {"size": st.st_size, "sha256": sha256_file(p), "mtime_ns": st.st_mtime_ns}
    return fp


def _guard_output(root: Path, out_dir: Path):
    out, r = out_dir.resolve(), root.resolve()
    for name in FORBIDDEN_OUTPUT_PARENTS:
        bad = (r / name).resolve()
        if out == bad or bad in out.parents:
            raise ValueError(f"refusing to write the export under {bad}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; exports never overwrite")


# ------------------------------------------------------------ writer
def load_json_lenient(path) -> object:
    """Parse a source JSON file as Python writes it (NaN / Infinity / -Infinity accepted), read-only."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_json_bytes(data: bytes) -> tuple[bytes, dict]:
    """Source JSON bytes -> strict JSON bytes (allow_nan=False) in which every non-finite number is null.
    Structure, keys, order and every finite value are kept; already-null values stay null. Returns the new bytes
    and {n_replaced, by_kind, paths}."""
    obj = json.loads(data.decode("utf-8"))
    clean, undefined = A.sanitize(obj)
    out = json.dumps(clean, indent=1, allow_nan=False, ensure_ascii=False).encode("utf-8")
    kinds = {k: 0 for k in ("nan", "inf", "-inf")}
    for u in undefined:
        kinds[u["kind"]] += 1
    return out, {"n_replaced": len(undefined), "by_kind": kinds, "paths": [u["path"] for u in undefined]}


class Bundle:
    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.files = {}
        self.normalized = {}          # bundle path -> normalisation record (every JSON copied from a source)

    def _put(self, rel: str, data: bytes):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(rel)
        p.write_bytes(data)
        self.files[rel] = {"size": len(data), "sha256": sha256_bytes(data)}
        return rel

    def json(self, rel, obj):
        return self._put(rel, dumps_strict(obj).encode("utf-8"))

    def copy(self, rel, src: Path):
        """Verbatim copy of a NON-JSON source (CSV, YAML). JSON sources must go through copy_json."""
        if str(rel).lower().endswith(".json") or Path(src).suffix.lower() == ".json":
            raise ValueError(f"{rel}: JSON sources are copied with copy_json (strict, non-finite -> null)")
        return self._put(rel, Path(src).read_bytes())

    def copy_json(self, rel, src: Path):
        """Copy a source JSON file as STRICT JSON: NaN, Infinity and -Infinity become null (recursively), nothing
        else changes. The source file is only read. The normalisation (count, kinds, JSON-pointer paths, and the
        sha256 of the untouched source) is recorded for the integrity report and normalization_index.json."""
        raw = Path(src).read_bytes()
        out, rec = normalize_json_bytes(raw)
        self._put(rel, out)
        self.normalized[rel] = {"source": str(src), "source_sha256": sha256_bytes(raw), "source_bytes": len(raw),
                                **rec}
        return rel

    def array(self, rel, a):
        blob, meta = encode_array(a)
        self._put(rel, blob)
        return {**meta, "file": rel}


# ------------------------------------------------------------ main export
def export(root, out_dir, report_name="08_headpath_experiment.json", plan_name="08_headpath_plan.json",
           processed_root="auto", context07=True, context07_report=None) -> dict:
    """Build the bundle in `out_dir` (new or empty), validate it, zip it to `<out_dir>.zip`.
    Returns {"status", "zip", "integrity_report", "summary"}; raises ExportValidationError after writing the
    integrity report if any check fails (no zip is produced then)."""
    root, out_dir = Path(root), Path(out_dir)
    _guard_output(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C, B = Checks(), Bundle(out_dir)
    reports, runs_root = root / "reports", root / "runs" / "headpath"
    sources = {}

    def src(rel_label, path):
        path = Path(path)
        sources[rel_label] = {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        return path

    # -------- report and plan
    report_path = src("reports/" + report_name, reports / report_name)
    report = json.loads(report_path.read_text(encoding="utf-8"))   # Python json accepts NaN
    plan = report["plan"]
    others = sorted(p.name for p in reports.glob("08_headpath_experiment*.json") if p.name != report_name)
    C.add("report.selected", True, f"authoritative report: {report_name} (sha256 {sources['reports/' + report_name]['sha256'][:16]}…)",
          status="INFO", other_08_reports_present=others)
    plan_files = sorted(reports.glob("08_headpath_plan*.json"))
    plan_cmp = {}
    for p in plan_files:
        pj = json.loads(p.read_text(encoding="utf-8"))
        src("reports/" + p.name, p)
        plan_cmp[p.name] = all(A.same(pj.get(k), plan.get(k)) for k in ("train", "val", "runs", "thresholds", "eval_steps"))
    C.add("plan.matches_report", bool(plan_cmp) and all(plan_cmp.values()),
          "every 08_headpath_plan*.json agrees with report.plan (train, val, runs, thresholds, eval steps)", files=plan_cmp)
    names = [r["name"] for r in plan["runs"]]
    res_by = {r["name"]: r for r in report["results"]}
    C.add("report.runs", len(names) == 14 and set(res_by) == set(names),
          f"{len(res_by)} results for {len(names)} planned runs (expected 14 = 12 primary + 2 retest)")
    C.add("report.status", report["analysis"]["experiment_status"] == "VALID",
          f"experiment_status = {report['analysis']['experiment_status']}, verdict = {report['analysis']['verdict']}")
    pa = report.get("protected_artifacts")
    if pa is None:
        C.add("report.protected_artifacts", True, "the report has no protected_artifacts record (notebook 08 writes it "
              "as its last report update; an earlier copy of the report lacks it)", status="WARN")
    else:
        C.add("report.protected_artifacts", pa.get("unchanged") is True,
              f"notebook 08 recorded its protected 05-07 artifacts as unchanged: {pa.get('unchanged')} "
              f"({pa.get('n_files')} files)")
    C.add("report.all_completed", all(res_by.get(n, {}).get("run_status") == "COMPLETED" for n in names),
          "every planned run has run_status COMPLETED", counts=report.get("run_status_counts"))

    # -------- processed samples (label provenance)
    if processed_root == "auto":
        pre = reports / "03_preprocessing_official9.json"
        processed_root = (root / "processed" / "ispd15" / json.loads(pre.read_text())["prep_hash"]) if pre.exists() else None
    proc_files = []
    if processed_root:
        for s in plan["val"]:
            d = Path(processed_root) / A_design(s) / s
            proc_files += [d / "sample.npz", d / "meta.json"]
    before = fingerprint(root, proc_files)
    C.add("sources.fingerprint_before", True, f"{len(before)} source files fingerprinted (sha256 of full content)",
          status="INFO", n_files=len(before), bytes=sum(v["size"] for v in before.values()))

    # -------- per run
    bundle_runs, shared, shared_meta, grid_cmp = {}, {}, {}, {}
    for name in names:
        r, rd = res_by[name], runs_root / name
        base = f"headpath/runs/{name}"
        steps_p, evals_p, hist_p = rd / "steps.csv", rd / "evals.csv", rd / "history.csv"
        for p in (steps_p, evals_p, hist_p):
            if not p.exists():
                C.add(f"{name}.files", False, f"missing {p.name}")
        if not (steps_p.exists() and evals_p.exists() and hist_p.exists()):
            continue
        steps, evals, hist = A.read_rows(src(f"runs/headpath/{name}/steps.csv", steps_p)), \
            A.read_rows(src(f"runs/headpath/{name}/evals.csv", evals_p)), A.read_rows(src(f"runs/headpath/{name}/history.csv", hist_p))
        der = A.rederive_run(steps, evals, plan)
        C.add(f"{name}.steps", der["n_steps"] == plan["steps_per_run"] and der["losses_finite"] and der["grad_norms_finite"],
              f"{der['n_steps']} steps (expected {plan['steps_per_run']}), losses finite {der['losses_finite']}, "
              f"gradient norms finite {der['grad_norms_finite']}")
        C.add(f"{name}.evals", der["eval_steps"] == plan["eval_steps"], f"evaluations at {der['eval_steps']}")
        mism = A.compare_run(r, der)
        C.add(f"{name}.rederived", not mism, "category, P1/P2/P3, H-sig, grid, per-design, zero reference and "
              "optimizer-step count re-derived from the CSVs " + ("agree with the report" if not mism else "DISAGREE"),
              mismatches=mism[:20])
        # checkpoint (read-only; not exported)
        from ..diagnostics.headpath import checkpoint_checks
        ck = checkpoint_checks(rd)
        if ck.get("present"):
            src(f"runs/headpath/{name}/checkpoints/last.pt", rd / "checkpoints" / "last.pt")
        C.add(f"{name}.checkpoint", ck == r.get("checkpoint") and ck.get("present") and ck.get("epoch") == plan["epochs"]
              and ck.get("adam_step_counts") == [der["optimizer_step_calls"]] and ck.get("params_finite")
              and ck.get("adam_moments_finite"),
              "last.pt re-read: equals the report's checkpoint record, epoch, Adam step count = optimizer steps, finite",
              rechecked=ck)
        # predictions
        pdir = rd / "predictions_val_official"
        from ..diagnostics.headpath import predictions_complete
        complete = predictions_complete(pdir)
        C.add(f"{name}.predictions_marker", complete, "_COMPLETE.json present and every listed file's sha256 matches")
        if not complete:
            continue
        marker = json.loads(src(f"runs/headpath/{name}/predictions_val_official/_COMPLETE.json",
                                pdir / "_COMPLETE.json").read_text())
        expected = {f"{s}.npz" for s in plan["val"]}
        C.add(f"{name}.predictions_set", set(marker["files"]) == expected,
              f"{len(marker['files'])} prediction files = the 12 validation samples")
        run_maps, gp_all, gt_all, by_design = {}, [], [], {}
        for s in plan["val"]:
            f = src(f"runs/headpath/{name}/predictions_val_official/{s}.npz", pdir / f"{s}.npz")
            with np.load(f, allow_pickle=False) as z:
                arr = {k: z[k] for k in z.files}
            ok_keys = set(arr) == NPZ_KEYS
            M, N = arr["grid_gt"].shape
            checks = {
                "keys": ok_keys,
                "grid_shapes": all(arr[k].shape == (M, N) for k in ("grid_pred", "grid_abs_err", "raw_h_overflow",
                                                                      "raw_v_overflow")),
                "cell_shapes": len({arr[k].shape for k in ("cell_index", "cell_gt", "cell_pred", "cell_tile")}) == 1,
                "finite": all(np.isfinite(arr[k]).all() for k in arr if arr[k].dtype.kind == "f"),
                "official_eval_nonnegative": bool((arr["grid_pred"] >= 0).all() and (arr["cell_pred"] >= 0).all()),
                "abs_err_exact": bitwise_equal(arr["grid_abs_err"], np.abs(arr["grid_pred"] - arr["grid_gt"])),
                "cell_gt_from_grid_gt": bitwise_equal(arr["cell_gt"], arr["grid_gt"].reshape(-1)[arr["cell_tile"]]),
                "cell_index_sorted_unique": bool(np.all(np.diff(arr["cell_index"]) > 0)),
            }
            C.add(f"{name}.{s}.npz", all(checks.values()), "array keys, shapes, finiteness, clamped (>= 0) official "
                  "eval outputs, |err| = |pred - gt| bit-exact, cell_gt = grid_gt[cell_tile]", **checks)
            # shared arrays: identical across runs, stored once
            for k in SHARED_ARRAYS:
                if s not in shared:
                    shared[s] = {}
                if k not in shared[s]:
                    shared[s][k] = arr[k]
                elif not bitwise_equal(shared[s][k], arr[k]):
                    C.add(f"{name}.{s}.{k}.shared", False, f"{k} differs from the first run's copy")
            rm = {}
            for k in RUN_ARRAYS:
                meta = B.array(f"{base}/maps/{s}/{k}.bin.gz", arr[k])
                rm[k] = meta
            run_maps[s] = rm
            gp_all.append(arr["grid_pred"].reshape(-1))
            gt_all.append(arr["grid_gt"].reshape(-1))
            by_design.setdefault(A_design(s), []).append((arr["grid_pred"].reshape(-1), arr["grid_gt"].reshape(-1)))
        # grid metrics recomputed from the full NPZ maps vs evals.csv at the final step
        final = {int(A._f(e["global_step"])): e for e in evals}[plan["steps_per_run"]]
        from ..evaluation.head_metrics import spearman_indep
        gp, gt = np.concatenate(gp_all), np.concatenate(gt_all)
        cmp = {"pooled_spearman_indep": (spearman_indep(gp, gt), A._f(final.get("val_grid_spearman_indep"))),
               "pooled_zero_frac": (float((gp == 0).mean()), A._f(final.get("val_grid_pred_zero_frac"))),
               "pooled_std": (float(gp.astype(np.float64).std(ddof=1)), A._f(final.get("val_grid_pred_std")))}
        for d, lst in by_design.items():
            p_, g_ = np.concatenate([x for x, _ in lst]), np.concatenate([y for _, y in lst])
            cmp[f"design_{d}_spearman_indep"] = (spearman_indep(p_, g_), A._f(final.get(f"design_{d}_grid_spearman_indep")))
        diffs = {k: (None if (math.isnan(a) and math.isnan(b)) else (abs(a - b) if math.isfinite(a) and math.isfinite(b)
                                                                       else float("inf"))) for k, (a, b) in cmp.items()}
        worst = max([v for v in diffs.values() if v is not None], default=0.0)
        grid_cmp[name] = {k: {"npz": None if math.isnan(a) else a, "evals_csv": None if math.isnan(b) else b,
                              "abs_diff": diffs[k]} for k, (a, b) in cmp.items()}
        C.add(f"{name}.grid_metrics_from_npz", worst <= NPZ_VS_EVAL_ATOL,
              f"grid Spearman / zero fraction / std recomputed from the full NPZ maps agree with evals.csv at step "
              f"{plan['steps_per_run']} (max |diff| {worst:.3g}, tolerance {NPZ_VS_EVAL_ATOL}: the NPZ comes from a "
              "second eval pass; undefined == undefined)", comparison=_finite_only(grid_cmp[name]))
        # bundle the run's tables (verbatim sources + column JSON)
        for nm, rows, p in (("steps", steps, steps_p), ("evals", evals, evals_p), ("history", hist, hist_p)):
            B.copy(f"{base}/sources/{nm}.csv", p)
            B.json(f"{base}/{nm}.json", A.columns(rows))
        for extra in ("config.yaml", "status.json", "env.json"):
            if (rd / extra).exists():
                put = B.copy_json if extra.endswith(".json") else B.copy
                put(f"{base}/sources/{extra}", src(f"runs/headpath/{name}/{extra}", rd / extra))
        B.copy_json(f"{base}/sources/predictions_COMPLETE.json", pdir / "_COMPLETE.json")
        bundle_runs[name] = {"name": name, "arm": r["arm"], "seed": r["seed"], "retest": r["retest"],
                             "cell_head_train_clamp": r["cell_head_train_clamp"],
                             "primary_inference": not r["retest"],
                             "checkpoint_record": ck, "maps": run_maps}

    # -------- shared arrays (labels) + provenance against the processed samples
    maps_index = {}
    c_max = plan["c_max"]
    from ..data.graph import grid_label
    for s, arrs in shared.items():
        entry = {"design": A_design(s), "role": "official validation design" if s in plan["val_main"]
                 else "held-out sample of a training design", "shape": list(arrs["grid_gt"].shape),
                 "n_cells_subsample": int(arrs["cell_index"].size), "arrays": {}}
        for k, a in arrs.items():
            entry["arrays"][k] = B.array(f"headpath/maps/{s}/{k}.bin.gz", a)
        maps_index[s] = entry
        if processed_root:
            d = Path(processed_root) / A_design(s) / s
            with np.load(d / "sample.npz", allow_pickle=False) as z:
                lh, lv, tile = z["label_h"], z["label_v"], z["cell_tile"]
            M, N = arrs["grid_gt"].shape
            exp_gt = (np.asarray(grid_label(lh, lv), dtype=np.float32).copy() / np.float32(c_max)).reshape(M, N)
            ok = {"grid_gt_equals_label_over_cmax": bitwise_equal(arrs["grid_gt"], exp_gt),
                  "raw_h_equals_label_h": bitwise_equal(arrs["raw_h_overflow"], lh.reshape(-1).astype(np.float32).reshape(M, N)),
                  "raw_v_equals_label_v": bitwise_equal(arrs["raw_v_overflow"], lv.reshape(-1).astype(np.float32).reshape(M, N)),
                  "cell_tile_equals_processed": bitwise_equal(arrs["cell_tile"].astype(np.int64),
                                                              tile.astype(np.int64)[arrs["cell_index"]])}
            C.add(f"labels.{s}", all(ok.values()), "labels in the predictions equal the processed sample: "
                  "grid_gt = log1p((H+V)/2)/c_max, raw H and V overflow (kept separate), cell tiles", **ok)
    if not processed_root:
        C.add("labels.provenance", True, "processed samples not available: label provenance not checked", status="WARN")
    C.add("maps.shared_once", len(shared) == len(plan["val"]),
          f"label / raw H / raw V / cell-subsample arrays identical across all runs, stored once for {len(shared)} samples")

    # -------- report-level re-derivations
    alpha = plan["thresholds"]["sign_test_alpha"]
    pt = A.primary_test_from_results(report["results"], alpha)
    st = report["analysis"]["primary_test"]
    same_pt = (pt["n_valid_pairs"] == st["n_valid_pairs"] and pt["U_only"] == st["U_only"] and pt["O_only"] == st["O_only"]
               and pt["direction"] == st["direction"] and pt["significant"] == st["significant"]
               and A.same(pt["p_two_sided_exact"], st["p_two_sided_exact"]) and
               [(x["seed"], x["O"], x["U"], x["discordant"]) for x in pt["pairs"]] ==
               [(x["seed"], x["O"], x["U"], x["discordant"]) for x in st["pairs"]])
    try:
        from scipy.stats import binomtest
        sp = binomtest(max(pt["U_only"], pt["O_only"]), pt["n_discordant"], 0.5).pvalue if pt["n_discordant"] else 1.0
    except Exception:  # noqa: BLE001
        sp = None
    C.add("primary_test.rederived", same_pt and (sp is None or A.same(sp, pt["p_two_sided_exact"], rtol=1e-9)),
          f"exact McNemar re-derived from the stored categories (seeds 42-47, retests excluded): "
          f"{pt['n_discordant']} discordant, p = {pt['p_two_sided_exact']:.5g}; scipy binomtest p = {sp}",
          rederived=pt)
    seeds = [x["seed"] for x in st["pairs"]] + [x["seed"] for x in st["excluded"]]
    C.add("primary_test.retests_excluded", set(seeds) <= set(A.PRIMARY_SEEDS) and len(seeds) == len(set(seeds)),
          "the primary test has one pair per seed 42-47 and no other entry; the HEADPATH_*_RETEST__seed42 runs "
          "are not in it (re-derivation above uses non-retest runs only)", pairs_and_excluded_seeds=seeds)
    note = A.threshold_note(plan, report["results"])
    C.add("thresholds.recorded_vs_implemented", True,
          f"documentation inconsistency flagged: recorded p2_min_improvement = {note['recorded_value']} is not used; "
          f"implemented: {note['implemented_rule']}. Runs in the disputed interval: "
          f"{[x['name'] for x in note['runs_in_disputed_interval']] or 'none'}. The report is not modified.",
          status="WARN", note=note)

    # -------- notebook 07 context (separate experiment)
    ctx = None
    if context07:
        ctx = _export_context07(root, B, C, src, context07_report)

    # -------- headpath JSON outputs
    clean, undefined = A.sanitize(report)
    B.copy_json("headpath/sources/" + report_name, report_path)
    B.json("headpath/report.json", {"report": clean, "undefined_paths": undefined,
                                    "note": "Sanitised copy of the report: non-finite numbers replaced by null at "
                                            "undefined_paths (never 0). sources/ holds the same report as strict JSON (see "
                                            "normalization_index.json); the original's sha256 is in manifest.sources. "
                                            "Values unchanged otherwise."})
    if plan_files:
        pclean, pundef = A.sanitize(json.loads((reports / plan_name).read_text(encoding="utf-8"))
                                    if (reports / plan_name).exists() else plan)
        B.json("headpath/plan.json", {"plan": pclean, "undefined_paths": pundef})
        for p in plan_files:
            B.copy_json(f"headpath/sources/{p.name}", p)
    analysis = {"paired_table_primary": A.paired_table(report["results"]),
                "primary_test_rederived": pt, "retest": A.retest_table(report["results"]),
                "threshold_note": note, "exploratory": A.exploratory_paired_differences(report["results"]),
                "grid_metrics_npz_vs_evals": grid_cmp}
    aclean, aundef = A.sanitize(analysis)
    B.json("headpath/analysis.json", {**aclean, "undefined_paths": aundef})
    B.json("headpath/runs/index.json", bundle_runs)
    B.json("headpath/maps/index.json", maps_index)

    # -------- after-fingerprint (sources unchanged)
    after = fingerprint(root, proc_files)
    from ..diagnostics.headpath import compare_fingerprints
    fpc = compare_fingerprints(before, after)
    C.add("sources.unchanged", fpc["unchanged"], f"{fpc['n_files']} source files re-hashed after the export: "
          f"added {len(fpc['added'])}, removed {len(fpc['removed'])}, modified {len(fpc['modified'])}",
          added=fpc["added"], removed=fpc["removed"], modified=fpc["modified"], mtime_only=fpc["mtime_only_changes"])

    # -------- decode every exported array again (lossless round trip)
    n_arr = 0
    for rel in [f for f in B.files if f.endswith(".bin.gz")]:
        meta = _array_meta(rel, bundle_runs, maps_index)
        decode_array((out_dir / rel).read_bytes(), meta)
        n_arr += 1
    C.add("arrays.roundtrip", True, f"{n_arr} arrays decoded again; sha256 of the raw bytes matches (lossless)")

    # -------- JSON copied from sources: strict copies, non-finite -> null, recorded (sources untouched)
    by_source_sha = {v["path"]: v["sha256"] for v in sources.values()}
    for rel, rec in sorted(B.normalized.items()):
        unchanged = sha256_file(rec["source"]) == rec["source_sha256"]
        fp_ok = by_source_sha.get(rec["source"], rec["source_sha256"]) == rec["source_sha256"]
        C.add(f"json.normalized.{rel}", unchanged and fp_ok,
              f"{rel}: {rec['n_replaced']} non-finite values -> null (NaN {rec['by_kind']['nan']}, +Infinity "
              f"{rec['by_kind']['inf']}, -Infinity {rec['by_kind']['-inf']}); source untouched (sha256 "
              f"{rec['source_sha256'][:16]}… before and after)", status=None if not (unchanged and fp_ok) else
              ("INFO" if rec["n_replaced"] else "PASS"), n_replaced=rec["n_replaced"], by_kind=rec["by_kind"],
              source=rec["source"], source_sha256=rec["source_sha256"])
    B.json("normalization_index.json", {
        "note": ("Every JSON file copied from a source was re-serialised as strict JSON (allow_nan=False): NaN, "
                 "Infinity and -Infinity were replaced by null and nothing else changed. A null at a listed path means "
                 "UNDEFINED in the source (e.g. a correlation of a constant prediction), never 0. The source files "
                 "were only read; their sha256 is listed here and in manifest.sources."),
        "files": dict(sorted(B.normalized.items())),
        "total_replaced": sum(r["n_replaced"] for r in B.normalized.values())})
    bad_json = []
    for rel in sorted(f for f in B.files if f.endswith(".json")):
        try:
            json.loads((out_dir / rel).read_text(encoding="utf-8"), parse_constant=_reject_constant)
        except ValueError as e:
            bad_json.append(f"{rel}: {e}")
    C.add("bundle.strict_json", not bad_json, f"every JSON file in the bundle parses as strict JSON (no NaN / Infinity): "
          f"{sum(f.endswith('.json') for f in B.files)} files checked (manifest and integrity reports are written "
          "strictly afterwards and re-checked in the zip)", failures=bad_json[:20])

    # -------- manifest, integrity report, zip
    status = "FAILED" if C.failed else "PASSED"
    manifest = {"schema_version": SCHEMA_VERSION, "exporter_version": EXPORTER_VERSION, "status": status,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1),
                "experiment": "HEADPATH (notebook 08) + notebook-07 context",
                "integrity_notice": ("All values are copied or computed from the listed source files. Validation set only "
                                     "(12 diagnostic graphs); no test-set results exist. Arm U is a diagnostic deviation "
                                     "(train-time cell-head clamp removed); not a reproduction of the paper."),
                "array_encoding": {"name": ENCODING, "steps": ["little-endian float32 or int32 (float32 bit-identical; "
                                   "ints checked to fit int32)", "byte shuffle: bytes regrouped into itemsize planes",
                                   "gzip (deflate, level 9, mtime 0)"], "lossless": True,
                                   "derived_not_stored": "grid |error| = |grid_pred - grid_gt| in float32 (checked "
                                   "bit-exact against the NPZ grid_abs_err)"},
                "official_commit": _official_commit(), "sources": sources,
                "json_normalization": {"rule": "NaN, Infinity, -Infinity -> null in every JSON copied from a source "
                                               "(strict JSON, allow_nan=False); details in normalization_index.json",
                                       "files": len(B.normalized),
                                       "total_replaced": sum(r["n_replaced"] for r in B.normalized.values())},
                "checks_summary": C.summary()}
    integ = {"status": status, "summary": C.summary(), "checks": C.items, "fingerprint": fpc,
             "bundle_files": len(B.files) + 3, "bundle_bytes_excluding_reports": sum(v["size"] for v in B.files.values())}
    ic, iu = A.sanitize(integ)
    B.json("integrity_report.json", {**ic, "undefined_paths": iu})
    B._put("INTEGRITY_REPORT.md", integrity_markdown(integ, manifest).encode("utf-8"))
    manifest["files"] = dict(sorted(B.files.items()))           # every file except manifest.json itself
    B.json("manifest.json", manifest)
    if C.failed:
        raise ExportValidationError(f"{len(C.failed)} integrity checks failed; see {out_dir / 'integrity_report.json'}: "
                                    + "; ".join(f"{c['id']}: {c['message']}" for c in C.failed[:10]))
    zpath = out_dir.with_suffix(".zip")
    if zpath.exists():
        raise FileExistsError(zpath)
    with zipfile.ZipFile(zpath, "w") as z:
        for rel in sorted(B.files):
            comp = zipfile.ZIP_STORED if rel.endswith(".gz") else zipfile.ZIP_DEFLATED
            z.write(out_dir / rel, f"{out_dir.name}/{rel}", compress_type=comp)
    v = verify_bundle(zpath)
    if v["status"] != "PASSED":
        raise ExportValidationError(f"zip verification failed: {v['errors'][:5]}")
    return {"status": status, "zip": str(zpath), "zip_sha256": sha256_file(zpath), "zip_bytes": zpath.stat().st_size,
            "integrity_report": str(out_dir / "INTEGRITY_REPORT.md"), "summary": C.summary(), "verify": v}


def A_design(sample: str) -> str:
    return sample.split("-")[1]


def _finite_only(d):
    return A.sanitize(d)[0]


def _array_meta(rel, runs, maps_index):
    for r in runs.values():
        for arrs in r["maps"].values():
            for m in arrs.values():
                if m["file"] == rel:
                    return m
    for e in maps_index.values():
        for m in e["arrays"].values():
            if m["file"] == rel:
                return m
    raise KeyError(rel)


def _official_commit():
    try:
        from .. import official
        return official.official_commit()
    except Exception:  # noqa: BLE001
        return None


# docs/POST_COLLAPSE_FORENSIC_REPORT.md section 3.2: summary fields of the notebook-07 report that must not be
# shown at face value (reporting defects; training unaffected). Exported with the data so the dashboard can label them.
CONTEXT07_FIELD_CAVEATS = {
    "collapsed": "A1: detects only BOTH heads dead (exactly-zero gradient tail >= 20 steps); a run can be 'not "
                 "collapsed' with a dead cell head. Per-head end state: final_val_pred_zero_frac.",
    "collapse_step": "Start of the FINAL zero-gradient run only; earlier zero steps that recovered are ignored.",
    "steps_both_heads_ge_0999": "A2: actually counts steps where EITHER head is >= 99.9 % clamped (max of the two), "
                                "not both.",
    "min_cell_raw_le0": "A3: cell head's best step only; cannot classify per-head state.",
    "max_grid_raw_le0": "A3: grid head's worst step only; cannot classify per-head state.",
    "max_grad_norm": "A4: includes non-finite norms of scaler-skipped steps (history.csv grad_norm_max is finite-only).",
    "final_val_avg_spearman": "A5: the official metric returns 0.0 for a constant (dead-head) prediction; the "
                              "average hides a dead head.",
    "_source": "docs/POST_COLLAPSE_FORENSIC_REPORT.md section 3.2 (A1-A6)",
}


def _export_context07(root, B, C, src, report_name=None):
    """Notebook-07 collapse experiment: a SEPARATE, earlier experiment (official model, 1 epoch, different
    arms). Exported for context only; never mixed with HEADPATH."""
    reports, runs = root / "reports", root / "runs" / "collapse_diag"
    audit_p = reports / "07_forensic_audit.json"
    audit = json.loads(audit_p.read_text(encoding="utf-8")) if audit_p.exists() else None
    if report_name is None:
        if audit and audit.get("report"):
            report_name = Path(audit["report"]).name
        else:
            cands = sorted(reports.glob("07_collapse_experiment*.json"))
            report_name = cands[-1].name if cands else None
    if not report_name or not (reports / report_name).exists():
        C.add("context07.report", False, f"notebook-07 report {report_name} not found")
        return None
    rp = src("reports/" + report_name, reports / report_name)
    rep = json.loads(rp.read_text(encoding="utf-8"))
    n_plan = len(rep.get("plan", {}).get("runs", []))
    C.add("context07.report", n_plan > 0 and len(rep.get("results", [])) == n_plan,
          f"notebook-07 report {report_name}: {len(rep.get('results', []))} results for {n_plan} planned runs "
          "(the executed notebook-07 plan has 18)",
          forensic_audit_refers_to=audit.get("report") if audit else None)
    from ..diagnostics import collapse_experiment as CE
    runs_out = {}
    for r in rep.get("results", []):
        d = runs / r["name"]
        entry = {"name": r["name"], "arm": r["arm"], "seed": r["seed"], "precision": r["precision"]}
        for nm in ("steps", "history"):
            p = d / f"{nm}.csv"
            if p.exists():
                rows = A.read_rows(src(f"runs/collapse_diag/{r['name']}/{nm}.csv", p))
                B.copy(f"context07/runs/{r['name']}/sources/{nm}.csv", p)
                B.json(f"context07/runs/{r['name']}/{nm}.json", A.columns(rows))
                entry[nm] = True
                if nm == "steps":
                    zero, tail, cstep, collapsed = CE._collapse(rows, 20) if rows else ([], 0, None, False)
                    ok = collapsed == r["collapsed"] and cstep == r["collapse_step"] and sum(zero) == r["zero_grad_steps"]
                    C.add(f"context07.{r['name']}.collapse", ok, "collapsed / collapse step / zero-gradient steps "
                          "re-derived from steps.csv agree with the notebook-07 report",
                          rederived={"collapsed": collapsed, "collapse_step": cstep, "zero_grad_steps": sum(zero)})
            else:
                C.add(f"context07.{r['name']}.{nm}", True, f"{nm}.csv not found (context page shows report values only)",
                      status="WARN")
        runs_out[r["name"]] = entry
    clean, undef = A.sanitize(rep)
    B.json("context07/report.json", {"report": clean, "undefined_paths": undef, "report_file": report_name,
                                     "field_caveats": CONTEXT07_FIELD_CAVEATS})
    B.copy_json(f"context07/sources/{report_name}", rp)
    for extra in ("07_collapse_plan.json", "07_forensic_audit.json"):
        p = reports / extra
        if p.exists():
            src("reports/" + extra, p)
            B.copy_json(f"context07/sources/{extra}", p)
            c2, u2 = A.sanitize(json.loads(p.read_text(encoding="utf-8")))
            B.json(f"context07/{extra}", {"data": c2, "undefined_paths": u2})
    B.json("context07/runs/index.json", runs_out)
    return {"report": report_name, "runs": len(runs_out)}


# ------------------------------------------------------------ verification of a finished zip (laptop or Colab)
def verify_bundle(zip_path) -> dict:
    """Independent check of a bundle zip: every file listed in manifest.json is present with the listed size
    and sha256, no unlisted files, every JSON file parses as strict JSON (no NaN), every array decodes to its
    sha256_raw, and the manifest/integrity status is PASSED."""
    errors, n_arr = [], 0
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:
            errors.append("zip CRC error")
        names = z.namelist()
        top = names[0].split("/")[0]
        man = json.loads(z.read(f"{top}/manifest.json"))
        listed = set(man["files"]) | {"manifest.json"}
        present = {n[len(top) + 1:] for n in names}
        if present != listed:
            errors.append(f"unlisted or missing files: {sorted(present ^ listed)[:10]}")
        metas = {}
        for rel in ("headpath/runs/index.json", "headpath/maps/index.json"):
            metas[rel] = json.loads(z.read(f"{top}/{rel}"))
        for rel, info in man["files"].items():
            data = z.read(f"{top}/{rel}")
            if len(data) != info["size"] or sha256_bytes(data) != info["sha256"]:
                errors.append(f"{rel}: size/sha256 mismatch")
            if rel.endswith(".json"):
                try:
                    json.loads(data, parse_constant=_reject_constant)
                except ValueError as e:
                    errors.append(f"{rel}: not strict JSON ({e})")
            if rel.endswith(".bin.gz"):
                try:
                    decode_array(data, _array_meta(rel, metas["headpath/runs/index.json"],
                                                   metas["headpath/maps/index.json"]))
                    n_arr += 1
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{rel}: {e}")
        integ = json.loads(z.read(f"{top}/integrity_report.json"))
    status = "PASSED" if not errors and man["status"] == "PASSED" and integ["status"] == "PASSED" else "FAILED"
    return {"status": status, "errors": errors, "files": len(man["files"]), "arrays_decoded": n_arr,
            "manifest_status": man["status"], "integrity_status": integ["status"], "schema_version": man["schema_version"]}


def integrity_markdown(integ: dict, manifest: dict) -> str:
    s = integ["summary"]
    lines = [f"# Dashboard export integrity report: **{integ['status']}**", "",
             f"- Created (UTC): {manifest['created_utc']} · exporter {manifest['exporter_version']} · schema {manifest['schema_version']}",
             f"- Checks: {s['PASS']} PASS · {s['WARN']} WARN · {s['INFO']} INFO · {s['FAIL']} FAIL",
             f"- Source files fingerprinted before and after: {integ['fingerprint']['n_files']} "
             f"({integ['fingerprint']['bytes'] / 1e6:.1f} MB), unchanged: **{integ['fingerprint']['unchanged']}**",
             f"- Bundle: {integ['bundle_files']} files, {integ['bundle_bytes_excluding_reports'] / 1e6:.1f} MB "
             "(without the manifest and the two integrity-report files)",
             "", f"> {manifest['integrity_notice']}", "",
             "| Status | Check | Message |", "|---|---|---|"]
    order = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
    for c in sorted(integ["checks"], key=lambda c: order[c["status"]]):
        lines.append(f"| {c['status']} | `{c['id']}` | {c['message'].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ copy to Drive with verification
def copy_verified(src_path, dst_dir, name=None) -> dict:
    """Copy a finished file into dst_dir as `name` (default: its own name), never overwriting, and verify the
    copy's sha256."""
    src_path, dst_dir = Path(src_path), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src_path.name)
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copy2(src_path, dst)
    a, b = sha256_file(src_path), sha256_file(dst)
    if a != b:
        raise ExportValidationError(f"copy of {src_path.name} differs (sha256 {a} vs {b})")
    return {"path": str(dst), "sha256": b, "bytes": dst.stat().st_size}
