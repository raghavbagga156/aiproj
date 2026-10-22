"""Stage 1 of the dashboard (docs/DASHBOARD_PLAN.md): analysis module, read-only exporter, bundle verification
and notebook 09. Runs on a SYNTHETIC Drive layout produced by the real HEADPATH and notebook-07 code on tiny
CPU models (test data only; never results)."""
import copy
import gzip
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from conftest import make_design
from verihgn_repro.analysis import headpath_results as A
from verihgn_repro.colab import DriveLayout
from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.graph import build_graph
from verihgn_repro.diagnostics import collapse_experiment as CE
from verihgn_repro.diagnostics import headpath as H
from verihgn_repro.export import dashboard_bundle as DB

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"], "val": ["mgc_pci_bridge32_a"], "test": []}


def _tree_hash(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def drive(tmp_path_factory):
    """Synthetic MyDrive/VeriHGN: processed data, 14 HEADPATH runs + report (real run_plan), and a notebook-07
    context (real collapse_experiment.run_plan)."""
    tmp = tmp_path_factory.mktemp("drive")
    raw = tmp / "raw"
    for k, d in enumerate(["mgc_fft_a", "mgc_pci_bridge32_a", "mgc_des_perf_a", "mgc_edit_dist_a"]):
        make_design(raw, d, n_cells=30 + 5 * k, n_nets=25, samples=5, seed=k + 1)
    L = DriveLayout(root=tmp / "MyDrive" / "VeriHGN").ensure()
    P.run(raw, L.processed_for("h0"), scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    (L.reports / "03_preprocessing_official9.json").write_text(json.dumps({"prep_hash": "h0"}))
    split = {"name": "official_seed42", "designs": DESIGNS}
    processed = L.processed_for("h0")
    base = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2}})
    # notebook-07 context (written first, as on the real Drive)
    p7 = CE.plan_experiment(processed, split, seeds=(42,), val_from_val_design=2, per_design_mid=2, single_n=3,
                            replica_seeds=(42,))
    g7 = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=p7["c_max"])
          for n, d in p7["sample_dirs"].items()}
    rep7 = L.reports / "07_collapse_experiment__20990101_000000.json"
    CE.run_plan(p7, base, g7, L.runs / "collapse_diag", device="cpu", report_path=rep7)
    (L.reports / "07_collapse_plan.json").write_text(json.dumps({k: v for k, v in p7.items() if k != "sample_dirs"}))
    (L.reports / "07_forensic_audit.json").write_text(json.dumps({"report": str(rep7), "runs": {}}))
    # HEADPATH: all 14 runs
    plan = H.plan_headpath(processed, split, val_from_val_design=2, eval_every=6)
    (L.reports / "08_headpath_plan.json").write_text(json.dumps(H._public(plan)))
    gb = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=plan["c_max"])
          for n, d in plan["sample_dirs"].items()}
    H.run_plan(plan, base, gb, L.runs / "headpath", device="cpu", report_path=L.reports / "08_headpath_experiment.json")
    rep = json.loads((L.reports / "08_headpath_experiment.json").read_text())
    assert rep["analysis"]["experiment_status"] == "VALID", rep["analysis"]["status_reasons"]
    return L


@pytest.fixture
def drive_copy(drive, tmp_path):
    root = tmp_path / "MyDrive" / "VeriHGN"
    shutil.copytree(drive.root, root)
    return DriveLayout(root=root)


@pytest.fixture(scope="module")
def exported(drive, tmp_path_factory):
    before = _tree_hash(drive.root)
    out = tmp_path_factory.mktemp("export") / "headpath_dashboard_test"
    res = DB.export(drive.root, out)
    return {"res": res, "out": out, "before": before}


# ------------------------------------------------------------ end to end
def test_export_passes_and_never_touches_the_sources(drive, exported):
    res = exported["res"]
    assert res["status"] == "PASSED" and Path(res["zip"]).exists()
    assert _tree_hash(drive.root) == exported["before"]                     # independent of the exporter's check
    integ = json.loads((exported["out"] / "integrity_report.json").read_text())
    assert integ["status"] == "PASSED" and integ["fingerprint"]["unchanged"] is True
    ids = {c["id"]: c["status"] for c in integ["checks"]}
    assert ids["thresholds.recorded_vs_implemented"] == "WARN"              # flagged, not fixed
    assert all(v == "PASS" for k, v in ids.items() if k.endswith(".rederived") or k.startswith("labels."))
    assert ids["primary_test.rederived"] == "PASS" and ids["sources.unchanged"] == "PASS"
    n_val = len(json.loads((drive.reports / "08_headpath_experiment.json").read_text())["plan"]["val"])
    assert n_val == 5 and sum(k.endswith(".npz") for k in ids) == 14 * n_val   # 2 val-design + 3 held-out samples
    v = DB.verify_bundle(res["zip"])
    assert v["status"] == "PASSED" and v["arrays_decoded"] > 0 and not v["errors"]


def test_every_exported_array_is_bit_identical_to_the_npz(drive, exported):
    out = exported["out"]
    runs = json.loads((out / "headpath/runs/index.json").read_text())
    maps = json.loads((out / "headpath/maps/index.json").read_text())
    rep = json.loads((drive.reports / "08_headpath_experiment.json").read_text())
    assert set(runs) == {r["name"] for r in rep["plan"]["runs"]}
    for name, r in runs.items():
        for s, arrs in r["maps"].items():
            with np.load(drive.runs / "headpath" / name / "predictions_val_official" / f"{s}.npz") as z:
                for k, meta in arrs.items():
                    a = DB.decode_array((out / meta["file"]).read_bytes(), meta)
                    assert DB.bitwise_equal(a, z[k]), (name, s, k)
                for k, meta in maps[s]["arrays"].items():
                    a = DB.decode_array((out / meta["file"]).read_bytes(), meta)
                    assert np.array_equal(a.astype(z[k].dtype), z[k]) and a.dtype.itemsize == 4, (s, k)
    assert all(m["role"] for m in maps.values())


def test_bundle_json_is_strict_and_undefined_values_are_listed_not_zeroed(drive, exported):
    out = exported["out"]
    src = json.loads((drive.reports / "08_headpath_experiment.json").read_text())
    _, undefined = A.sanitize(src)
    b = json.loads((out / "headpath/report.json").read_text(),
                   parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert b["undefined_paths"] == undefined
    for u in undefined:                                                      # each one is null, never 0
        node = b["report"]
        for part in u["path"].strip("/").split("/"):
            node = node[int(part)] if isinstance(node, list) else node[part.replace("~1", "/").replace("~0", "~")]
        assert node is None
    copy = json.loads((out / "headpath/sources/08_headpath_experiment.json").read_text(),
                      parse_constant=lambda c: pytest.fail(c))                # strict copy of the source report
    assert copy == A.sanitize(src)[0]                                          # = source with non-finite -> null only
    idx = json.loads((out / "normalization_index.json").read_text())
    rec = idx["files"]["headpath/sources/08_headpath_experiment.json"]
    assert rec["source_sha256"] == hashlib.sha256((drive.reports / "08_headpath_experiment.json").read_bytes()).hexdigest()
    assert rec["n_replaced"] == len(undefined)


def test_analysis_tables_follow_the_report(drive, exported):
    an = json.loads((exported["out"] / "headpath/analysis.json").read_text())
    rep = json.loads((drive.reports / "08_headpath_experiment.json").read_text())
    st = rep["analysis"]["primary_test"]
    assert an["primary_test_rederived"]["p_two_sided_exact"] == pytest.approx(st["p_two_sided_exact"])
    assert [r["seed"] for r in an["paired_table_primary"]] == list(A.PRIMARY_SEEDS)
    assert an["exploratory"]["label"].startswith("EXPLORATORY")
    assert all(r["excluded_from_inference"] for r in an["retest"])
    assert an["threshold_note"]["recorded_value"] == 0.10 and not an["threshold_note"]["recorded_value_used_by_code"]


def test_context07_is_exported_separately(exported):
    out = exported["out"]
    c = json.loads((out / "context07/report.json").read_text())
    assert c["report_file"] == "07_collapse_experiment__20990101_000000.json"
    assert "steps_both_heads_ge_0999" in c["field_caveats"] and "EITHER" in c["field_caveats"]["steps_both_heads_ge_0999"]
    idx = json.loads((out / "context07/runs/index.json").read_text())
    assert idx and all(e.get("steps") for e in idx.values())
    assert not any(k.startswith("context07") for k in json.loads((out / "headpath/runs/index.json").read_text()))


# ------------------------------------------------------------ refusals and tamper detection
def test_refuses_output_inside_experiment_folders_or_non_empty(drive, tmp_path):
    for bad in (drive.runs / "x", drive.reports / "x", drive.processed / "x"):
        with pytest.raises(ValueError, match="refusing"):
            DB.export(drive.root, bad)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    with pytest.raises(FileExistsError):
        DB.export(drive.root, tmp_path / "full")


def _expect_fail(L, tmp_path, check_prefix):
    out = tmp_path / "out"
    with pytest.raises(DB.ExportValidationError):
        DB.export(L.root, out)
    integ = json.loads((out / "integrity_report.json").read_text())
    assert integ["status"] == "FAILED" and not out.with_suffix(".zip").exists()
    failed = [c["id"] for c in integ["checks"] if c["status"] == "FAIL"]
    assert any(f.startswith(check_prefix) or check_prefix in f for f in failed), failed
    return failed


def test_tampered_prediction_file_is_detected(drive_copy, tmp_path):
    d = drive_copy.runs / "headpath" / "HEADPATH_U__seed43" / "predictions_val_official"
    f = sorted(d.glob("*.npz"))[0]
    with np.load(f) as z:
        arr = {k: z[k].copy() for k in z.files}
    arr["grid_pred"].flat[0] += 1.0
    np.savez_compressed(f, **arr)
    _expect_fail(drive_copy, tmp_path, "HEADPATH_U__seed43.predictions_marker")


def test_consistently_tampered_prediction_content_is_detected(drive_copy, tmp_path):
    """Content changed AND the marker rewritten to match: the content checks still catch it."""
    d = drive_copy.runs / "headpath" / "HEADPATH_O__seed44" / "predictions_val_official"
    f = sorted(d.glob("*.npz"))[0]
    with np.load(f) as z:
        arr = {k: z[k].copy() for k in z.files}
    arr["grid_pred"] = arr["grid_pred"] + np.float32(0.5)
    np.savez_compressed(f, **arr)
    m = json.loads((d / "_COMPLETE.json").read_text())
    m["files"][f.name] = H._sha256(f)
    (d / "_COMPLETE.json").write_text(json.dumps(m))
    failed = _expect_fail(drive_copy, tmp_path, "HEADPATH_O__seed44")
    assert any(x.endswith(".npz") or x.endswith("grid_metrics_from_npz") for x in failed)


def test_tampered_report_category_is_detected(drive_copy, tmp_path):
    p = drive_copy.reports / "08_headpath_experiment.json"
    rep = json.loads(p.read_text())
    r = next(x for x in rep["results"] if x["name"] == "HEADPATH_U__seed45")
    r["category"] = "LEARNS_AND_GENERALIZES" if r["category"] != "LEARNS_AND_GENERALIZES" else "INCONCLUSIVE"
    p.write_text(json.dumps(rep))
    failed = _expect_fail(drive_copy, tmp_path, "HEADPATH_U__seed45.rederived")
    assert "primary_test.rederived" in failed


def test_tampered_training_log_is_detected(drive_copy, tmp_path):
    p = drive_copy.runs / "headpath" / "HEADPATH_O__seed46" / "steps.csv"
    df = pd.read_csv(p)
    df.loc[len(df) - 1, "cell_wmse_clamped"] *= 0.5
    df.to_csv(p, index=False)
    _expect_fail(drive_copy, tmp_path, "HEADPATH_O__seed46.rederived")


def test_label_provenance_is_checked_against_the_processed_samples(drive_copy, tmp_path):
    rep = json.loads((drive_copy.reports / "08_headpath_experiment.json").read_text())
    s = rep["plan"]["val"][0]
    f = drive_copy.processed_for("h0") / s.split("-")[1] / s / "sample.npz"
    with np.load(f) as z:
        arr = {k: z[k].copy() for k in z.files}
    arr["label_h"] = arr["label_h"] + 1
    np.savez(f, **arr)
    _expect_fail(drive_copy, tmp_path, f"labels.{s}")


def test_verify_bundle_detects_a_modified_member(exported, tmp_path):
    src = Path(exported["res"]["zip"])
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(bad, "w") as zo:
        for n in zi.namelist():
            data = zi.read(n)
            if n.endswith("headpath/analysis.json"):
                data = data.replace(b"EXPLORATORY", b"exploratory")
            zo.writestr(n, data)
    v = DB.verify_bundle(bad)
    assert v["status"] == "FAILED" and any("analysis.json" in e for e in v["errors"])


# ------------------------------------------------------------ unit tests
def test_lossless_encoding_edge_cases():
    a = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1e-45, 3.4e38, 1.0 / 3], dtype=np.float32).reshape(2, 4)
    blob, meta = DB.encode_array(a)
    assert DB.bitwise_equal(DB.decode_array(blob, meta), a) and meta["dtype"] == "float32"
    assert gzip.decompress(blob) != a.tobytes() or a.size < 2                  # shuffled, not plain
    i = np.array([0, 5, 2 ** 31 - 1], dtype=np.int64)
    blob, meta = DB.encode_array(i)
    assert np.array_equal(DB.decode_array(blob, meta), i) and meta["dtype"] == "int32"
    with pytest.raises(DB.ExportValidationError, match="int32"):
        DB.encode_array(np.array([2 ** 31], dtype=np.int64))
    with pytest.raises(DB.ExportValidationError, match="float32"):
        DB.encode_array(np.zeros(3, dtype=np.float64))
    assert DB.encode_array(a)[0] == DB.encode_array(a)[0]                     # deterministic bytes


def test_sanitize_and_strict_json():
    clean, undef = A.sanitize({"a": [1.0, float("nan")], "b/c": {"d": float("inf")}, "e": -float("inf"), "f": 0.0})
    assert clean == {"a": [1.0, None], "b/c": {"d": None}, "e": None, "f": 0.0}
    assert undef == [{"path": "/a/1", "kind": "nan"}, {"path": "/b~1c/d", "kind": "inf"}, {"path": "/e", "kind": "-inf"}]
    with pytest.raises(ValueError):
        DB.dumps_strict({"x": float("nan")})
    col = A.columns([{"a": "1", "b": "nan", "c": "True", "d": ""}, {"a": "2.5", "b": "inf", "c": "False", "d": "x"}])
    assert col["data"] == {"a": [1, 2.5], "b": [None, None], "c": [True, False], "d": [None, "x"]}
    assert col["undefined"] == {"b": {"0": "nan", "1": "inf"}}


def test_exact_mcnemar_equals_scipy():
    for n in range(0, 9):
        for k in range(0, n + 1):
            p = A.sign_test_exact(k, n - k)
            ref = ss.binomtest(max(k, n - k), n, 0.5).pvalue if n else 1.0
            assert p == pytest.approx(ref, rel=1e-12, abs=1e-15)
    assert A.sign_test_exact(2, 0) == 0.5 and A.sign_test_exact(6, 0) == 0.03125 and A.sign_test_exact(5, 0) == 0.0625


def test_threshold_note_flags_without_changing_anything():
    plan = {"thresholds": dict(H.THRESHOLDS)}
    res = [{"name": n, "P2": {"cell_wmse_clamped_improvement_vs_zero": v}}
           for n, v in (("a", 0.09), ("b", 0.11), ("c", 0.2), ("d", 0.05), ("e", float("nan")))]
    before = json.dumps(res)
    note = A.threshold_note(plan, res)
    assert [x["name"] for x in note["runs_in_disputed_interval"]] == ["a", "b"]
    assert note["runs_in_disputed_interval"][0] == {"name": "a", "improvement": 0.09, "implemented_outcome": "BORDERLINE",
                                                    "single_threshold_reading": "FAIL"}
    assert json.dumps(res) == before                                          # inputs not modified
    for v, exp in ((0.08, "FAIL"), (0.0800001, "BORDERLINE"), (0.12, "PASS"), (float("nan"), "FAIL")):
        assert A.p2_outcome_effective(v, plan["thresholds"]) == exp == H.p2_outcome(v, plan["thresholds"])


def test_exploratory_differences_keep_undefined_as_none():
    def r(arm, seed, imp, rho, grho):
        return {"name": f"HEADPATH_{arm}__seed{seed}", "arm": arm, "seed": seed, "retest": False,
                "run_status": "COMPLETED", "P2": {"cell_wmse_clamped_improvement_vs_zero": imp},
                "P3": {"final_val_cell_spearman_indep": rho},
                "grid": {"final_val_spearman_indep": grho, "wmse_improvement_vs_zero": 0.0}}
    res = [r("O", 42, 0.0, float("nan"), 0.1), r("U", 42, 0.3, 0.2, float("nan")),
           {**r("U", 42, 9.0, 9.0, 9.0), "retest": True}]
    ex = A.exploratory_paired_differences(res)
    m = ex["metrics"]
    assert m["P2_cell_wmse_clamped_improvement"]["per_seed"][0]["U_minus_O"] == pytest.approx(0.3)
    assert m["final_val_cell_spearman_indep"]["per_seed"][0] == {"seed": 42, "O": None, "U": 0.2, "U_minus_O": None}
    assert m["final_val_grid_spearman_indep"]["n_defined_pairs"] == 0
    assert "not pre-registered" in ex["label"]


# ------------------------------------------------------------ notebook 09
def test_notebook_09_cells_run_on_synthetic_layout(drive_copy, tmp_path, monkeypatch):
    from verihgn_repro import colab
    nb = json.loads((ROOT / "notebooks" / "09_results_export.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][1:]
    flushed = []
    monkeypatch.setattr(colab, "flush_drive", lambda paths, remount_verify=True: flushed.append([str(p) for p in paths])
                        or {"flushed_and_unmounted": True, "remount_verified": remount_verify})
    before = _tree_hash(drive_copy.runs) | {("reports/" + k): v for k, v in _tree_hash(drive_copy.reports).items()}
    ns = {"L": drive_copy, "CODE_DIR": str(ROOT), "LOCAL_EXPORT_ROOT": str(tmp_path / "local")}
    for src in cells:
        exec(compile(src, "nb09", "exec"), ns)
    zips = list(drive_copy.export.glob("headpath_dashboard_*.zip"))
    assert len(zips) == 1 and DB.verify_bundle(zips[0])["status"] == "PASSED"
    assert list(drive_copy.export.glob("headpath_dashboard_*_INTEGRITY_REPORT.md"))
    after = _tree_hash(drive_copy.runs) | {("reports/" + k): v for k, v in _tree_hash(drive_copy.reports).items()}
    assert after == before                                                   # runs/ and reports/ untouched
    assert len(flushed) == 1 and str(zips[0]) in flushed[0]


# ------------------------------------------------------------ non-finite JSON in source files (Colab failure, 2026-10-22)
def _strict(text):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_normalize_json_nested_nan_inf_null_and_finite():
    src = {"a": float("nan"), "b": [1.5, float("inf"), {"c": -float("inf"), "d": None}], "e": {"f": [[float("nan")]]},
           "g": 0, "h": -2.25e-300, "i": "NaN", "j": True, "k": None}
    raw = json.dumps(src).encode()                                         # how Python wrote the reports
    assert b"NaN" in raw and b"Infinity" in raw and b"-Infinity" in raw
    out, rec = DB.normalize_json_bytes(raw)
    got = _strict(out.decode())
    assert got == {"a": None, "b": [1.5, None, {"c": None, "d": None}], "e": {"f": [[None]]},
                   "g": 0, "h": -2.25e-300, "i": "NaN", "j": True, "k": None}   # strings / null / finite untouched
    assert rec["n_replaced"] == 4 and rec["by_kind"] == {"nan": 2, "inf": 1, "-inf": 1}
    assert sorted(rec["paths"]) == ["/a", "/b/1", "/b/2/c", "/e/f/0/0"]
    assert list(got) == list(src)                                          # key order preserved
    out2, rec2 = DB.normalize_json_bytes(out)                              # idempotent: nothing left to replace
    assert out2 == out and rec2["n_replaced"] == 0


def test_normalization_does_not_mutate_the_input_object_or_the_source_file(tmp_path):
    obj = {"x": [float("nan"), {"y": float("inf")}], "z": 1.0}
    before = json.dumps(obj)
    clean, undef = A.sanitize(obj)
    assert json.dumps(obj) == before and math.isnan(obj["x"][0]) and obj["x"][1]["y"] == float("inf")
    assert clean == {"x": [None, {"y": None}], "z": 1.0} and len(undef) == 2
    f = tmp_path / "source.json"
    f.write_text(json.dumps(obj))
    raw, st = f.read_bytes(), f.stat()
    B = DB.Bundle(tmp_path / "bundle")
    B.copy_json("sources/source.json", f)
    assert f.read_bytes() == raw and f.stat().st_mtime_ns == st.st_mtime_ns          # read only
    rec = B.normalized["sources/source.json"]
    assert rec["source_sha256"] == hashlib.sha256(raw).hexdigest() and rec["n_replaced"] == 2
    assert _strict((tmp_path / "bundle/sources/source.json").read_text()) == {"x": [None, {"y": None}], "z": 1.0}


def test_bundle_refuses_verbatim_json_copies(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"v": NaN}')
    B = DB.Bundle(tmp_path / "b")
    with pytest.raises(ValueError, match="copy_json"):
        B.copy("sources/a.json", f)
    with pytest.raises(ValueError, match="copy_json"):
        B.copy("sources/a.txt", f)
    with pytest.raises(ValueError):
        DB.dumps_strict({"v": float("-inf")})


def test_export_with_nan_and_infinity_in_every_kind_of_source_json(drive_copy, tmp_path):
    """Reproduces the Colab failure: the real 08 report holds NaN, the real 07 report Infinity. The export must
    pass with strict copies, record every replacement, keep the sources byte-identical and never turn
    undefined into 0."""
    L = drive_copy
    rp = L.reports / "08_headpath_experiment.json"
    rep = json.loads(rp.read_text())
    rep["analysis"]["test_retest"]["O"]["abs_diff_final_cell_spearman"] = float("nan")   # as in the real report
    rep["results"][0]["seconds_this_session"] = float("inf")
    rep["extra_check"] = {"deep": [{"neg": -float("inf"), "null": None, "finite": 0.25}]}
    rp.write_text(json.dumps(rep))
    p7 = L.reports / "07_collapse_experiment__20990101_000000.json"
    r7 = json.loads(p7.read_text())
    r7["results"][0]["max_grad_norm"] = float("inf")                                   # as in the real 07 report (A4)
    p7.write_text(json.dumps(r7))
    st = L.runs / "headpath" / "HEADPATH_O__seed43" / "status.json"
    s = json.loads(st.read_text())
    s["best_avg_spearman"] = -float("inf")
    st.write_text(json.dumps(s))
    sources_before = _tree_hash(L.root)
    out = tmp_path / "bundle"
    res = DB.export(L.root, out)
    assert res["status"] == "PASSED" and _tree_hash(L.root) == sources_before          # sources byte-identical
    v = DB.verify_bundle(res["zip"])
    assert v["status"] == "PASSED" and not v["errors"]
    with zipfile.ZipFile(res["zip"]) as z:                                             # every .json strictly parsed
        jsons = [n for n in z.namelist() if n.endswith(".json")]
        for n in jsons:
            _strict(z.read(n).decode("utf-8"))
    assert len(jsons) > 20
    idx = _strict((out / "normalization_index.json").read_text())
    f8, f7 = idx["files"]["headpath/sources/08_headpath_experiment.json"], \
        idx["files"]["context07/sources/07_collapse_experiment__20990101_000000.json"]
    assert f8["n_replaced"] >= 3 and f8["by_kind"]["nan"] >= 1 and f8["by_kind"]["inf"] >= 1 and f8["by_kind"]["-inf"] >= 1
    assert f7["by_kind"]["inf"] >= 1 and "/results/0/max_grad_norm" in f7["paths"]
    assert f8["source_sha256"] == hashlib.sha256(rp.read_bytes()).hexdigest()
    sj = idx["files"]["headpath/runs/HEADPATH_O__seed43/sources/status.json"]
    assert sj["by_kind"]["-inf"] == 1
    integ = _strict((out / "integrity_report.json").read_text())
    ids = {c["id"]: c for c in integ["checks"]}
    assert ids["json.normalized.headpath/sources/08_headpath_experiment.json"]["details"]["n_replaced"] == f8["n_replaced"]
    assert ids["bundle.strict_json"]["status"] == "PASS" and ids["sources.unchanged"]["status"] == "PASS"
    assert "08_headpath_experiment.json" in (out / "INTEGRITY_REPORT.md").read_text()
    man = _strict((out / "manifest.json").read_text())
    assert man["json_normalization"]["total_replaced"] == idx["total_replaced"] >= 5
    # undefined stays null (never 0) in the normalised copy and in the sanitised report
    src_copy = _strict((out / "headpath/sources/08_headpath_experiment.json").read_text())
    assert src_copy["analysis"]["test_retest"]["O"]["abs_diff_final_cell_spearman"] is None
    assert src_copy["extra_check"]["deep"][0] == {"neg": None, "null": None, "finite": 0.25}
    b = _strict((out / "headpath/report.json").read_text())
    assert b["report"]["results"][0]["seconds_this_session"] is None
    assert {"path": "/results/0/seconds_this_session", "kind": "inf"} in b["undefined_paths"]
    c7 = _strict((out / "context07/report.json").read_text())
    assert c7["report"]["results"][0]["max_grad_norm"] is None
