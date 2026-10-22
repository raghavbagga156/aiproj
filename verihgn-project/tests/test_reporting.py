import json

import numpy as np
import pytest

from verihgn_repro import reporting
from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.splits import build_split_manifest

SIGS = {"mgc_fft_a": "a", "mgc_pci_bridge32_a": "b"}
MANIFESTS = {
    "official_seed42": build_split_manifest("official_seed42", {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                                "test": []}, SIGS, {}),
    "pair_aware": build_split_manifest("pair_aware", {"train": ["mgc_pci_bridge32_a"], "val": ["mgc_fft_a"],
                                                      "test": []}, SIGS, {}),
}


@pytest.fixture
def processed(synthetic_raw, tmp_path):
    raw, samples = synthetic_raw
    out = tmp_path / "processed"
    man = P.run(raw, out, cfg=P.prep_config(knn_k=4), workers=1)
    assert man["written"] == 3 and man["failed_count"] == 0
    man2 = P.run(raw, out, cfg=P.prep_config(knn_k=4), workers=1)
    assert man2["skipped_already_processed"] == 3 and man2["written"] == 0
    return out, samples


def test_audit_detects_missing_and_corrupt(processed):
    out, samples = processed
    names = [P.parse_sample_name(s)["sample"] for s in samples]
    rep = reporting.audit_processed(out, "official9", expected_samples=names + ["999-mgc_fft_a-util_0.9"])
    assert rep["audited"] == 3 and rep["ok"] == 3
    assert rep["missing"] == ["999-mgc_fft_a-util_0.9"] and not rep["passed"]
    # corrupt one meta: column order swapped -> flagged
    meta_p = P.list_processed(out)[0] / "meta.json"
    m = json.loads(meta_p.read_text())
    m["feature_columns"]["cell_x"] = list(reversed(m["feature_columns"]["cell_x"]))
    meta_p.write_text(json.dumps(m))
    rep2 = reporting.audit_processed(out, "official9", expected_samples=names)
    assert rep2["ok"] == 2 and any("column order" in e for f in rep2["failed"] for e in f["errors"])


def test_validate_variants_covers_all_combinations(processed):
    out, _ = processed
    rep = reporting.validate_variants(out, MANIFESTS, device="cpu")
    assert rep["passed"], rep
    assert len(rep["variants"]) == 6          # 2 feature sets x (per_sample + per_dataset for 2 splits)
    assert all(v["ok"] == 3 for v in rep["variants"].values())
    assert set(rep["c_max"]) == {"official_seed42", "pair_aware"}
    assert {c["grid_features"] for c in rep["official_model_checks"]} == {"paper_eq3", "official_code_available"}


def test_profiles_and_stage_report(processed, tmp_path):
    out, _ = processed
    rows = reporting.data_loading_profile(out, ["mgc_fft_a"], device="cpu", per_design=1)
    assert rows[0]["total_mb"] > 0 and rows[0]["h2d_s"] is None
    pc = reporting.persistence_check({"processed": out, "nothing": tmp_path / "nope"})
    assert pc["processed"]["files"] == 7          # 3 samples x 2 files + manifest
    assert not pc["nothing"]["exists"]
    rdir = tmp_path / "reports"
    rdir.mkdir()
    (rdir / "05_data_loading.json").write_text(json.dumps(rows))
    text = reporting.summarize_reports(rdir, rdir / "STAGE_REPORT.md")
    assert "## 7. Data loading and graph memory" in text and "not run" in text   # other sections not run
    assert (rdir / "STAGE_REPORT.md").exists()


def test_repeatability_check_exact_on_cpu(processed, tmp_path):
    from verihgn_repro.training.benchmark import repeatability_check
    out, _ = processed
    res = build_experiment_graphs(out, {"name": "t", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                                 "test": []}}, verbose=False)
    cfg = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 1}})
    r = repeatability_check(cfg, res["graphs"], tmp_path / "rep", "fp32", epochs=2, device="cpu")
    assert r["max_abs_param_diff"] == 0.0 and r["max_abs_val_spearman_diff"] == 0.0
