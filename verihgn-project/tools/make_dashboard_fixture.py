"""Build the dashboard's SYNTHETIC test bundle (app/test-fixtures/), independent of the real experiment.

It runs the real HEADPATH and notebook-07 code on tiny synthetic CPU data (the same generator the Python tests
use: tests/conftest.py make_design), then the real Stage 1 exporter (export.dashboard_bundle.export). The
result has the genuine bundle format but values that come only from synthetic data: test data, never results.
Also writes golden vectors for the JavaScript array decoder (encoded by the Python encoder).

usage: python tools/make_dashboard_fixture.py          (about 2-4 min on a laptop CPU)
"""
import base64
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
OUT = ROOT / "app" / "test-fixtures"


def build_synthetic_drive(tmp: Path):
    from conftest import make_design
    from verihgn_repro.colab import DriveLayout
    from verihgn_repro.config import load_config
    from verihgn_repro.data import preprocess as P
    from verihgn_repro.data.graph import build_graph
    from verihgn_repro.diagnostics import collapse_experiment as CE
    from verihgn_repro.diagnostics import headpath as H
    raw = tmp / "raw"
    for k, d in enumerate(["mgc_fft_a", "mgc_pci_bridge32_a", "mgc_des_perf_a", "mgc_edit_dist_a"]):
        make_design(raw, d, n_cells=30 + 5 * k, n_nets=25, samples=5, seed=k + 101)   # seeds differ from the tests
    L = DriveLayout(root=tmp / "MyDrive" / "VeriHGN").ensure()
    P.run(raw, L.processed_for("synthetic"), scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    (L.reports / "03_preprocessing_official9.json").write_text(json.dumps({"prep_hash": "synthetic"}))
    split = {"name": "official_seed42", "designs": {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"],
                                                    "val": ["mgc_pci_bridge32_a"], "test": []}}
    processed = L.processed_for("synthetic")
    base = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2}})
    p7 = CE.plan_experiment(processed, split, seeds=(42,), val_from_val_design=2, per_design_mid=2, single_n=3,
                            replica_seeds=(42,))
    g7 = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=p7["c_max"]) for n, d in p7["sample_dirs"].items()}
    rep7 = L.reports / "07_collapse_experiment__synthetic.json"
    CE.run_plan(p7, base, g7, L.runs / "collapse_diag", device="cpu", report_path=rep7)
    (L.reports / "07_collapse_plan.json").write_text(json.dumps({k: v for k, v in p7.items() if k != "sample_dirs"}))
    (L.reports / "07_forensic_audit.json").write_text(json.dumps({"report": str(rep7), "runs": {}}))
    plan = H.plan_headpath(processed, split, val_from_val_design=2, eval_every=6)
    (L.reports / "08_headpath_plan.json").write_text(json.dumps(H._public(plan)))
    gb = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=plan["c_max"]) for n, d in plan["sample_dirs"].items()}
    H.run_plan(plan, base, gb, L.runs / "headpath", device="cpu", report_path=L.reports / "08_headpath_experiment.json")
    return L


def golden_vectors() -> dict:
    from verihgn_repro.export.dashboard_bundle import encode_array
    cases = {
        "float_edges": np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1e-45, 3.4028235e38, 1.0 / 3, -2.5], dtype=np.float32).reshape(3, 3),
        "float_ramp": np.linspace(-1, 1, 1000, dtype=np.float32).reshape(25, 40),
        "int_range": np.array([0, 1, -1, 2 ** 31 - 1, -(2 ** 31), 123456789], dtype=np.int64),
    }
    out = {}
    for name, a in cases.items():
        blob, meta = encode_array(a)
        raw = np.ascontiguousarray(a.astype(np.float32 if a.dtype.kind == "f" else np.int32)).astype("<f4" if a.dtype.kind == "f" else "<i4").tobytes()
        out[name] = {"blob_b64": base64.b64encode(blob).decode(), "meta": {**meta, "file": f"golden/{name}"},
                     "raw_hex": raw.hex()}
    return out


def main():
    from verihgn_repro.export import dashboard_bundle as DB
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verihgn_fixture_") as t:
        tmp = Path(t)
        L = build_synthetic_drive(tmp)
        res = DB.export(L.root, tmp / "export" / "headpath_dashboard_synthetic")
        if res["status"] != "PASSED":
            raise SystemExit(res)
        dst = OUT / "synthetic_bundle.zip"
        if dst.exists():
            dst.unlink()                       # the fixture is regenerated; it is test data, not a result
        shutil.copy2(res["zip"], dst)
        report = json.loads((L.reports / "08_headpath_experiment.json").read_text())
    (OUT / "golden_arrays.json").write_text(json.dumps(golden_vectors(), indent=1))
    meta = {"note": "SYNTHETIC test data produced by tools/make_dashboard_fixture.py on random synthetic designs. "
                    "Not experimental results.",
            "zip": dst.name, "zip_sha256": DB.sha256_file(dst), "primary_p": report["analysis"]["primary_test"]["p_two_sided_exact"],
            "n_discordant": report["analysis"]["primary_test"]["n_discordant"]}
    (OUT / "fixture_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
