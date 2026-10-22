"""Integration on synthetic CircuitNet-shaped data: preprocess -> save/reload ->
graphs -> validation gates -> official model -> training with resume."""
import json

import numpy as np
import pytest
import torch

from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.graph import build_graph
from verihgn_repro.data.validate import validate_dataset, validate_graph
from verihgn_repro.training.checkpointing import CheckpointManager, RunStatus
from verihgn_repro.training.trainer import Trainer

SPLIT = {"name": "synthetic", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"], "test": []}}
# training test reuses the val design as test purely to exercise the code path
TRAIN_SPLIT = {"name": "synthetic_train", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                      "test": ["mgc_pci_bridge32_a"]}}


@pytest.fixture
def processed(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    man = P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 3 and not man["failed"], man
    assert man["cell_density_mismatch"] == []
    return out


def test_preprocess_roundtrip_and_skip(processed, synthetic_raw):
    raw, _ = synthetic_raw
    dirs = P.list_processed(processed)
    arrays, meta = P.load_sample(dirs[0])
    assert arrays["cell_x"].shape[1] == 13 and arrays["net_x"].shape[1] == 6
    assert arrays["label_h"].shape == (meta["grid_M"], meta["grid_N"])
    assert meta["feature_columns"]["cell_x"][0:2] == ["x_um", "y_um"]
    assert meta["checks"]["cell_density_exact_match"]
    man = P.run(raw, processed, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 3      # second run: cached, nothing failed


@pytest.mark.parametrize("norm", ["per_sample", "per_dataset"])
@pytest.mark.parametrize("gf,dim", [("paper_eq3", 5), ("official_code_available", 3)])
def test_graphs_pass_validation_and_official_model(processed, norm, gf, dim, tmp_path):
    res = build_experiment_graphs(processed, SPLIT, grid_features=gf, label_norm=norm,
                                  cache_dir=tmp_path / "cache", verbose=False)
    rep = validate_dataset(res["graphs"], grid_feat_dim=dim, model_check_samples=1)
    assert rep["passed"], rep["errors"]
    # cache reload gives identical graphs
    res2 = build_experiment_graphs(processed, SPLIT, grid_features=gf, label_norm=norm,
                                   cache_dir=tmp_path / "cache", verbose=False)
    a, b = res["graphs"]["train"][0], res2["graphs"]["train"][0]
    assert torch.equal(a["cell"].x, b["cell"].x) and torch.equal(a["grid_l0"].y, b["grid_l0"].y)


def test_validation_catches_corruption(processed):
    arrays, meta = P.load_sample(P.list_processed(processed)[0])
    g = build_graph(arrays, meta)
    assert validate_graph(g, 5)["errors"] == []
    bad = build_graph(arrays, meta)
    ei = bad["cell", "pin", "net"].edge_index.clone()   # tensors share memory with `arrays`
    ei[1, 0] = 10 ** 6
    bad["cell", "pin", "net"].edge_index = ei
    assert any("out of range" in e for e in validate_graph(bad, 5)["errors"])
    bad2 = build_graph(arrays, meta)
    bad2["cell"].y = bad2["cell"].y.clone()
    bad2["cell"].y[0] += 1.0
    assert validate_graph(bad2, 5)["errors"]
    assert validate_graph(g, 3)["errors"]       # wrong grid feature dimension detected


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_training_resume_is_exact_on_cpu(processed, tmp_path):
    res = build_experiment_graphs(processed, TRAIN_SPLIT, verbose=False)
    over = {"model": {"hidden_dim": 16, "num_layers": 2},
            "training": {"precision": "fp32", "num_epochs": 3, "max_epochs_this_session": None}}
    cfg = load_config(overrides=over)
    t_full = Trainer(cfg, res["graphs"], tmp_path / "full", device="cpu")
    t_full.run()
    cfg2 = load_config(overrides={**over, "training": {**over["training"], "max_epochs_this_session": 2}})
    Trainer(cfg2, res["graphs"], tmp_path / "split", device="cpu").run()
    assert RunStatus(tmp_path / "split").read()["epoch"] == 2
    cfg3 = load_config(overrides=over)
    t_res = Trainer(cfg3, res["graphs"], tmp_path / "split", device="cpu")
    st = t_res.run()
    assert st["state"] == "completed" and st["epoch"] == 3
    torch.testing.assert_close(_params(t_full.model), _params(t_res.model), rtol=0, atol=0)
    hist = (tmp_path / "split" / "history.csv").read_text().strip().splitlines()
    assert len(hist) == 1 + 3
    result = t_res.test()
    assert result["precision_eval"] == "fp32"
    assert (tmp_path / "split" / "eval" / "test_metrics.json").exists()
    preds = list((tmp_path / "split" / "predictions").glob("*.npz"))
    assert preds
    z = np.load(preds[0])
    assert z["grid_pred"].shape == z["grid_gt"].shape == z["raw_h_overflow"].shape

    # demo bundle: decodes back to exactly the saved predictions
    import base64
    from verihgn_repro.export.demo_bundle import export_run
    info = export_run(tmp_path / "split", processed, tmp_path / "bundle")
    man = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert all("provenance" in it for it in man["items"])
    rec = json.loads(next((tmp_path / "bundle" / "predictions").glob("*.json")).read_text())
    dec = np.frombuffer(base64.b64decode(rec["grid_pred"]), dtype="<f4").reshape(rec["grid_M"], rec["grid_N"])
    np.testing.assert_array_equal(dec, np.load(tmp_path / "split" / "predictions" / f"{rec['sample']}.npz")["grid_pred"])
    assert info["size_mb"] < 50


def test_corrupted_last_checkpoint_falls_back(tmp_path):
    cm = CheckpointManager(tmp_path, keep_every=1)
    cm.save({"epoch": 1}, 1, is_best=True)
    cm.save({"epoch": 2}, 2, is_best=False)
    cm.last_path.write_bytes(b"truncated")
    state, path = cm.load_latest()
    assert state["epoch"] == 2 and path.name == "epoch_0002.pt"


def test_status_reports_interrupted_when_stale(tmp_path):
    s = RunStatus(tmp_path, stale_after_s=0)
    s.write("running", epoch=5)
    assert s.describe() == "interrupted (resumable)"
    s.write("completed", epoch=9)
    assert s.describe() == "completed"


def test_benchmark_and_memory_probe_run_on_cpu(processed, tmp_path):
    from verihgn_repro.training.benchmark import measure_peak_memory, run_precision_benchmark
    res = build_experiment_graphs(processed, TRAIN_SPLIT, verbose=False)
    cfg = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 1}})
    rows = measure_peak_memory(cfg, res["graphs"], precisions=("fp32",), device="cpu")
    assert {r["design"] for r in rows} == {"mgc_fft_a", "mgc_pci_bridge32_a"} and all(r["loss_finite"] for r in rows)
    out = run_precision_benchmark(cfg, res["graphs"], tmp_path / "bench", epochs=1, seeds=(1,),
                                  precisions=("fp32",), device="cpu")
    assert out["summary"]["fp32"]["skipped_nonfinite_total"] == 0
    assert (tmp_path / "bench" / "precision_benchmark.json").exists()
