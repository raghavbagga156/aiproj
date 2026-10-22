"""Smoke-test investigation (docs/SMOKE_TRAINING_DIAGNOSIS.md): the signature seen
on the A100 (exactly-zero gradient norm, identical training loss in consecutive
epochs, all correlations exactly 0) is what the official clamp(min=0) heads
produce once every raw output is <= 0. These tests pin that mechanism, the
metric conventions, the collapse monitoring and the diagnostic tool."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats as ss

from verihgn_repro import official
from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.evaluation.evaluate import prediction_health
from verihgn_repro.training import trainer as T

ROOT = Path(__file__).resolve().parents[1]
SPLIT = {"name": "synthetic_train", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                "test": ["mgc_pci_bridge32_a"]}}


def _tool():
    spec = importlib.util.spec_from_file_location("diagnose_training", ROOT / "tools" / "diagnose_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def processed(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    man = P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 3 and not man["failed"]
    return out


def small_cfg(epochs=2):
    return load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2},
                                  "training": {"precision": "fp32", "num_epochs": epochs,
                                               "max_epochs_this_session": None}})


# ------------------------------------------------------------ metrics
def test_independent_metrics_agree_with_scipy_and_official():
    tool = _tool()
    ev = official.evaluation_module()
    rng = np.random.RandomState(0)
    y = np.where(rng.rand(5000) < 0.9, 0.0, rng.rand(5000)).astype(np.float32)   # zero-inflated, many ties
    p = (y + 0.05 * rng.randn(5000)).clip(min=0).astype(np.float32)
    np.testing.assert_array_equal(tool._ranks(y[:400]), ss.rankdata(y[:400]))
    np.testing.assert_array_equal(tool._ranks_fast(y), ss.rankdata(y))
    p64, y64 = p.astype(np.float64), y.astype(np.float64)          # scipy keeps float32 inputs in float32
    assert abs(tool.pearson_indep(p, y) - ss.pearsonr(p64, y64)[0]) < 1e-9
    assert abs(tool.spearman_indep(p, y) - ss.spearmanr(p64, y64)[0]) < 1e-9
    assert abs(ev.compute_pearson(p, y) - tool.pearson_indep(p, y)) < 1e-5     # official: float32 inputs
    assert abs(ev.compute_spearman(p, y) - tool.spearman_indep(p, y)) < 1e-5


def test_constant_prediction_gives_exact_zero_in_official_metrics():
    """Official convention: correlation of a constant prediction is reported as 0.0
    (mathematically undefined; the independent implementation returns nan)."""
    tool = _tool()
    ev = official.evaluation_module()
    y = np.r_[np.zeros(90), np.linspace(0.1, 1, 10)].astype(np.float32)
    p = np.zeros_like(y)
    m = ev.evaluate_predictions(p, y)
    assert m["pearson"] == 0.0 and m["spearman"] == 0.0 and m["kendall"] == 0.0
    assert np.isnan(tool.pearson_indep(p, y)) and np.isnan(tool.spearman_indep(p, y))


# ------------------------------------------------------------ mechanism
def test_all_clamped_outputs_give_exactly_zero_finite_gradient():
    lf = official.loss_module().verihgn_loss
    raw = (-torch.rand(500) - 0.01).requires_grad_(True)
    y = torch.zeros(500)
    y[:20] = torch.rand(20)
    pred = raw.clamp(min=0)                  # what PredictionHead returns
    loss = lf(pred, y, pred, y)["total"]
    loss.backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad == 0).all()
    zero_loss = lf(torch.zeros(500), y, torch.zeros(500), y)["total"]
    assert float(loss) == float(zero_loss)   # the loss no longer depends on the parameters


def _collapsed_build_model(monkeypatch, bias=-10.0):
    orig = T.build_model

    def build(cfg):
        m = orig(cfg)
        with torch.no_grad():
            m.cell_head.mlp[-1].bias.fill_(bias)
            m.grid_head.mlp[-1].bias.fill_(bias)
        return m
    monkeypatch.setattr(T, "build_model", build)


def test_collapse_reproduces_smoke_signature(processed, tmp_path, monkeypatch):
    """Force every raw head output below 0 at initialisation: the trainer then logs
    exactly the A100 smoke-test signature."""
    res = build_experiment_graphs(processed, SPLIT, verbose=False)
    _collapsed_build_model(monkeypatch)
    tr = T.Trainer(small_cfg(3), res["graphs"], tmp_path / "run", device="cpu")
    tr.run()
    h = pd.read_csv(tmp_path / "run" / "history.csv")
    assert (h["grad_norm_mean"] == 0).all() and (h["grad_zero_steps"] == h["train_steps"]).all()
    assert h["train_loss"].nunique() == 1                                  # identical every epoch
    lf = official.loss_module().verihgn_loss
    zl = [float(lf(torch.zeros_like(g["cell"].y), g["cell"].y, torch.zeros_like(g["grid_l0"].y),
                   g["grid_l0"].y)["total"]) for g in res["graphs"]["train"]]
    assert abs(h["train_loss"].iloc[0] - np.mean(zl)) < 1e-7               # = loss of the all-zero predictor
    for c in ("val_cell_spearman", "val_grid_spearman", "val_cell_pearson", "val_grid_pearson",
              "val_cell_kendall", "val_grid_kendall", "val_avg_spearman"):
        assert (h[c] == 0).all(), c
    assert (h["val_cell_pred_zero_frac"] == 1).all() and (h["val_grid_pred_std"] == 0).all()
    assert (h["skipped_nonfinite"] == 0).all()


def test_parameters_still_move_while_outputs_stay_collapsed(processed, tmp_path, monkeypatch):
    """With zero gradients AdamW still applies decoupled weight decay, so the
    parameters change but the loss and predictions do not."""
    res = build_experiment_graphs(processed, SPLIT, verbose=False)
    _collapsed_build_model(monkeypatch)
    tr = T.Trainer(small_cfg(1), res["graphs"], tmp_path / "run", device="cpu")
    tr._init_state()
    before = torch.cat([p.detach().flatten().clone() for p in tr.model.parameters()])
    out = tr._train_epoch()
    after = torch.cat([p.detach().flatten() for p in tr.model.parameters()])
    assert out["grad_norm_mean"] == 0
    assert 0 < float((after - before).abs().max()) < 1e-4    # lr*wd*|p| per step (|p| <= 10 here)


def test_healthy_model_is_not_flagged(processed, tmp_path):
    res = build_experiment_graphs(processed, SPLIT, verbose=False)
    tr = T.Trainer(small_cfg(1), res["graphs"], tmp_path / "run", device="cpu")
    tr._init_state()
    with torch.no_grad():                    # all raw outputs > 0
        tr.model.cell_head.mlp[-1].bias.fill_(5.0)
        tr.model.grid_head.mlp[-1].bias.fill_(5.0)
    out = tr._train_epoch()
    assert out["grad_zero_steps"] == 0 and out["grad_norm_mean"] > 0


def test_prediction_health():
    recs = [{"cell_pred": torch.tensor([0.0, 0.0, 1.0]), "grid_pred": torch.zeros(4)}]
    h = prediction_health(recs)
    assert h["grid_pred_zero_frac"] == 1.0 and h["grid_pred_std"] == 0.0
    assert abs(h["cell_pred_zero_frac"] - 2 / 3) < 1e-12 and h["cell_pred_std"] > 0


# ------------------------------------------------------------ diagnostic tool
def test_diagnostic_tool_runs_the_production_step(processed, tmp_path):
    tool = _tool()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"name": "s", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                          "test": []}}))
    rep = tool.main(["--processed", str(processed), "--split", str(split), "--out", str(tmp_path / "d.json"),
                     "--steps", "3", "--metric-every", "2", "--device", "cpu", "--train-samples", "2"])
    run = rep["runs"]["fp32"]
    assert len(run["steps"]) == 3 and [m["step"] for m in run["val_metrics"]] == [0, 2, 3]
    s = run["steps"][0]
    for k in ("grad_norm_pre_clip", "grad_norm_post_clip", "param_delta_l2", "eval_loss_before",
              "eval_loss_after", "cell_raw_head", "loss_components", "zero_predictor_loss"):
        assert k in s, k
    assert s["optimizer_step_ran"] and s["grad_norm_post_clip"] <= max(s["grad_norm_pre_clip"], 1.0) + 1e-6
    assert abs(s["loss_components"]["total"] - s["train_loss"]) < 1e-6
    # instrumentation is removed afterwards
    assert torch.nn.utils.clip_grad_norm_.__module__ == "torch.nn.utils.clip_grad"
    assert official.loss_module().verihgn_loss.__module__ == official.loss_module().__name__
    with pytest.raises(SystemExit, match="needs CUDA"):
        if not torch.cuda.is_available():
            tool.main(["--processed", str(processed), "--split", str(split), "--out", str(tmp_path / "e.json"),
                       "--steps", "1", "--device", "cpu", "--precisions", "fp16"])
        else:
            raise SystemExit("needs CUDA (skipped on a GPU machine)")


# ------------------------------------------------------------ equivalence with the official loop
def _official_train_module():
    import sys
    official.model_module()                   # official packages already imported under their names
    path = str(official.OFFICIAL_DIR)
    sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location("official_train", official.OFFICIAL_DIR / "train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(path)
    return mod


def test_trainer_step_equals_official_train_one_epoch(processed, tmp_path):
    """Our FP32 training step is the official train.py step: same model init, same
    graph order -> bit-identical parameters and losses after 3 epochs. So the
    collapse is not introduced by our training loop."""
    otr = _official_train_module()
    res = build_experiment_graphs(processed, {"name": "s", "designs": {"train": ["mgc_fft_a", "mgc_pci_bridge32_a"],
                                                                       "val": [], "test": []}}, verbose=False)
    graphs = {"train": res["graphs"]["train"], "val": []}
    collate = official.dataset_module().collate_fn
    cfg = small_cfg()
    ours = T.Trainer(cfg, graphs, tmp_path / "a", device="cpu")
    ours._init_state()
    order_gen = torch.Generator()
    order_gen.manual_seed(cfg.training.seed)              # replicates ours.shuffle_gen
    our_losses = [ours._train_epoch()["train_loss"] for _ in range(3)]

    theirs = T.Trainer(cfg, graphs, tmp_path / "b", device="cpu")
    theirs._init_state()                                   # same seed -> same init and RNG state
    their_losses = []
    for _ in range(3):
        order = torch.randperm(len(graphs["train"]), generator=order_gen).tolist()
        loader = [collate([graphs["train"][i]]) for i in order]
        loss, _, _ = otr.train_one_epoch(theirs.model, loader, theirs.optimizer, "cpu",
                                         {"wmse_alpha": cfg.training.wmse_alpha, "lambda_var": cfg.training.lambda_var},
                                         scaler=None)
        their_losses.append(loss)
    assert our_losses == their_losses
    for a, b in zip(ours.model.parameters(), theirs.model.parameters()):
        assert torch.equal(a, b)
