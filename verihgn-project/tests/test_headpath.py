"""HEADPATH (docs/HEADPATH_PLAN.md): tests T1-T20 that must pass before the experiment may run.

The official model, labels, graphs, optimizer and evaluation functions are untouched; arm U differs
from arm O only by the training-mode cell-head clamp. Training here is short CPU runs on synthetic
data only (test data, never results)."""
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats as ss
from torch_geometric.data import HeteroData

from conftest import make_design
from verihgn_repro import official
from verihgn_repro.config import HEADPATH_DEVIATION, deviations, load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.graph import build_graph
from verihgn_repro.diagnostics import collapse_experiment as CE
from verihgn_repro.diagnostics import headpath as H
from verihgn_repro.evaluation import head_metrics as HM
from verihgn_repro.evaluation.evaluate import predict
from verihgn_repro.model_deviations import apply_cell_head_train_unclamped, is_official_head, parameter_partition
from verihgn_repro.training import trainer as T
from verihgn_repro.training.checkpointing import CheckpointError

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "VeriHGN"
SPLIT = {"name": "s", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"], "test": []}}
OFFICIAL_SHA256 = {   # third_party/VeriHGN at 28a6f1e (recorded 2026-10-20 before any HEADPATH change)
    "models/verihgn.py": "d107c493b8eab51a3f31e43c926832f314e88e07cfac22bb07a0deda199fb7cf",
    "utils/loss.py": "d14508b036fa645c2b9a730ba276f8719de631bfca4c08f66f03f2a6e0fe4dad",
    "utils/evaluation.py": "f4f9b9e4434cb81b7c7c2f8fce66a337c79d6c4663d9cf0c8d10c05232d23ae8",
    "train.py": "37d4d062b64286173b290c6a910531a69a61ced86382455e263e9f734e4dcdc0",
}
BASE_RUN_ID = "verihgn_ispd15__official_seed42__paper_eq3__per_dataset__fp16__seed42"
SMOKE_RUN_ID = "verihgn_ispd15_smoke__official_seed42__paper_eq3__per_dataset__fp16__seed42"


def hp_cfg(arm="O", epochs=2, session=None, step_log=True, heads=True, eval_every=1, seed=42):
    return load_config(overrides={
        "model": {"hidden_dim": 16, "num_layers": 2, "cell_head_train_clamp": arm == "O"},
        "training": {"precision": "fp32", "num_epochs": epochs, "max_epochs_this_session": session, "seed": seed,
                     "step_log": step_log, "step_log_heads": heads, "eval_every_steps": eval_every, "keep_every": 0}})


@pytest.fixture
def graphs(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    return build_experiment_graphs(out, SPLIT, verbose=False)["graphs"]


def _multi(tmp_path):
    raw = tmp_path / "raw"
    for k, d in enumerate(["mgc_fft_a", "mgc_pci_bridge32_a", "mgc_des_perf_a", "mgc_edit_dist_a"]):
        make_design(raw, d, n_cells=30 + 5 * k, n_nets=25, samples=5, seed=k + 1)
    return raw


@pytest.fixture
def multi_design(tmp_path):
    out = tmp_path / "processed"
    man = P.run(_multi(tmp_path), out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 20
    return out, {"name": "m", "designs": {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"],
                                          "val": ["mgc_pci_bridge32_a"], "test": []}}


def hand_graph(nc=400, nn=300, M=10, N=10, seed=0):
    """Random graph with every edge type the official model reads (no preprocessing)."""
    g = torch.Generator().manual_seed(seed)
    d = HeteroData()
    G0, M1, N1 = M * N, (M + 1) // 2, (N + 1) // 2
    d["cell"].x, d["net"].x = torch.randn(nc, 13, generator=g), torch.randn(nn, 6, generator=g)
    d["grid_l0"].x, d["grid_l1"].x = torch.randn(G0, 5, generator=g), torch.randn(M1 * N1, 5, generator=g)
    c, n = torch.randint(0, nc, (1200,), generator=g), torch.randint(0, nn, (1200,), generator=g)
    d["cell", "pin", "net"].edge_index, d["cell", "pin", "net"].edge_attr = torch.stack([c, n]), torch.randn(1200, 3, generator=g)
    d["net", "pin_rev", "cell"].edge_index = torch.stack([n, c])
    tile = torch.randint(0, G0, (nc,), generator=g)
    d["cell", "in", "grid_l0"].edge_index = torch.stack([torch.arange(nc), tile])
    ns, gs = torch.randint(0, nn, (800,), generator=g), torch.randint(0, G0, (800,), generator=g)
    d["net", "spans", "grid_l0"].edge_index = torch.stack([ns, gs])
    d["cell", "geom", "cell"].edge_index = torch.randint(0, nc, (2, 1600), generator=g)
    d["cell", "geom", "cell"].edge_attr = torch.randn(1600, 4, generator=g)
    i = torch.arange(G0)
    p = (i // N // 2) * N1 + (i % N) // 2
    d["grid_l0", "child_of", "grid_l1"].edge_index = torch.stack([i, p])
    d["grid_l1", "parent_of", "grid_l0"].edge_index = torch.stack([p, i])
    src, dst = zip(*[(r * N1 + q, (r + a) * N1 + q + b) for r in range(M1) for q in range(N1)
                     for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)) if 0 <= r + a < M1 and 0 <= q + b < N1])
    d["grid_l1", "adj", "grid_l1"].edge_index = torch.tensor([src, dst])
    y = torch.where(torch.rand(G0, generator=g) < 0.2, torch.rand(G0, generator=g) * 0.3, torch.zeros(G0))
    d["grid_l0"].y, d["cell"].y = y, y[tile]
    return d


def real_model(seed, arm, layers=3, hidden=128):
    cfg = load_config(overrides={"model": {"hidden_dim": hidden, "num_layers": layers,
                                           "cell_head_train_clamp": arm == "O"}})
    torch.manual_seed(seed)
    return T.build_model(cfg)


def params(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def run(cfg, graphs, d, **kw):
    tr = T.Trainer(cfg, graphs, d, device="cpu")
    tr.run(**kw)
    return tr


# ------------------------------------------------------------ T1, T2: official code and defaults untouched
def test_T1_third_party_unchanged():
    for rel, h in OFFICIAL_SHA256.items():
        assert hashlib.sha256((OFFICIAL / rel).read_bytes()).hexdigest() == h, rel
    assert official.official_commit() == official.OFFICIAL_COMMIT
    if (OFFICIAL / ".git").exists():
        head = subprocess.run(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(OFFICIAL), "status", "--porcelain"], capture_output=True, text=True)
        assert head.stdout.strip() == official.OFFICIAL_COMMIT and status.stdout.strip() == ""


def test_T2_defaults_preserve_official_behaviour():
    base = load_config(ROOT / "configs" / "base.yaml")
    assert base.run_id() == BASE_RUN_ID
    for extra in ("smoke.yaml", "norm_per_sample.yaml", "split_pair_aware.yaml", "ablation_official_code_features.yaml"):
        c = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / extra)
        e, t = c.experiment, c.training                   # the run-id formula before HEADPATH
        assert c.run_id() == f"{e.name}__{e.split}__{e.grid_features}__{e.label_norm}__{t.precision}__seed{t.seed}"
    smoke = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "smoke.yaml")
    smoke.experiment.name = "verihgn_ispd15_smoke"        # as notebook 05 names the smoke run
    assert smoke.run_id() == SMOKE_RUN_ID
    t = base.training
    assert base.model.cell_head_train_clamp and not t.step_log_heads and t.eval_every_steps == 0
    assert HEADPATH_DEVIATION not in deviations(base)
    m = T.build_model(base)
    assert is_official_head(m.cell_head) and is_official_head(m.grid_head)
    dev = copy.deepcopy(base)
    dev.model.cell_head_train_clamp = False
    assert dev.run_id() == BASE_RUN_ID + "__DEV-cellhead-train-unclamped"
    assert deviations(dev) == deviations(base) + [HEADPATH_DEVIATION]
    md = T.build_model(dev)
    assert not is_official_head(md.cell_head) and is_official_head(md.grid_head)
    old = base.to_dict()                      # config stored in a checkpoint written before HEADPATH
    del old["model"]["cell_head_train_clamp"], old["training"]["step_log_heads"], old["training"]["eval_every_steps"]
    assert load_config(overrides=old).run_id() == BASE_RUN_ID
    with pytest.raises(AssertionError):
        load_config(overrides={"training": {"step_log_heads": True}})       # needs step_log
    with pytest.raises(TypeError):
        apply_cell_head_train_unclamped(md)                                  # never applied twice


# ------------------------------------------------------------ T3: identical initial state, seeds 42-47
def test_T3_arms_have_identical_initial_state(graphs, tmp_path):
    base = load_config(ROOT / "configs" / "base.yaml", overrides={"training": {"precision": "fp32"}})
    for seed in H.SEEDS:
        states = []
        for arm in ("O", "U"):
            cfg = copy.deepcopy(base)
            cfg.training.seed, cfg.model.cell_head_train_clamp = seed, arm == "O"
            tr = T.Trainer(cfg, graphs, tmp_path / f"{arm}{seed}", device="cpu")
            tr._init_state()
            states.append({"sd": tr.model.state_dict(), "names": [n for n, _ in tr.model.named_parameters()],
                           "opt": tr.optimizer.state_dict()["param_groups"], "torch": torch.get_rng_state(),
                           "np": np.random.get_state()[1].copy(), "py": random.getstate(),
                           "shuffle": tr.shuffle_gen.get_state()})
        a, b = states
        assert a["names"] == b["names"] and list(a["sd"]) == list(b["sd"])
        assert all(torch.equal(a["sd"][k], b["sd"][k]) for k in a["sd"]), seed
        assert a["opt"] == b["opt"] and torch.equal(a["torch"], b["torch"]) and torch.equal(a["shuffle"], b["shuffle"])
        assert np.array_equal(a["np"], b["np"]) and a["py"] == b["py"]


# ------------------------------------------------------------ T4 (requirement 7), T5: eval and train paths
def test_T4_arm_u_official_evaluation_bit_identical_to_fresh_official_model(graphs, tmp_path, monkeypatch):
    for when in ("init", "trained"):
        tr = T.Trainer(hp_cfg("U", eval_every=0, heads=False), graphs, tmp_path / when, device="cpu")
        tr.run() if when == "trained" else tr._init_state()
        ref = T.build_model(hp_cfg("O"))
        ref.load_state_dict(tr.model.state_dict())
        a, b = predict(tr.model, tr.batches["val"], "cpu"), predict(ref, tr.batches["val"], "cpu")
        for x, y in zip(a, b):
            assert torch.equal(x["cell_pred"], y["cell_pred"]) and torch.equal(x["grid_pred"], y["grid_pred"]), when
            assert (x["cell_pred"] >= 0).all()
        fc = H.fresh_official_check(tr, atol=0.0, rtol=0.0)                    # exact on the CPU
        assert fc["bit_identical"] and fc["equivalent"] and fc["max_abs_diff_self_repeat"] == 0.0
        assert fc["max_abs_diff_vs_fresh_official"] == fc["max_rel_diff_vs_fresh_official"] == 0.0
    import verihgn_repro.evaluation.evaluate as EV        # the tolerance rule itself: |a-b| <= atol + rtol*|b|
    orig_build, orig_predict = T.build_model, EV.predict

    def build(c):
        m = orig_build(c)
        m._perturb = c.model.cell_head_train_clamp       # mark the freshly built official model
        return m

    def predict_perturbed(model, batches, device):       # adds 5e-7 to that model's predictions only
        recs = orig_predict(model, batches, device)
        if getattr(model, "_perturb", False):
            for r in recs:
                r["cell_pred"], r["grid_pred"] = r["cell_pred"] + 5e-7, r["grid_pred"] + 5e-7
        return recs
    monkeypatch.setattr(T, "build_model", build)
    monkeypatch.setattr(EV, "predict", predict_perturbed)
    fc = H.fresh_official_check(tr, atol=1e-6, rtol=1e-6)
    assert not fc["bit_identical"] and fc["equivalent"]
    assert 4e-7 < fc["max_abs_diff_vs_fresh_official"] < 6e-7 and fc["max_tolerance_ratio"] <= 1.0   # float32 rounding
    assert not H.fresh_official_check(tr, atol=1e-7, rtol=0.0)["equivalent"]
    monkeypatch.setattr(T, "build_model", orig_build)
    monkeypatch.setattr(EV, "predict", orig_predict)
    base = official.model_module().PredictionHead       # eval-mode arm U calls the official forward itself
    calls, orig = [], base.forward
    monkeypatch.setattr(base, "forward", lambda self, h: (calls.append(type(self).__name__), orig(self, h))[1])
    predict(tr.model, tr.batches["val"], "cpu")
    assert calls.count("TrainUnclampedPredictionHead") == len(tr.batches["val"])


def test_T5_train_mode_outputs():
    data = hand_graph()
    for seed in (42, 43):
        mo, mu = real_model(seed, "O"), real_model(seed, "U")
        raw = {}
        hk = [m.cell_head.mlp.register_forward_hook(lambda _m, i, o, k=k: raw.__setitem__(k, o.detach().squeeze(-1)))
              for k, m in (("O", mo), ("U", mu))]
        out = {}
        for k, m in (("O", mo), ("U", mu)):
            m.train()
            torch.manual_seed(1000 + seed)
            out[k] = m(data)
        for h in hk:
            h.remove()
        assert torch.equal(raw["O"], raw["U"])                                 # same weights, same dropout masks
        assert torch.equal(out["U"]["cell_pred"].detach(), raw["U"]) and (raw["U"] < 0).any()
        assert torch.equal(out["O"]["cell_pred"].detach(), raw["O"].clamp(min=0))
        assert torch.equal(out["O"]["grid_pred"].detach(), out["U"]["grid_pred"].detach())


# ------------------------------------------------------------ T6: paired single step, positive / negative control
def _one_step(cfg, graphs, d, bias, monkeypatch):
    orig = T.build_model

    def build(c):
        m = orig(c)
        with torch.no_grad():
            m.cell_head.mlp[-1].bias.fill_(bias)
        return m
    monkeypatch.setattr(T, "build_model", build)
    tr = T.Trainer(cfg, {"train": graphs["train"][:1], "val": graphs["val"]}, d, device="cpu")
    tr._init_state()
    tr.epoch = 1
    tr._train_epoch()
    monkeypatch.setattr(T, "build_model", orig)
    return tr


def test_T6_paired_step_positive_and_negative_controls(graphs, tmp_path, monkeypatch):
    pos = {a: _one_step(hp_cfg(a, eval_every=0), graphs, tmp_path / f"p{a}", 5.0, monkeypatch) for a in "OU"}
    assert all(t._step_rows[0]["cell_raw_le0"] == 0.0 for t in pos.values())   # clamp is the identity here
    assert torch.equal(params(pos["O"].model), params(pos["U"].model))           # => the step is bit-identical
    assert pos["O"]._step_rows[0]["loss"] == pos["U"]._step_rows[0]["loss"]
    neg = {a: _one_step(hp_cfg(a, eval_every=0), graphs, tmp_path / f"n{a}", -10.0, monkeypatch) for a in "OU"}
    ro, ru = neg["O"]._step_rows[0], neg["U"]._step_rows[0]
    assert ro["cell_raw_le0"] == ru["cell_raw_le0"] == 1.0
    assert ro["grad_norm_cell_only"] == 0.0                                       # official: dead head, no gradient
    assert 0.0 < ru["grad_norm_cell_only"] < float("inf")                         # arm U: gradient reaches it
    assert ro["grad_norm_grid_only"] > 0 and ro["grad_norm_shared"] > 0


# ------------------------------------------------------------ T7, T11, T12: order, budget, logged terms
def test_T7_T11_T12_order_budget_and_logged_terms(graphs, tmp_path):
    lam = hp_cfg().training.lambda_var
    trs = {a: run(hp_cfg(a), graphs, tmp_path / a) for a in "OU"}
    st = {a: pd.read_csv(tmp_path / a / "steps.csv") for a in "OU"}
    ev = {a: pd.read_csv(tmp_path / a / "evals.csv") for a in "OU"}
    n = len(graphs["train"])
    for a in "OU":
        assert st[a]["global_step"].tolist() == list(range(1, 2 * n + 1))
        assert ev[a]["global_step"].tolist() == list(range(0, 2 * n + 1))
    assert st["O"]["sample"].tolist() == st["U"]["sample"].tolist()
    assert st["O"]["graph_index"].tolist() == st["U"]["graph_index"].tolist()
    for a, s in st.items():
        parts = s["grad_norm_cell_only"] ** 2 + s["grad_norm_grid_only"] ** 2 + s["grad_norm_shared"] ** 2
        assert np.allclose(parts, s["grad_norm"] ** 2, rtol=1e-5, atol=1e-12), a
        assert np.allclose(s["cell_wmse_loss"] + s["grid_wmse_loss"] + s["var_reg_loss"], s["loss"], rtol=1e-5, atol=1e-9)
        assert np.allclose(s["var_reg_loss"], lam * (s["var_cell_loss"] + s["var_grid_loss"]), rtol=1e-5, atol=1e-12)
        assert (s["cell_wmse_clamped"] <= s["cell_wmse_raw"] + 1e-12).all()
        # clamp(raw) == 0 <=> raw <= 0 (cell_raw_le0 is a float32 mean, the stream statistic float64)
        assert np.allclose(s["cell_clamped_zero_frac"], s["cell_raw_le0"], rtol=0, atol=1e-6)
        for _, r in s.iterrows():                          # zero-predictor terms recomputed independently
            y = trs[a].graphs_by_name[r["sample"]]["cell"].y
            z = official.loss_module().weighted_mse(torch.zeros_like(y), y, 10.0)
            assert abs(float(z) - r["cell_wmse_zero"]) < 1e-12
    assert np.allclose(st["O"]["cell_wmse_clamped"], st["O"]["cell_wmse_loss"])   # official: loss uses clamped
    assert np.allclose(st["U"]["cell_wmse_raw"], st["U"]["cell_wmse_loss"])       # arm U: loss uses raw
    hu, ho = pd.read_csv(tmp_path / "U" / "history.csv"), pd.read_csv(tmp_path / "O" / "history.csv")
    assert "train_cell_raw_spearman" in hu and "train_cell_raw_spearman" not in ho   # R6: raw kept separately
    for a in "OU":                                         # official eval stream vs eval-mode raw stream
        for h in ("cell", "grid"):
            e = ev[a]
            assert e[f"val_{h}_activity"].isin(["DEAD", "DEGENERATE", "HEALTHY"]).all()
            assert (e[f"val_{h}_pred_zero_frac"] - e[f"val_{h}_evalraw_le0_frac"]).abs().max() < 1e-12


# ------------------------------------------------------------ T8: logging and mid-epoch evaluation are observation only
def test_T8_logging_and_mid_epoch_eval_do_not_change_training(graphs, tmp_path):
    for a in "OU":
        out = []
        for k, kw in enumerate((dict(step_log=False, heads=False, eval_every=0), dict(step_log=True, heads=True, eval_every=1))):
            tr = run(hp_cfg(a, **kw), graphs, tmp_path / f"{a}{k}")
            out.append((params(tr.model), tr.optimizer.state_dict()["state"], torch.get_rng_state(),
                        pd.read_csv(tmp_path / f"{a}{k}" / "history.csv")))
        (p0, s0, r0, h0), (p1, s1, r1, h1) = out
        assert torch.equal(p0, p1) and torch.equal(r0, r1), a
        assert all(torch.equal(s0[i][k], s1[i][k]) for i in s0 for k in s0[i]), a
        for c in ("train_loss", "grad_norm_mean", "val_loss", "val_avg_spearman", "train_cell_spearman"):
            assert h0[c].tolist() == h1[c].tolist(), (a, c)


# ------------------------------------------------------------ T9: train/eval mode restoration
def test_T9_train_and_eval_mode_restored_around_every_evaluation(graphs, tmp_path, monkeypatch):
    calls = []
    orig_build, orig_eval = T.build_model, T.Trainer._step_eval

    def build(c):
        m = orig_build(c)

        def hook(mod, inp, out):
            raw = mod.mlp.forward(inp[0]).squeeze(-1)      # .forward: does not fire the capture hooks again
            calls.append((mod.training, all(x.training == mod.training for x in m.modules()),
                          torch.equal(out, raw if mod.training else raw.clamp(min=0))))
        m.cell_head.register_forward_hook(hook)
        return m

    def step_eval(self, gstep):
        row = orig_eval(self, gstep)
        assert self.model.training and all(x.training for x in self.model.modules()), gstep
        return row
    monkeypatch.setattr(T, "build_model", build)
    monkeypatch.setattr(T.Trainer, "_step_eval", step_eval)
    run(hp_cfg("U", eval_every=1), graphs, tmp_path / "u")
    n, nv = len(graphs["train"]), len(graphs["val"])
    train_calls = [c for c in calls if c[0]]
    assert len(train_calls) == 2 * n                       # one training forward per step, all in train mode
    assert len(calls) - len(train_calls) == (1 + 2 * n) * nv + 2 * nv   # step-0 + per-step evals + epoch-end evals
    assert all(consistent and correct for _, consistent, correct in calls)   # whole model in one mode; U clamps
    # only in eval mode (every training call returned the raw output)


# ------------------------------------------------------------ T10: parameter partition
def _nonzero(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n for n, p in model.named_parameters() if p.grad is not None and bool((p.grad != 0).any())}


def test_T10_partition_rule_matches_autograd():
    L, data = official.loss_module(), hand_graph()
    for layers in (3, 2):
        m = real_model(0, "O", layers=layers, hidden=32)
        m.train()
        sets = {}
        for h, y in (("cell", data["cell"].y), ("grid", data["grid_l0"].y)):
            torch.manual_seed(7)
            o = m(data)[f"{h}_pred"]
            sets[h] = _nonzero(m, L.weighted_mse(o, y) + 0.5 * L.variance_reg(o, y))
        auto = {n: "shared" if n in sets["cell"] and n in sets["grid"] else "cell_only" if n in sets["cell"]
                else "grid_only" if n in sets["grid"] else "none" for n, _ in m.named_parameters()}
        assert auto == parameter_partition(m), layers


# ------------------------------------------------------------ T13, T14: metrics and the tile-projection reference
def test_T13_independent_metrics_and_activity_state():
    rng = np.random.RandomState(0)
    y = np.where(rng.rand(5000) < 0.9, 0.0, rng.rand(5000)).astype(np.float32)
    p = (y + 0.05 * rng.randn(5000)).clip(min=0).astype(np.float32)
    assert abs(HM.spearman_indep(p, y) - ss.spearmanr(p.astype(np.float64), y.astype(np.float64))[0]) < 1e-9
    assert abs(HM.pearson_indep(p, y) - ss.pearsonr(p.astype(np.float64), y.astype(np.float64))[0]) < 1e-9
    assert abs(HM.spearman_indep(p, y) - official.evaluation_module().compute_spearman(p, y)) < 1e-5
    spec = importlib.util.spec_from_file_location("dt", ROOT / "tools" / "diagnose_training.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert HM.spearman_indep(p, y) == tool.spearman_indep(p, y)
    assert np.isnan(HM.spearman_indep(np.zeros(10), y[:10])) and np.isnan(HM.pearson_indep(np.ones(10), y[:10]))
    a = HM.activity_state
    assert a(0.0, 1.0) == "DEAD" and a(1.6e-5, 0.99999) == "DEGENERATE" and a(5e-5, 0.5) == "DEGENERATE"
    assert a(0.01, 0.999) == "DEGENERATE" and a(0.01, 0.9989) == "HEALTHY" and a(1e-4, 0.5) == "HEALTHY"
    assert a(float("nan"), 0.5) == "DEGENERATE"


def test_T14_tile_projection_reference(graphs):
    g = graphs["val"][0]
    tiles = HM.cell_tiles(g)
    assert torch.equal(g["cell"].y, g["grid_l0"].y[tiles])                 # cell label = its tile's label
    rec = {"sample": g.placement_name, "design": "d", "grid_pred": g["grid_l0"].y.clone(), "grid_gt": g["grid_l0"].y,
           "cell_pred": torch.zeros_like(g["cell"].y), "cell_gt": g["cell"].y}
    proj = HM.tile_projection([rec], {g.placement_name: g})[0]
    assert torch.equal(proj["cell_pred"], g["cell"].y)
    if float(g["cell"].y.std()) > 0:
        assert HM.spearman_indep(proj["cell_pred"].numpy(), g["cell"].y.numpy()) == pytest.approx(1.0)
    base = HM.cell_feature_baseline(graphs["val"])
    assert set(base) >= {"_note", "pooled_cell_density", "pooled_mean_net_hpwl_um"}
    assert any(k.startswith("sample_") for k in base) and any(k.endswith("_mean_of_samples") for k in base)


def test_T14b_baseline_per_graph_values_are_standardisation_invariant(graphs):
    """Decision 6: a positive affine map per graph (what per-graph standardisation is) leaves every
    per-graph baseline unchanged; pooled values over several graphs are not protected by that."""
    gs = list(graphs["train"]) + list(graphs["val"])
    ref = HM.cell_feature_baseline(gs)
    moved = []
    for k, g in enumerate(gs):
        h = copy.copy(g)
        h["grid_l0"].x = g["grid_l0"].x * (1.0 + 3.0 * k) + 10.0 * k     # different scale and shift per graph
        moved.append(h)
    new = HM.cell_feature_baseline(moved)
    per = [k for k in ref if k.startswith("sample_") or k.endswith("_mean_of_samples")]
    assert per and all(np.isclose(ref[k], new[k], equal_nan=True) for k in per)


# ------------------------------------------------------------ T15: resume, run identity
def test_T15_resume_without_duplicates_and_distinct_run_ids(graphs, tmp_path):
    for a in "OU":
        run(hp_cfg(a), graphs, tmp_path / f"full{a}")
        run(hp_cfg(a, session=1), graphs, tmp_path / f"split{a}")
        run(hp_cfg(a), graphs, tmp_path / f"split{a}", mode="resume")
        for f in ("steps.csv", "evals.csv"):
            x, y = pd.read_csv(tmp_path / f"full{a}" / f), pd.read_csv(tmp_path / f"split{a}" / f)
            assert x["global_step"].tolist() == y["global_step"].tolist(), (a, f)
            assert len(set(y["global_step"])) == len(y)                       # no duplicated rows
            col = "loss" if f == "steps.csv" else "val_cell_spearman_indep"
            assert np.array_equal(x[col].to_numpy(), y[col].to_numpy(), equal_nan=True), (a, f)
        assert (pd.read_csv(tmp_path / f"split{a}" / "evals.csv")["global_step"] == 0).sum() == 1
    with pytest.raises(CheckpointError, match="belongs to run"):            # never resume the other arm
        T.Trainer(hp_cfg("O"), graphs, tmp_path / "splitU", device="cpu").run(mode="resume")
    man = json.loads((ROOT / "configs" / "headpath_manifest.json").read_text())
    base = load_config(ROOT / "configs" / "base.yaml")
    ids = [H.run_config(base, r, man).run_id() for r in man["runs"]]
    assert len(set(ids)) == len(ids) == 14
    assert all(("__DEV-cellhead-train-unclamped" in i) == (r["arm"] == "U") for i, r in zip(ids, man["runs"]))


# ------------------------------------------------------------ T16: pre-registered manifest
@pytest.fixture(scope="module")
def manifest():
    return json.loads((ROOT / "configs" / "headpath_manifest.json").read_text())


def test_T16_manifest_is_generated_and_matches_notebook_07(manifest, tmp_path):
    spec = importlib.util.spec_from_file_location("mhm", ROOT / "tools" / "make_headpath_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from verihgn_repro.data import circuitnet as cn, splits as sp
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    by = defaultdict(list)
    for n in names:
        by[CE.design_of(n)].append(n)
    regen = H.manifest_from_names(by, {"designs": sp.official_split(cn.SCOPES["official9"])})
    assert {k: manifest[k] for k in regen} == regen
    c07 = json.loads((ROOT / "configs" / "collapse_experiment_manifest.json").read_text())
    assert manifest["train"] == c07["arms"]["FULL"] and manifest["val"] == c07["val"]
    assert len(manifest["train"]) == 245 and len(manifest["val"]) == 12
    assert manifest["n_runs"] == len(manifest["runs"]) == 14 and manifest["total_optimizer_steps"] == 14 * 490
    assert [(r["arm"], r["seed"]) for r in manifest["runs"] if not r["retest"]] == \
        [(a, s) for s in range(42, 48) for a in ("O", "U")]
    assert [(r["arm"], r["seed"]) for r in manifest["runs"] if r["retest"]] == [("O", 42), ("U", 42)]
    assert all(r["epochs"] == 2 and r["steps"] == 490 for r in manifest["runs"])
    assert manifest["eval_steps"] == list(range(0, 491, 49)) and manifest["replication_step"] == 245
    assert all(r["cell_head_train_clamp"] == (r["arm"] == "O") for r in manifest["runs"])
    test_d = set(manifest["split"]["test"]) | {"mgc_matrix_mult_a"}
    assert not set(manifest["train"]) & set(manifest["val"]) and not {CE.design_of(n) for n in manifest["train"]} & test_d
    assert not {CE.design_of(n) for n in manifest["val"]} & set(manifest["split"]["test"])
    assert manifest["pending_research_decisions"] == []                 # all seven decided (2026-10-20)
    assert "ACTIVE_WITHOUT_USEFUL_SIGNAL" in manifest["categories"] and "does not invalidate" in manifest["replication_policy"]
    primary = [r for r in manifest["runs"] if not r["retest"]]
    retest = [r for r in manifest["runs"] if r["retest"]]
    assert (len(primary), len(retest)) == (12, 2)                       # 12 primary + 2 separate retest runs
    assert "RETEST runs are excluded" in manifest["statistical_test"]["pairs"]
    assert manifest["thresholds"]["fresh_official_atol"] == manifest["thresholds"]["fresh_official_rtol"] == 1e-6
    for n in names:                                       # stub tree with the real names: plan == manifest
        d = tmp_path / CE.design_of(n) / n
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        np.savez(d / "sample.npz", label_h=np.ones((3, 3), np.float32), label_v=np.zeros((3, 3), np.float32))
    split = {"name": "official_seed42", "designs": manifest["split"]}
    plan = H.plan_headpath(tmp_path, split, expected=manifest, collapse_manifest=c07)
    assert H.manifest_diff(plan, manifest) == []
    bad = json.loads(json.dumps(manifest))
    bad["thresholds"]["p3_min_cell_spearman"] = 0.05
    with pytest.raises(ValueError, match="pre-registered"):
        H.plan_headpath(tmp_path, split, expected=bad)


# ------------------------------------------------------------ T17: the arms differ in exactly one setting
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_T17_run_configs_differ_only_in_arm_seed_and_name(manifest):
    base = load_config(ROOT / "configs" / "base.yaml")
    cfgs = {r["name"]: _flat(H.run_config(base, r, manifest).to_dict()) for r in manifest["runs"]}
    for s in H.SEEDS:
        o, u = cfgs[f"HEADPATH_O__seed{s}"], cfgs[f"HEADPATH_U__seed{s}"]
        assert {k for k in o if o[k] != u[k]} == {"model.cell_head_train_clamp", "experiment.name"}
    o42, o43 = cfgs["HEADPATH_O__seed42"], cfgs["HEADPATH_O__seed43"]
    assert {k for k in o42 if o42[k] != o43[k]} == {"training.seed"}
    r = cfgs["HEADPATH_O_RETEST__seed42"]
    assert {k for k in o42 if o42[k] != r[k]} == {"experiment.name"}
    fb = _flat(base.to_dict())
    for k in ("training.learning_rate", "training.weight_decay", "training.grad_clip", "training.step_size",
              "training.gamma", "training.wmse_alpha", "training.lambda_var", "training.compute_train_metrics",
              "model.hidden_dim", "model.num_layers", "model.dropout", "model.use_checkpoint", "experiment.split",
              "experiment.grid_features", "experiment.label_norm", "experiment.dataset_scope"):
        assert all(c[k] == fb[k] for c in cfgs.values()), k
    assert all(c["training.precision"] == "fp32" and c["training.allow_tf32"] is False and c["training.num_epochs"] == 2
               and c["training.eval_every_steps"] == 49 and c["training.step_log_heads"] for c in cfgs.values())
    for rr in manifest["runs"]:
        dv = deviations(H.run_config(base, rr, manifest))
        assert (HEADPATH_DEVIATION in dv) == (rr["arm"] == "U")
    bad = copy.deepcopy(base)
    bad.experiment.label_norm = "per_sample"
    with pytest.raises(ValueError):
        H.run_config(bad, manifest["runs"][0], manifest)
    bad = copy.deepcopy(base)
    bad.model.cell_head_train_clamp = False
    with pytest.raises(ValueError):
        H.run_config(bad, manifest["runs"][0], manifest)


# ------------------------------------------------------------ T18: pre-registered decision rule and statistics
def test_T18_decision_rule_and_paired_sign_test(manifest):
    th = manifest["thresholds"]
    # decision 1: explicit P2 bounds
    assert H.p2_outcome(0.12, th) == "PASS" and H.p2_outcome(0.30, th) == "PASS"
    assert H.p2_outcome(0.1199, th) == "BORDERLINE" and H.p2_outcome(0.0801, th) == "BORDERLINE"
    assert H.p2_outcome(0.08, th) == "FAIL" and H.p2_outcome(-0.5, th) == "FAIL" and H.p2_outcome(float("nan"), th) == "FAIL"
    # decision 2: H-sig
    assert H.is_hsig(True, 0.03, 0.05, th) and H.is_hsig(True, -0.03, -0.2, th) and H.is_hsig(True, 0.0, 0.0, th)
    assert not H.is_hsig(False, 0.0, 0.0, th) and not H.is_hsig(True, 0.031, 0.0, th)
    assert not H.is_hsig(True, 0.0, 0.051, th) and not H.is_hsig(True, float("nan"), 0.0, th)
    assert not H.is_hsig(True, 0.0, float("nan"), th)
    c = H.categorize
    assert c(True, True, "PASS", True) == "LEARNS_AND_GENERALIZES"
    assert c(True, True, "PASS", False) == "LEARNS_WITHOUT_DEMONSTRATED_GENERALIZATION"
    assert c(True, False, "PASS", True, True) == "CELL_HEAD_FAILURE_PERSISTS"
    assert c(True, True, "FAIL", False, True) == "ACTIVE_WITHOUT_USEFUL_SIGNAL"
    assert all(c(True, True, p2, p3) == "INCONCLUSIVE" for p2 in ("FAIL", "BORDERLINE") for p3 in (True, False))
    assert c(False, True, "PASS", True) == "INVALID"
    # exact McNemar = two-sided exact binomial test on discordant pairs
    assert H.sign_test_two_sided(6, 0) == 0.03125 and H.sign_test_two_sided(0, 6) == 0.03125
    assert H.sign_test_two_sided(5, 0) == 0.0625 and H.sign_test_two_sided(5, 1) == 0.21875
    assert H.sign_test_two_sided(0, 0) == 1.0
    for a_, b_ in ((6, 0), (5, 1), (4, 2), (3, 0)):
        assert H.sign_test_two_sided(a_, b_) == pytest.approx(ss.binomtest(a_, a_ + b_, 0.5).pvalue)

    zref = {"samples_sha256": "x", "sum_cell_wmse_zero": 1.0}

    def res(arm, seed, cat, valid=True, retest=False, rep=("DEAD", "DEAD"), hsig=False, zero=zref):
        return {"name": f"HEADPATH_{arm}{'_RETEST' if retest else ''}__seed{seed}", "arm": arm, "seed": seed,
                "retest": retest, "valid": valid, "invalid_reasons": [] if valid else ["x"], "category": cat,
                "hsig_criteria_met": hsig, "p2_zero_reference": dict(zero),
                "P1": {"final_cell_activity": "HEALTHY"},
                "grid": {"final_activity": "HEALTHY", "final_val_spearman_indep": 0.3},
                "P2": {"cell_wmse_clamped_improvement_vs_zero": 0.2}, "P3": {"final_val_cell_spearman_indep": 0.2},
                "state_at_replication_step": {"cell": rep[0], "grid": rep[1]}}
    ok_rep = {42: ("DEAD", "DEAD"), 43: ("DEAD", "DEAD"), 44: ("DEAD", "HEALTHY")}

    def results(u_cats, rep=ok_rep, retest_o="LEARNS_AND_GENERALIZES"):
        out = []
        for s, uc in zip(H.SEEDS, u_cats):
            out += [res("O", s, "CELL_HEAD_FAILURE_PERSISTS", rep=rep.get(s, ("DEAD", "DEAD"))),
                    res("U", s, uc, hsig=uc == "ACTIVE_WITHOUT_USEFUL_SIGNAL")]
        # retest runs: deliberately the opposite outcome; they must not enter the test
        out += [res("O", 42, retest_o, retest=True), res("U", 42, "CELL_HEAD_FAILURE_PERSISTS", retest=True)]
        return out
    a = H.analyse(results(["LEARNS_AND_GENERALIZES"] * 6), manifest)
    t = a["primary_test"]
    assert a["experiment_status"] == "VALID" and a["historical_replication"] == "PASSED" and a["verdict"] == "VALID"
    assert t["significant"] and (t["U_only"], t["O_only"], t["n_discordant"], t["n_valid_pairs"]) == (6, 0, 6, 6)
    assert t["direction"] == "U > O" and t["p_two_sided_exact"] == 0.03125
    assert a["conclusion"].startswith("Supports H-dyn") and "exact two-sided p = 0.03125" in a["conclusion"]
    assert a["test_retest"]["O"]["excluded_from_inference"] and not a["test_retest"]["O"]["same_category"]
    a = H.analyse(results(["LEARNS_AND_GENERALIZES"] * 5 + ["ACTIVE_WITHOUT_USEFUL_SIGNAL"]), manifest)
    assert not a["primary_test"]["significant"] and a["primary_test"]["n_discordant"] == 5
    assert a["conclusion"].startswith("Not statistically conclusive")
    assert a["hsig"]["U"] == {"n_seeds_meeting_hsig": 1, "seeds": [47], "n_valid_runs": 6}   # separate, counted
    assert a["category_counts"]["U"]["ACTIVE_WITHOUT_USEFUL_SIGNAL"] == 1
    # decision 5: a failed historical replication is flagged, not invalidating
    a = H.analyse(results(["LEARNS_AND_GENERALIZES"] * 6, rep={**ok_rep, 44: ("DEAD", "DEAD")}), manifest)
    assert a["experiment_status"] == "VALID" and a["historical_replication"] == "FAILED"
    assert a["verdict"] == "VALID - HISTORICAL REPLICATION FAILED (investigate)"
    assert a["conclusion"].startswith("WARNING - HISTORICAL REPLICATION") and "Supports H-dyn" in a["conclusion"]
    assert a["primary_test"]["significant"]
    # an invalid run makes the experiment INVALID (distinct from a failed replication); its pair is excluded
    rs = results(["LEARNS_AND_GENERALIZES"] * 6)
    rs[1]["valid"], rs[1]["category"] = False, "INVALID"
    a = H.analyse(rs, manifest)
    assert a["experiment_status"] == "INVALID" and a["primary_test"]["excluded"][0]["seed"] == 42
    assert not a["primary_test"]["significant"] and a["conclusion"].startswith("INVALID EXPERIMENT")
    # decision 1: the two arms of a pair must share the same zero-predictor reference
    rs = results(["LEARNS_AND_GENERALIZES"] * 6)
    rs[3]["p2_zero_reference"]["sum_cell_wmse_zero"] = 1.5
    a = H.analyse(rs, manifest)
    assert a["experiment_status"] == "INVALID" and "zero-predictor" in a["primary_test"]["excluded"][0]["reason"]
    # a missing primary run: INCOMPLETE, no conclusion
    a = H.analyse(results(["LEARNS_AND_GENERALIZES"] * 6)[2:], manifest)
    assert a["experiment_status"] == "INCOMPLETE" and "HEADPATH_O__seed42" in a["missing_runs"]


# ------------------------------------------------------------ T19: validity checks never drop a run
def test_T19_invalid_runs_are_reported_not_dropped(multi_design, tmp_path):
    processed, split = multi_design
    plan = H.plan_headpath(processed, split, val_from_val_design=2, eval_every=6)
    gb = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=plan["c_max"])
          for n, d in plan["sample_dirs"].items()}
    base = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2}})
    rep = H.run_plan(plan, base, gb, tmp_path / "runs", device="cpu", only=["HEADPATH_U__seed42"])
    assert [x["name"] for x in rep["results"]] == [x["name"] for x in plan["runs"]]   # every planned run listed
    r = next(x for x in rep["results"] if x["name"] == "HEADPATH_U__seed42")
    assert r["run_status"] == "COMPLETED" and r["predictions_export"]["status"] == "written"
    assert all(x["run_status"] == "NOT_STARTED" and "not selected" in x["reason"]
               for x in rep["results"] if x["name"] != "HEADPATH_U__seed42")
    assert r["valid"], r["invalid_reasons"]
    assert r["checkpoint"]["adam_step_counts"] == [plan["steps_per_run"]] == [r["optimizer_step_calls"]]
    fc = r["fresh_official_check"]
    assert fc["bit_identical"] and fc["atol"] == fc["rtol"] == 1e-6 and fc["max_abs_diff_vs_fresh_official"] == 0.0
    assert rep["analysis"]["experiment_status"] == "INCOMPLETE"            # 13 of 14 runs not run here
    d = tmp_path / "runs" / "HEADPATH_U__seed42"
    st = pd.read_csv(d / "steps.csv")
    st.loc[3, "loss"] = float("inf")
    st.drop(index=len(st) - 1).to_csv(d / "steps.csv", index=False)
    bad = H.summarize_run(d, plan, plan["runs"][1])
    assert not bad["valid"] and bad["category"] == "INVALID"
    assert any("non-finite training loss" in x for x in bad["invalid_reasons"])
    assert any("Adam steps" in x for x in bad["invalid_reasons"]) and any("steps logged" in x for x in bad["invalid_reasons"])
    with pytest.raises(RuntimeError, match="pending research decisions"):   # the gate still works if reopened
        H.run_plan({**plan, "pending_research_decisions": ["x"]}, base, gb, tmp_path / "runs2", device="cpu")


# ------------------------------------------------------------ T20: notebook 08 on a synthetic Drive layout
def test_T20_notebook_08_cells_run_on_synthetic_layout(tmp_path, monkeypatch):
    from verihgn_repro import colab
    from verihgn_repro.colab import DriveLayout
    L = DriveLayout(root=tmp_path / "drive" / "VeriHGN").ensure()
    P.run(_multi(tmp_path), L.processed_for("h0"), scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    (L.reports / "03_preprocessing_official9.json").write_text(json.dumps({"prep_hash": "h0"}))
    designs = {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"], "val": ["mgc_pci_bridge32_a"], "test": []}
    split = {"name": "official_seed42", "designs": designs}
    (L.splits / "official9__official_seed42.json").write_text(json.dumps(split))
    (L.reports / "04_validation_official9.json").write_text(json.dumps({"c_max": {"official_seed42": CE.compute_c_max(
        P.list_processed(L.processed_for("h0"), set(designs["train"])))}}))
    kw = dict(val_from_val_design=2, eval_every=6)
    ref = H.plan_headpath(L.processed_for("h0"), split, **kw)
    p07 = {"arms": {"FULL": ref["train"]}, "val": ref["val"], "c_max": ref["c_max"]}
    (L.reports / "07_collapse_plan.json").write_text(json.dumps(p07))            # pre-existing 07 results
    (L.reports / "07_collapse_experiment.json").write_text("{}")
    (L.runs / "collapse_diag" / "FULL__fp32__seed42" / "checkpoints").mkdir(parents=True)
    (L.runs / "collapse_diag" / "FULL__fp32__seed42" / "checkpoints" / "last.pt").write_bytes(b"07 checkpoint")
    (L.runs / "collapse_diag" / "sentinel.txt").write_text("07")
    before = H.protected_artifacts(L)
    nb = json.loads((ROOT / "notebooks" / "08_headpath_experiment.ipynb").read_text(encoding="utf-8"))
    pre, plan_cell, run_cell = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][1:]
    assert "CUDA_VISIBLE_DEVICES=''" in pre and "subset_raw_tree" in pre        # preflight: CPU only (not run here)
    assert "RUN_HEADPATH = False" in run_cell and "H.execute(" in run_cell and "colab.flush_drive(" in run_cell
    assert "expected=EXPECTED, collapse_manifest=C07" in plan_cell              # the real notebook enforces both
    flushed = []                                                               # Drive flush mock: no real unmount

    def fake_flush(paths, remount_verify=True):
        flushed.append({str(p): p.read_bytes() for p in map(Path, paths)})     # content at flush time
        return {"flushed_and_unmounted": True, "remount_verified": remount_verify}
    monkeypatch.setattr(colab, "flush_drive", fake_flush)
    plan_cell = plan_cell.replace("expected=EXPECTED, collapse_manifest=C07",
                                  "expected=None, collapse_manifest=None, val_from_val_design=2, eval_every=6")
    ns = {"L": L, "CODE_DIR": str(ROOT)}
    for src in (plan_cell, run_cell):                                         # default: plan only, nothing trained
        exec(compile(src, "nb08", "exec"), ns)
    assert (L.reports / "08_headpath_plan.json").exists() and not (L.runs / "headpath").exists()
    ns["cfg"].model.hidden_dim, ns["cfg"].model.num_layers = 16, 2
    sel = ["HEADPATH_O__seed42", "HEADPATH_U__seed42", "HEADPATH_O_RETEST__seed42"]
    assert "verify_paths=[plan_path])" in run_cell
    gated = run_cell.replace("RUN_HEADPATH = False", "RUN_HEADPATH = True").replace(
        "verify_paths=[plan_path])", f"verify_paths=[plan_path], device='cpu', only={sel})")
    exec(compile(gated, "nb08-run", "exec"), ns)
    report = L.reports / "08_headpath_experiment.json"
    rep = json.loads(report.read_text())
    assert [r["name"] for r in rep["results"]] == [r["name"] for r in ns["plan"]["runs"]]   # all 14 listed
    done = [r for r in rep["results"] if r["name"] in sel]
    assert all(r["run_status"] == "COMPLETED" and r["valid"] for r in done), [r.get("invalid_reasons") for r in done]
    assert rep["run_status_counts"] == {"COMPLETED": 3, "INCOMPLETE": 0, "FAILED": 0, "NOT_STARTED": 11}
    a = rep["analysis"]
    assert a["deviation_notice"] == HEADPATH_DEVIATION and a["experiment_status"] == "INCOMPLETE"
    assert a["primary_test"]["n_valid_pairs"] == 1 and a["test_retest"]["O"]["excluded_from_inference"]
    assert rep["protected_artifacts"]["unchanged"] is True and rep["protected_artifacts"]["n_files"] == len(before)
    assert H.protected_artifacts(L) == before                                  # 05-07 artifacts untouched
    assert len(flushed) == 1 and set(flushed[0]) == {str(report), str(ns["plan_path"])}
    assert flushed[0][str(report)] == report.read_bytes()                     # the flush was the last write
    exec(compile(gated, "nb08-rerun", "exec"), ns)                            # re-running: new report, no overwrite
    assert len(list(L.reports.glob("08_headpath_experiment*.json"))) == 2
    assert json.loads((L.reports / "08_headpath_experiment.json").read_text()) == rep
    exec(compile(plan_cell, "nb08-again", "exec"), ns)                        # a second plan never overwrites
    assert len(list(L.reports.glob("08_headpath_plan*.json"))) == 2
    with pytest.raises(ValueError, match="collapse_diag"):
        H.run_plan(ns["plan"], ns["cfg"], {}, L.runs / "collapse_diag")
    (L.reports / "07_collapse_plan.json").write_text(json.dumps({**p07, "val": p07["val"][::-1]}))
    with pytest.raises(AssertionError, match="executed 07 plan"):             # cross-check against the real 07 plan
        exec(compile(plan_cell, "nb08-mismatch", "exec"), ns)
    (L.runs / "collapse_diag" / "sentinel.txt").write_text("changed")
    assert H.protected_artifacts(L) != before                                  # the fingerprint detects a change


def test_real_fixture_subset_tree(synthetic_raw, tmp_path, monkeypatch):
    """Notebook 08 preflight: a 3-sample RawLayout tree of symlinks for tests/test_real_data.py.
    Where symlinks need a privilege (Windows), the path logic is checked with copies instead."""
    import shutil
    from verihgn_repro.data import circuitnet as cn
    raw, samples = synthetic_raw
    pick = sorted(samples)[:3]
    try:
        out = cn.subset_raw_tree(raw, tmp_path / "fx", pick)
        linked = True
    except OSError:
        monkeypatch.setattr(Path, "symlink_to", lambda self, target, target_is_directory=False:
                            shutil.copytree(target, self) if target_is_directory else shutil.copy2(target, self))
        out = cn.subset_raw_tree(raw, tmp_path / "fx_copy", pick)
        linked = False
    lay = cn.RawLayout(out)
    assert lay.samples() == pick
    for s in pick:
        assert lay.routability("gr_h_overflow", s).exists() and lay.placement_gcell(s).exists()
        assert lay.placement_micron(s).is_symlink() == linked
    man = P.run(out, tmp_path / "processed_fx", scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == len(pick) and not man["failed"]                       # usable by test_real_data.py
    assert set(cn.RawLayout(raw).samples()) >= set(pick)                      # the source tree is untouched
    with pytest.raises(FileNotFoundError):
        cn.subset_raw_tree(raw, tmp_path / "fx2", ["999-mgc_fft_a-util_0.7"])
