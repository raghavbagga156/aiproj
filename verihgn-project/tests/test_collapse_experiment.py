"""Collapse-experiment code (docs/COLLAPSE_EXPERIMENT_PLAN.md): the per-step log
must not change training, the Part B instrumentation must not change training,
the subset selection must follow its fixed rules, and the runner must detect a
collapse with the normal engine."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import make_design
from verihgn_repro import official
from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.graph import build_graph
from verihgn_repro.diagnostics import collapse_experiment as CE
from verihgn_repro.training import trainer as T

ROOT = Path(__file__).resolve().parents[1]
SPLIT = {"name": "s", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"], "test": []}}


def small_cfg(epochs=3, session=None, step_log=False):
    return load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2},
                                  "training": {"precision": "fp32", "num_epochs": epochs, "step_log": step_log,
                                               "max_epochs_this_session": session}})


@pytest.fixture
def graphs(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    return build_experiment_graphs(out, SPLIT, verbose=False)["graphs"]


@pytest.fixture
def multi_design(tmp_path):
    """4 designs x 5 samples, preprocessed."""
    raw = tmp_path / "raw"
    for k, d in enumerate(["mgc_fft_a", "mgc_pci_bridge32_a", "mgc_des_perf_a", "mgc_edit_dist_a"]):
        make_design(raw, d, n_cells=30 + 5 * k, n_nets=25, samples=5, seed=k + 1)
    out = tmp_path / "processed"
    man = P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 20 and not man["failed"]
    manifest = {"name": "m", "designs": {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"],
                                         "val": ["mgc_pci_bridge32_a"], "test": []}}
    return out, manifest


def params(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


# ------------------------------------------------------------ step log
def test_step_log_does_not_change_training(graphs, tmp_path):
    a = T.Trainer(small_cfg(step_log=False), graphs, tmp_path / "a", device="cpu")
    a.run()
    b = T.Trainer(small_cfg(step_log=True), graphs, tmp_path / "b", device="cpu")
    b.run()
    assert torch.equal(params(a.model), params(b.model))
    ha, hb = pd.read_csv(tmp_path / "a" / "history.csv"), pd.read_csv(tmp_path / "b" / "history.csv")
    for c in ("train_loss", "grad_norm_mean", "val_loss", "val_avg_spearman"):
        assert ha[c].tolist() == hb[c].tolist(), c
    assert not (tmp_path / "a" / "steps.csv").exists()
    st = pd.read_csv(tmp_path / "b" / "steps.csv")
    n = len(graphs["train"])
    assert len(st) == 3 * n and st["global_step"].tolist() == list(range(1, 3 * n + 1))
    for e, grp in st.groupby("epoch"):
        h = hb[hb["epoch"] == e].iloc[0]
        assert abs(grp["grad_norm"].mean() - h["grad_norm_mean"]) <= 1e-6 * h["grad_norm_mean"]   # history: float32 mean
        assert abs(grp["loss"].mean() - h["train_loss"]) < 1e-12
        assert len(grp) == h["train_steps"]
    assert st["cell_raw_le0"].between(0, 1).all() and st["grid_raw_le0"].between(0, 1).all()
    assert set(st["sample"]) == {g.placement_name for g in graphs["train"]}


def test_step_log_survives_resume_without_duplicates(graphs, tmp_path):
    full = T.Trainer(small_cfg(step_log=True), graphs, tmp_path / "full", device="cpu")
    full.run()
    T.Trainer(small_cfg(session=2, step_log=True), graphs, tmp_path / "split", device="cpu").run()
    T.Trainer(small_cfg(step_log=True), graphs, tmp_path / "split", device="cpu").run(mode="resume")
    a, b = pd.read_csv(tmp_path / "full" / "steps.csv"), pd.read_csv(tmp_path / "split" / "steps.csv")
    assert a["global_step"].tolist() == b["global_step"].tolist()
    assert a["loss"].tolist() == b["loss"].tolist() and a["sample"].tolist() == b["sample"].tolist()


def test_part_b_instrumentation_does_not_change_the_step(graphs, tmp_path):
    """tools/diagnose_training.Probe wraps clip/loss and adds hooks; the step itself
    must be bit-identical to an un-instrumented step."""
    spec = importlib.util.spec_from_file_location("diagnose_training", ROOT / "tools" / "diagnose_training.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    collate = official.dataset_module().collate_fn
    out = []
    for probe in (False, True):
        tr = T.Trainer(small_cfg(), graphs, tmp_path / f"p{probe}", device="cpu")
        tr._init_state()
        for g in graphs["train"]:
            tr.batches = {"train": [collate([g])]}
            pr = tool.Probe(tr) if probe else None
            if pr:
                pr.start()
            try:
                tr._train_epoch()
            finally:
                if pr:
                    pr.close()
        out.append(params(tr.model))
    assert torch.equal(out[0], out[1])


# ------------------------------------------------------------ plan
def test_plan_follows_fixed_rules(multi_design):
    processed, manifest = multi_design
    plan = CE.plan_experiment(processed, manifest, seeds=(42, 43), val_from_val_design=2, per_design_mid=2,
                              single_n=3)
    names = {d: [p.name for p in P.list_processed(processed, {d})] for d in ["mgc_fft_a", "mgc_des_perf_a",
                                                                               "mgc_edit_dist_a", "mgc_pci_bridge32_a"]}
    for d in manifest["designs"]["train"]:
        assert plan["holdout"][d] == [names[d][2]]                 # middle of 5
    runs = {r["name"]: r for r in plan["runs"]}
    full = runs["FULL__fp32__seed42"]
    assert len(full["train"]) == 12 and full["epochs"] == 1 and plan["budget_steps"] == 12
    d6 = runs["DIVERSE6__fp32__seed42"]
    assert d6["train"] == [names[d][0] for d in sorted(manifest["designs"]["train"])]
    assert d6["epochs"] == 4 and d6["steps"] == 12
    assert len(runs["DIVERSE36__fp32__seed42"]["train"]) == 6
    s6 = runs["SINGLE6__fp32__seed42"]["train"]
    assert plan["single_design"] == "mgc_fft_a"                                   # pre-registered, not data-chosen
    assert {n.split("-")[1] for n in s6} == {"mgc_fft_a"} and len(s6) == 3
    nz = plan["grid_nonzero_fraction_by_design"]
    ranked = sorted(nz, key=lambda d: (nz[d], d))
    assert plan["lower_median_sparsity_design"] == ranked[(len(ranked) - 1) // 2]   # reported only
    val = full["val"]
    assert sum(n.split("-")[1] == "mgc_pci_bridge32_a" for n in val) == 2
    assert all(h in val for hs in plan["holdout"].values() for h in hs)
    for r in plan["runs"]:
        if r["arm"] != "REPLICA":
            assert not set(r["train"]) & set(r["val"]), r["name"]
            assert r["precision"] == "fp32"
    reps = [r for r in plan["runs"] if r["arm"] == "REPLICA"]
    assert [(r["precision"], r["seed"]) for r in reps] == [(p, s) for p in ("fp16", "fp32") for s in (42, 43, 44)]
    assert all(len(r["train"]) == 15 for r in reps) and plan["runs"][:6] == reps          # replicas run first
    assert all(n.split("-")[1] == "mgc_pci_bridge32_a" for n in reps[0]["val"])
    assert plan == CE.plan_experiment(processed, manifest, seeds=(42, 43), val_from_val_design=2,
                                      per_design_mid=2, single_n=3)               # deterministic


def test_plan_rejects_train_test_overlap(multi_design):
    processed, manifest = multi_design
    bad = {**manifest, "designs": {**manifest["designs"], "test": ["mgc_fft_a"]}}
    with pytest.raises(ValueError, match="overlap"):
        CE.plan_experiment(processed, bad)


# ------------------------------------------------------------ runner
def _graphs_for(plan):
    out = {}
    for name, d in plan["sample_dirs"].items():
        a, m = P.load_sample(Path(d))
        out[name] = build_graph(a, m, label_norm="per_dataset", c_max=plan["c_max"])
    return out


def test_run_plan_end_to_end_and_collapse_detection(multi_design, tmp_path, monkeypatch):
    processed, manifest = multi_design
    plan = CE.plan_experiment(processed, manifest, seeds=(42,), val_from_val_design=2, per_design_mid=2, single_n=3)
    gb = _graphs_for(plan)
    base = load_config(overrides={"model": {"hidden_dim": 16, "num_layers": 2}})
    rep = CE.run_plan(plan, base, gb, tmp_path / "runs", device="cpu", only=["DIVERSE6", "FULL"],
                      report_path=tmp_path / "r.json")
    assert [r["arm"] for r in rep["results"]] == ["FULL", "DIVERSE6"]
    for r in rep["results"]:
        assert r["steps_logged"] == r["steps"] == 12
        assert set(r["final_val_per_design"]) == {"mgc_pci_bridge32_a", "mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"}
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["provenance"] == "measured (this run)" and len(saved["results"]) == 2
    cfg = CE.run_config(base, plan["runs"][0])
    assert cfg.training.patience == 10 ** 9 and cfg.training.keep_every == 0 and cfg.training.step_log
    assert cfg.training.learning_rate == base.training.learning_rate and cfg.model == base.model

    orig = T.build_model                       # forced collapse: every raw head output < 0 from the start

    def dead(c):
        m = orig(c)
        with torch.no_grad():
            m.cell_head.mlp[-1].bias.fill_(-10.0)
            m.grid_head.mlp[-1].bias.fill_(-10.0)
        return m
    monkeypatch.setattr(T, "build_model", dead)
    rep2 = CE.run_plan(plan, base, gb, tmp_path / "runs_dead", device="cpu", only=["DIVERSE6"])
    r = rep2["results"][0]
    assert r["collapsed"] and r["collapse_step"] == 1 and r["zero_grad_steps"] == 12
    assert r["final_val_pred_zero_frac"] == {"cell": 1.0, "grid": 1.0}


def test_summarize_run_collapse_rule(tmp_path):
    rows = [{"epoch": 1, "global_step": i, "grad_norm": 0.0 if i > 5 else 0.1, "grad_exactly_zero": i > 5,
             "sample": f"s{i}", "cell_raw_le0": 1.0, "grid_raw_le0": 1.0 if i > 5 else 0.9} for i in range(1, 31)]
    pd.DataFrame(rows).to_csv(tmp_path / "steps.csv", index=False)
    s = CE.summarize_run(tmp_path)
    assert s["collapse_step"] == 6 and s["zero_tail_len"] == 25 and s["collapsed"]
    assert s["steps_both_heads_ge_0999"] == 30 and s["max_grad_step"] == 1
    rows[-1].update(grad_norm=0.2, grad_exactly_zero=False)                 # revived at the end
    pd.DataFrame(rows).to_csv(tmp_path / "steps.csv", index=False)
    s = CE.summarize_run(tmp_path)
    assert s["collapse_step"] is None and not s["collapsed"] and s["zero_grad_steps"] == 24
    rows2 = rows[:10]
    for r in rows2[5:]:
        r.update(grad_norm=0.0, grad_exactly_zero=True)
    pd.DataFrame(rows2).to_csv(tmp_path / "steps.csv", index=False)
    s = CE.summarize_run(tmp_path, min_tail=20)                              # tail 5 < 20 and not whole run
    assert s["collapse_step"] == 6 and not s["collapsed"]


# ------------------------------------------------------------ notebook 07
def test_notebook_07_cells_run_on_synthetic_layout(tmp_path):
    """Executes the notebook's own code cells (bootstrap excluded) against a synthetic
    Drive layout on the CPU: planning only by default, then the gated run."""
    from verihgn_repro.colab import DriveLayout
    raw = tmp_path / "raw"
    for k, d in enumerate(["mgc_fft_a", "mgc_pci_bridge32_a", "mgc_des_perf_a", "mgc_edit_dist_a"]):
        make_design(raw, d, n_cells=30 + 5 * k, n_nets=25, samples=5, seed=k + 1)
    L = DriveLayout(root=tmp_path / "drive" / "VeriHGN").ensure()
    P.run(raw, L.processed_for("h0"), scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    (L.reports / "03_preprocessing_official9.json").write_text(json.dumps({"prep_hash": "h0"}))
    (L.splits / "official9__official_seed42.json").write_text(json.dumps(
        {"name": "official_seed42", "designs": {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"],
                                                "val": ["mgc_pci_bridge32_a"], "test": []}}))
    nb = json.loads((ROOT / "notebooks" / "07_collapse_experiment.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][1:]
    assert "RUN_COLLAPSE_EXPERIMENT = False" in cells[-1]
    (L.reports / "04_validation_official9.json").write_text(json.dumps({"c_max": {"official_seed42": CE.compute_c_max(
        P.list_processed(L.processed_for("h0"), {"mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"}))}}))
    assert "expected=EXPECTED" in cells[0]                    # the real notebook enforces the manifest
    cells[0] = cells[0].replace("expected=EXPECTED", "expected=None")   # synthetic data != real manifest
    ns = {"L": L, "CODE_DIR": str(ROOT)}
    for src in cells:                                         # default: plan only, nothing trained
        exec(compile(src, "nb07", "exec"), ns)
    assert (L.reports / "07_collapse_plan.json").exists() and not (L.runs / "collapse_diag").exists()
    gated = cells[-1].replace("RUN_COLLAPSE_EXPERIMENT = False", "RUN_COLLAPSE_EXPERIMENT = True").replace(
        "report_path=report_path)", "report_path=report_path, device='cpu', only=['REPLICA__fp32__seed42', 'SINGLE6'])")
    ns["cfg"].model.hidden_dim, ns["cfg"].model.num_layers = 16, 2
    exec(compile(gated, "nb07-run", "exec"), ns)
    rep = json.loads((L.reports / "07_collapse_experiment.json").read_text())
    assert [r["name"] for r in rep["results"]] == ["REPLICA__fp32__seed42", "SINGLE6__fp32__seed42",
                                                    "SINGLE6__fp32__seed43", "SINGLE6__fp32__seed44"]
    exec(compile(cells[0], "nb07-again", "exec"), ns)          # a second plan never overwrites the first
    assert len(list(L.reports.glob("07_collapse_plan*.json"))) == 2


# ------------------------------------------------------------ pre-registered manifest (real sample names)
PART_B_TRAIN = ["108-mgc_des_perf_a-util_0.9", "100-mgc_edit_dist_a-util_0.7", "101-mgc_fft_a-util_0.85",
                "240-mgc_matrix_mult_b-util_0.85", "10-mgc_pci_bridge32_a-util_0.7", "1-mgc_pci_bridge32_b-util_0.9"]


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ROOT / "configs" / "collapse_experiment_manifest.json").read_text())


def test_manifest_file_is_what_the_generator_produces(manifest):
    spec = importlib.util.spec_from_file_location("mcm", ROOT / "tools" / "make_collapse_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import verihgn_repro.diagnostics.collapse_experiment as ce
    from collections import defaultdict
    from verihgn_repro.data import circuitnet as cn, splits as sp
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    by = defaultdict(list)
    for n in names:
        by[ce.design_of(n)].append(n)
    regenerated = ce.manifest_from_names(by, {"designs": sp.official_split(cn.SCOPES["official9"])})
    assert {k: manifest[k] for k in regenerated} == regenerated


def test_manifest_counts_designs_and_steps(manifest):
    from collections import Counter
    dz = CE.design_of
    assert manifest["split"] == {"train": ["mgc_pci_bridge32_a", "mgc_edit_dist_a", "mgc_matrix_mult_b",
                                           "mgc_des_perf_a", "mgc_pci_bridge32_b", "mgc_fft_a"],
                                 "val": ["mgc_matrix_mult_a"], "test": ["mgc_fft_b", "mgc_matrix_mult_c"]}
    assert sum(manifest["samples_per_design"].values()) == 383 and len(manifest["samples_per_design"]) == 9
    train_d = set(manifest["split"]["train"])
    a = manifest["arms"]
    assert Counter(map(dz, a["FULL"])) == {"mgc_des_perf_a": 44, "mgc_edit_dist_a": 44, "mgc_fft_a": 44,
                                          "mgc_matrix_mult_b": 43, "mgc_pci_bridge32_a": 26, "mgc_pci_bridge32_b": 44}
    assert a["DIVERSE6"] == PART_B_TRAIN and Counter(map(dz, a["DIVERSE6"])) == {d: 1 for d in train_d}
    assert len(a["DIVERSE36"]) == 36 and Counter(map(dz, a["DIVERSE36"])) == {d: 6 for d in train_d}
    assert len(a["SINGLE6"]) == 6 and set(map(dz, a["SINGLE6"])) == {"mgc_fft_a"}
    assert set(a["DIVERSE6"]) <= set(a["DIVERSE36"]) <= set(a["FULL"]) and set(a["SINGLE6"]) <= set(a["FULL"])
    # validation: 6 x matrix_mult_a + the 6 held-out samples, 7 designs, never trained on by an arm
    assert len(manifest["val"]) == 12 and Counter(map(dz, manifest["val"]))["mgc_matrix_mult_a"] == 6
    assert set(map(dz, manifest["val"])) == train_d | {"mgc_matrix_mult_a"}
    held = [h for hs in manifest["holdout"].values() for h in hs]
    assert len(held) == 6 and set(held) <= set(manifest["val"]) and not set(held) & set(a["FULL"])
    test_d = set(manifest["split"]["test"])
    for arm, names in a.items():
        assert not set(names) & set(manifest["val"]) and not set(map(dz, names)) & (test_d | {"mgc_matrix_mult_a"})
    steps = {r["name"]: (r["n_train"], r["epochs"], r["steps"]) for r in manifest["runs"]}
    for s in (42, 43, 44):
        for p in ("fp16", "fp32"):
            assert steps[f"REPLICA__{p}__seed{s}"] == (251, 1, 251)
        assert steps[f"FULL__fp32__seed{s}"] == (245, 1, 245)
        assert steps[f"DIVERSE6__fp32__seed{s}"] == (6, 41, 246)
        assert steps[f"DIVERSE36__fp32__seed{s}"] == (36, 7, 252)
        assert steps[f"SINGLE6__fp32__seed{s}"] == (6, 41, 246)
    assert len(manifest["runs"]) == 18 and manifest["common_steps"] == 245 == manifest["budget_steps"]
    for r in manifest["runs"]:
        assert r["n_val"] == (6 if r["arm"] == "REPLICA" else 12)


def test_replica_order_equals_the_smoke_runs_graph_order(multi_design):
    """The smoke run indexed its shuffled order into build_experiment_graphs' train list;
    the replica must present the same list in the same order."""
    processed, manifest = multi_design
    plan = CE.plan_experiment(processed, manifest, seeds=(42,), val_from_val_design=2, per_design_mid=2, single_n=3)
    smoke_order = [g.placement_name for g in build_experiment_graphs(processed, manifest, label_norm="per_dataset",
                                                                     verbose=False)["graphs"]["train"]]
    rep = next(r for r in plan["runs"] if r["name"] == "REPLICA__fp16__seed42")
    assert rep["train"] == smoke_order and rep["seed"] == 42 and rep["epochs"] == 1
    assert abs(plan["c_max"] - build_experiment_graphs(processed, manifest, label_norm="per_dataset",
                                                       verbose=False)["c_max"]) == 0
    cfg = CE.run_config(load_config(ROOT / "configs" / "base.yaml"), rep)
    smoke = load_config(ROOT / "configs" / "base.yaml", ROOT / "configs" / "smoke.yaml")
    for k in ("learning_rate", "weight_decay", "grad_clip", "step_size", "gamma", "wmse_alpha", "lambda_var",
              "precision", "allow_tf32", "seed", "compute_train_metrics"):
        assert getattr(cfg.training, k) == getattr(smoke.training, k), k
    assert cfg.model == smoke.model and cfg.training.keep_dataset_on_gpu == "false"
    assert (cfg.experiment.split, cfg.experiment.grid_features, cfg.experiment.label_norm) == \
           (smoke.experiment.split, smoke.experiment.grid_features, smoke.experiment.label_norm)


def test_plan_refuses_to_differ_from_the_manifest(multi_design):
    processed, manifest = multi_design
    kw = dict(seeds=(42,), val_from_val_design=2, per_design_mid=2, single_n=3)
    plan = CE.plan_experiment(processed, manifest, **kw)
    exp = {"comment": "x", "split": manifest["designs"], "samples_per_design": {}, **CE.manifest_of(plan)}
    exp = json.loads(json.dumps(exp))                                   # as loaded from the shipped JSON file
    assert CE.plan_experiment(processed, manifest, expected=exp, **kw)["runs"] == plan["runs"]
    with pytest.raises(ValueError, match="pre-registered"):
        CE.plan_experiment(processed, manifest, expected={k: v for k, v in exp.items() if k != "val"}, **kw)
    exp["arms"]["DIVERSE6"] = exp["arms"]["DIVERSE6"][::-1]
    with pytest.raises(ValueError, match="pre-registered"):
        CE.plan_experiment(processed, manifest, expected=exp, **kw)


def test_common_step_budget_rule(tmp_path):
    rows = [{"epoch": 1, "global_step": i, "grad_norm": 0.0 if 220 <= i <= 245 else 0.1,
             "grad_exactly_zero": 220 <= i <= 245, "sample": "s", "cell_raw_le0": 1, "grid_raw_le0": 1}
            for i in range(1, 253)]
    pd.DataFrame(rows).to_csv(tmp_path / "steps.csv", index=False)
    s = CE.summarize_run(tmp_path, common_steps=245)
    assert not s["collapsed"]                                       # revived after step 245
    assert s["at_common_steps"] == {"common_steps": 245, "collapsed": True, "collapse_step": 220, "zero_tail_len": 26}


def test_real_manifest_is_accepted_for_the_real_sample_names(manifest, tmp_path):
    """Stub processed tree with the real 383 directory names (tiny label arrays): the
    notebook-07 plan must reproduce the shipped manifest exactly, so the run cell is not
    blocked on Colab by a spurious mismatch."""
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    rng = np.random.RandomState(0)
    for n in names:
        d = tmp_path / CE.design_of(n) / n
        d.mkdir(parents=True)
        (d / "meta.json").write_text("{}")
        np.savez(d / "sample.npz", label_h=rng.rand(3, 3).astype(np.float32), label_v=np.zeros((3, 3), np.float32))
    split = {"name": "official_seed42", "designs": manifest["split"]}
    plan = CE.plan_experiment(tmp_path, split, expected=manifest)
    assert CE.manifest_diff(plan, manifest) == []
    assert [r["train"] for r in plan["runs"] if r["arm"] == "REPLICA"][0] == \
        [n.name for n in P.list_processed(tmp_path, set(manifest["split"]["train"]))]


def test_replica_and_arm_validation_intersections(manifest):
    """REPLICA trains on all 251 official training samples, which include the 6 held-out
    samples = validation rows 7-12; it therefore validates on rows 1-6 only."""
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    replica = {n for n in names if CE.design_of(n) in manifest["split"]["train"]}
    val = manifest["val"]
    rows_1_6, rows_7_12 = set(val[:6]), set(val[6:])
    held = {h for hs in manifest["holdout"].values() for h in hs}
    assert len(replica) == 251 and rows_7_12 == held
    assert {CE.design_of(n) for n in rows_1_6} == {"mgc_matrix_mult_a"}
    assert replica & rows_1_6 == set() and replica & rows_7_12 == held
    for arm in ("FULL", "DIVERSE6", "DIVERSE36", "SINGLE6"):
        assert set(manifest["arms"][arm]) & set(val) == set(), arm
    for r in manifest["runs"]:
        assert r["n_val"] == (6 if r["arm"] == "REPLICA" else 12)
    assert "Primary outcome" in CE.aggregate([])["_note"]
