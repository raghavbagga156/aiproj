"""Notebook-08 execution-path hardening (docs/HEADPATH_PLAN.md section 11): per-run failure isolation and
complete reporting, Drive flush, protected-artifact content hashes, resume-config safety, the epoch-1
checkpoint window, and atomic prediction export. CPU only, synthetic data only (test data, never results)."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from test_headpath import _multi, hp_cfg, params
from verihgn_repro import colab
from verihgn_repro.colab import DriveLayout, DrivePersistenceError
from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.graph import build_graph
from verihgn_repro.diagnostics import headpath as H
from verihgn_repro.evaluation import evaluate as EV
from verihgn_repro.training import checkpointing as C
from verihgn_repro.training import trainer as T

SPLIT = {"name": "s", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"], "test": []}}
TINY = {"model": {"hidden_dim": 16, "num_layers": 2}}


@pytest.fixture
def graphs(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    return build_experiment_graphs(out, SPLIT, verbose=False)["graphs"]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """A synthetic 4-design plan (14 runs x 24 steps) and its graphs, shared by the failure tests."""
    tmp = tmp_path_factory.mktemp("hp")
    out = tmp / "processed"
    P.run(_multi(tmp), out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    split = {"name": "m", "designs": {"train": ["mgc_fft_a", "mgc_des_perf_a", "mgc_edit_dist_a"],
                                      "val": ["mgc_pci_bridge32_a"], "test": []}}
    plan = H.plan_headpath(out, split, val_from_val_design=2, eval_every=6)
    gb = {n: build_graph(*P.load_sample(Path(d)), label_norm="per_dataset", c_max=plan["c_max"])
          for n, d in plan["sample_dirs"].items()}
    return plan, gb


def _fail_at(monkeypatch, target, exc_factory, when="before"):
    """Inject a failure into the run whose directory name is `target`: before training starts, or
    inside epoch 2 (the Trainer records 'failed' in status.json and re-raises)."""
    if when == "before":
        orig = T.Trainer.run

        def run(self, *a, **k):
            if self.run_dir.name == target:
                raise exc_factory()
            return orig(self, *a, **k)
        monkeypatch.setattr(T.Trainer, "run", run)
    else:
        orig = T.Trainer._train_epoch

        def epoch(self):
            if self.run_dir.name == target and self.epoch == 2:
                raise exc_factory()
            return orig(self)
        monkeypatch.setattr(T.Trainer, "_train_epoch", epoch)


def _disk(report):
    return json.loads(Path(report).read_text())


def _status(rep):
    return {r["name"]: r["run_status"] for r in rep["results"]}


# ------------------------------------------------------------ 1. failure isolation and complete reporting
def test_failure_in_the_middle_is_isolated_and_all_14_runs_are_reported(synth, tmp_path, monkeypatch):
    plan, gb = synth
    names = [r["name"] for r in plan["runs"]]
    target = names[6]                                           # HEADPATH_O__seed45, fails inside epoch 2
    _fail_at(monkeypatch, target, lambda: RuntimeError("synthetic failure in epoch 2"), when="during")
    report = tmp_path / "r.json"
    rep = H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", report_path=report)
    disk = _disk(report)
    assert [r["name"] for r in disk["results"]] == names        # every planned run, in plan order
    st = _status(disk)
    assert st[target] == "FAILED" and all(st[n] == "COMPLETED" for n in names if n != target)   # later runs ran
    f = next(r for r in disk["results"] if r["name"] == target)
    assert f["error"]["type"] == "RuntimeError" and "synthetic failure" in f["error"]["message"]
    assert "Traceback" in f["error"]["traceback"] and f["error"]["runtime_fatal"] is False
    assert f["trainer_status"]["state"] == "failed" and f["category"] is None and f["valid"] is False
    a = disk["analysis"]
    assert a["experiment_status"] == "FAILED" and a["verdict"].startswith("FAILED")       # never VALID
    assert any(target in s for s in a["status_reasons"]) and target in a["missing_runs"]
    assert [x["seed"] for x in a["primary_test"]["excluded"]] == [45]
    assert disk["run_status_counts"] == {"COMPLETED": 13, "INCOMPLETE": 0, "FAILED": 1, "NOT_STARTED": 0}
    assert rep["results"] == disk["results"]
    assert "FAILED EXPERIMENT" in a["conclusion"] and "no conclusion is drawn" in a["conclusion"]


@pytest.mark.parametrize("position", ["first", "final"])
def test_failure_in_first_or_final_position(synth, tmp_path, monkeypatch, position):
    plan, gb = synth
    names = [r["name"] for r in plan["runs"]]
    sel = names[:3] if position == "first" else names[-3:]
    target = sel[0] if position == "first" else sel[-1]          # plan-first / plan-final run
    _fail_at(monkeypatch, target, lambda: ValueError(f"synthetic {position} failure"))
    report = tmp_path / "r.json"
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", only=sel, report_path=report)
    st = _status(_disk(report))
    assert len(st) == 14 and st[target] == "FAILED"
    assert all(st[n] == "COMPLETED" for n in sel if n != target)                      # the others still ran
    assert all(st[n] == "NOT_STARTED" for n in names if n not in sel)
    assert _disk(report)["analysis"]["experiment_status"] == "FAILED"


def test_runtime_fatal_failure_marks_remaining_runs_not_started(synth, tmp_path, monkeypatch):
    plan, gb = synth
    names = [r["name"] for r in plan["runs"]]
    _fail_at(monkeypatch, names[3], lambda: RuntimeError("CUDA error: an illegal memory access was encountered"))
    report = tmp_path / "r.json"
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", report_path=report)
    disk = _disk(report)
    st = _status(disk)
    assert [st[n] for n in names[:3]] == ["COMPLETED"] * 3 and st[names[3]] == "FAILED"
    assert all(st[n] == "NOT_STARTED" for n in names[4:])
    assert all("runtime was unusable" in r["reason"] for r in disk["results"][4:])
    assert disk["results"][3]["error"]["runtime_fatal"] is True
    assert not any((tmp_path / "runs" / n).exists() for n in names[4:])              # really not started
    assert disk["analysis"]["experiment_status"] == "FAILED"


def test_failed_device_probe_is_fatal_but_cuda_oom_is_not(synth, tmp_path, monkeypatch):
    plan, gb = synth
    names = [r["name"] for r in plan["runs"]]
    sel = names[:3]
    _fail_at(monkeypatch, sel[1], lambda: torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"))
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "a", device="cpu", only=sel, report_path=tmp_path / "a.json")
    st = _status(_disk(tmp_path / "a.json"))
    assert (st[sel[1]], st[sel[2]]) == ("FAILED", "COMPLETED")                       # OOM alone: continue
    monkeypatch.setattr(H, "device_usable", lambda device=None: False)              # ...unless the device is broken
    _fail_at(monkeypatch, sel[1], lambda: ValueError("anything"))
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "b", device="cpu", only=sel, report_path=tmp_path / "b.json")
    st = _status(_disk(tmp_path / "b.json"))
    assert (st[sel[1]], st[sel[2]]) == ("FAILED", "NOT_STARTED")
    assert H.runtime_fatal(RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED")) and H.runtime_fatal(KeyboardInterrupt())


def test_keyboard_interrupt_is_recorded_then_reraised(synth, tmp_path, monkeypatch):
    plan, gb = synth
    names = [r["name"] for r in plan["runs"]]
    _fail_at(monkeypatch, names[1], KeyboardInterrupt)
    report = tmp_path / "r.json"
    with pytest.raises(KeyboardInterrupt):
        H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", only=names[:3],
                   report_path=report)
    disk = _disk(report)
    st = _status(disk)
    assert (st[names[0]], st[names[1]], st[names[2]]) == ("COMPLETED", "FAILED", "NOT_STARTED")
    assert "interrupted" in disk["results"][2]["reason"] and disk["results"][1]["error"]["type"] == "KeyboardInterrupt"


def test_report_exists_before_the_first_run_finishes(synth, tmp_path, monkeypatch):
    plan, gb = synth
    seen = {}
    orig = T.Trainer.run

    def run(self, *a, **k):
        seen.setdefault("at_first_run", _disk(tmp_path / "r.json"))
        return orig(self, *a, **k)
    monkeypatch.setattr(T.Trainer, "run", run)
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu",
               only=[plan["runs"][0]["name"]], report_path=tmp_path / "r.json")
    first = seen["at_first_run"]
    assert len(first["results"]) == 14 and all(r["run_status"] == "NOT_STARTED" for r in first["results"])
    assert first["analysis"]["experiment_status"] == "INCOMPLETE"


def test_verdict_rules_never_report_valid_with_a_missing_failed_or_invalid_run(synth):
    plan, _ = synth
    base = [{"name": r["name"], "arm": r["arm"], "seed": r["seed"], "retest": r["retest"], "run_status": "COMPLETED",
             "valid": True, "category": "CELL_HEAD_FAILURE_PERSISTS", "hsig_criteria_met": False,
             "p2_zero_reference": {"x": 1}, "invalid_reasons": [], "P1": {"final_cell_activity": "DEAD"},
             "grid": {"final_activity": "DEAD", "final_val_spearman_indep": 0.0},
             "P2": {"cell_wmse_clamped_improvement_vs_zero": 0.0}, "P3": {"final_val_cell_spearman_indep": 0.0},
             "state_at_replication_step": {"cell": "DEAD", "grid": "DEAD"}} for r in plan["runs"]]
    assert H.analyse(base, plan)["experiment_status"] == "VALID"
    for i, change in ((13, {"run_status": "FAILED"}), (13, {"run_status": "NOT_STARTED"}),
                      (13, {"run_status": "INCOMPLETE"}), (12, {"valid": False, "invalid_reasons": ["x"]})):
        rs = [dict(r) for r in base]
        rs[i].update(change)                                       # a RETEST run counts too
        assert H.analyse(rs, plan)["experiment_status"] != "VALID", change
    assert H.analyse(base[:-1], plan)["experiment_status"] == "INCOMPLETE"


# ------------------------------------------------------------ atomic report writes
def test_report_writes_are_atomic_and_an_interrupted_write_keeps_the_previous_report(synth, tmp_path, monkeypatch):
    plan, _ = synth
    p = tmp_path / "r.json"
    recs = [H.not_started_record(r, "x") for r in plan["runs"]]
    H._write(p, plan, recs, None)
    good = p.read_bytes()
    orig_dump = json.dump

    def partial_dump(obj, f, **k):                                 # writes half, then the process "dies"
        s = json.dumps(obj, **k)
        f.write(s[: len(s) // 2])
        raise OSError("disk vanished mid-write")
    monkeypatch.setattr(C.json, "dump", partial_dump)
    with pytest.raises(OSError):
        H._write(p, plan, recs, None)
    assert p.read_bytes() == good and json.loads(good)             # the report is intact and parseable
    monkeypatch.setattr(C.json, "dump", orig_dump)
    monkeypatch.setattr(C.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(OSError):
        H._write(p, plan, recs, None)
    assert p.read_bytes() == good
    monkeypatch.undo()
    H.update_report(p, {"protected_artifacts": {"unchanged": True}})               # leftover .tmp is harmless
    assert _disk(p)["protected_artifacts"] == {"unchanged": True} and len(_disk(p)["results"]) == 14


# ------------------------------------------------------------ protected artifacts (content hashes)
def _layout_with_protected(tmp_path):
    L = DriveLayout(root=tmp_path / "drive" / "VeriHGN").ensure()
    files = {
        L.runs / "verihgn_ispd15_smoke__official_seed42__paper_eq3__per_dataset__fp16__seed42" / "checkpoints" / "last.pt":
            os.urandom(2 << 20),                                   # 2 MB: large files are content-hashed too
        L.runs / "precision_benchmark" / "fp32__seed42" / "history.csv": b"epoch,loss\n1,0.1\n",
        L.runs / "collapse_diag" / "FULL__fp32__seed42" / "checkpoints" / "best.pt": os.urandom(1 << 20) + b"x",
        L.reports / "05_smoke.json": b"{}", L.reports / "06_step_diagnostic.json": b"{}",
        L.reports / "07_collapse_experiment.json": b"{}", L.reports / "STAGE_REPORT.md": b"# stage",
        L.runs / "headpath" / "HEADPATH_O__seed42" / "steps.csv": b"not protected",
        L.reports / "08_headpath_experiment.json": b"{}",
    }
    for p, b in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return L, files


def test_protected_artifacts_cover_every_pattern_with_full_content_hashes(tmp_path):
    L, files = _layout_with_protected(tmp_path)
    fp = H.protected_artifacts(L)
    assert len(fp) == 7 and not any("headpath" in k or k.startswith("reports/08_") for k in fp)
    for p, b in files.items():
        key = f"{'runs' if L.runs in p.parents else 'reports'}/{p.relative_to(L.runs if L.runs in p.parents else L.reports).as_posix()}"
        if key in fp:
            assert fp[key]["sha256"] == hashlib.sha256(b).hexdigest() and fp[key]["size"] == len(b)
    big = next(p for p in files if p.name == "last.pt")
    st = big.stat()
    data = bytearray(big.read_bytes())
    data[123456] ^= 0xFF                                           # same size ...
    big.write_bytes(bytes(data))
    os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns))            # ... and same mtime
    cmp = H.compare_fingerprints(fp, H.protected_artifacts(L))
    assert not cmp["unchanged"] and len(cmp["modified"]) == 1 and "last.pt" in cmp["modified"][0]
    big.write_bytes(files[big])
    os.utime(big, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))  # content restored, mtime differs
    cmp = H.compare_fingerprints(fp, H.protected_artifacts(L))
    assert cmp["unchanged"] and len(cmp["mtime_only_changes"]) == 1
    (L.runs / "headpath" / "new.txt").write_text("ignored")       # outside the protected set
    (L.reports / "07_new.json").write_text("{}")                   # a new protected file is a change
    cmp = H.compare_fingerprints(fp, H.protected_artifacts(L))
    assert not cmp["unchanged"] and cmp["added"] == ["reports/07_new.json"]


class FakeDrive:
    def __init__(self, fail_flush=False, corrupt_on_mount=None):
        self.calls, self.fail_flush, self.corrupt = [], fail_flush, corrupt_on_mount

    def flush_and_unmount(self):
        self.calls.append("flush_and_unmount")
        if self.fail_flush:
            raise OSError("drive daemon died")

    def mount(self, mountpoint):
        self.calls.append(("mount", mountpoint))
        if self.corrupt:
            Path(self.corrupt).write_text("changed on the way")


def test_execute_runs_protected_check_and_flush_even_when_a_run_fails(synth, tmp_path, monkeypatch):
    plan, gb = synth
    L, _ = _layout_with_protected(tmp_path)
    before = H.protected_artifacts(L)
    report, events = L.reports / "08_headpath_experiment__t.json", []

    def flush(paths):
        events.append(("flush", report.read_bytes()))
        return {"ok": True}
    _fail_at(monkeypatch, plan["runs"][0]["name"], lambda: RuntimeError("boom"))
    out = H.execute(plan, load_config(overrides=TINY), L, report, lambda: gb, flush=flush, device="cpu",
                    only=[plan["runs"][0]["name"], plan["runs"][1]["name"]])
    assert out["report"]["analysis"]["experiment_status"] == "FAILED"   # an isolated failure is reported, not raised
    disk = _disk(report)
    assert disk["protected_artifacts"]["unchanged"] is True and H.protected_artifacts(L) == before
    assert _status(disk)[plan["runs"][0]["name"]] == "FAILED" and _status(disk)[plan["runs"][1]["name"]] == "COMPLETED"
    assert len(events) == 1 and events[0][1] == report.read_bytes()       # flushed once, after the last write


def test_execute_reports_all_runs_when_graph_building_fails(synth, tmp_path):
    plan, _ = synth
    L, _ = _layout_with_protected(tmp_path)
    report, flushed = L.reports / "08_r.json", []

    def broken():
        raise MemoryError("no RAM for the graphs")
    with pytest.raises(MemoryError):
        H.execute(plan, load_config(overrides=TINY), L, report, broken, flush=lambda p: flushed.append(p) or {})
    disk = _disk(report)
    assert len(disk["results"]) == 14 and all(r["run_status"] == "NOT_STARTED" for r in disk["results"])
    assert "building the graphs failed" in disk["results"][0]["reason"]
    assert disk["protected_artifacts"]["unchanged"] is True and len(flushed) == 1


def test_execute_detects_a_protected_change_and_still_flushes(synth, tmp_path, monkeypatch):
    plan, gb = synth
    L, _ = _layout_with_protected(tmp_path)
    report, flushed = L.reports / "08_r.json", []

    def tamper(*a, **k):
        (L.reports / "06_step_diagnostic.json").write_text('{"tampered": 1}')
        H._write(report, plan, [H.not_started_record(r, "x") for r in plan["runs"]], None)
        return {"analysis": {}}
    monkeypatch.setattr(H, "run_plan", tamper)
    with pytest.raises(H.ProtectedArtifactError):
        H.execute(plan, load_config(overrides=TINY), L, report, lambda: gb, flush=lambda p: flushed.append(p) or {})
    prot = _disk(report)["protected_artifacts"]
    assert prot["unchanged"] is False and prot["modified"] == ["reports/06_step_diagnostic.json"] and len(flushed) == 1


def test_execute_raises_a_flush_failure_and_keeps_the_run_error_first(synth, tmp_path, monkeypatch):
    plan, gb = synth
    L, _ = _layout_with_protected(tmp_path)
    monkeypatch.setattr(H, "run_plan", lambda *a, **k: {"analysis": {}})

    def bad_flush(paths):
        raise DrivePersistenceError("flush failed")
    with pytest.raises(DrivePersistenceError):
        H.execute(plan, load_config(overrides=TINY), L, L.reports / "08_a.json", lambda: gb, flush=bad_flush)
    monkeypatch.setattr(H, "run_plan", lambda *a, **k: (_ for _ in ()).throw(ValueError("run error")))
    with pytest.raises(ValueError, match="run error") as ei:
        H.execute(plan, load_config(overrides=TINY), L, L.reports / "08_b.json", lambda: gb, flush=bad_flush)
    assert any("drive flush failed" in n for n in getattr(ei.value, "__notes__", []))


# ------------------------------------------------------------ Drive flush (mocked; no real unmount)
def test_flush_drive_flushes_then_verifies_through_a_remount(tmp_path):
    a, b = tmp_path / "report.json", tmp_path / "plan.json"
    a.write_text('{"r": 1}')
    b.write_text('{"p": 1}')
    d = FakeDrive()
    rec = colab.flush_drive([a, b], drive_module=d, mountpoint="/content/drive")
    assert d.calls == ["flush_and_unmount", ("mount", "/content/drive")] and rec["remount_verified"] is True
    assert rec["sha256_before_flush"][str(a)] == hashlib.sha256(a.read_bytes()).hexdigest()
    d = FakeDrive()
    assert colab.flush_drive([a], remount_verify=False, drive_module=d)["remount_verified"] is None
    assert d.calls == ["flush_and_unmount"]
    with pytest.raises(DrivePersistenceError, match="flush_and_unmount"):
        colab.flush_drive([a], drive_module=FakeDrive(fail_flush=True))
    with pytest.raises(DrivePersistenceError, match="differ"):
        colab.flush_drive([a], drive_module=FakeDrive(corrupt_on_mount=a))
    d = FakeDrive()
    with pytest.raises(DrivePersistenceError, match="missing before the flush"):
        colab.flush_drive([tmp_path / "nope.json"], drive_module=d)
    assert d.calls == []                                            # never unmounts when a file is missing


# ------------------------------------------------------------ resume configuration safety
def _tree_bytes(d):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def test_resume_refuses_a_changed_training_configuration(graphs, tmp_path):
    run = tmp_path / "run"
    T.Trainer(hp_cfg("O", session=1), graphs, run, device="cpu").run()
    snap = _tree_bytes(run)
    bad = hp_cfg("O")
    bad.training.learning_rate = 1e-3
    with pytest.raises(C.CheckpointError, match="training.learning_rate"):
        T.Trainer(bad, graphs, run, device="cpu").run()
    bad = hp_cfg("O", eval_every=2)                                 # observation settings are protected too
    with pytest.raises(C.CheckpointError, match="training.eval_every_steps"):
        T.Trainer(bad, graphs, run, device="cpu").run()
    assert _tree_bytes(run) == snap                                 # a refused resume writes nothing
    env1 = (run / "env.json").read_bytes()
    T.Trainer(hp_cfg("O"), graphs, run, device="cpu").run()        # only the session length differs: allowed
    assert (run / "env.json").read_bytes() == env1                  # first-session environment preserved
    assert (run / "config.yaml").read_bytes() == snap["config.yaml"]
    assert sorted(p.name for p in run.glob("env_session_*.json")) == ["env_session_001.json", "env_session_002.json"]
    assert json.loads((run / "status.json").read_text())["state"] == "completed"


def test_resume_accepts_an_older_checkpoint_and_rejects_unknown_keys(graphs, tmp_path):
    run = tmp_path / "run"
    T.Trainer(hp_cfg("O", session=1), graphs, run, device="cpu").run()
    p = run / "checkpoints" / "last.pt"
    s = C.load_checkpoint(p)
    for k in ("cell_head_train_clamp",):                            # written before the key existed
        del s["config"]["model"][k]
    for k in ("step_log_heads", "eval_every_steps"):
        s["config"]["training"].pop(k)
    old = hp_cfg("O", heads=False, eval_every=0)                    # those keys' defaults
    assert T.resume_config_diff(s["config"], old) == []
    s["config"]["training"]["mystery"] = 1
    assert T.resume_config_diff(s["config"], old)[0].startswith("<stored configuration cannot be loaded")
    assert T.resume_config_diff(None, old) == ["<checkpoint has no stored configuration>"]


# ------------------------------------------------------------ the epoch-1 checkpoint window
def test_epoch1_history_without_checkpoint_is_archived_and_the_run_restarts(graphs, tmp_path, monkeypatch):
    ref = T.Trainer(hp_cfg("U"), graphs, tmp_path / "ref", device="cpu")
    ref.run()
    run = tmp_path / "run"
    orig = C.CheckpointManager.save

    def crash_before_first_checkpoint(self, state, epoch, is_best):
        raise OSError("runtime died between epoch 1's history write and its checkpoint")
    monkeypatch.setattr(C.CheckpointManager, "save", crash_before_first_checkpoint)
    with pytest.raises(OSError):
        T.Trainer(hp_cfg("U"), graphs, run, device="cpu").run()
    assert (run / "history.csv").exists() and not (run / "checkpoints" / "last.pt").exists()
    env1 = (run / "env.json").read_bytes()
    monkeypatch.setattr(C.CheckpointManager, "save", orig)
    tr = T.Trainer(hp_cfg("U"), graphs, run, device="cpu")
    tr.run()
    arch = [d for d in run.iterdir() if d.name.startswith("aborted_attempt_")]
    assert len(arch) == 1 and tr.resume_info["archived_prior_attempt"] == str(arch[0])
    assert {p.name for p in arch[0].iterdir()} >= {"history.csv", "steps.csv", "evals.csv", "status.json"}
    assert torch.equal(params(tr.model), params(ref.model))         # same result as an uninterrupted run (CPU)
    for f in ("steps.csv", "evals.csv", "history.csv"):
        a, b = pd.read_csv(run / f), pd.read_csv(tmp_path / "ref" / f)
        assert a["epoch"].tolist() == b["epoch"].tolist(), f        # no duplicated rows
    assert (run / "env.json").read_bytes() == env1


def test_lost_checkpoints_after_completion_or_two_epochs_are_still_refused(graphs, tmp_path):
    run = tmp_path / "run"
    T.Trainer(hp_cfg("O", epochs=1), graphs, run, device="cpu").run()       # completed after 1 epoch
    for p in (run / "checkpoints").glob("*.pt"):
        p.unlink()
    with pytest.raises(C.CheckpointNotFoundError, match="refusing to restart"):
        T.Trainer(hp_cfg("O", epochs=1), graphs, run, device="cpu").run()
    run2 = tmp_path / "run2"
    T.Trainer(hp_cfg("O", epochs=3, session=2), graphs, run2, device="cpu").run()
    for p in (run2 / "checkpoints").glob("*.pt"):
        p.unlink()
    with pytest.raises(C.CheckpointNotFoundError, match="refusing to restart"):
        T.Trainer(hp_cfg("O", epochs=3), graphs, run2, device="cpu").run()


# ------------------------------------------------------------ atomic prediction export
def _recs(graphs, tmp_path):
    tr = T.Trainer(hp_cfg("O"), graphs, tmp_path / "pred_model", device="cpu")
    tr._init_state()
    gb = {g.placement_name: g for gs in graphs.values() for g in gs}
    return EV.predict(tr.model, tr.batches["val"], "cpu"), gb


def test_interrupted_prediction_export_is_never_treated_as_complete(graphs, tmp_path, monkeypatch):
    recs, gb = _recs(graphs, tmp_path)
    final = tmp_path / "run" / H.PREDICTIONS_DIR
    orig = EV.save_predictions

    def dies_after_one_file(recs_, gb_, out_dir, *a, **k):
        orig(recs_[:1], gb_, out_dir)
        raise OSError("interrupted mid-export")
    monkeypatch.setattr(EV, "save_predictions", dies_after_one_file)
    with pytest.raises(OSError):
        H.export_predictions_atomic(recs, gb, final)
    assert not final.exists() and list(final.parent.glob(final.name + ".tmp_*"))
    monkeypatch.setattr(EV, "save_predictions", orig)
    out = H.export_predictions_atomic(recs, gb, final)               # recovery on the next attempt
    assert out["status"] == "written" and len(out["moved_aside"]) == 1 and H.predictions_complete(final)
    assert out["n_files"] == len(recs) and not list(final.parent.glob(final.name + ".tmp_*"))
    assert H.export_predictions_atomic(recs, gb, final)["status"] == "existing_complete"
    victim = next(p for p in final.iterdir() if p.suffix == ".npz")
    victim.write_bytes(b"corrupt")                                   # a tampered export is not complete either
    assert not H.predictions_complete(final)
    out = H.export_predictions_atomic(recs, gb, final)
    assert out["status"] == "written" and H.predictions_complete(final) and len(out["moved_aside"]) == 1


def test_legacy_partial_prediction_directory_is_moved_aside_by_run_plan(synth, tmp_path):
    plan, gb = synth
    name = plan["runs"][0]["name"]
    H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", only=[name])
    final = tmp_path / "runs" / name / H.PREDICTIONS_DIR
    (final / H.PREDICTIONS_MARKER).unlink()                          # as an export interrupted before the marker
    rep = H.run_plan(plan, load_config(overrides=TINY), gb, tmp_path / "runs", device="cpu", only=[name])
    r = next(x for x in rep["results"] if x["name"] == name)
    assert r["run_status"] == "COMPLETED" and r["predictions_export"]["status"] == "written"
    assert len(r["predictions_export"]["moved_aside"]) == 1 and H.predictions_complete(final)


def test_moved_aside_and_report_names_never_collide_within_one_second(graphs, tmp_path, monkeypatch):
    """Regression: two exports (or reports) within the same second used to choose the same name."""
    recs, gb = _recs(graphs, tmp_path)
    monkeypatch.setattr(H.time, "strftime", lambda fmt, *a: "20261021_000000")   # every call in the same second
    final = tmp_path / "run" / H.PREDICTIONS_DIR
    H.export_predictions_atomic(recs, gb, final)
    asides = []
    for _ in range(3):                                              # tamper and re-export three times
        (final / H.PREDICTIONS_MARKER).unlink()
        asides += H.export_predictions_atomic(recs, gb, final)["moved_aside"]
    assert len(set(asides)) == 3 and all(Path(a).is_dir() for a in asides) and H.predictions_complete(final)
    seen = []
    for _ in range(3):
        p = H.new_report_path(tmp_path, "08_headpath_experiment.json")
        assert not p.exists()
        p.write_text("{}")
        seen.append(p.name)
    assert seen == ["08_headpath_experiment.json", "08_headpath_experiment__20261021_000000.json",
                    "08_headpath_experiment__20261021_000000_1.json"]


def test_execute_lists_all_runs_when_the_plan_is_refused_before_the_first_run(synth, tmp_path):
    plan, gb = synth
    L, _ = _layout_with_protected(tmp_path)
    report, flushed = L.reports / "08_r.json", []
    with pytest.raises(RuntimeError, match="pending research decisions"):
        H.execute({**plan, "pending_research_decisions": ["open question"]}, load_config(overrides=TINY), L, report,
                  lambda: gb, flush=lambda p: flushed.append(p) or {})
    disk = _disk(report)
    assert len(disk["results"]) == 14 and all(r["run_status"] == "NOT_STARTED" for r in disk["results"])
    assert "pending research decisions" in disk["results"][0]["reason"]
    assert disk["protected_artifacts"]["unchanged"] is True and len(flushed) == 1
