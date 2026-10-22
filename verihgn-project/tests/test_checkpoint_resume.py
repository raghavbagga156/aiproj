"""Regression tests for checkpoint save/load and resume (notebook 05 incident:
'TypeError: RNG state must be a torch.ByteTensor' when a new Trainer resumed on
CUDA, because the checkpoint was deserialised with map_location=<cuda device>)."""
import random
import zipfile

import numpy as np
import pytest
import torch

from verihgn_repro.config import load_config
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.training import checkpointing as C
from verihgn_repro.training import trainer as T
from verihgn_repro.training.trainer import Trainer
from verihgn_repro.utils.seed import as_cpu_byte_tensor, rng_state, set_rng_state

SPLIT = {"name": "synthetic_train", "designs": {"train": ["mgc_fft_a"], "val": ["mgc_pci_bridge32_a"],
                                                "test": ["mgc_pci_bridge32_a"]}}
SMALL = {"model": {"hidden_dim": 16, "num_layers": 2}}     # dropout stays at the official 0.15


@pytest.fixture
def graphs(synthetic_raw, tmp_path):
    raw, _ = synthetic_raw
    out = tmp_path / "processed"
    man = P.run(raw, out, scope="official9", cfg=P.prep_config(knn_k=4), workers=1)
    assert man["ok"] == 3 and not man["failed"]
    return build_experiment_graphs(out, SPLIT, verbose=False)["graphs"]


def cfg_for(epochs=3, session=None, seed=42):
    return load_config(overrides={**SMALL, "training": {"precision": "fp32", "num_epochs": epochs,
                                                        "max_epochs_this_session": session, "seed": seed}})


def params(model):
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def draws():
    out = {"python": random.random(), "numpy": float(np.random.rand()), "torch": torch.rand(4)}
    if torch.cuda.is_available():
        out["cuda"] = torch.rand(4, device="cuda").cpu()
    return out


def assert_same_draws(a, b):
    assert a["python"] == b["python"] and a["numpy"] == b["numpy"]
    assert torch.equal(a["torch"], b["torch"])
    if "cuda" in a:
        assert torch.equal(a["cuda"], b["cuda"])


# ------------------------------------------------------------- 1, 2, 3: RNG
def test_rng_state_types_survive_checkpoint_roundtrip(tmp_path):
    st = rng_state()
    assert isinstance(st["python"], tuple) and isinstance(st["numpy"], tuple)
    assert st["torch"].type() == "torch.ByteTensor" and st["torch"].device.type == "cpu"
    for s in st.get("cuda", []):
        assert s.type() == "torch.ByteTensor" and s.device.type == "cpu"
    C.atomic_torch_save({"rng": st}, tmp_path / "x.pt")
    back = C.load_checkpoint(tmp_path / "x.pt", required=("rng",))["rng"]
    assert back["torch"].type() == "torch.ByteTensor" and torch.equal(back["torch"], st["torch"])
    assert back["python"] == st["python"]
    assert back["numpy"][0] == st["numpy"][0] and np.array_equal(back["numpy"][1], st["numpy"][1])
    for a, b in zip(st.get("cuda", []), back.get("cuda", [])):
        assert b.type() == "torch.ByteTensor" and torch.equal(a, b)


def test_python_numpy_torch_cuda_rng_restored_from_checkpoint(tmp_path):
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    C.atomic_torch_save({"rng": rng_state()}, tmp_path / "x.pt")
    expected = draws()
    draws(); draws()                                    # advance every generator
    report = set_rng_state(C.load_checkpoint(tmp_path / "x.pt", required=("rng",))["rng"])
    assert_same_draws(draws(), expected)
    assert report["torch"] and report["cuda_devices_restored"] == torch.cuda.device_count() * \
        int(torch.cuda.is_available())


def test_loader_always_deserialises_onto_cpu(tmp_path, monkeypatch):
    """The original bug: torch.load(map_location=cuda) moved the CPU generator state
    to the GPU. Every load path must use map_location='cpu'."""
    C.atomic_torch_save({"epoch": 1}, tmp_path / "x.pt")
    seen = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: seen.append(k.get("map_location")) or real(*a, **k))
    C.load_checkpoint(tmp_path / "x.pt", required=("epoch",))
    assert seen == ["cpu"]


def test_rng_state_on_wrong_device_or_dtype_is_handled_explicitly():
    st = torch.get_rng_state()
    with pytest.raises(TypeError, match="meta tensor"):
        as_cpu_byte_tensor(st.to("meta"), "torch")          # what map_location to a non-CPU device yields
    with pytest.raises(TypeError, match="uint8"):
        as_cpu_byte_tensor(st.float(), "torch")
    with pytest.raises(TypeError, match="uint8"):
        set_rng_state({"python": random.getstate(), "numpy": np.random.get_state(), "torch": st.numpy()})
    with pytest.raises(KeyError, match="numpy"):
        set_rng_state({"python": random.getstate(), "torch": st})
    if torch.cuda.is_available():                           # a GPU copy is restored losslessly
        assert torch.equal(as_cpu_byte_tensor(st.cuda(), "torch"), st)


# ------------------------------------------------- 4, 5: optimizer/scheduler/epoch/best
def test_resume_restores_optimizer_scheduler_epoch_best(graphs, tmp_path):
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=2), graphs, run, device="cpu").run()
    saved = C.load_checkpoint(run / "checkpoints" / "last.pt")
    assert saved["epoch"] == 2

    tr = Trainer(cfg_for(3), graphs, run, device="cpu")
    s, path = tr.resolve_start("resume")
    assert path == run / "checkpoints" / "last.pt"
    tr._init_state()
    tr._restore(s)
    assert tr.epoch == 2 and tr.best == saved["best_avg_spearman"] and tr.patience_counter == saved["patience_counter"]
    assert tr.scheduler.state_dict() == saved["scheduler"]
    assert tr.optimizer.param_groups[0]["lr"] == saved["optimizer"]["param_groups"][0]["lr"]
    got, exp = tr.optimizer.state_dict()["state"], saved["optimizer"]["state"]
    assert got.keys() == exp.keys() and len(got) > 0
    for k in exp:
        for name in exp[k]:
            assert torch.equal(torch.as_tensor(got[k][name]), torch.as_tensor(exp[k][name])), (k, name)
    for a, b in zip(tr.model.state_dict().values(), saved["model"].values()):
        assert torch.equal(a, b)
    assert tr.scaler.state_dict() == saved["scaler"]
    assert torch.equal(tr.shuffle_gen.get_state(), saved["shuffle_gen"])
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])


# ------------------------------------------------- 6, 7: interrupted == uninterrupted
def test_interrupted_then_resumed_equals_uninterrupted(graphs, tmp_path):
    full = Trainer(cfg_for(3), graphs, tmp_path / "full", device="cpu")
    full.run()
    Trainer(cfg_for(3, session=2), graphs, tmp_path / "split", device="cpu").run()
    res = Trainer(cfg_for(3), graphs, tmp_path / "split", device="cpu")
    st = res.run(mode="resume")
    assert st["state"] == "completed" and st["epoch"] == 3
    assert res.resume_info["resumed"] and res.resume_info["start_epoch"] == 2
    torch.testing.assert_close(params(full.model), params(res.model), rtol=0, atol=0)
    a = C.load_checkpoint(tmp_path / "full" / "checkpoints" / "last.pt")
    b = C.load_checkpoint(tmp_path / "split" / "checkpoints" / "last.pt")
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"]) and a["rng"]["python"] == b["rng"]["python"]
    import pandas as pd
    ha, hb = pd.read_csv(tmp_path / "full" / "history.csv"), pd.read_csv(tmp_path / "split" / "history.csv")
    for col in ("train_loss", "val_loss", "val_avg_spearman"):
        assert ha[col].tolist() == hb[col].tolist(), col


def test_equivalence_test_is_sensitive_to_rng_restore(graphs, tmp_path, monkeypatch):
    """Negative control: without restoring the RNG streams the resumed run diverges,
    so the equality above really depends on the RNG restore."""
    full = Trainer(cfg_for(3), graphs, tmp_path / "full", device="cpu")
    full.run()
    Trainer(cfg_for(3, session=2), graphs, tmp_path / "split", device="cpu").run()
    monkeypatch.setattr(T, "set_rng_state", lambda st: {})
    res = Trainer(cfg_for(3), graphs, tmp_path / "split", device="cpu")
    res.run(mode="resume")
    assert not torch.equal(params(full.model), params(res.model))


# ------------------------------------------------- 8, 9: missing / corrupted
def test_missing_checkpoint_is_an_error_when_resume_requested(graphs, tmp_path):
    with pytest.raises(C.CheckpointNotFoundError, match="no last.pt"):
        Trainer(cfg_for(), graphs, tmp_path / "empty", device="cpu").run(mode="resume")
    assert not (tmp_path / "empty" / "status.json").exists()      # nothing written on failure


def test_auto_refuses_silent_fresh_start_when_history_exists(graphs, tmp_path):
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=2), graphs, run, device="cpu").run()
    for p in (run / "checkpoints").glob("*.pt"):
        p.rename(p.with_suffix(".moved"))                          # checkpoints gone, history remains
    with pytest.raises(C.CheckpointNotFoundError, match="refusing to restart"):
        Trainer(cfg_for(3), graphs, run, device="cpu").run()


def test_auto_on_empty_dir_starts_fresh_and_says_so(graphs, tmp_path):
    tr = Trainer(cfg_for(1), graphs, tmp_path / "new", device="cpu")
    tr.run()
    assert tr.resume_info == {"mode": "auto", "resumed": False, "checkpoint": None, "start_epoch": 0}


def test_fresh_never_overwrites_existing_checkpoints(graphs, tmp_path):
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=1), graphs, run, device="cpu").run()
    before = (run / "checkpoints" / "last.pt").read_bytes()
    with pytest.raises(FileExistsError):
        Trainer(cfg_for(3), graphs, run, device="cpu").run(fresh=True)
    assert (run / "checkpoints" / "last.pt").read_bytes() == before


def test_corrupted_last_checkpoint_raises_and_is_not_modified(graphs, tmp_path):
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=2), graphs, run, device="cpu").run()
    last = run / "checkpoints" / "last.pt"
    last.write_bytes(last.read_bytes()[: last.stat().st_size // 2])    # truncated upload
    for mode in ("auto", "resume"):
        with pytest.raises(C.CheckpointCorruptError):
            Trainer(cfg_for(3), graphs, run, device="cpu").run(mode=mode)
    assert last.stat().st_size > 0 and C.RunStatus(run).read()["epoch"] == 2


def test_corruption_kinds_detected(tmp_path):
    p = tmp_path / "x.pt"
    C.atomic_torch_save({"epoch": 1, "w": torch.arange(1000.0)}, p)
    p.write_bytes(b"")
    with pytest.raises(C.CheckpointCorruptError, match="empty"):
        C.load_checkpoint(p, required=("epoch",))
    C.atomic_torch_save({"epoch": 1, "w": torch.arange(1000.0)}, p)
    raw = bytearray(p.read_bytes())
    with zipfile.ZipFile(p) as z:
        info = next(i for i in z.infolist() if i.file_size > 1000)
    off = info.header_offset + 30 + len(info.filename) + len(info.extra) + 100
    raw[off] ^= 0xFF                                            # flip one byte of tensor data
    p.write_bytes(bytes(raw))
    with pytest.raises(C.CheckpointCorruptError):
        C.load_checkpoint(p, required=("epoch",))
    C.atomic_torch_save({"epoch": 1}, p)
    with pytest.raises(C.CheckpointCorruptError, match="missing keys"):
        C.load_checkpoint(p)


def test_fallback_only_when_explicitly_allowed(tmp_path):
    cm = C.CheckpointManager(tmp_path, keep_every=1)
    full = {k: 0 for k in C.REQUIRED_KEYS}
    cm.save({**full, "epoch": 1}, 1, is_best=True)
    cm.save({**full, "epoch": 2}, 2, is_best=False)
    cm.last_path.write_bytes(b"truncated")
    with pytest.raises(C.CheckpointCorruptError, match="epoch_0002.pt"):
        cm.load_for_resume()
    s, path = cm.load_for_resume(allow_fallback=True)
    assert s["epoch"] == 2 and path.name == "epoch_0002.pt"


def test_atomic_save_leaves_previous_checkpoint_when_write_fails(tmp_path, monkeypatch):
    p = tmp_path / "ck" / "last.pt"
    C.atomic_torch_save({"epoch": 1}, p)
    good = p.read_bytes()

    def broken_save(obj, f):
        f.write(b"partial")
        raise OSError("drive disconnected")
    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        C.atomic_torch_save({"epoch": 2}, p)
    assert p.read_bytes() == good


# ------------------------------------------------- 10: path consistency
def test_checkpoint_paths_consistent_between_save_load_and_diagnostics(graphs, tmp_path):
    cfg = cfg_for(3, session=2)
    run = tmp_path / "runs" / cfg.run_id()                    # exactly how notebook 05 names it
    tr = Trainer(cfg, graphs, run, device="cpu")
    tr.run()
    assert tr.ckpt.last_path == run / "checkpoints" / "last.pt" and tr.ckpt.last_path.exists()
    assert not (run / "last.pt").exists()                     # the path the failed diagnostic looked at
    info = C.inspect_run_dir(run)
    assert info["checkpoints"]["last.pt"]["ok"] and info["checkpoints"]["last.pt"]["epoch"] == 2
    assert info["checkpoints"]["last.pt"]["run_id"] == cfg.run_id() == info["status"]["run_id"]
    assert info["checkpoints"]["last.pt"]["rng_types"]["torch"].startswith("torch.ByteTensor")
    assert info["history_epochs"] == [1, 2]
    res = Trainer(cfg_for(3), graphs, run, device="cpu")
    res.run(mode="resume")
    assert res.resume_info["checkpoint"] == str(tr.ckpt.last_path)


def test_checkpoint_from_another_run_is_rejected(graphs, tmp_path):
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=1, seed=42), graphs, run, device="cpu").run()
    with pytest.raises(C.CheckpointError, match="belongs to run"):
        Trainer(cfg_for(3, seed=43), graphs, run, device="cpu").run(mode="resume")


# ------------------------------------------------- CUDA (runs on Colab, skipped on CPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resume_on_cuda_new_trainer(graphs, tmp_path):
    """The exact notebook-05 sequence on a GPU: session of 2 epochs, then a new
    Trainer resumes epoch 3. GPU scatter is not bit-deterministic, so equality of
    parameters is not asserted here; RNG/optimizer restoration is."""
    run = tmp_path / "run"
    Trainer(cfg_for(3, session=2), graphs, run, device="cuda").run()
    saved = C.load_checkpoint(run / "checkpoints" / "last.pt")
    assert len(saved["rng"]["cuda"]) == torch.cuda.device_count()
    tr = Trainer(cfg_for(3), graphs, run, device="cuda")
    s, _ = tr.resolve_start("resume")
    tr._init_state()
    tr._restore(s)
    assert torch.equal(torch.cuda.get_rng_state(), saved["rng"]["cuda"][0])
    assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
    st = Trainer(cfg_for(3), graphs, run, device="cuda").run(mode="resume")
    assert st["state"] == "completed" and st["epoch"] == 3
