"""Atomic, resumable checkpoints and run status on persistent storage (Drive).

Layout of a run directory (the only place checkpoints are written):

    <run_dir>/checkpoints/last.pt        state after the most recent epoch
    <run_dir>/checkpoints/best.pt        state at the best validation epoch
    <run_dir>/checkpoints/epoch_XXXX.pt  periodic copies (every `keep_every` epochs)
    <run_dir>/status.json, history.csv, config.yaml, env.json

A checkpoint is written to `<name>.tmp`, flushed + fsynced, verified (zip CRCs),
then renamed over the target, so an interrupted write never replaces a good
checkpoint. Checkpoints are always deserialised onto the CPU: generator states
must be CPU ByteTensors, and model/optimizer states are moved to the training
device by load_state_dict.
"""
import json
import os
import time
import zipfile
from pathlib import Path

import torch

STATES = ("running", "completed", "early_stopped", "failed")
REQUIRED_KEYS = ("epoch", "model", "optimizer", "scheduler", "scaler", "best_avg_spearman",
                 "patience_counter", "rng", "shuffle_gen", "run_id")


class CheckpointError(RuntimeError):
    pass


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    pass


class CheckpointCorruptError(CheckpointError):
    pass


def _fsync_dir(path: Path) -> None:
    try:                                    # not supported on Windows / some FUSE mounts
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def verify_checkpoint_file(path) -> int:
    """Cheap integrity check without unpickling: a torch checkpoint is a zip
    archive; every member's CRC must match. Returns the file size."""
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise CheckpointCorruptError(f"{path} is empty")
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
    except (zipfile.BadZipFile, OSError) as e:
        raise CheckpointCorruptError(f"{path} is not a readable torch checkpoint: {e}") from e
    if bad is not None:
        raise CheckpointCorruptError(f"{path}: CRC mismatch in member {bad}")
    return size


def atomic_torch_save(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    size = verify_checkpoint_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    if path.stat().st_size != size:
        raise CheckpointError(f"{path}: size after rename {path.stat().st_size} != written {size}")


def atomic_write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, required=REQUIRED_KEYS) -> dict:
    """Load one checkpoint onto the CPU. Missing -> CheckpointNotFoundError,
    unreadable or incomplete -> CheckpointCorruptError (both name the path)."""
    path = Path(path)
    if not path.exists():
        raise CheckpointNotFoundError(f"checkpoint not found: {path}")
    verify_checkpoint_file(path)
    try:
        s = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - re-raised with the path
        raise CheckpointCorruptError(f"could not deserialise {path}: {type(e).__name__}: {e}") from e
    if not isinstance(s, dict):
        raise CheckpointCorruptError(f"{path} does not contain a state dict (got {type(s).__name__})")
    missing = [k for k in required if k not in s]
    if missing:
        raise CheckpointCorruptError(f"{path} is missing keys {missing}")
    return s


class CheckpointManager:
    def __init__(self, run_dir, keep_every: int = 10, keep_last_n: int = 3):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.keep_every = keep_every
        self.keep_last_n = keep_last_n

    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    def periodic(self) -> list:
        return sorted(self.ckpt_dir.glob("epoch_*.pt"))

    def has_any(self) -> bool:
        return self.last_path.exists() or self.best_path.exists() or bool(self.periodic())

    def save(self, state: dict, epoch: int, is_best: bool) -> None:
        atomic_torch_save(state, self.last_path)
        if is_best:
            atomic_torch_save(state, self.best_path)
        if self.keep_every and epoch % self.keep_every == 0:
            atomic_torch_save(state, self.ckpt_dir / f"epoch_{epoch:04d}.pt")
            for old in self.periodic()[:-self.keep_last_n]:
                old.unlink(missing_ok=True)

    def load_latest(self, map_location="cpu"):
        """Lenient: newest loadable of last.pt, epoch_*.pt (newest first); (None, None)
        if none loads. Used for inspection. Resume uses load_for_resume()."""
        candidates = [self.last_path] + self.periodic()[::-1]
        for p in candidates:
            if not p.exists():
                continue
            try:
                return torch.load(p, map_location=map_location, weights_only=False), p
            except Exception as e:  # noqa: BLE001 - corrupted file: report and fall back
                print(f"[checkpoint] could not load {p}: {type(e).__name__}: {e}; trying older")
        return None, None

    def load_for_resume(self, allow_fallback: bool = False):
        """Strict: returns (state, path) of last.pt. If last.pt is missing or corrupt
        this raises, naming the files that do exist; an older periodic checkpoint is
        used only when `allow_fallback=True` (it loses the epochs after it)."""
        existing = sorted(p.name for p in self.ckpt_dir.glob("*")) if self.ckpt_dir.exists() else []
        try:
            return load_checkpoint(self.last_path), self.last_path
        except CheckpointNotFoundError:
            if not allow_fallback or not self.periodic():
                raise CheckpointNotFoundError(
                    f"no last.pt in {self.ckpt_dir} (files there: {existing or 'none'})") from None
            err = None
        except CheckpointCorruptError as e:
            if not allow_fallback:
                raise CheckpointCorruptError(
                    f"{e}. Older checkpoints present: {[p.name for p in self.periodic()] or 'none'}; "
                    "pass allow_fallback=True to resume from the newest of them") from e
            err = e
        for p in self.periodic()[::-1]:
            try:
                s = load_checkpoint(p)
                print(f"[checkpoint] last.pt unusable ({err}); falling back to {p.name} as requested")
                return s, p
            except CheckpointCorruptError as e:
                print(f"[checkpoint] {e}")
        raise CheckpointCorruptError(f"no loadable checkpoint in {self.ckpt_dir} (files: {existing})")


def describe_state(s: dict) -> dict:
    """Types of the restorable parts of a loaded checkpoint (for diagnostics)."""
    rng = s.get("rng", {})

    def t(x):
        return f"{x.type()} {tuple(x.shape)} on {x.device}" if torch.is_tensor(x) else type(x).__name__
    return {"epoch": s.get("epoch"), "run_id": s.get("run_id"), "best_avg_spearman": s.get("best_avg_spearman"),
            "patience_counter": s.get("patience_counter"),
            "optimizer_steps": sorted({int(v["step"]) for v in s["optimizer"]["state"].values()})
            if s.get("optimizer", {}).get("state") else [],
            "scheduler": {k: v for k, v in s.get("scheduler", {}).items() if k in ("last_epoch", "_last_lr")},
            "scaler_scale": s.get("scaler", {}).get("scale"),
            "rng_types": {"python": type(rng.get("python")).__name__, "numpy": type(rng.get("numpy")).__name__,
                          "torch": t(rng.get("torch")), "cuda": [t(x) for x in rng.get("cuda", [])]},
            "shuffle_gen": t(s.get("shuffle_gen"))}


def inspect_run_dir(run_dir) -> dict:
    """What is actually in a run directory: every file with its size, the status,
    the epochs in history.csv, and for each checkpoint whether it loads and what it
    holds. Reads only; never modifies anything."""
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir), "exists": run_dir.exists(), "files": {}, "checkpoints": {}}
    if not run_dir.exists():
        return out
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            out["files"][f.relative_to(run_dir).as_posix()] = f.stat().st_size
    st = run_dir / "status.json"
    out["status"] = json.loads(st.read_text()) if st.exists() else None
    hist = run_dir / "history.csv"
    if hist.exists():
        import csv
        with open(hist, newline="") as f:
            out["history_epochs"] = [int(r["epoch"]) for r in csv.DictReader(f)]
    for p in sorted((run_dir / "checkpoints").glob("*.pt")):
        try:
            out["checkpoints"][p.name] = {"ok": True, **describe_state(load_checkpoint(p))}
        except CheckpointError as e:
            out["checkpoints"][p.name] = {"ok": False, "error": str(e)}
    return out


class RunStatus:
    """status.json with a heartbeat. A 'running' run whose heartbeat is older than
    `stale_after_s` is reported as interrupted (resumable), never as training."""

    def __init__(self, run_dir, stale_after_s: int = 900):
        self.path = Path(run_dir) / "status.json"
        self.stale_after_s = stale_after_s

    def write(self, state: str, **fields) -> None:
        assert state in STATES, state
        rec = self.read() or {}
        rec.update(fields)
        rec.update({"state": state, "updated_at": time.time()})
        atomic_write_json(rec, self.path)

    def read(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def describe(self) -> str:
        rec = self.read()
        if rec is None:
            return "not started"
        if rec["state"] == "running" and time.time() - rec["updated_at"] > self.stale_after_s:
            return "interrupted (resumable)"
        return rec["state"]
