"""Access to the unmodified official VeriHGN code in third_party/VeriHGN.

The official repo uses generic top-level package names (`data`, `models`,
`utils`). They are imported once, under that repo's directory, and exposed here
so the rest of this project never touches sys.path itself.
"""
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

from .compat.torch_scatter_shim import install as _install_scatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "third_party" / "VeriHGN"
OFFICIAL_COMMIT = "28a6f1ebbd617c7aec0d0b39bf63f1c772193e40"


@lru_cache(maxsize=1)
def _load():
    if not (OFFICIAL_DIR / "models" / "verihgn.py").exists():
        raise FileNotFoundError(f"official VeriHGN code not found at {OFFICIAL_DIR}")
    scatter_impl = _install_scatter()
    path = str(OFFICIAL_DIR)
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    try:
        mods = {
            "model": importlib.import_module("models.verihgn"),
            "graph_builder": importlib.import_module("data.graph_builder"),
            "dataset": importlib.import_module("data.dataset"),
            "loss": importlib.import_module("utils.loss"),
            "evaluation": importlib.import_module("utils.evaluation"),
        }
    finally:
        if added:
            sys.path.remove(path)
    mods["scatter_impl"] = scatter_impl
    return mods


def model_module():
    return _load()["model"]


def graph_builder():
    return _load()["graph_builder"]


def dataset_module():
    return _load()["dataset"]


def loss_module():
    return _load()["loss"]


def evaluation_module():
    return _load()["evaluation"]


def scatter_impl() -> str:
    return _load()["scatter_impl"]


def VeriHGN(**kwargs):
    """Construct the official model."""
    return model_module().VeriHGN(**kwargs)


def official_commit() -> str:
    stamp = OFFICIAL_DIR / "OFFICIAL_COMMIT"
    if stamp.exists():
        return stamp.read_text().strip()
    head = OFFICIAL_DIR / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            ref_file = OFFICIAL_DIR / ".git" / ref.split(" ", 1)[1]
            if ref_file.exists():
                return ref_file.read_text().strip()
            packed = OFFICIAL_DIR / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(ref.split(" ", 1)[1]):
                        return line.split()[0]
        return ref
    return os.environ.get("VERIHGN_OFFICIAL_COMMIT", OFFICIAL_COMMIT)
