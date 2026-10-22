"""Minimal stand-in for the two `torch_scatter` functions the official VeriHGN
model imports (`scatter_mean`, `scatter_add`).

Used only when no `torch_scatter` wheel exists for the installed torch build
(e.g. torch 2.14 on Windows/Colab). Both functions delegate to
`torch_geometric.utils.scatter`, which is pure PyTorch, and keep
torch_scatter's semantics: empty segments produce 0 and `dim_size` fixes the
output length. Equivalence is checked in tests/test_compat.py.
"""
import sys
import types

import torch
from torch_geometric.utils import scatter as _pyg_scatter


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                out=None, dim_size=None) -> torch.Tensor:
    if out is not None:
        raise NotImplementedError("shim does not support out=")
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 out=None, dim_size=None) -> torch.Tensor:
    if out is not None:
        raise NotImplementedError("shim does not support out=")
    return _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")


def install() -> str:
    """Make `import torch_scatter` work. Returns which implementation is active."""
    try:
        import torch_scatter  # noqa: F401
        if getattr(torch_scatter, "__verihgn_shim__", False):
            return "shim"
        return "torch_scatter " + getattr(torch_scatter, "__version__", "?")
    except ImportError:
        mod = types.ModuleType("torch_scatter")
        mod.scatter_add = scatter_add
        mod.scatter_mean = scatter_mean
        mod.__verihgn_shim__ = True
        mod.__version__ = "shim"
        sys.modules["torch_scatter"] = mod
        return "shim"
