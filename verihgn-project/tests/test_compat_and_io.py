import io
import os
import pickle

import numpy as np
import pytest
import torch

from verihgn_repro.compat import torch_scatter_shim as shim
from verihgn_repro.utils.safe_npy import load_npy


def _reference(src, index, dim_size, reduce):
    out = torch.zeros(dim_size, src.shape[1])
    cnt = torch.zeros(dim_size)
    for i, j in enumerate(index.tolist()):
        out[j] += src[i]
        cnt[j] += 1
    if reduce == "mean":
        out = out / cnt.clamp(min=1).unsqueeze(1)
    return out


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_shim_matches_reference_including_empty_segments(reduce):
    g = torch.Generator().manual_seed(0)
    src = torch.randn(50, 4, generator=g)
    index = torch.randint(0, 7, (50,), generator=g)
    index[index == 3] = 4             # segment 3 empty
    fn = shim.scatter_add if reduce == "sum" else shim.scatter_mean
    got = fn(src, index, dim=0, dim_size=9)   # segments 7, 8 empty too
    assert got.shape == (9, 4)
    torch.testing.assert_close(got, _reference(src, index, 9, reduce))


def test_shim_install_is_idempotent():
    impl = shim.install()
    assert impl == shim.install()
    import torch_scatter
    assert hasattr(torch_scatter, "scatter_mean") and hasattr(torch_scatter, "scatter_add")


def test_safe_npy_loads_object_dict_and_numeric(tmp_path):
    p = tmp_path / "d.npy"
    arr = np.empty((), dtype=object)
    arr[()] = {"a": [1, 2, 3, 4]}
    np.save(p, arr, allow_pickle=True)
    assert load_npy(p) == {"a": [1, 2, 3, 4]}
    q = tmp_path / "n.npy"
    np.save(q, np.arange(6.0).reshape(2, 3))
    np.testing.assert_array_equal(load_npy(q), np.arange(6.0).reshape(2, 3))


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_npy_blocks_code_execution(tmp_path):
    p = tmp_path / "evil.npy"
    arr = np.empty((), dtype=object)
    arr[()] = _Evil()
    np.save(p, arr, allow_pickle=True)
    with pytest.raises(pickle.UnpicklingError):
        load_npy(p)
