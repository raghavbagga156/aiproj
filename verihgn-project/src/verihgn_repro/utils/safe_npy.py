"""Load CircuitNet .npy files without allowing arbitrary pickle code execution.

CircuitNet stores graph information and instance placements as pickled object
arrays (dicts / string arrays). `np.load(allow_pickle=True)` would execute any
pickled callable; this loader only reconstructs numpy arrays and plain
containers, and refuses every other global.
"""
import pickle

import numpy as np
from numpy.lib import format as npf

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("builtins", "list"), ("builtins", "dict"), ("builtins", "tuple"), ("builtins", "set"),
    ("builtins", "str"), ("builtins", "int"), ("builtins", "float"),
    ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global {module}.{name}")


def load_npy(path):
    """Return the array stored at `path`. Object arrays of shape () are unwrapped
    to their single element (e.g. the placement dict)."""
    with open(path, "rb") as f:
        version = npf.read_magic(f)
        reader = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
        shape, _fortran, dtype = reader(f)
        if dtype.hasobject:
            obj = _RestrictedUnpickler(f).load()
        else:
            f.seek(0)
            obj = np.load(f, allow_pickle=False)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    return obj
