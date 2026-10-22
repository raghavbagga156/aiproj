"""Shared fixtures.

`synthetic_raw` writes a tiny CircuitNet-shaped raw tree (graph_information,
gcell/micron placements, GR overflow maps, cell_density, code maps) in the same
file formats as the real dataset. It is test data only and is never used for
reported results. Real-data tests use the files under $VERIHGN_FIXTURES.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verihgn_repro.data.circuitnet import ROUTABILITY_MEMBERS  # noqa: E402

M, N = 12, 10
PITCH = 2.0   # micrometres per gcell in the synthetic design


def _save_obj(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty((), dtype=object)
    arr[()] = obj
    with open(path, "wb") as f:
        np.save(f, arr, allow_pickle=True)


def _save_obj_array(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.save(f, np.asarray(arr, dtype=object), allow_pickle=True)


def make_design(root: Path, design: str, n_cells: int, n_nets: int, samples: int, seed: int, M=M, N=N):
    rng = np.random.RandomState(seed)
    masters = rng.choice(["in01f01", "na02f01", "ms00f80"], size=n_cells)
    names = [f"c{i}" for i in range(n_cells)]
    masters[0] = "h0"           # one macro, named after its own master as in mgc designs
    names[0] = "h0"
    pins_per_master = {"in01f01": ["a", "o"], "na02f01": ["a", "b", "o"], "ms00f80": ["ck", "d", "o"]}
    pin_name, pin_net, pin_node = [], [], []
    for c in range(n_cells):
        if masters[c] == "h0":
            nets = rng.choice(n_nets, size=3, replace=False)
            for n in nets:
                pin_name.append(f"net{n}")
                pin_net.append(int(n))
                pin_node.append(c + 1)
            continue
        for p in pins_per_master[masters[c]]:
            n = int(rng.randint(n_nets))
            pin_name.append(p)
            pin_net.append(n)
            pin_node.append(c + 1)
    used = set(pin_net)
    for n in range(n_nets):            # every net gets at least one pin
        if n not in used:
            pin_name.append("a"); pin_net.append(n); pin_node.append(1 + (n % (n_cells - 1)) + 0)
    g = root / "graph_infromation"
    _save_obj_array(g / "node_attr" / f"{design}_node_attr.npy", [["top"] + names, ["module"] + list(masters)])
    _save_obj_array(g / "net_attr" / f"{design}_net_attr.npy", [[f"net{i}" for i in range(n_nets)]])
    _save_obj_array(g / "pin_attr" / f"{design}_pin_attr.npy", [pin_name, pin_net, pin_node])

    sample_names = []
    for s in range(samples):
        sample = f"{seed * 100 + s}-{design}-util_0.{7 + s}"
        sample_names.append(sample)
        gcell, micron = {}, {}
        for c, nm in enumerate(names):
            if c == 0:
                x0, y0, x1, y1 = 0, 0, 3, 2
            else:
                x0 = int(rng.randint(0, M - 1)); y0 = int(rng.randint(0, N - 1))
                x1 = x0 + int(rng.randint(0, 2)); y1 = y0 + int(rng.randint(0, 2))
                x1 = min(x1, M - 1); y1 = min(y1, N - 1)
            gcell[nm] = [x0, y0, x1, y1]
            micron[nm] = [x0 * PITCH + 0.1, y0 * PITCH + 0.1, max(x1, x0 + 1) * PITCH - 0.1,
                          max(y1, y0 + 1) * PITCH - 0.1]
        _save_obj(root / "instance_placement_gcell" / sample, gcell)
        _save_obj(root / "instance_placement_micron" / f"{sample}.npy", micron)
        h = np.where(rng.rand(M, N) < 0.1, rng.rand(M, N) * 3, 0.0)
        v = np.where(rng.rand(M, N) < 0.1, rng.rand(M, N) * 2, 0.0)
        cd = np.zeros((M, N))
        for x0, y0, x1, y1 in gcell.values():
            cd[x0:max(x1, x0 + 1), y0:max(y1, y0 + 1)] += 1
        maps = {"gr_h_overflow": h, "gr_v_overflow": v, "cell_density": cd,
                "rudy": rng.rand(M, N), "rudy_pin": rng.rand(M, N), "macro_map": (cd > 3).astype(float)}
        for key, arr in maps.items():
            p = root / ROUTABILITY_MEMBERS[key] / sample
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                np.save(f, arr)
    return sample_names


@pytest.fixture
def synthetic_raw(tmp_path):
    root = tmp_path / "raw"
    samples = []
    samples += make_design(root, "mgc_fft_a", n_cells=40, n_nets=30, samples=2, seed=1)
    samples += make_design(root, "mgc_pci_bridge32_a", n_cells=30, n_nets=25, samples=1, seed=2)
    return root, samples


@pytest.fixture(scope="session")
def real_fixtures():
    p = os.environ.get("VERIHGN_FIXTURES")
    if not p or not Path(p).exists():
        pytest.skip("set VERIHGN_FIXTURES to a raw CircuitNet sample tree")
    return Path(p)
