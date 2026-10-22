import numpy as np
import pytest

from verihgn_repro.data import features as F
from verihgn_repro.data.circuitnet import RawLayout
from verihgn_repro.data.netlist import Netlist, load_netlist
from verihgn_repro.utils.safe_npy import load_npy

from conftest import M, N


def _tiny_netlist():
    # cells: 0 macro, 1..3 std; nets: 0={1,2}, 1={1,3}, 2={0,3}
    return Netlist(
        design="mgc_x", cell_names=np.array(["h0", "c1", "c2", "c3"]),
        masters=np.array(["h0", "in01f01", "in01f01", "na02f01"]),
        is_macro=np.array([True, False, False, False]), num_nets=3,
        pin_name=np.array(["o", "a", "o", "a", "b", "net2"]),
        pin_net=np.array([0, 1, 0, 1, 2, 2]),
        pin_cell=np.array([1, 1, 2, 3, 3, 0]),
        pin_is_output=np.array([1, 0, 1, 0, 0, 0], dtype=np.float32),
        macro_pin_direction="input")


def _tiny_placement():
    micron = {"h0": [0, 0, 4, 4], "c1": [10, 0, 12, 2], "c2": [20, 0, 22, 2], "c3": [10, 10, 12, 12]}
    gcell = {"h0": [0, 0, 2, 2], "c1": [5, 0, 5, 0], "c2": [10, 0, 10, 0], "c3": [5, 5, 5, 5]}
    return micron, gcell


def test_hand_computed_cell_net_pin_features():
    nl = _tiny_netlist()
    micron, gcell = _tiny_placement()
    f = F.compute_sample_features(nl, micron, gcell, M=12, N=8, knn_k=2)
    cx = {0: 2, 1: 11, 2: 21, 3: 11}
    cy = {0: 2, 1: 1, 2: 1, 3: 11}
    # net 0 = cells 1,2: w=10, h=0 ; net 1 = cells 1,3: w=0,h=10 ; net 2 = cells 3,0: w=9,h=9
    np.testing.assert_allclose(f["net_x"][:, 0], [10, 10, 18])                  # hpwl
    np.testing.assert_allclose(f["net_x"][:, 3], [2, 2, 2])                     # degree
    np.testing.assert_allclose(f["net_x"][:, 5], [0, 0, 81])                    # bbox area
    cell = f["cell_x"]
    np.testing.assert_allclose(cell[:, 0], [cx[i] for i in range(4)])
    np.testing.assert_allclose(cell[:, 1], [cy[i] for i in range(4)])
    np.testing.assert_allclose(cell[:, 4], [16, 4, 4, 4])                       # area
    np.testing.assert_allclose(cell[:, 5], [1, 2, 1, 2])                        # incident nets
    np.testing.assert_allclose(cell[:, 6], [1, 2, 1, 2])                        # pins
    np.testing.assert_allclose(cell[:, 7], [18, 10, 10, 14])                    # mean incident hpwl
    np.testing.assert_allclose(cell[:, 8], [18, 10, 10, 18])                    # max incident hpwl
    np.testing.assert_allclose(cell[:, 10], [1, 0, 0, 0])                       # macro
    np.testing.assert_allclose(cell[:, 11], [1, 0.5, 0, 1])                     # input-pin ratio
    np.testing.assert_allclose(cell[:, 12], [0.25, 0.5, 0.5, 0.25])             # type frequency
    # pin offsets: cell 1 on net 0 (centroid x=16, w=10) -> dx=-0.5 ; on net 1 (w=0) -> 0
    np.testing.assert_allclose(f["pin_x"][0], [1, -0.5, 0])
    np.testing.assert_allclose(f["pin_x"][1], [0, 0, -0.5])
    # h0 covers tiles x,y in [0,2) -> centre tile (0,0); 1-tile cells map to their own tile
    assert f["cell_tile"].tolist() == [0, 5 * 8 + 0, 10 * 8 + 0, 5 * 8 + 5]


def test_coverage_sum_matches_brute_force():
    rng = np.random.RandomState(0)
    x0 = rng.randint(0, 9, 40); y0 = rng.randint(0, 7, 40)
    x1 = x0 + rng.randint(1, 3, 40); y1 = y0 + rng.randint(1, 3, 40)
    x1 = np.minimum(x1, 10); y1 = np.minimum(y1, 8)
    w = rng.rand(40)
    brute = np.zeros((10, 8))
    for a, b, c, d, ww in zip(x0, x1, y0, y1, w):
        brute[a:b, c:d] += ww
    np.testing.assert_allclose(F.coverage_sum(x0, x1, y0, y1, w, 10, 8), brute, atol=1e-12)


def test_knn_edges_symmetric_no_self_loops():
    rng = np.random.RandomState(1)
    ei = F.knn_edges(rng.rand(200), rng.rand(200), k=8)
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert all((b, a) in pairs for a, b in pairs)
    assert all(a != b for a, b in pairs)
    assert np.bincount(ei[1], minlength=200).min() >= 8   # every node keeps its 8 neighbours


def test_synthetic_design_cell_density_matches_circuitnet_convention(synthetic_raw):
    root, samples = synthetic_raw
    raw = RawLayout(root)
    nl = load_netlist(raw, "mgc_fft_a")
    s = samples[0]
    f = F.compute_sample_features(nl, load_npy(raw.placement_micron(s)), load_npy(raw.placement_gcell(s)), M, N)
    np.testing.assert_array_equal(f["cell_density_recomputed"], load_npy(raw.routability("cell_density", s)))
    assert f["grid_x_paper"].shape == (M * N, 5)
    np.testing.assert_array_equal(f["grid_x_paper"][:, 0], f["cell_density_recomputed"].reshape(-1))
    assert f["grid_x_paper"][:, 1].sum() == nl.num_pins          # every pin counted once
    assert np.isfinite(f["cell_x"]).all() and np.isfinite(f["grid_x_paper"]).all()


def test_netlist_direction_rules(synthetic_raw):
    root, _ = synthetic_raw
    nl = load_netlist(RawLayout(root), "mgc_fft_a")
    std = ~nl.is_macro[nl.pin_cell]
    assert np.array_equal(nl.pin_is_output[std] == 1, nl.pin_name[std] == "o")
    assert (nl.pin_is_output[~std] == 0).all()                    # macro pins default to input
    nl_out = load_netlist(RawLayout(root), "mgc_fft_a", macro_pin_direction="output")
    assert (nl_out.pin_is_output[~std] == 1).all()


def test_superblue_macro_detection_refused():
    from verihgn_repro.data.netlist import macro_flags
    with pytest.raises(NotImplementedError):
        macro_flags(np.array(["a"]), np.array(["b"]), "mgc_superblue12")
