"""Integration on real CircuitNet-ISPD15 samples (skipped unless VERIHGN_FIXTURES
points at a raw tree in the RawLayout format, e.g. the 3-sample tree built in
Phase 1: fft_a / pci_bridge32_a / matrix_mult_a)."""
import numpy as np
import pytest

from verihgn_repro.data import preprocess as P
from verihgn_repro.data.circuitnet import RawLayout, parse_sample_name
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.netlist import load_netlist
from verihgn_repro.data.validate import validate_dataset

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def processed_real(real_fixtures, tmp_path_factory):
    out = tmp_path_factory.mktemp("real_processed")
    raw = RawLayout(real_fixtures)
    has_code = all(raw.routability(k, s).exists() for s in raw.samples() for k in ("rudy", "rudy_pin", "macro_map"))
    man = P.run(real_fixtures, out, cfg=P.prep_config(with_code_features=has_code), workers=1)
    assert not man["failed"], man["failed"]
    return out, has_code, man


def test_real_cell_density_matches_exactly(processed_real):
    _, _, man = processed_real
    assert man["ok"] >= 1
    assert man["cell_density_mismatch"] == []


def test_real_netlist_pin_semantics(real_fixtures):
    raw = RawLayout(real_fixtures)
    for s in raw.samples():
        nl = load_netlist(raw, parse_sample_name(s)["design"])
        std = ~nl.is_macro[nl.pin_cell]
        # every standard cell has exactly one output pin
        outs = np.bincount(nl.pin_cell[std], weights=nl.pin_is_output[std], minlength=nl.num_cells)
        assert (outs[~nl.is_macro] == 1).all()


@pytest.mark.parametrize("norm", ["per_sample", "per_dataset"])
def test_real_graphs_pass_all_gates(processed_real, norm):
    out, has_code, _ = processed_real
    designs = sorted({p.parent.name for p in P.list_processed(out)})
    split = {"name": "real_fixture", "designs": {"train": designs[:1], "val": designs[1:2], "test": designs[2:]}}
    for gf, dim in [("paper_eq3", 5)] + ([("official_code_available", 3)] if has_code else []):
        res = build_experiment_graphs(out, split, grid_features=gf, label_norm=norm, verbose=False)
        rep = validate_dataset(res["graphs"], grid_feat_dim=dim, model_check_samples=1)
        assert rep["passed"], rep["errors"]
