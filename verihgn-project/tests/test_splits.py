import numpy as np

from verihgn_repro import official
from verihgn_repro.data.circuitnet import SAMPLES_PER_DESIGN, SCOPES
from verihgn_repro.data.splits import (build_split_manifest, leakage_report, official_split,
                                       pair_aware_split, twin_groups)

# Twin structure measured in Phase 0 (identical cell/net/pin/master counts) and
# confirmed by Netlist.signature() on the real graph_information files.
SIGS = {d: d for d in SCOPES["official9"]}
SIGS["mgc_fft_b"] = SIGS["mgc_fft_a"]
SIGS["mgc_matrix_mult_c"] = SIGS["mgc_matrix_mult_b"]


def test_official_split_matches_official_method():
    ds_cls = official.dataset_module().VeriHGNDataset
    obj = ds_cls.__new__(ds_cls)
    obj.designs = SCOPES["official9"]
    assert official_split(SCOPES["official9"]) == obj._split_designs()


def test_official_split_membership_and_leakage():
    sp = official_split(SCOPES["official9"])
    assert sp["test"] == ["mgc_fft_b", "mgc_matrix_mult_c"]
    assert sp["val"] == ["mgc_matrix_mult_a"]
    m = build_split_manifest("official_seed42", sp, SIGS, SAMPLES_PER_DESIGN)
    assert m["num_samples"] == {"train": 251, "val": 45, "test": 87}
    assert not m["leakage"]["leak_free"]
    assert sorted(m["leakage"]["netlist_twins_across_splits"]) == [
        ["mgc_fft_a", "mgc_fft_b"], ["mgc_matrix_mult_b", "mgc_matrix_mult_c"]]


def test_pair_aware_split_is_leak_free_and_complete():
    sp = pair_aware_split(SIGS)
    rep = leakage_report(sp, SIGS)
    assert rep["leak_free"]
    assert sorted(sum(sp.values(), [])) == SCOPES["official9"]
    assert len(sp["test"]) == 2 and len(sp["val"]) == 1
    for g in twin_groups(SIGS):
        assert len({p for p, ds in sp.items() for d in ds if d in g}) == 1


def test_pair_aware_split_deterministic():
    assert pair_aware_split(SIGS) == pair_aware_split(SIGS)
    np.random.seed(123)   # global RNG state must not matter
    assert pair_aware_split(SIGS) == pair_aware_split(dict(reversed(list(SIGS.items()))))
