"""Our `assemble` must reproduce the official `build_verihgn_graph` exactly when
given the same inputs and the official cell->tile assignment.

The official function can only read layout.csv files carrying all five
code-feature columns; this test writes such a file with random synthetic values
(test-only data) so both implementations can be compared tensor by tensor.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from verihgn_repro import official
from verihgn_repro.data.graph import assemble


def _write_official_sample(d, M, N, n_cells=60, n_nets=40, seed=0):
    rng = np.random.RandomState(seed)
    cell = rng.rand(n_cells, 13).astype(np.float32)
    cell[:, 0] = rng.uniform(0, M * 3.0, n_cells)
    cell[:, 1] = rng.uniform(0, N * 3.0, n_cells)
    net = rng.rand(n_nets, 6).astype(np.float32)
    pins = np.stack([rng.randint(0, n_cells, 150), rng.randint(0, n_nets, 150)]).astype(np.int64)
    pin_f = rng.rand(150, 3).astype(np.float32)
    geom = np.stack([rng.randint(0, n_cells, 200), rng.randint(0, n_cells, 200)]).astype(np.int64)
    geom_f = rng.rand(200, 4).astype(np.float32)
    xs, ys = np.meshgrid(np.arange(M) * 3.0, np.arange(N) * 3.0, indexing="ij")
    layout = pd.DataFrame({
        "x": xs.reshape(-1), "y": ys.reshape(-1),
        "RUDY": rng.rand(M * N), "PinRUDY": rng.rand(M * N), "MacroRegion": rng.randint(0, 2, M * N),
        "H_MacroMargin": rng.rand(M * N), "V_MacroMargin": rng.rand(M * N),
        "H_Congestion": np.where(rng.rand(M * N) < 0.2, rng.rand(M * N) * 4, 0),
        "V_Congestion": np.where(rng.rand(M * N) < 0.2, rng.rand(M * N) * 4, 0),
    })
    for name, arr in [("cell_features", cell), ("net_features", net), ("pin_edge_index", pins),
                      ("pin_features", pin_f), ("geom_edge_index", geom), ("geom_edge_features", geom_f)]:
        np.save(d / f"{name}.npy", arr)
    layout.to_csv(d / "layout.csv", index=False)
    (d / "metadata.txt").write_text(f"grid_M={M}\ngrid_N={N}\n")
    return cell, net, pins, pin_f, geom, geom_f, layout


@pytest.mark.parametrize("M,N", [(8, 6), (7, 5)])   # even and odd (padding path)
def test_assemble_equals_official_builder(tmp_path, M, N):
    cell, net, pins, pin_f, geom, geom_f, layout = _write_official_sample(tmp_path, M, N)
    ref = official.graph_builder().build_verihgn_graph(str(tmp_path))

    # official cell->tile assignment (graph_builder.py:195-206), recomputed here
    cx, cy = torch.from_numpy(cell[:, 0]).float(), torch.from_numpy(cell[:, 1]).float()
    lx, ly = layout["x"].values, layout["y"].values
    xs = (lx.max() - lx.min()) / M
    ys = (ly.max() - ly.min()) / N
    idx = ((cx - lx.min()) / xs).long().clamp(0, M - 1) * N + ((cy - ly.min()) / ys).long().clamp(0, N - 1)

    feat = torch.zeros(M * N, 5)
    for i, c in enumerate(["RUDY", "PinRUDY", "MacroRegion", "H_MacroMargin", "V_MacroMargin"]):
        feat[:, i] = torch.from_numpy(layout[c].values.astype(np.float32))
    cong = np.log1p((layout["H_Congestion"].values.astype(np.float32) +
                     layout["V_Congestion"].values.astype(np.float32)) / 2.0)
    ours = assemble(torch.from_numpy(cell), torch.from_numpy(net), torch.from_numpy(pins), torch.from_numpy(pin_f),
                    torch.from_numpy(geom), torch.from_numpy(geom_f), feat, cong, idx, M, N, label_norm="per_sample")

    assert set(ref.node_types) == set(ours.node_types)
    assert set(ref.edge_types) == set(ours.edge_types)
    for nt in ref.node_types:
        for key in ("x", "y"):
            if key in ref[nt]:
                torch.testing.assert_close(ours[nt][key], ref[nt][key], rtol=0, atol=0)
        assert ours[nt].num_nodes == ref[nt].num_nodes
    for et in ref.edge_types:
        assert torch.equal(ours[et].edge_index, ref[et].edge_index), et
        if "edge_attr" in ref[et]:
            assert torch.equal(ours[et].edge_attr, ref[et].edge_attr), et
    assert (ours.grid_M, ours.grid_N, ours.grid_M1, ours.grid_N1) == (ref.grid_M, ref.grid_N, ref.grid_M1, ref.grid_N1)


def test_per_dataset_normalisation_divides_by_cmax():
    M, N = 4, 4
    cong = np.linspace(0, 2, M * N).astype(np.float32)
    args = (torch.rand(5, 13), torch.rand(3, 6), torch.tensor([[0, 1], [0, 1]]), torch.rand(2, 3),
            torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4), torch.rand(M * N, 5), cong,
            torch.zeros(5, dtype=torch.long), M, N)
    g = assemble(*args, label_norm="per_dataset", c_max=4.0)
    torch.testing.assert_close(g["grid_l0"].y, torch.from_numpy(cong / 4.0))
    with pytest.raises(ValueError):
        assemble(*args, label_norm="per_dataset", c_max=None)
