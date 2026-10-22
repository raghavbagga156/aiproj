"""Assemble the VeriHGN HeteroData graph from a preprocessed sample.

This mirrors `build_verihgn_graph` (official data/graph_builder.py:154-297)
statement by statement and calls the official helpers for every structural part
(`standardize`, `build_grid_adjacency`, `build_hierarchical_edges`,
`build_grid_net_edges`). It differs from the official function only where the
official one cannot be fed honestly:
  * grid feature columns are configurable (paper Eq. 3, or the 3 available
    official-code columns) instead of the hard-coded RUDY..V_MacroMargin;
  * the cell -> L0 tile assignment comes from CircuitNet gcell placement
    (verified exact against cell_density) instead of re-deriving it from
    layout.csv x/y extents;
  * the label normaliser is selectable: per-sample max (official code) or a
    per-dataset c_max (paper).
tests/test_graph_equivalence.py checks that, given the same inputs and the
official tile assignment, the result is identical to the official builder.
"""
import time

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .. import official

LABEL_NORMS = ("per_sample", "per_dataset")
GRID_FEATURE_SETS = {"paper_eq3": "grid_x_paper", "official_code_available": "grid_x_code"}


def grid_label(label_h: np.ndarray, label_v: np.ndarray) -> np.ndarray:
    """Official aggregation (graph_builder.py:233-236): mean of H and V, then log1p."""
    return np.log1p((label_h.astype(np.float32) + label_v.astype(np.float32)) / 2.0).reshape(-1)


def assemble(cell_feat, net_feat, pin_edge_index, pin_feat, geom_edge_index, geom_edge_feat,
             layout_feat, grid_congestion_log, cell_grid_idx, M, N,
             label_norm="per_sample", c_max=None) -> HeteroData:
    """Core of the official builder with explicit inputs (all torch tensors / ints)."""
    gb = official.graph_builder()
    t0 = time.time()
    num_cells = cell_feat.shape[0]
    num_nets = net_feat.shape[0]
    F = layout_feat.shape[1]

    cell_x = gb.standardize(cell_feat.float())
    net_x = gb.standardize(net_feat.float())

    grid_l0_feat = layout_feat
    grid_l0_x = gb.standardize(grid_l0_feat)

    M1 = (M + 1) // 2
    N1 = (N + 1) // 2
    grid_l0_raw_2d = grid_l0_feat.reshape(M, N, F)
    padM = M1 * 2 - M
    padN = N1 * 2 - N
    if padM > 0 or padN > 0:
        grid_l0_padded = torch.nn.functional.pad(grid_l0_raw_2d, (0, 0, 0, padN, 0, padM))
    else:
        grid_l0_padded = grid_l0_raw_2d
    grid_l1_feat = grid_l0_padded.reshape(M1, 2, N1, 2, F).mean(dim=(1, 3))
    grid_l1_x = gb.standardize(grid_l1_feat.reshape(M1 * N1, F))

    grid_congestion = np.asarray(grid_congestion_log, dtype=np.float32).copy()
    if label_norm == "per_sample":
        local_max = grid_congestion.max()
        if local_max > 0:
            grid_congestion = grid_congestion / local_max
    elif label_norm == "per_dataset":
        if not c_max or c_max <= 0:
            raise ValueError("per_dataset normalisation needs a positive c_max")
        grid_congestion = grid_congestion / np.float32(c_max)
    else:
        raise ValueError(f"label_norm must be one of {LABEL_NORMS}")
    grid_y = torch.from_numpy(grid_congestion)
    cell_y = grid_y[cell_grid_idx]

    data = HeteroData()
    data['cell'].x = cell_x
    data['cell'].y = cell_y
    data['cell'].num_nodes = num_cells
    data['net'].x = net_x
    data['net'].num_nodes = num_nets
    data['grid_l0'].x = grid_l0_x
    data['grid_l0'].y = grid_y
    data['grid_l0'].num_nodes = M * N
    data['grid_l1'].x = grid_l1_x
    data['grid_l1'].num_nodes = M1 * N1

    data['cell', 'pin', 'net'].edge_index = pin_edge_index
    data['cell', 'pin', 'net'].edge_attr = pin_feat.float()
    data['net', 'pin_rev', 'cell'].edge_index = pin_edge_index.flip(0)
    data['net', 'pin_rev', 'cell'].edge_attr = pin_feat.float()
    data['cell', 'geom', 'cell'].edge_index = geom_edge_index
    data['cell', 'geom', 'cell'].edge_attr = geom_edge_feat.float()
    data['grid_l0', 'adj', 'grid_l0'].edge_index = gb.build_grid_adjacency(M, N)
    data['grid_l1', 'adj', 'grid_l1'].edge_index = gb.build_grid_adjacency(M1, N1)
    l0_idx, l1_idx = gb.build_hierarchical_edges(M, N, M1, N1)
    data['grid_l0', 'child_of', 'grid_l1'].edge_index = torch.stack([l0_idx, l1_idx])
    data['grid_l1', 'parent_of', 'grid_l0'].edge_index = torch.stack([l1_idx, l0_idx])
    cell_indices = torch.arange(num_cells, dtype=torch.long)
    data['cell', 'in', 'grid_l0'].edge_index = torch.stack([cell_indices, cell_grid_idx])
    data['grid_l0', 'contains', 'cell'].edge_index = torch.stack([cell_grid_idx, cell_indices])
    net_src, grid_dst = gb.build_grid_net_edges(cell_grid_idx, pin_edge_index, M, N, num_nets,
                                                max_bbox_tiles=100, max_total_edges=5_000_000)
    data['net', 'spans', 'grid_l0'].edge_index = torch.stack([net_src, grid_dst])
    data['grid_l0', 'spans_rev', 'net'].edge_index = torch.stack([grid_dst, net_src])

    data.grid_M = M
    data.grid_N = N
    data.grid_M1 = M1
    data.grid_N1 = N1
    data.build_time = time.time() - t0
    return data


def build_graph(arrays: dict, meta: dict, grid_features="paper_eq3", label_norm="per_sample",
                c_max=None, keep_raw_labels=True) -> HeteroData:
    key = GRID_FEATURE_SETS[grid_features]
    if key not in arrays:
        raise KeyError(f"sample {meta['sample']} has no {key} (grid feature set '{grid_features}')")
    M, N = int(meta["grid_M"]), int(meta["grid_N"])
    data = assemble(
        cell_feat=torch.from_numpy(arrays["cell_x"]),
        net_feat=torch.from_numpy(arrays["net_x"]),
        pin_edge_index=torch.from_numpy(arrays["pin_edge_index"]),
        pin_feat=torch.from_numpy(arrays["pin_x"]),
        geom_edge_index=torch.from_numpy(arrays["geom_edge_index"]),
        geom_edge_feat=torch.from_numpy(arrays["geom_x"]),
        layout_feat=torch.from_numpy(arrays[key]).float(),
        grid_congestion_log=grid_label(arrays["label_h"], arrays["label_v"]),
        cell_grid_idx=torch.from_numpy(arrays["cell_tile"]),
        M=M, N=N, label_norm=label_norm, c_max=c_max,
    )
    data.design_name = meta["design"]
    data.placement_name = meta["sample"]
    data.label_norm = label_norm
    data.grid_feature_set = grid_features
    if keep_raw_labels:
        # raw overflow maps kept for visualisation / H-V analysis; not used by the model
        data['grid_l0'].raw_hv = torch.from_numpy(
            np.stack([arrays["label_h"].reshape(-1), arrays["label_v"].reshape(-1)], 1).astype(np.float32))
    return data


def compute_c_max(sample_dirs) -> float:
    """Paper normaliser: max of log(1 + (H+V)/2) over the given samples (pass the
    training split only, so no test information enters preprocessing)."""
    c_max = 0.0
    for d in sample_dirs:
        with np.load(d / "sample.npz", allow_pickle=False) as z:
            c_max = max(c_max, float(grid_label(z["label_h"], z["label_v"]).max()))
    return c_max
