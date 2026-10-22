"""Feature extraction: CircuitNet raw arrays -> VeriHGN node/edge features.

Traceability tags follow docs/RESEARCH_TO_CODE_MAPPING.md:
  P = paper, C = official code, D = dataset documentation/inspection,
  A = engineering adaptation (author preprocessing unavailable), U = unresolved assumption.

All coordinates are CircuitNet's. Gcell boxes use the convention verified to
reproduce CircuitNet `cell_density` exactly: an instance [x0, y0, x1, y1] covers
tiles [x0, max(x1, x0+1)) x [y0, max(y1, y0+1)); arrays are indexed [x, y].
"""
import numpy as np
from scipy.spatial import cKDTree

# Column order of each feature matrix (written into every sample's metadata).
CELL_FEATURES = [            # paper Eq. 1 order [P]; cols 0-1 are raw x, y as the official builder expects [C]
    "x_um", "y_um", "w_um", "h_um", "area_um2",
    "num_incident_nets", "num_pins", "mean_incident_hpwl_um", "max_incident_hpwl_um", "mean_incident_net_degree",
    "is_macro", "input_pin_ratio", "type_frequency",
]
NET_FEATURES = ["hpwl_um", "bbox_w_um", "bbox_h_um", "degree", "log_degree", "bbox_area_um2"]   # Eq. 2 [P]
PIN_FEATURES = ["is_output", "dx_norm", "dy_norm"]                                               # Eq. 4 [P]
GEOM_FEATURES = ["dx_um", "dy_um", "manhattan_um", "euclidean_um"]                               # Eq. 5 [P]
GRID_FEATURES_PAPER = ["cell_density", "pin_density", "net_density", "mean_cell_area_um2", "mean_net_hpwl_um"]  # Eq. 3 [P]
GRID_FEATURES_CODE_AVAILABLE = ["rudy", "rudy_pin", "macro_map"]   # official set minus H/V MacroMargin (absent) [C/D]

FEATURE_PROVENANCE = {
    "cell": "P (definitions), C (dim 13, x/y in cols 0-1), A (computed from CircuitNet placement + graph_information); "
            "HPWL from cell centres (no pin offsets in data) = A",
    "net": "P (definitions), A (cell-centre bounding boxes; degree counts every pin of the net)",
    "pin": "P (definitions); direction D (std cells: pin 'o' is the only output) / U (macro pins: no direction "
           "source); dx,dy normalised by net bbox width/height = U",
    "geom_edges": "U (rule not specified anywhere): symmetric kNN on cell centres; features in micrometres (unscaled, "
                  "the official builder applies no scaling) = U",
    "grid_paper": "P (definitions), A (counts per gcell; pins located at their cell's centre tile; nets by the tile "
                  "bbox of their cells' centre tiles, consistent with the official grid-net edge rule)",
    "grid_code_available": "C (column set) + D (CircuitNet RUDY, RUDY_pin, macro_map); H/V MacroMargin not available",
    "labels": "D + A: CircuitNet GR overflow maps, H and V kept separately (aggregation happens at graph build)",
}


def _boxes(placement: dict, names) -> np.ndarray:
    missing = [n for n in names if n not in placement]
    if missing:
        raise KeyError(f"{len(missing)} instances missing from placement, e.g. {missing[:3]}")
    return np.asarray([placement[n] for n in names], dtype=np.float64)


def coverage_sum(x0, x1, y0, y1, weights, M, N) -> np.ndarray:
    """Sum `weights` over every tile in [x0, x1) x [y0, y1) (2-D difference array)."""
    d = np.zeros((M + 1, N + 1), dtype=np.float64)
    np.add.at(d, (x0, y0), weights)
    np.add.at(d, (x1, y0), -weights)
    np.add.at(d, (x0, y1), -weights)
    np.add.at(d, (x1, y1), weights)
    return d.cumsum(0).cumsum(1)[:M, :N]


def gcell_extent(gcell_boxes: np.ndarray, M: int, N: int):
    g = gcell_boxes.astype(np.int64)
    x0 = np.clip(g[:, 0], 0, M - 1)
    y0 = np.clip(g[:, 1], 0, N - 1)
    x1 = np.clip(np.maximum(g[:, 2], g[:, 0] + 1), 1, M)
    y1 = np.clip(np.maximum(g[:, 3], g[:, 1] + 1), 1, N)
    return x0, x1, y0, y1


def knn_edges(cx, cy, k: int):
    """Symmetric kNN graph on points; returns (2, E) both directions, no self loops."""
    n = len(cx)
    if n < 2 or k <= 0:
        return np.zeros((2, 0), dtype=np.int64)
    pts = np.stack([cx, cy], 1)
    kk = min(k + 1, n)
    _, idx = cKDTree(pts).query(pts, k=kk)
    src = np.repeat(np.arange(n), kk)
    dst = idx.reshape(-1)
    keep = src != dst
    a, b = src[keep], dst[keep]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(lo * n + hi)
    lo, hi = pairs // n, pairs % n
    return np.stack([np.concatenate([lo, hi]), np.concatenate([hi, lo])]).astype(np.int64)


def compute_sample_features(nl, micron: dict, gcell: dict, M: int, N: int,
                            knn_k: int = 8, code_maps: dict | None = None,
                            pin_offset_norm: str = "bbox") -> dict:
    """pin_offset_norm: "bbox" divides the cell-to-net-centroid offset by the net bbox
    width/height (default, U); "none" keeps micrometres (ablation)."""
    if pin_offset_norm not in ("bbox", "none"):
        raise ValueError(f"pin_offset_norm must be bbox|none, got {pin_offset_norm}")
    names = nl.cell_names
    C, E = nl.num_cells, nl.num_nets
    pc, pn = nl.pin_cell, nl.pin_net

    mb = _boxes(micron, names)
    gb = _boxes(gcell, names)
    w = mb[:, 2] - mb[:, 0]
    h = mb[:, 3] - mb[:, 1]
    cx = (mb[:, 0] + mb[:, 2]) / 2
    cy = (mb[:, 1] + mb[:, 3]) / 2
    area = w * h

    gx0, gx1, gy0, gy1 = gcell_extent(gb, M, N)
    tx = (gx0 + gx1 - 1) // 2          # centre tile of the instance's gcell extent
    ty = (gy0 + gy1 - 1) // 2
    cell_tile = tx * N + ty            # row-major [x, y], same as official grid index row*N+col

    # ---- nets: bounding box of member cell centres
    xmin = np.full(E, np.inf); xmax = np.full(E, -np.inf)
    ymin = np.full(E, np.inf); ymax = np.full(E, -np.inf)
    np.minimum.at(xmin, pn, cx[pc]); np.maximum.at(xmax, pn, cx[pc])
    np.minimum.at(ymin, pn, cy[pc]); np.maximum.at(ymax, pn, cy[pc])
    has = np.isfinite(xmin)
    nw = np.where(has, xmax - xmin, 0.0)
    nh = np.where(has, ymax - ymin, 0.0)
    ncx = np.where(has, (xmin + xmax) / 2, 0.0)
    ncy = np.where(has, (ymin + ymax) / 2, 0.0)
    hpwl = nw + nh
    deg = np.bincount(pn, minlength=E).astype(np.float64)
    net_x = np.stack([hpwl, nw, nh, deg, np.log(np.maximum(deg, 1)), nw * nh], 1)

    # ---- cells: topology statistics over distinct incident nets
    pair = np.unique(pc * E + pn)
    pair_c, pair_n = pair // E, pair % E
    n_net = np.bincount(pair_c, minlength=C).astype(np.float64)
    n_pin = np.bincount(pc, minlength=C).astype(np.float64)
    denom = np.maximum(n_net, 1)
    mean_hpwl = np.bincount(pair_c, weights=hpwl[pair_n], minlength=C) / denom
    max_hpwl = np.zeros(C); np.maximum.at(max_hpwl, pair_c, hpwl[pair_n])
    mean_deg = np.bincount(pair_c, weights=deg[pair_n], minlength=C) / denom
    n_out = np.bincount(pc, weights=nl.pin_is_output, minlength=C)
    in_ratio = np.where(n_pin > 0, (n_pin - n_out) / np.maximum(n_pin, 1), 0.0)
    _, inv, counts = np.unique(nl.masters, return_inverse=True, return_counts=True)
    type_freq = counts[inv] / C
    cell_x = np.stack([cx, cy, w, h, area, n_net, n_pin, mean_hpwl, max_hpwl, mean_deg,
                       nl.is_macro.astype(np.float64), in_ratio, type_freq], 1)

    # ---- pin edges (one edge per pin) and features
    pin_ei = np.stack([pc, pn]).astype(np.int64)
    if pin_offset_norm == "bbox":
        dxn = np.where(nw[pn] > 0, (cx[pc] - ncx[pn]) / np.where(nw[pn] > 0, nw[pn], 1), 0.0)
        dyn = np.where(nh[pn] > 0, (cy[pc] - ncy[pn]) / np.where(nh[pn] > 0, nh[pn], 1), 0.0)
    else:
        dxn, dyn = cx[pc] - ncx[pn], cy[pc] - ncy[pn]
    pin_x = np.stack([nl.pin_is_output.astype(np.float64), dxn, dyn], 1)

    # ---- geometric cell-cell edges
    geom_ei = knn_edges(cx, cy, knn_k)
    gdx = cx[geom_ei[0]] - cx[geom_ei[1]]
    gdy = cy[geom_ei[0]] - cy[geom_ei[1]]
    geom_x = np.stack([gdx, gdy, np.abs(gdx) + np.abs(gdy), np.hypot(gdx, gdy)], 1)

    # ---- grid features, paper Eq. 3
    ones = np.ones(C)
    cell_density = coverage_sum(gx0, gx1, gy0, gy1, ones, M, N)
    area_sum = coverage_sum(gx0, gx1, gy0, gy1, area, M, N)
    pin_density = np.zeros((M, N)); np.add.at(pin_density, (tx[pc], ty[pc]), 1)
    ntx0 = np.full(E, M, np.int64); ntx1 = np.full(E, -1, np.int64)
    nty0 = np.full(E, N, np.int64); nty1 = np.full(E, -1, np.int64)
    np.minimum.at(ntx0, pn, tx[pc]); np.maximum.at(ntx1, pn, tx[pc])
    np.minimum.at(nty0, pn, ty[pc]); np.maximum.at(nty1, pn, ty[pc])
    v = ntx1 >= 0
    net_density = coverage_sum(ntx0[v], ntx1[v] + 1, nty0[v], nty1[v] + 1, np.ones(v.sum()), M, N)
    hpwl_sum = coverage_sum(ntx0[v], ntx1[v] + 1, nty0[v], nty1[v] + 1, hpwl[v], M, N)
    mean_area = np.divide(area_sum, cell_density, out=np.zeros_like(area_sum), where=cell_density > 0)
    mean_net_hpwl = np.divide(hpwl_sum, net_density, out=np.zeros_like(hpwl_sum), where=net_density > 0)
    grid_paper = np.stack([cell_density, pin_density, net_density, mean_area, mean_net_hpwl], -1).reshape(M * N, 5)

    out = {
        "cell_x": cell_x.astype(np.float32),
        "net_x": net_x.astype(np.float32),
        "pin_edge_index": pin_ei,
        "pin_x": pin_x.astype(np.float32),
        "geom_edge_index": geom_ei,
        "geom_x": geom_x.astype(np.float32),
        "cell_tile": cell_tile.astype(np.int64),
        "grid_x_paper": grid_paper.astype(np.float32),
        "cell_density_recomputed": cell_density.astype(np.float32),
    }
    if code_maps:
        cols = []
        for key in GRID_FEATURES_CODE_AVAILABLE:
            a = np.asarray(code_maps[key], dtype=np.float32)
            if a.shape != (M, N):
                raise ValueError(f"{key} map shape {a.shape} != label grid {(M, N)}")
            cols.append(a.reshape(-1))
        out["grid_x_code"] = np.stack(cols, 1)
    return out
