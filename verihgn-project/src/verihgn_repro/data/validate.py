"""Validation gates that must pass before any training.

`validate_graph` checks one HeteroData object; `validate_dataset` runs it over
every graph of every split, adds split-leakage checks and an official-model
forward/backward check, and returns a JSON-serialisable report. Any error in
the report means training must not start.
"""
import math

import torch

from .. import official

EXPECTED_NODE_TYPES = {"cell", "net", "grid_l0", "grid_l1"}
EXPECTED_EDGE_TYPES = {
    ("cell", "pin", "net"), ("net", "pin_rev", "cell"), ("cell", "geom", "cell"),
    ("grid_l0", "adj", "grid_l0"), ("grid_l1", "adj", "grid_l1"),
    ("grid_l0", "child_of", "grid_l1"), ("grid_l1", "parent_of", "grid_l0"),
    ("cell", "in", "grid_l0"), ("grid_l0", "contains", "cell"),
    ("net", "spans", "grid_l0"), ("grid_l0", "spans_rev", "net"),
}
EXPECTED_DIMS = {"cell": 13, "net": 6, ("cell", "pin", "net"): 3, ("cell", "geom", "cell"): 4}


def validate_graph(data, grid_feat_dim: int) -> dict:
    errors, warn = [], []
    name = getattr(data, "placement_name", "?")

    if set(data.node_types) != EXPECTED_NODE_TYPES:
        errors.append(f"node types {sorted(data.node_types)}")
    if set(data.edge_types) != EXPECTED_EDGE_TYPES:
        errors.append(f"edge types differ: missing {EXPECTED_EDGE_TYPES - set(data.edge_types)}, "
                      f"extra {set(data.edge_types) - EXPECTED_EDGE_TYPES}")

    for nt, dim in (("cell", 13), ("net", 6), ("grid_l0", grid_feat_dim), ("grid_l1", grid_feat_dim)):
        x = data[nt].x
        if x.dim() != 2 or x.shape[1] != dim:
            errors.append(f"{nt}.x shape {tuple(x.shape)}, expected (*, {dim})")
        if x.shape[0] != data[nt].num_nodes:
            errors.append(f"{nt}.x rows {x.shape[0]} != num_nodes {data[nt].num_nodes}")
        if not torch.isfinite(x).all():
            errors.append(f"{nt}.x has NaN/Inf")
    for et, dim in ((("cell", "pin", "net"), 3), (("net", "pin_rev", "cell"), 3), (("cell", "geom", "cell"), 4)):
        ea = data[et].edge_attr
        if ea.shape != (data[et].edge_index.shape[1], dim):
            errors.append(f"{et}.edge_attr shape {tuple(ea.shape)}")
        if not torch.isfinite(ea).all():
            errors.append(f"{et}.edge_attr has NaN/Inf")

    for et in data.edge_types:
        ei = data[et].edge_index
        s, _, d = et
        if ei.dim() != 2 or ei.shape[0] != 2 or ei.dtype != torch.long:
            errors.append(f"{et} edge_index malformed {tuple(ei.shape)} {ei.dtype}")
            continue
        if ei.numel() and (ei[0].min() < 0 or ei[0].max() >= data[s].num_nodes or
                           ei[1].min() < 0 or ei[1].max() >= data[d].num_nodes):
            errors.append(f"{et} index out of range")

    M, N, M1, N1 = data.grid_M, data.grid_N, data.grid_M1, data.grid_N1
    if data["grid_l0"].num_nodes != M * N or data["grid_l1"].num_nodes != M1 * N1:
        errors.append("grid node counts inconsistent with grid_M/N")
    if (M1, N1) != ((M + 1) // 2, (N + 1) // 2):
        errors.append("L1 size is not ceil(M/2) x ceil(N/2)")
    ch = data["grid_l0", "child_of", "grid_l1"].edge_index
    if ch.shape[1] != M * N or torch.unique(ch[0]).numel() != M * N:
        errors.append("every L0 tile must have exactly one L1 parent")
    else:
        r0, c0 = ch[0] // N, ch[0] % N
        r1, c1 = ch[1] // N1, ch[1] % N1
        if not (torch.equal(r0 // 2, r1) and torch.equal(c0 // 2, c1)):
            errors.append("hierarchy parent is not the 2x2 block containing the child")
    pr = data["grid_l1", "parent_of", "grid_l0"].edge_index
    if not torch.equal(pr, ch.flip(0)):
        errors.append("parent_of is not the reverse of child_of")
    adj = data["grid_l0", "adj", "grid_l0"].edge_index
    if adj.shape[1] != 2 * (M * (N - 1) + (M - 1) * N):
        errors.append("L0 adjacency count != 4-connected count")

    cg = data["cell", "in", "grid_l0"].edge_index
    if cg.shape[1] != data["cell"].num_nodes or not torch.equal(cg[0], torch.arange(data["cell"].num_nodes)):
        errors.append("each cell must have exactly one cell->grid edge, in order")
    if not torch.equal(data["grid_l0", "contains", "cell"].edge_index, cg.flip(0)):
        errors.append("contains is not the reverse of cell->grid")
    if not torch.equal(data["net", "pin_rev", "cell"].edge_index, data["cell", "pin", "net"].edge_index.flip(0)):
        errors.append("pin_rev is not the reverse of pin")
    geom = data["cell", "geom", "cell"].edge_index
    if geom.numel():
        if (geom[0] == geom[1]).any():
            errors.append("geom edges contain self loops")
        n = data["cell"].num_nodes
        fwd = set((geom[0] * n + geom[1]).tolist())
        rev = set((geom[1] * n + geom[0]).tolist())
        if fwd != rev:
            errors.append("geom edges are not symmetric")

    gy, cy = data["grid_l0"].y, data["cell"].y
    if gy.shape != (M * N,) or cy.shape != (data["cell"].num_nodes,):
        errors.append(f"label shapes grid {tuple(gy.shape)} cell {tuple(cy.shape)}")
    if not (torch.isfinite(gy).all() and torch.isfinite(cy).all()):
        errors.append("labels have NaN/Inf")
    if gy.min() < 0:
        errors.append("negative grid labels")
    if not torch.equal(cy, gy[cg[1]]):
        errors.append("cell labels != grid label of the cell's tile")
    if getattr(data, "label_norm", None) == "per_sample" and gy.numel() and gy.max() > 0 \
            and not math.isclose(float(gy.max()), 1.0, rel_tol=1e-6):
        errors.append("per_sample labels must peak at 1")
    if hasattr(data["grid_l0"], "raw_hv"):
        raw = data["grid_l0"].raw_hv
        if raw.shape != (M * N, 2) or (raw < 0).any():
            errors.append("raw_hv malformed")
        if not torch.equal(gy > 0, raw.sum(1) > 0):
            errors.append("label support (y>0) differs from raw H+V support")

    stats = {
        "sample": name, "grid": [M, N],
        "nodes": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edges": {"__".join(et): int(data[et].edge_index.shape[1]) for et in data.edge_types},
        "label_nonzero_frac": float((gy > 0).float().mean()),
        "label_max": float(gy.max()),
    }
    if stats["edges"].get("net__spans__grid_l0", 0) == 0:
        warn.append("no grid-net edges")
    return {"sample": name, "errors": errors, "warnings": warn, "stats": stats}


def official_model_check(data, grid_feat_dim: int, device="cpu", hidden_dim=128, num_layers=3) -> dict:
    """Forward + loss + backward of the unmodified official model on one graph."""
    model = official.VeriHGN(cell_feat_dim=13, net_feat_dim=6, grid_feat_dim=grid_feat_dim,
                             pin_feat_dim=3, geom_feat_dim=4, hidden_dim=hidden_dim,
                             num_layers=num_layers, dropout=0.15, use_checkpoint=True).to(device)
    model.train()
    batch = official.dataset_module().collate_fn([data]).to(device)
    out = model(batch)
    loss = official.loss_module().verihgn_loss(out["cell_pred"].float(), batch["cell"].y,
                                               out["grid_pred"].float(), batch["grid_l0"].y,
                                               alpha=10.0, lambda_var=0.5)
    loss["total"].backward()
    grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    ok = (out["cell_pred"].shape == batch["cell"].y.shape and out["grid_pred"].shape == batch["grid_l0"].y.shape
          and torch.isfinite(loss["total"]).item() and grads_finite)
    return {"ok": bool(ok), "loss": loss["total"].item(), "cell_pred_shape": list(out["cell_pred"].shape),
            "grid_pred_shape": list(out["grid_pred"].shape), "grads_finite": bool(grads_finite)}


def validate_dataset(graphs_by_split: dict, grid_feat_dim: int, split_manifest: dict | None = None,
                     model_check_device="cpu", model_check_samples=1) -> dict:
    report = {"per_sample": [], "errors": [], "splits": {}}
    seen = {}
    for part, graphs in graphs_by_split.items():
        designs = set()
        for g in graphs:
            r = validate_graph(g, grid_feat_dim)
            r["split"] = part
            report["per_sample"].append(r)
            report["errors"] += [f"{r['sample']}: {e}" for e in r["errors"]]
            designs.add(g.design_name)
            if g.placement_name in seen:
                report["errors"].append(f"{g.placement_name} in both {seen[g.placement_name]} and {part}")
            seen[g.placement_name] = part
        report["splits"][part] = {"samples": len(graphs), "designs": sorted(designs)}
    parts = list(report["splits"])
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            both = set(report["splits"][a]["designs"]) & set(report["splits"][b]["designs"])
            if both:
                report["errors"].append(f"designs {sorted(both)} in both {a} and {b}")
    if split_manifest is not None:
        report["split_leakage"] = split_manifest.get("leakage")
        if split_manifest["name"].startswith("pair_aware") and not split_manifest["leakage"]["leak_free"]:
            report["errors"].append("pair-aware split has netlist twins across splits")
    checks = []
    for part in ("train", "val", "test"):
        for g in graphs_by_split.get(part, [])[:model_check_samples]:
            c = official_model_check(g, grid_feat_dim, device=model_check_device)
            c["sample"] = g.placement_name
            checks.append(c)
            if not c["ok"]:
                report["errors"].append(f"official model check failed on {g.placement_name}: {c}")
    report["official_model_checks"] = checks
    report["passed"] = not report["errors"]
    return report
