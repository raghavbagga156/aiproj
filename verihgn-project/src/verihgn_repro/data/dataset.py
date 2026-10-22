"""Experiment datasets: processed samples + split manifest + graph options ->
cached lists of HeteroData per split."""
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .. import official
from .graph import build_graph, compute_c_max
from .preprocess import list_processed, load_sample


def graph_cache_key(prep_hash: str, split_name: str, grid_features: str, label_norm: str, c_max) -> str:
    payload = json.dumps([prep_hash, split_name, grid_features, label_norm,
                          None if c_max is None else round(float(c_max), 8)])
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


class GraphListDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


def build_experiment_graphs(processed_root, split_manifest: dict, grid_features="paper_eq3",
                            label_norm="per_sample", cache_dir=None, prep_hash=None, verbose=True) -> dict:
    processed_root = Path(processed_root)
    dirs = {part: list_processed(processed_root, set(designs))
            for part, designs in split_manifest["designs"].items()}
    c_max = compute_c_max(dirs["train"]) if label_norm == "per_dataset" else None
    if prep_hash is None:
        metas = [json.loads((d / "meta.json").read_text())["prep_hash"] for ds in dirs.values() for d in ds[:1]]
        prep_hash = metas[0] if metas else "unknown"
    key = graph_cache_key(prep_hash, split_manifest["name"], grid_features, label_norm, c_max)
    out = {"c_max": c_max, "cache_key": key, "graphs": {}}
    for part, sample_dirs in dirs.items():
        path = Path(cache_dir) / f"graphs_{key}_{part}.pt" if cache_dir else None
        if path and path.exists():
            if verbose:
                print(f"[{part}] loading cached graphs {path}")
            out["graphs"][part] = torch.load(path, weights_only=False)
            continue
        graphs = []
        for i, d in enumerate(sample_dirs, 1):
            arrays, meta = load_sample(d)
            graphs.append(build_graph(arrays, meta, grid_features=grid_features,
                                      label_norm=label_norm, c_max=c_max))
            if verbose and i % 25 == 0:
                print(f"[{part}] built {i}/{len(sample_dirs)} graphs")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(graphs, tmp)
            os.replace(tmp, path)
        out["graphs"][part] = graphs
    return out


def make_loaders(graphs_by_split: dict, seed: int = 42):
    """batch_size=1 and shuffle on train only, as in the official get_dataloaders."""
    collate = official.dataset_module().collate_fn
    gen = torch.Generator()
    gen.manual_seed(seed)
    loaders = {}
    for part, graphs in graphs_by_split.items():
        loaders[part] = DataLoader(GraphListDataset(graphs), batch_size=1, shuffle=(part == "train"),
                                   num_workers=0, collate_fn=collate,
                                   generator=gen if part == "train" else None)
    return loaders, gen
