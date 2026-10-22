"""Read-only check of an existing run for prediction collapse (docs/SMOKE_TRAINING_DIAGNOSIS.md).

For a run directory it
  1. loads a checkpoint (default checkpoints/last.pt) onto the CPU, never writing anything,
  2. runs the model in FP32 eval mode on a few train and val samples and reports the raw
     head outputs *before* the official clamp(min=0): fraction <= 0, max, and the
     prediction spread after the clamp,
  3. computes the loss of the all-zero predictor on every training graph and compares
     its mean with the training loss in history.csv (equal => the model predicted 0
     everywhere during that epoch).
No training, no optimizer step, no GPU required (the A100 makes step 2 faster).

usage: python tools/inspect_collapse.py --run-dir <runs/<run_id>> --processed <processed/ispd15/<hash>>
                                        --split <splits/official9__official_seed42.json> --out <file.json>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verihgn_repro import official  # noqa: E402
from verihgn_repro.config import load_config  # noqa: E402
from verihgn_repro.data.graph import build_graph, compute_c_max  # noqa: E402
from verihgn_repro.data.preprocess import list_processed, load_sample  # noqa: E402
from verihgn_repro.training.checkpointing import load_checkpoint  # noqa: E402
from verihgn_repro.training.trainer import build_model  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--processed", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward-samples", type=int, default=3, help="per split, for the raw-output check")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    s = load_checkpoint(run_dir / "checkpoints" / a.checkpoint)
    cfg = load_config(overrides=s["config"])
    e, t = cfg.experiment, cfg.training
    split = json.loads(Path(a.split).read_text())
    dirs = {p: list_processed(a.processed, set(ds)) for p, ds in split["designs"].items()}
    c_max = compute_c_max(dirs["train"]) if e.label_norm == "per_dataset" else None
    model = build_model(cfg)
    model.load_state_dict(s["model"])
    model.to(a.device).eval()
    raw = {}
    hooks = [model.cell_head.mlp.register_forward_hook(lambda m, i, o: raw.__setitem__("cell", o.detach().float().squeeze(-1))),
             model.grid_head.mlp.register_forward_hook(lambda m, i, o: raw.__setitem__("grid", o.detach().float().squeeze(-1)))]
    lf = official.loss_module().verihgn_loss
    collate = official.dataset_module().collate_fn
    report = {"run_dir": str(run_dir), "checkpoint": a.checkpoint, "epoch": s["epoch"], "run_id": s["run_id"],
              "c_max": c_max, "head_bias": {"cell": model.cell_head.mlp[-1].bias.item(),
                                            "grid": model.grid_head.mlp[-1].bias.item()},
              "forward": [], "zero_predictor": {}}
    zero = []
    for part in ("train", "val"):
        for i, d in enumerate(dirs[part]):
            arrays, meta = load_sample(d)
            g = build_graph(arrays, meta, grid_features=e.grid_features, label_norm=e.label_norm, c_max=c_max)
            cy, gy = g["cell"].y.float(), g["grid_l0"].y.float()
            if part == "train":
                zero.append(float(lf(torch.zeros_like(cy), cy, torch.zeros_like(gy), gy, alpha=t.wmse_alpha,
                                     lambda_var=t.lambda_var)["total"]))
            if i < a.forward_samples:
                with torch.no_grad():
                    out = model(collate([g]).to(a.device))
                rec = {"split": part, "sample": d.name}
                for nt, key in (("cell", "cell_pred"), ("grid", "grid_pred")):
                    r, p = raw[nt].cpu(), out[key].float().cpu()
                    rec[nt] = {"raw_frac_le0": float((r <= 0).float().mean()), "raw_max": float(r.max()),
                               "raw_mean": float(r.mean()), "pred_std": float(p.std()),
                               "pred_frac_zero": float((p == 0).float().mean())}
                report["forward"].append(rec)
                print(json.dumps(rec))
    for h in hooks:
        h.remove()
    report["zero_predictor"] = {"train_graphs": len(zero), "mean_loss": float(np.mean(zero))}
    hist = run_dir / "history.csv"
    if hist.exists():
        with open(hist, newline="") as f:
            rows = list(csv.DictReader(f))
        report["history_train_loss"] = {r["epoch"]: float(r["train_loss"]) for r in rows}
        report["history_minus_zero_predictor"] = {k: v - report["zero_predictor"]["mean_loss"]
                                                  for k, v in report["history_train_loss"].items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("epoch", "head_bias", "zero_predictor")}, indent=1))
    print("history train_loss - zero-predictor loss:", report.get("history_minus_zero_predictor"))
    print("wrote", a.out)
    return report


if __name__ == "__main__":
    main()
