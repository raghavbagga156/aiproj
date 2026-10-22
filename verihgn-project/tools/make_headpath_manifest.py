"""Write configs/headpath_manifest.json: the pre-registered HEADPATH experiment (docs/HEADPATH_PLAN.md),
computed from sample NAMES and the pre-registration constants in diagnostics/headpath.py only.

The training set and validation set must equal notebook 07's FULL arm and validation set
(configs/collapse_experiment_manifest.json). Notebook 08 recomputes the plan from the processed
data on Drive and refuses to run if it differs from this file, or while research decisions are pending.

usage: python tools/make_headpath_manifest.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verihgn_repro.data import circuitnet as cn  # noqa: E402
from verihgn_repro.data import splits as sp  # noqa: E402
from verihgn_repro.diagnostics import collapse_experiment as CE  # noqa: E402
from verihgn_repro.diagnostics import headpath as H  # noqa: E402


def main():
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    by = defaultdict(list)
    for n in names:
        by[CE.design_of(n)].append(n)
    assert {d: len(v) for d, v in by.items()} == {d: cn.SAMPLES_PER_DESIGN[d] for d in cn.SCOPES["official9"]}
    split = {"name": "official_seed42", "designs": sp.official_split(cn.SCOPES["official9"])}
    man = H.manifest_from_names(by, split)
    c07 = json.loads((ROOT / "configs" / "collapse_experiment_manifest.json").read_text())
    assert man["train"] == c07["arms"]["FULL"] and man["val"] == c07["val"], "must equal notebook 07 FULL / val"
    man = {"comment": "Pre-registered HEADPATH experiment (tools/make_headpath_manifest.py). Generated from sample "
                      "names and the constants in diagnostics/headpath.py; no labels or results were used. "
                      "DIAGNOSTIC: arm U is a documented deviation from the official model, not a reproduction.",
           "split": split["designs"], "n_runs": len(man["runs"]),
           "total_optimizer_steps": sum(r["steps"] for r in man["runs"]), **man}
    out = ROOT / "configs" / "headpath_manifest.json"
    out.write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out, f"({man['n_runs']} runs, {man['total_optimizer_steps']} steps)")
    return man


if __name__ == "__main__":
    main()
