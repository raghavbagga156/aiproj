"""Write configs/collapse_experiment_manifest.json: the pre-registered sample selection and
step budgets of the notebook-07 collapse experiment, computed from sample NAMES only.

Input: configs/official9_sample_names.txt (the 383 official-scope sample names, taken from the
public CircuitNet-ISPD15 archive listing; they equal the processed directory names).
Notebook 07 recomputes the plan from the processed data on Drive and refuses to run if it
differs from this file in any selection, budget or run.

usage: python tools/make_collapse_manifest.py
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


def main():
    names = (ROOT / "configs" / "official9_sample_names.txt").read_text().split()
    by = defaultdict(list)
    for n in names:
        by[CE.design_of(n)].append(n)
    counts = {d: len(v) for d, v in sorted(by.items())}
    assert counts == {d: cn.SAMPLES_PER_DESIGN[d] for d in cn.SCOPES["official9"]}, counts
    split = {"name": "official_seed42", "designs": sp.official_split(cn.SCOPES["official9"])}
    man = CE.manifest_from_names(by, split)
    man = {"comment": "Pre-registered selection for notebook 07 (tools/make_collapse_manifest.py). "
                      "Generated from sample names only; no labels or results were used.",
           "split": split["designs"], "samples_per_design": counts, **man}
    out = ROOT / "configs" / "collapse_experiment_manifest.json"
    out.write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out)
    return man


if __name__ == "__main__":
    main()
