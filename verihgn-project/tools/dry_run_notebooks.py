"""Execute notebooks 03-06 locally on a small raw sample tree (CPU) to catch bugs
before spending Colab GPU time. Colab-only parts are patched:
  * bootstrap/Drive mount -> a temporary DriveLayout,
  * scope = the designs present in the fixture tree, expected counts = what is there,
  * CUDA-only calls -> CPU, benchmark length 20 epochs x 2 seeds -> 1 epoch x 1 seed.
Notebooks 01-02 (GPU report, archive download) are Colab-only and not executed.
Outputs of this dry run are test data, never results.

usage: python tools/dry_run_notebooks.py <raw_fixture_root> <work_dir>
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(raw_root, work):
    raw_root, work = Path(raw_root), Path(work)
    if work.exists():
        shutil.rmtree(work)
    from verihgn_repro.colab import DriveLayout
    from verihgn_repro.data import circuitnet as cn
    from verihgn_repro.data import splits as sp
    from verihgn_repro.utils import hardware

    present = Counter(cn.parse_sample_name(s)["design"] for s in cn.RawLayout(raw_root).samples())
    cn.SCOPES["official9"] = sorted(present)
    cn.SAMPLES_PER_DESIGN.clear()
    cn.SAMPLES_PER_DESIGN.update(present)
    designs = sorted(present)
    sp.official_split = lambda ds: {"train": designs[:-2], "val": designs[-2:-1], "test": designs[-1:]}

    class L(DriveLayout):
        pass
    layout = L(root=work / "drive" / "VeriHGN", local_raw=raw_root, local_archives=work / "archives")
    layout.ensure()
    (work / "graph_cache").mkdir(parents=True)
    hardware.detect = (lambda orig: (lambda: orig().__class__(device="cuda", name="DRY-RUN (CPU)")))(hardware.detect)

    ns = {"L": layout, "CODE_DIR": str(ROOT), "display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
    patches = [
        ("precisions=tuple(prof.candidate_precisions))", "precisions=('fp32',), device='cpu')"),
        ("device='cuda', per_design=2", "device='cpu', per_design=1"),
        ("epochs=20, seeds=(42, 43),", "epochs=1, seeds=(42,),"),
        ("precisions=tuple(prof.candidate_precisions), repeat_epochs=3", "precisions=('fp32',), repeat_epochs=1, device='cpu'"),
        ("Trainer(scfg, data['graphs'], run_dir)", "Trainer(scfg, data['graphs'], run_dir, device='cpu')"),
        ("device = 'cuda' if torch.cuda.is_available() else 'cpu'", "device = 'cpu'"),
        ("hardware.GPUMonitor(interval_s=0.5).start()", "hardware.GPUMonitor(interval_s=0.5)"),
        ("max(r['peak_allocated_gb'] for r in rows)", "max((r['peak_allocated_gb'] or 0) for r in rows)"),
        ("RUN_PRECISION_BENCHMARK = False", "RUN_PRECISION_BENCHMARK = True"),     # exercise the code path
        ("RUN_GPU_DIAGNOSTIC = False", "RUN_GPU_DIAGNOSTIC = True"),
        ("'--precisions', 'fp32', 'fp16', '--steps', 60", "'--precisions', 'fp32', '--steps', 2"),
        ("'--metric-every', 20, '--device', 'cuda')", "'--metric-every', 1, '--device', 'cpu')"),
        ("from google.colab import drive\ndrive.flush_and_unmount()", "pass"),
    ]
    for nb in ("03_preprocessing", "04_graph_construction_validation", "05_gpu_profiling_and_precision_benchmark",
               "06_training_diagnosis"):
        cells = json.loads((ROOT / "notebooks" / f"{nb}.ipynb").read_text(encoding="utf-8"))["cells"]
        code_cells = [c for c in cells if c["cell_type"] == "code"][1:]      # skip bootstrap
        print(f"\n===== {nb}: {len(code_cells)} cells")
        for i, c in enumerate(code_cells, 1):
            src = "".join(c["source"])
            for a, b in patches:
                src = src.replace(a, b)
            src = src.replace("cfg.training.max_epochs_this_session = 3, None",
                              "cfg.training.max_epochs_this_session = 3, None")
            print(f"--- cell {i}")
            exec(compile(src, f"{nb}[{i}]", "exec"), ns)
    print("\nDRY RUN OK — reports:", sorted(p.name for p in layout.reports.iterdir()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
