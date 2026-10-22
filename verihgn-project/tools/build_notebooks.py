"""Generate the Colab notebooks in notebooks/ (run: python tools/build_notebooks.py).

Keeping the notebooks as generated artefacts guarantees every notebook shares
the identical bootstrap cell and stays in sync with the package API.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

BOOTSTRAP = r'''# ── Bootstrap (identical in every notebook; safe to re-run) ──────────────────────
# Mounts YOUR Google Drive (you authorise in the popup; no credentials are stored),
# unpacks the project code from Drive and puts it on the Python path.
import os, sys, zipfile, subprocess
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/VeriHGN'            # persistent project storage
CODE_ZIP   = f'{DRIVE_ROOT}/code/verihgn_code.zip'        # uploaded from the laptop (tools/make_colab_bundle.py)
CODE_DIR   = '/content/verihgn-project'

if not os.path.exists(CODE_ZIP):
    raise FileNotFoundError(f'Upload verihgn_code.zip to {CODE_ZIP} first (see README, "Colab").')
stamp = f'{CODE_DIR}/.zip_mtime'
if not os.path.exists(stamp) or open(stamp).read() != str(os.path.getmtime(CODE_ZIP)):
    subprocess.run(['rm', '-rf', CODE_DIR], check=True)
    with zipfile.ZipFile(CODE_ZIP) as z:
        z.extractall(CODE_DIR)
    open(stamp, 'w').write(str(os.path.getmtime(CODE_ZIP)))
sys.path.insert(0, f'{CODE_DIR}/src')

marker = '/content/.verihgn_deps_ok'
if not os.path.exists(marker):
    from verihgn_repro.colab import install_dependencies
    install_dependencies()
    open(marker, 'w').write('ok')

from verihgn_repro.colab import DriveLayout
L = DriveLayout().ensure()
print('code:', CODE_DIR, '| drive root:', L.root)'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


NB = {}

NB["01_environment_setup"] = notebook([
    md("""
# 01 · Environment setup, GPU check and dataset access test

**Purpose.** Verify the GPU, CUDA and libraries, then test (I-1) whether the CircuitNet folder is reachable
through *your* Drive mount and check free storage (I-2). Nothing is downloaded or trained here.

**Before running:** Runtime → Change runtime type → **A100 GPU**. Upload `verihgn_code.zip` to
`MyDrive/VeriHGN/code/`. For the Drive route, open the
[CircuitNet folder](https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB) and choose
*Organize → Add shortcut → My Drive*.
"""),
    code(BOOTSTRAP),
    md("## 1. GPU, CUDA, library versions"),
    code(r'''
import json, subprocess, torch
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
from verihgn_repro.utils import hardware
from verihgn_repro import official
env = hardware.env_report({'scatter_impl': official.scatter_impl(), 'official_commit': official.official_commit()})
print(json.dumps(env, indent=1))
if env['hardware']['device'] != 'cuda':
    print('⚠ No GPU: select Runtime → Change runtime type → A100. CPU is only usable for smoke tests.')
'''),
    md("## 2. Dataset access test (I-1) and storage check (I-2)"),
    code(r'''
from verihgn_repro.colab import access_check, storage_report, write_json
acc = access_check('/content/drive')
print(json.dumps(acc, indent=1))
st = storage_report(L)
print(json.dumps(st, indent=1))
write_json({'env': env, 'access': acc, 'storage': st}, L.reports / '01_environment.json')
if acc['ok']:
    print('✔ Drive route works: archives found via', acc['found']['method'])
else:
    print('✘ Drive route not available yet.', acc.get('instructions', ''))
'''),
    md("""
## 3. Public-link fallback test (only if the Drive route failed)
Downloads the 124 MB `graph_infromation.tar.gz` with `gdown` and checks its byte size.
"""),
    code(r'''
TEST_GDOWN = not acc['ok']          # set True to test it anyway
if TEST_GDOWN:
    import time
    from verihgn_repro.data import circuitnet as cn
    t0 = time.time()
    p = cn.download_with_gdown(cn.ARCHIVES['graph_information'], L.local_archives)
    print('✔ gdown ok:', p, f'{time.time()-t0:.0f}s')
'''),
    md("""
**Decide I-2 now.** Persistent data this stage writes to Drive: processed samples ≈ 3–4 GB (estimated from
real samples: 2–14 MB each), reports, and benchmark checkpoints (a few MB each). Caching the raw archives too
would add ≈ 12.4 GB (graph info 0.12 + gcell 1.96 + micron 2.72 + routability 7.59). Set
`CACHE_ON_DRIVE = True` in notebook 02 only if the free Drive space above comfortably exceeds ~17 GB.
"""),
])

NB["02_dataset_fetch_and_audit"] = notebook([
    md("""
# 02 · Fetch and extract the official-scope data, then audit it

Reads the archives through your Drive shortcut (or downloads them with `gdown` if the shortcut is
unavailable) and extracts **only** the 9 official-scope designs and the 6 routability maps the project uses,
into `/content/circuitnet_ispd15` (local disk, rebuilt each session; ≈ 15–30 min). Nothing on Drive is
modified. Archive integrity: byte sizes are checked against the Phase 0 audit, and every gzip stream is
read to its end (tar reports truncation/CRC errors). The audit then checks sample counts and map shapes.
"""),
    code(BOOTSTRAP),
    code(r'''
import json
from verihgn_repro.colab import access_check, fetch_archives, extract_for_scope, write_json
from verihgn_repro import reporting
SCOPE = 'official9'          # D6 primary scope (never mixed with all14)
CACHE_ON_DRIVE = False       # I-2: True only if notebook 01 showed > ~15 GB free on Drive
acc = access_check('/content/drive')
METHOD = 'drive' if acc['ok'] else 'gdown'
print('method:', METHOD, '| archive size checks:', [(a['archive'], a['ok']) for a in acc['archives']])
paths = fetch_archives(L, method=METHOD, ispd15_dir=acc['found']['path'], cache_on_drive=CACHE_ON_DRIVE)
'''),
    code(r'''
counts = extract_for_scope(paths, L, scope=SCOPE)
print(json.dumps(counts, indent=1))
'''),
    md("## Audit: discovery (expected vs found) and label/feature map shapes for every sample"),
    code(r'''
import numpy as np
from verihgn_repro.data import circuitnet as cn
from verihgn_repro.utils.safe_npy import load_npy
raw = cn.RawLayout(L.local_raw)
disc = reporting.discovery_report(L.local_raw, SCOPE)
problems = [] if disc['complete'] else [f'discovery incomplete: {disc["per_design"]}']
shapes = {}
for s in raw.samples():
    try:
        h = load_npy(raw.routability('gr_h_overflow', s)); v = load_npy(raw.routability('gr_v_overflow', s))
        if h.shape != v.shape:
            problems.append(f'{s}: H {h.shape} != V {v.shape}')
        for k in ('cell_density', 'rudy', 'rudy_pin', 'macro_map'):
            if load_npy(raw.routability(k, s)).shape != h.shape:
                problems.append(f'{s}: {k} shape differs from label')
        if not raw.placement_micron(s).exists():
            problems.append(f'{s}: micron placement missing')
        shapes.setdefault(cn.parse_sample_name(s)['design'], set()).add(h.shape)
    except Exception as e:
        problems.append(f'{s}: {type(e).__name__}: {e}')
audit = {'discovery': disc, 'extraction': counts, 'archive_checks': acc['archives'], 'method': METHOD,
         'label_shapes': {d: sorted(map(list, v)) for d, v in shapes.items()}, 'problems': problems}
write_json(audit, L.reports / f'02_audit_{SCOPE}.json')
print(json.dumps(disc, indent=1)); print('problems:', problems[:10])
assert not problems, 'audit failed — do not continue'
print(f"✔ {disc['found_total']}/{disc['expected_total']} samples discovered, all maps aligned")
'''),
])

NB["03_preprocessing"] = notebook([
    md("""
# 03 · Preprocessing, processed-data audit and splits

Writes `MyDrive/VeriHGN/processed/ispd15/<prep_hash>/<design>/<sample>/{sample.npz, meta.json}`.
The hash covers the code and the assumptions below, so changing any of them creates a **new** folder and
never overwrites earlier data. Every sample is accounted for as written, already-processed (skipped), or
failed — nothing is dropped silently. Run notebook 02 first in the same session.
"""),
    code(BOOTSTRAP),
    code(r'''
import json, os
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.circuitnet import RawLayout, parse_sample_name, SCOPES
from verihgn_repro.data.splits import make_split_manifests
from verihgn_repro import reporting
from verihgn_repro.colab import write_json
SCOPE = 'official9'
# Unresolved research assumptions — kept at the approved defaults (Stage 5); change only for ablations.
cfg = P.prep_config(knn_k=8, macro_pin_direction='input', with_code_features=True, pin_offset_norm='bbox')
PHASH = P.prep_hash(cfg)
PROCESSED = L.processed_for(PHASH)
print('prep hash:', PHASH, '->', PROCESSED)
print(json.dumps(P.ASSUMPTIONS, indent=1))
'''),
    code(r'''
man = P.run(L.local_raw, PROCESSED, scope=SCOPE, cfg=cfg, workers=max(1, (os.cpu_count() or 2) - 1))
disc = reporting.discovery_report(L.local_raw, SCOPE)
prep_report = {'prep_hash': PHASH, 'processed_dir': str(PROCESSED), 'assumptions': P.ASSUMPTIONS,
               'expected': disc['expected_total'], 'requested': man['requested'], 'written': man['written'],
               'skipped_already_processed': man['skipped_already_processed'], 'failed_count': man['failed_count'],
               'failed': man['failed'], 'cell_density_mismatch': man['cell_density_mismatch'], 'seconds': man['seconds']}
write_json(prep_report, L.reports / f'03_preprocessing_{SCOPE}.json')
print(json.dumps({k: v for k, v in prep_report.items() if k != 'assumptions'}, indent=1)[:3000])
assert man['failed_count'] == 0 and not man['cell_density_mismatch'], 'preprocessing failures — inspect before continuing'
assert man['requested'] == disc['expected_total'], 'sample count mismatch'
'''),
    md("## Audit every processed sample (feature order, shapes, index ranges, H/V alignment, finiteness)"),
    code(r'''
expected = [parse_sample_name(s)['sample'] for s in RawLayout(L.local_raw).samples()]
aud = reporting.audit_processed(PROCESSED, SCOPE, expected_samples=expected)
write_json(aud, L.reports / f'03_processed_audit_{SCOPE}.json')
print('audited', aud['audited'], '| ok', aud['ok'], '| failed', len(aud['failed']), '| missing', aud['missing'])
print(json.dumps(aud['per_design'], indent=1))
assert aud['passed'], aud['failed'][:3]
'''),
    md("## Splits: official seed-42 and pair-aware (from real netlist signatures)"),
    code(r'''
splits = make_split_manifests(L.local_raw, SCOPE, L.splits)
srep = reporting.split_report({k: v for k, v in splits.items()}, PROCESSED, SCOPE)
srep['prep_hash'] = PHASH
write_json(srep, L.reports / f'03_splits_{SCOPE}.json')
for name, r in srep.items():
    if name == 'prep_hash': continue
    print(name, r['designs'], r['processed_samples'], 'leak-free:', r['leakage']['leak_free'],
          '| twins across splits:', r['leakage']['netlist_twins_across_splits'])
assert splits['pair_aware']['leakage']['leak_free'] and all(r['all_designs_assigned_once'] for k, r in srep.items() if k != 'prep_hash')
'''),
])

NB["04_graph_construction_validation"] = notebook([
    md("""
# 04 · Graph construction and the mandatory validation gates (all variants, all samples)

For **every** sample, builds the heterogeneous graph under each variant —
{paper Eq. 3 features, official-code features} × {per-sample normalisation, per-dataset normalisation with
the c_max of each split's training set} — and runs every validation gate. The unmodified official model
runs forward + backward on the largest-first sample of each design on the GPU. Graphs are not stored
(≈ 30 GB per variant); they are rebuilt deterministically when needed. Training refuses to start unless
this notebook's report says `passed`. ≈ 15–40 min.
"""),
    code(BOOTSTRAP),
    code(r'''
import json, torch
from verihgn_repro.data import preprocess as P
from verihgn_repro.data.splits import load_manifest
from verihgn_repro import reporting
from verihgn_repro.colab import write_json
SCOPE = 'official9'
PHASH = json.load(open(L.reports / f'03_preprocessing_{SCOPE}.json'))['prep_hash']
PROCESSED = L.processed_for(PHASH)
manifests = {n: load_manifest(L.splits / f'{SCOPE}__{n}.json') for n in ('official_seed42', 'pair_aware')}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rep = reporting.validate_variants(PROCESSED, manifests, device=device, model_checks_per_design=1)
rep['prep_hash'] = PHASH
write_json(rep, L.reports / f'04_validation_{SCOPE}.json')
write_json([{'variant': k, 'passed': v['failed_count'] == 0 and v['skipped_count'] == 0, **v}
            for k, v in rep['variants'].items()], L.reports / '04_validation_summary.json')
for k, v in rep['variants'].items():
    print(f"{k:55s} ok {v['ok']:4d}  failed {v['failed_count']}  skipped {v['skipped_count']}")
print('c_max per split:', rep['c_max'])
print('official model checks ok:', all(c['ok'] for c in rep['official_model_checks']), len(rep['official_model_checks']))
assert rep['passed'], 'validation failed — training must not start'
print('✔ all validation gates passed for all variants')
'''),
])

NB["05_gpu_profiling_and_precision_benchmark"] = notebook([
    md("""
# 05 · A100 profiling, smoke test and precision benchmark (I-3) — no full training

Order (each step writes a JSON report to `MyDrive/VeriHGN/reports/`):
1. **Gate** — refuses to run without a passing notebook-04 report.
2. **Data loading + graph memory** — Drive read, graph build, collate, host→device per sample (≈ 2 min).
3. **Peak VRAM per design** — one full training step (batch = 1 graph, official) on the largest sample of
   each design in FP32 / FP16 / BF16, with GPU utilisation sampled (≈ 2–5 min).
4. **Smoke test** — read-only inspection of the actual run directory, the checkpoint/resume regression tests
   (seconds, includes a CUDA test), then 2 epochs on the full training split and a 3rd epoch in a **new**
   Trainer that must resume from `<run_dir>/checkpoints/last.pt` on Drive (≈ 5–15 min). If an earlier
   session already left a valid epoch-2 checkpoint, it is reused and only the resume is run.
5. **Precision benchmark** — FP32 vs official FP16-AMP vs BF16-AMP, 20 epochs × 2 seeds, plus a repeatability
   check; validation in FP32, test split never touched (≈ 2–4 h; resumable). **Off by default**: set
   `RUN_PRECISION_BENCHMARK = True` only after the smoke test passed and the benchmark is approved.
6. **Persistence check and stage report** — `reports/STAGE_REPORT.md` collects everything.

Batch size stays 1 graph throughout. Metrics here are diagnostics, not results. Full training is **not**
part of this notebook.
"""),
    code(BOOTSTRAP),
    code(r'''
import json, time, torch, psutil
from pathlib import Path
from verihgn_repro.config import load_config
from verihgn_repro.data.dataset import build_experiment_graphs
from verihgn_repro.data.splits import load_manifest
from verihgn_repro.training.trainer import Trainer
from verihgn_repro.training.checkpointing import RunStatus, inspect_run_dir
from verihgn_repro.training.benchmark import measure_peak_memory, run_precision_benchmark
from verihgn_repro.utils import hardware
from verihgn_repro import reporting
from verihgn_repro.colab import write_json
CFG_DIR = Path(CODE_DIR) / 'configs'
SCOPE = 'official9'
summary = json.load(open(L.reports / '04_validation_summary.json'))
assert all(s['passed'] for s in summary), 'GATE: validation has not passed (run notebook 04)'
PHASH = json.load(open(L.reports / f'03_preprocessing_{SCOPE}.json'))['prep_hash']
PROCESSED = L.processed_for(PHASH)
hw = hardware.detect(); prof = hardware.choose_profile(hw)
print(hw); print(prof)
assert hw.device == 'cuda', 'select an A100 runtime'
'''),
    md("### Primary configuration and graphs (built in RAM from the processed data on Drive)"),
    code(r'''
cfg = load_config(CFG_DIR / 'base.yaml')          # official hyper-parameters; primary variant (D3/D4/D5 defaults)
man = load_manifest(L.splits / f'{SCOPE}__{cfg.experiment.split}.json')
t0 = time.time()
data = build_experiment_graphs(PROCESSED, man, grid_features=cfg.experiment.grid_features,
                               label_norm=cfg.experiment.label_norm, cache_dir=None, prep_hash=PHASH)
print(cfg.run_id(), '| c_max:', data['c_max'], '| samples:', {k: len(v) for k, v in data['graphs'].items()},
      f'| built in {time.time()-t0:.0f}s | RAM available {psutil.virtual_memory().available/1e9:.1f} GB')
'''),
    md("### 2 · Data loading and graph memory"),
    code(r'''
dl = reporting.data_loading_profile(PROCESSED, sum(man['designs'].values(), []), device='cuda', per_design=2)
write_json(dl, L.reports / '05_data_loading.json')
import pandas as pd; display(pd.DataFrame([{k: v for k, v in r.items() if k not in ('nodes_mb', 'edges_mb')} for r in dl]))
'''),
    md("### 3 · Peak VRAM per design (largest sample of each design, one full training step)"),
    code(r'''
mon = hardware.GPUMonitor(interval_s=0.5).start()
rows = measure_peak_memory(cfg, data['graphs'], precisions=tuple(prof.candidate_precisions))
util = mon.stop_and_summary()
pm = {'gpu': hw.name, 'total_vram_gb': hw.total_vram_gb, 'batch_size_graphs': 1, 'rows': rows, 'gpu_util': util,
      'all_steps_finite': all(r['loss_finite'] for r in rows),
      'max_peak_gb': max(r['peak_allocated_gb'] for r in rows)}
write_json(pm, L.reports / '05_peak_memory.json')
display(pd.DataFrame(rows)); print(util, '| max peak', pm['max_peak_gb'], 'GB')
'''),
    md("### 4a · Smoke run directory — read-only inspection (the actual files on Drive)"),
    code(r'''
scfg = load_config(CFG_DIR / 'base.yaml', CFG_DIR / 'smoke.yaml')
scfg.experiment.name += '_smoke'
run_dir = L.runs / scfg.run_id()          # checkpoints live in run_dir / 'checkpoints' / 'last.pt'
pre = inspect_run_dir(run_dir)
print('run directory:', run_dir, '| exists:', pre['exists'])
print('files:', json.dumps(pre['files'], indent=1))
print('status:', json.dumps(pre.get('status'), indent=1, default=str))
print('history epochs:', pre.get('history_epochs'))
for name, c in pre['checkpoints'].items():
    print(name, json.dumps(c, indent=1, default=str))
'''),
    md("### 4b · Checkpoint/resume regression tests (≈ 1 min; the CUDA test runs only here)"),
    code(r'''
import subprocess, sys
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', 'tests/test_checkpoint_resume.py'],
                   cwd=CODE_DIR, capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
assert r.returncode == 0, 'checkpoint/resume tests failed: do not continue'
'''),
    md("### 4c · Smoke test: 2 epochs, then a new Trainer must resume epoch 3 from Drive"),
    code(r'''
ck = pre['checkpoints'].get('last.pt')
if not pre['files']:
    session1 = 'fresh'                    # nothing there yet: run epochs 1-2 now
elif ck and ck['ok'] and ck['epoch'] == 2 and pre.get('history_epochs') == [1, 2]:
    session1 = 'previous_session'         # epochs 1-2 from an earlier Colab session: reuse them, test the resume
else:                                     # anything else is left untouched; a new directory is used
    run_dir = run_dir.with_name(f'{run_dir.name}__rerun_{time.strftime("%Y%m%d_%H%M%S")}')
    session1 = 'fresh'
print('session 1:', session1, '| run directory:', run_dir)
t0 = time.time()
env_session1 = None
if session1 == 'fresh':
    st1 = Trainer(scfg, data['graphs'], run_dir).run(mode='fresh')         # epochs 1-2 (session limit)
else:
    st1 = RunStatus(run_dir).read()
    env_session1 = json.load(open(run_dir / 'env.json')) if (run_dir / 'env.json').exists() else None
mid = inspect_run_dir(run_dir)
assert mid['checkpoints'].get('last.pt', {}).get('ok') and mid['checkpoints']['last.pt']['epoch'] == 2, mid
scfg.training.max_epochs_this_session = None
tr = Trainer(scfg, data['graphs'], run_dir)                                # new object: must resume
st2 = tr.run(mode='resume')                                                # raises if it cannot resume
post = inspect_run_dir(run_dir)
hist = pd.read_csv(run_dir / 'history.csv')
smoke = {'run_dir': str(run_dir), 'session1_source': session1, 'status_after_session1': st1,
         'status_after_resume': st2, 'resume_info': tr.resume_info,
         'checkpoint_after_session1': mid['checkpoints']['last.pt'],
         'checkpoint_after_resume': post['checkpoints'].get('last.pt'),
         'resumed_correctly': (tr.resume_info['resumed'] and tr.resume_info['start_epoch'] == 2
                               and tr.resume_info['checkpoint'] == str(run_dir / 'checkpoints' / 'last.pt')
                               and st2['epoch'] == 3 and hist['epoch'].tolist() == [1, 2, 3]
                               and post['checkpoints']['last.pt']['epoch'] == 3),
         'seconds': round(time.time() - t0), 'history': hist.to_dict('records'),
         'checkpoints': sorted(post['checkpoints']), 'files_before': pre['files'],
         'env_session1': env_session1, 'env': json.load(open(run_dir / 'env.json'))}
write_json(smoke, L.reports / '05_smoke.json')
cols = ['epoch', 'train_loss', 'val_loss', 'val_avg_spearman', 'train_seconds', 'train_peak_allocated_gb',
        'gpu_util_mean', 'grad_norm_mean', 'skipped_nonfinite']
display(hist[[c for c in cols if c in hist.columns]])
print('resume info:', tr.resume_info)
print('resume ok:', smoke['resumed_correctly'])
assert smoke['resumed_correctly']
'''),
    md("### 5 · Precision benchmark — measurement only (≈ 2–4 h, resumable)"),
    code(r'''
RUN_PRECISION_BENCHMARK = False    # set True only after the smoke test passed and the benchmark is approved
if RUN_PRECISION_BENCHMARK:
    bench = run_precision_benchmark(cfg, data['graphs'], L.runs / 'precision_benchmark', epochs=20, seeds=(42, 43),
                                    precisions=tuple(prof.candidate_precisions), repeat_epochs=3)
    write_json(bench, L.reports / '05_precision_benchmark.json')
    display(pd.DataFrame(bench['summary']).T)
    display(pd.DataFrame(bench['repeatability']))
'''),
    md("### 6 · Drive persistence check and stage report"),
    code(r'''
pc = reporting.persistence_check({'processed': PROCESSED, 'splits': L.splits, 'runs': L.runs, 'reports': L.reports})
write_json(pc, L.reports / '05_persistence.json')
print(json.dumps(pc, indent=1))
text = reporting.summarize_reports(L.reports, L.reports / 'STAGE_REPORT.md')
print(text[:4000])
print('\n→ Download MyDrive/VeriHGN/reports/STAGE_REPORT.md (and the JSON files) and share them back.')
'''),
    md("### 7 · Flush Drive before leaving (last cell)\n\nWrites to `/content/drive` are uploaded asynchronously. "
       "`flush_and_unmount` blocks until everything written in this session has reached Drive. Run it before "
       "closing or disconnecting the runtime; re-run the bootstrap cell afterwards if you continue working."),
    code(r'''
from google.colab import drive
drive.flush_and_unmount()
print('Drive flushed and unmounted; all checkpoints and reports written in this session are on Drive.')
'''),
])


NB["06_training_diagnosis"] = notebook([
    md("""
# 06 · Smoke-test training diagnosis (read-only by default)

Background: in the notebook-05 smoke test the gradient norm became exactly 0 after epoch 1, the training loss
was identical in epochs 2 and 3, and every validation correlation was exactly 0. The local investigation
(`docs/SMOKE_TRAINING_DIAGNOSIS.md`) attributes this to **prediction collapse**: every raw output of the
official prediction heads falls below their `clamp(min=0)`, after which the gradient is exactly zero.

* **Part A, read-only, no training (≈ 5–10 min, GPU optional).** Loads the existing smoke checkpoint,
  reports the raw head outputs before the clamp, and compares the logged training loss with the loss of
  the all-zero predictor on all training graphs. Nothing in `runs/` is modified.
* **Part B, short GPU diagnostic, OFF by default (≈ 5–15 min on the A100).** 60 production training steps
  on 6 training graphs in FP32 and in FP16, instrumented step by step. Set `RUN_GPU_DIAGNOSTIC = True`
  only after you have authorised it. It writes only `reports/06_*.json`.

No full training and no precision benchmark is started from this notebook. Notebook 02 is not needed
(only processed data on Drive is read).
"""),
    code(BOOTSTRAP),
    code(r'''
import json, subprocess, sys, time
from pathlib import Path
from verihgn_repro.config import load_config
SCOPE = 'official9'
CFG_DIR = Path(CODE_DIR) / 'configs'
PHASH = json.load(open(L.reports / f'03_preprocessing_{SCOPE}.json'))['prep_hash']
PROCESSED = L.processed_for(PHASH)
cfg = load_config(CFG_DIR / 'base.yaml')
SPLIT = L.splits / f'{SCOPE}__{cfg.experiment.split}.json'
scfg = load_config(CFG_DIR / 'base.yaml', CFG_DIR / 'smoke.yaml'); scfg.experiment.name += '_smoke'
smoke_dirs = sorted(p for p in L.runs.glob(scfg.run_id() + '*') if (p / 'checkpoints' / 'last.pt').exists())
print('processed:', PROCESSED, '\nsplit:', SPLIT, '\nsmoke runs with a checkpoint:', *smoke_dirs, sep='\n  ')

def new_report(name):
    """Never overwrite an existing report: add a timestamp if the name is taken."""
    p = L.reports / name
    return p if not p.exists() else p.with_name(f'{p.stem}__{time.strftime("%Y%m%d_%H%M%S")}{p.suffix}')

def tool(script, *args):
    r = subprocess.run([sys.executable, f'tools/{script}', *map(str, args)], cwd=CODE_DIR, capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-3000:])
    assert r.returncode == 0, f'{script} failed'
'''),
    md("## Part A · Read-only inspection of the existing smoke checkpoint(s)"),
    code(r'''
for run_dir in smoke_dirs:
    out = new_report(f'06_collapse_inspection__{run_dir.name}.json')
    tool('inspect_collapse.py', '--run-dir', run_dir, '--processed', PROCESSED, '--split', SPLIT, '--out', out)
'''),
    md("## Part B · Short instrumented FP32 vs FP16 diagnostic (only when authorised)"),
    code(r'''
RUN_GPU_DIAGNOSTIC = False      # set True only after authorising this short GPU job
if RUN_GPU_DIAGNOSTIC:
    out = new_report('06_step_diagnostic.json')
    tool('diagnose_training.py', '--processed', PROCESSED, '--split', SPLIT, '--out', out,
         '--precisions', 'fp32', 'fp16', '--steps', 60, '--train-samples', 6, '--val-samples', 2,
         '--metric-every', 20, '--device', 'cuda')
else:
    print('Part B skipped (RUN_GPU_DIAGNOSTIC = False)')
print('→ Send back reports/06_*.json')
'''),
])

NB["07_collapse_experiment"] = notebook([
    md("""
# 07 · Controlled prediction-collapse experiment (diagnostic; OFF by default)

Plan and rationale: `docs/COLLAPSE_EXPERIMENT_PLAN.md`. The question is why the 251-graph smoke run collapsed while
the 6-graph Part B diagnostic kept learning.

* **Normal training engine** (`Trainer.run`). Official model, loss, AdamW, clipping, labels and features. The labels
  use the c_max of the full official training split in every run.
* **Runs.** Smoke replicas (full training split, 1 epoch) in FP16 and FP32 × seeds 42/43/44, then 4 training-subset
  arms × 3 seeds in FP32, each about one smoke epoch of optimizer steps (18 runs).
* **Validation.** 6 samples of the official validation design plus 1 held-out sample of each training design (7 designs).
  The test split is never loaded.
* **Outputs.** `runs/collapse_diag/<run>/` (history, per-step `steps.csv`, last/best checkpoints) and
  `reports/07_collapse_experiment*.json`. Existing files are never overwritten; an interrupted run resumes.
* **Estimated A100 time.** About 35–65 min, plus 5–10 min of graph building. This is an estimate, not a measurement.

Cell 1 only plans: it reads the processed data and writes the plan, with no GPU work. Set `RUN_COLLAPSE_EXPERIMENT = True`
in the last cell **only after approval**. Notebook 02 is not needed.
"""),
    code(BOOTSTRAP),
    code(r'''
import json, time
from pathlib import Path
from verihgn_repro.config import load_config
from verihgn_repro.data.splits import load_manifest
from verihgn_repro.diagnostics import collapse_experiment as CE
SCOPE = 'official9'
CFG_DIR = Path(CODE_DIR) / 'configs'
PHASH = json.load(open(L.reports / f'03_preprocessing_{SCOPE}.json'))['prep_hash']
PROCESSED = L.processed_for(PHASH)
cfg = load_config(CFG_DIR / 'base.yaml')                  # official hyper-parameters; primary variant
assert (cfg.experiment.split, cfg.experiment.label_norm, cfg.experiment.grid_features) == \
       ('official_seed42', 'per_dataset', 'paper_eq3')
man = load_manifest(L.splits / f'{SCOPE}__{cfg.experiment.split}.json')
EXPECTED = json.load(open(CFG_DIR / 'collapse_experiment_manifest.json'))   # pre-registered selection and budgets
plan = CE.plan_experiment(PROCESSED, man, expected=EXPECTED)                  # raises if anything differs
val04 = json.load(open(L.reports / f'04_validation_{SCOPE}.json'))
assert abs(plan['c_max'] - val04['c_max'][cfg.experiment.split]) < 1e-9, 'labels differ from the validated ones'

def new_report(name):
    p = L.reports / name
    return p if not p.exists() else p.with_name(f'{p.stem}__{time.strftime("%Y%m%d_%H%M%S")}{p.suffix}')

plan_path = new_report('07_collapse_plan.json')
plan_path.write_text(json.dumps({k: v for k, v in plan.items() if k != 'sample_dirs'}, indent=1))
print('c_max', plan['c_max'], '| single-design arm:', plan['single_design'],
      '| lower-median-sparsity design (reported only):', plan['lower_median_sparsity_design'])
print('budget', plan['budget_steps'], 'steps; common budget for comparisons', plan['common_steps'])
print('validation:')
for n in plan['val']:
    print(f"  {n}  ({CE.design_of(n)})")
for r in plan['runs']:
    ds = len({CE.design_of(n) for n in r['train']})
    print(f"{r['name']:26s} train {len(r['train']):3d} graphs / {ds} designs  epochs {r['epochs']:3d}  steps {r['steps']:4d}  val {len(r['val'])}")
print('plan written to', plan_path)
'''),
    code(r'''
RUN_COLLAPSE_EXPERIMENT = False      # set True only after the experiment is approved
if RUN_COLLAPSE_EXPERIMENT:
    from verihgn_repro.data.graph import build_graph
    from verihgn_repro.data.preprocess import load_sample
    t0 = time.time()
    graphs = {}
    for name, d in plan['sample_dirs'].items():
        a, m = load_sample(Path(d))
        graphs[name] = build_graph(a, m, grid_features=cfg.experiment.grid_features,
                                   label_norm=cfg.experiment.label_norm, c_max=plan['c_max'])
    print(f'{len(graphs)} graphs built in {time.time() - t0:.0f}s')
    report_path = new_report('07_collapse_experiment.json')
    rep = CE.run_plan(plan, cfg, graphs, L.runs / 'collapse_diag', report_path=report_path)
    print(json.dumps(rep['by_arm'], indent=1, default=str))
    print('→ send back', report_path.name, 'and', plan_path.name)
else:
    print('not run (RUN_COLLAPSE_EXPERIMENT = False)')
'''),
])

NB["08_headpath_experiment"] = notebook([
    md("""
# 08 · HEADPATH: train-time cell-head clamp experiment (DIAGNOSTIC DEVIATION; OFF by default)

Plan, decisions and outcomes: `docs/HEADPATH_PLAN.md`. Rationale: `docs/POST_COLLAPSE_FORENSIC_REPORT.md` §11.

> **Not a reproduction.** Arm **U** deviates from the official model: the cell head's `clamp(min=0)` is omitted in
> training forward passes, so the official loss sees raw cell outputs. Arm **O** is the unmodified official model.
> Every evaluation of both arms uses the official clamped forward and the official metrics.

* **Runs.** 12 primary runs (2 arms × 6 independent paired seeds, 42–47) + 2 separate test–retest runs (both arms,
  seed 42; never used for statistical inference) = **14 runs × 490 steps** (2 epochs of the notebook-07 FULL set,
  245 graphs), FP32. Validation: the notebook-07 12-graph / 7-design set, every 49 steps and at step 0.
* **Cell 1 (preflight, CPU only, GPU hidden):** the full test suite, including the 4 real-data tests on a 3-sample
  subset of the raw tree extracted by notebook 02 in this session.
* **Cell 2 (plan, no GPU):** recomputes the plan from the processed data on Drive and refuses to continue if it differs
  from `configs/headpath_manifest.json`, from the notebook-07 manifest, from the notebook-07 plan on Drive, or if the
  labels differ from the notebook-04 report.
* **Cell 3 (run):** off until approved. Content-hashes every protected notebook 05–07 file (smoke and benchmark run
  folders, `runs/collapse_diag/`, the 05–07 reports) before and, always, after, and fails if anything changed. Every
  run is isolated: a failure is recorded (FAILED, with its traceback) and the next run starts; after a runtime-fatal
  error the rest are recorded as NOT_STARTED. The report lists all 14 runs and is rewritten atomically after every
  run. The **last** step always flushes Drive (`flush_and_unmount`) and, by default, remounts it to verify the
  reports. Outputs: `runs/headpath/<run>/` and `reports/08_headpath_*.json`; existing files are never overwritten.
* **Estimated A100 time.** About 55–110 min including graph building (≈ 4–7 min per run), and ≈ 0.7 GB of Drive.
  This is an estimate, not a measurement.

Set `RUN_HEADPATH = True` in the last cell **only after explicit approval**.
"""),
    code(BOOTSTRAP),
    code(r'''
# Preflight (CPU only: CUDA is hidden from the test process, so no GPU work happens here).
import os, subprocess, sys
from pathlib import Path
from verihgn_repro.data import circuitnet as cn
REQUIRE_REAL_DATA_TESTS = True     # the 4 tests in tests/test_real_data.py need the raw tree from notebook 02
env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
if (L.local_raw / 'instance_placement_gcell').exists():
    fx = cn.subset_raw_tree(L.local_raw, Path('/content/real_fixture_tree'))   # symlinks only; nothing copied
    env['VERIHGN_FIXTURES'] = str(fx)
    print('real-data fixture tree:', fx, sorted(p.name for p in (fx / 'instance_placement_gcell').iterdir()))
else:
    print('no raw tree at', L.local_raw, '- run notebook 02 in this session to enable the real-data tests')
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', 'tests'],
                   cwd=CODE_DIR, env=env, capture_output=True, text=True)
print(r.stdout[-6000:], r.stderr[-2000:])
assert r.returncode == 0, 'test suite failed: do not continue'
real_skipped = 'VERIHGN_FIXTURES' in r.stdout
assert not (REQUIRE_REAL_DATA_TESTS and real_skipped), 'the real-data tests were skipped: run notebook 02 first'
print('preflight passed | real-data tests ran:', not real_skipped)
'''),
    code(r'''
import json, time
from pathlib import Path
from verihgn_repro.config import load_config
from verihgn_repro.data.splits import load_manifest
from verihgn_repro.diagnostics import headpath as H
SCOPE = 'official9'
CFG_DIR = Path(CODE_DIR) / 'configs'
PHASH = json.load(open(L.reports / f'03_preprocessing_{SCOPE}.json'))['prep_hash']
PROCESSED = L.processed_for(PHASH)
cfg = load_config(CFG_DIR / 'base.yaml')                  # official hyper-parameters; primary variant
H.check_base_config(cfg)
man = load_manifest(L.splits / f'{SCOPE}__{cfg.experiment.split}.json')
EXPECTED = json.load(open(CFG_DIR / 'headpath_manifest.json'))               # pre-registration
C07 = json.load(open(CFG_DIR / 'collapse_experiment_manifest.json'))         # FULL / validation of notebook 07
plan = H.plan_headpath(PROCESSED, man, expected=EXPECTED, collapse_manifest=C07)   # raises if anything differs
val04 = json.load(open(L.reports / f'04_validation_{SCOPE}.json'))
assert abs(plan['c_max'] - val04['c_max'][cfg.experiment.split]) < 1e-9, 'labels differ from the validated ones'
p07 = L.reports / '07_collapse_plan.json'                 # the plan notebook 07 actually ran with (read only)
if p07.exists():
    p07 = json.load(open(p07))
    assert p07['arms']['FULL'] == plan['train'] and p07['val'] == plan['val'], 'differs from the executed 07 plan'
    assert abs(p07['c_max'] - plan['c_max']) < 1e-9, 'c_max differs from the executed 07 plan'
    print('matches the executed notebook-07 plan (FULL, validation set, c_max)')
missing = [n for n, d in plan['sample_dirs'].items() if not (Path(d) / 'sample.npz').exists()]
assert not missing, f'processed samples missing on Drive: {missing}'
primary = [r for r in plan['runs'] if not r['retest']]
retest = [r for r in plan['runs'] if r['retest']]
assert (len(primary), len(retest), len(plan['runs'])) == (12, 2, 14)
assert sorted({r['seed'] for r in primary}) == list(range(42, 48)) and [(r['arm'], r['seed']) for r in retest] == [('O', 42), ('U', 42)]
assert not plan['pending_research_decisions'], plan['pending_research_decisions']

def new_report(name):                     # never an existing file (timestamp + counter if needed)
    return H.new_report_path(L.reports, name)

plan_path = new_report('08_headpath_plan.json')
plan_path.write_text(json.dumps({k: v for k, v in plan.items() if k != 'sample_dirs'}, indent=1))
print('c_max', plan['c_max'], '| train', len(plan['train']), '| val', len(plan['val']),
      '| steps per run', plan['steps_per_run'], '| evaluations at', plan['eval_steps'])
for r in plan['runs']:
    print(f"{r['name']:28s} arm {r['arm']}  seed {r['seed']}  {'RETEST (excluded from inference)' if r['retest'] else 'primary'}"
          f"  cell_head_train_clamp={r['cell_head_train_clamp']}  steps {r['steps']}")
print('plan written to', plan_path)
'''),
    code(r'''
RUN_HEADPATH = False      # set True only after explicit approval
VERIFY_AFTER_REMOUNT = True   # after flush_and_unmount, remount Drive and re-hash the reports (may show the auth popup)
if RUN_HEADPATH:
    from verihgn_repro import colab
    from verihgn_repro.data.graph import build_graph
    from verihgn_repro.data.preprocess import load_sample

    def build_graphs():
        t0 = time.time()
        graphs = {}
        for name, d in plan['sample_dirs'].items():
            a, m = load_sample(Path(d))
            graphs[name] = build_graph(a, m, grid_features=cfg.experiment.grid_features,
                                       label_norm=cfg.experiment.label_norm, c_max=plan['c_max'])
        print(f'{len(graphs)} graphs built in {time.time() - t0:.0f}s')
        return graphs

    report_path = new_report('08_headpath_experiment.json')
    # execute(): protected-artifact hashes -> graphs -> every run isolated, report after every run -> (always)
    # re-hash + record -> (always, last) Drive flush_and_unmount + verification. Raises at the end on any failure.
    out = H.execute(plan, cfg, L, report_path, build_graphs, flush=lambda paths: colab.flush_drive(paths, remount_verify=VERIFY_AFTER_REMOUNT), verify_paths=[plan_path])
    rep = out['report']
    print(json.dumps({k: rep['analysis'][k] for k in ('verdict', 'run_statuses', 'category_counts', 'primary_test', 'hsig',
                                                      'historical_replication', 'conclusion')}, indent=1, default=str))
    if rep['analysis']['experiment_status'] != 'VALID':
        print('\n' + '!' * 100 + f"\nEXPERIMENT STATUS {rep['analysis']['experiment_status']}: "
              f"{rep['analysis']['status_reasons']}\nFailed runs carry their traceback in the report; re-run this cell to "
              "resume (completed runs are not retrained).\n" + '!' * 100)
    print('Drive was flushed' + (' and remounted' if VERIFY_AFTER_REMOUNT else ' and UNMOUNTED (re-run the bootstrap cell to continue)'))
    print('→ send back', report_path.name, 'and', plan_path.name)
else:
    print('not run (RUN_HEADPATH = False)')
'''),
])

NB["09_results_export"] = notebook([
    md("""
# 09 · Results export for the offline dashboard (Stage 1; CPU only, read-only on the experiment)

Exports the **HEADPATH** results (notebook 08) and, separately, the **notebook-07 collapse experiment** context into
one validated bundle for the offline dashboard (`docs/DASHBOARD_PLAN.md`, `docs/DASHBOARD_EXPORT.md`).

* **Read-only.**
  * The exporter only reads `reports/`, `runs/headpath/`, `runs/collapse_diag/` and the 12 processed validation
    samples. `checkpoints/last.pt` is read for the validity checks only.
  * Every source file is content-hashed before and after; any change fails the export.
  * Nothing is written under `runs/`, `reports/` or `processed/`.
* **Validated.**
  * Every stored HEADPATH outcome is re-derived from the CSVs, and the McNemar test from the stored categories.
  * Labels are checked against the processed samples, grid metrics are recomputed from the full prediction maps, and
    every exported array is decoded again and compared bit for bit.
  * The recorded 0.10 vs implemented 0.08 / 0.12 P2 threshold inconsistency is **flagged**; the historical report
    is not modified.
* **No GPU, no training.** Runtime: a CPU runtime is enough. Estimate ≈ 10–25 min, mostly hashing about 1 GB of
  Drive files twice. Notebook 02 is not needed.
* **Output:** `MyDrive/VeriHGN/export/headpath_dashboard_<timestamp>.zip`, plus its integrity report (`.md` and
  `.json`). All values in the bundle come from the experiment files. This is validation data only; no test-set
  results exist.
"""),
    code(BOOTSTRAP),
    code(r'''
import json, shutil, time
from pathlib import Path
from verihgn_repro.export import dashboard_bundle as DB
REPORT = '08_headpath_experiment.json'      # the authoritative HEADPATH report (other 08 versions are listed, not used)
CONTEXT07_REPORT = None                     # None: the notebook-07 report named in reports/07_forensic_audit.json
try:
    LOCAL_EXPORT_ROOT
except NameError:
    LOCAL_EXPORT_ROOT = '/content/dashboard_export'    # local VM disk: the bundle is built here, then copied to Drive
out = Path(LOCAL_EXPORT_ROOT) / f'headpath_dashboard_{time.strftime("%Y%m%d_%H%M%S")}'
t0 = time.time()
try:
    res = DB.export(L.root, out, report_name=REPORT, context07_report=CONTEXT07_REPORT)
except DB.ExportValidationError as e:
    failed_md = out / 'INTEGRITY_REPORT.md'
    if failed_md.exists():                  # keep the evidence on Drive (export/ only), then stop
        for f, suffix in ((failed_md, '_FAILED_INTEGRITY_REPORT.md'), (out / 'integrity_report.json', '_FAILED_integrity_report.json')):
            DB.copy_verified(f, L.export, name=out.name + suffix)
        print(failed_md.read_text()[:8000])
    raise
print(f"export {res['status']} in {time.time() - t0:.0f}s | checks {res['summary']} | zip {res['zip_bytes'] / 1e6:.1f} MB")
print(Path(res['integrity_report']).read_text()[:8000])
'''),
    code(r'''
# Copy the zip and its integrity report to Drive (export/ only), verify the copies, then flush Drive (last step).
from verihgn_repro import colab
copies = [DB.copy_verified(res['zip'], L.export),
          DB.copy_verified(out / 'INTEGRITY_REPORT.md', L.export, name=out.name + '_INTEGRITY_REPORT.md'),
          DB.copy_verified(out / 'integrity_report.json', L.export, name=out.name + '_integrity_report.json')]
v = DB.verify_bundle(copies[0]['path'])
assert v['status'] == 'PASSED', v
print(json.dumps({'zip': copies[0], 'verify_on_drive': v}, indent=1))
persistence = colab.flush_drive([c['path'] for c in copies])        # LAST: nothing is written to Drive after this
print('drive persistence:', persistence)
print('\n→ Download', Path(copies[0]['path']).name, '(sha256', copies[0]['sha256'][:16] + '…) and',
      Path(copies[1]['path']).name, 'from MyDrive/VeriHGN/export/, and share the integrity report.')
'''),
])
# notebook 09 needs no GPU: request a CPU runtime (the shared helper asks for an A100)
NB["09_results_export"]["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}

def main():
    OUT.mkdir(exist_ok=True)
    for name, nb in NB.items():
        (OUT / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
        print("wrote", OUT / f"{name}.ipynb")


if __name__ == "__main__":
    main()
