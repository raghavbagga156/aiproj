# Incident: notebook 05 smoke test could not resume (2026-10-19)

## What was observed (A100 session, reported by the user)

1. Smoke run `verihgn_ispd15_smoke__official_seed42__paper_eq3__per_dataset__fp16__seed42`: epochs 1 and 2 completed.
2. A new `Trainer` for epoch 3 failed in `Trainer._restore → set_rng_state → torch.set_rng_state` with
   `TypeError: RNG state must be a torch.ByteTensor`.
3. A separate diagnostic checked `MyDrive/VeriHGN/runs/<run_id>/last.pt`, reported `exists: False`, and `torch.load` raised `FileNotFoundError`.

## Root cause 1: the RNG error

- **Where.** `Trainer.run` loaded the checkpoint with `torch.load(..., map_location=self.device)`. On Colab that device is `cuda`.
- **What `map_location` does.** It moves *every* tensor in the file to the GPU. That includes three tensors that were saved on the CPU:
  - `s["rng"]["torch"]`, the CPU generator state: a `torch.ByteTensor` of 5056 bytes on `cpu` when saved
  - each entry of `s["rng"]["cuda"]`, which `torch.cuda.get_rng_state_all()` also returns as CPU ByteTensors
  - `s["shuffle_gen"]`
- **Why it fails.** After loading, `s["rng"]["torch"]` was a `torch.cuda.ByteTensor`. `torch.set_rng_state` requires a CPU uint8 tensor, so it raised. CUDA generators require the same.
- **What was not wrong.** The state was saved correctly and the dtype was correct (uint8). Only the device was wrong.
- **Why tests and the dry run missed it.** Both ran on the CPU, where `map_location="cpu"`.
- **Reproduced locally.** Loading a saved state with a non-CPU `map_location` and calling `torch.set_rng_state` raises the identical message.

## Root cause 2: the missing-`last.pt` diagnostic

- **Where checkpoints are written.** `CheckpointManager` writes to `<run_dir>/checkpoints/last.pt` (plus `best.pt`), never `<run_dir>/last.pt`. The diagnostic inspected a path that is never written, so it reports "missing" whatever the state on Drive.
- **Evidence the checkpoint existed.** The RNG traceback itself proves it. `_restore` is only called after `load_latest` returned a deserialised state.
- **Which file was loaded.** With `keep_every: 10` and 2 epochs, no `epoch_*.pt` exists, so the file loaded was `checkpoints/last.pt`.
- **The two symptoms are independent.** One is a wrong device in restoration; the other is a wrong path in an ad-hoc check.

## Recoverability of epochs 1–2

- **Nothing was overwritten by the failed resume.** It raised before any write: `config.yaml`, `env.json`, `status.json` and the checkpoints are only written after a successful restore.
- **Expected state on Drive:**
  - `checkpoints/last.pt` (epoch 2) and `checkpoints/best.pt`
  - `history.csv` with epochs 1–2
  - `status.json` with `state: running`, `epoch: 2`, `session_ended: true`
- **To be confirmed on Colab.** Notebook 05 cell 4a lists the actual files and loads each checkpoint, reading only.
- **Cross-version check (CPU).** A checkpoint written by the *previous* bundle was resumed by the fixed code through the new notebook cells, and the resume succeeded. The save format did not change.

## Fix

| File | Change |
|---|---|
| `training/checkpointing.py` | `load_checkpoint()` always deserialises onto the CPU. `load_state_dict` moves model and optimizer state to the device. It verifies the zip CRCs and the required keys. Errors are typed: `CheckpointNotFoundError` and `CheckpointCorruptError`, each naming the path. `atomic_torch_save` verifies the temporary file before renaming, fsyncs the directory where supported, and checks the size after the rename. `load_for_resume()` is strict: a missing or corrupt `last.pt` raises and lists the older checkpoints; it falls back only with `allow_fallback=True`. `inspect_run_dir()` is a read-only diagnostic of the real run directory. |
| `utils/seed.py` | `set_rng_state` validates every generator state. A GPU copy is moved back losslessly; anything that is not uint8 raises a named error. CUDA states must match the device count. It returns what was restored. |
| `training/trainer.py` | `run(mode=...)`:<br>• `auto` resumes when checkpoints exist, and refuses to start fresh if `history.csv` records finished epochs but no checkpoint exists.<br>• `resume` must resume.<br>• `fresh` refuses to write over existing checkpoints.<br>A checkpoint from a different `run_id` is rejected. `resume_info` records the mode, the checkpoint path, the start epoch and the RNG restore, and is written to `status.json`. `test()` uses the same loader. |
| notebook 05 | **4a** inspects the actual run directory (read-only).<br>**4b** runs the checkpoint/resume regression tests, including the CUDA test.<br>**4c** reuses a valid epoch-2 checkpoint from an earlier session, or runs epochs 1–2 fresh. The resume uses `mode='resume'`, and the cell checks the resumed checkpoint path. If the directory is in any other state it is left untouched and a new `__rerun_<time>` directory is used.<br>The precision benchmark is now off by default. A final cell runs `drive.flush_and_unmount()`. |

The methodology, configuration, model, loss and metrics are unchanged.

## Determinism

- **CPU.** An interrupted run (2 epochs, then a new Trainer) equals an uninterrupted 3-epoch run bit for bit: parameters, RNG states and every history column (`tests/test_checkpoint_resume.py`).
- **Negative control.** Disabling the RNG restore makes the runs diverge, so the equality really depends on the restore.
- **GPU.** Scatter-add is not bit-deterministic on the GPU, so interrupted and uninterrupted runs can differ at roundoff level even with exact state restoration. The precision benchmark's repeatability check measures the size of that floor. The CUDA test asserts exact restoration of the RNG and optimizer state, not parameter equality.
