"""Resumable training loop around the unmodified official model, loss and metrics.

Optimisation follows official train.py exactly: AdamW, StepLR stepped once per
epoch, batch size 1 graph shuffled each epoch, gradient clipping at 1.0,
non-finite losses skipped, early stopping on mean validation Spearman.
Engineering additions: precision modes, atomic checkpoints with full state on
Drive, resume, heartbeat status, per-epoch history with timing and VRAM.
Validation and test always run in FP32 (official test.py uses FP32).
"""
import contextlib
import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from .. import official
from ..config import Config, load_config, save_config
from ..evaluation import head_metrics as HM
from ..evaluation.evaluate import breakdown, loss_of, metrics_of, predict, prediction_health, save_predictions
from ..model_deviations import PARTITIONS, apply_cell_head_train_unclamped, parameter_partition
from ..utils import hardware
from ..utils.seed import as_cpu_byte_tensor, rng_state, seed_everything, set_rng_state
from .checkpointing import (CheckpointError, CheckpointManager, CheckpointNotFoundError, RunStatus,
                            atomic_write_json, load_checkpoint)

RESUME_MODES = ("auto", "resume", "fresh")
# Keys that may differ between the session that wrote a checkpoint and the one resuming it: session length and
# storage locations. Everything else is training-relevant and must be identical.
RESUME_IGNORED_KEYS = ("training.max_epochs_this_session",)
RESUME_IGNORED_PREFIXES = ("paths.",)


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def resume_config_diff(saved, cfg: Config) -> list:
    """Configuration keys in which `cfg` differs from the config stored in a checkpoint. The stored config is
    re-loaded through load_config, so keys added since it was written take their defaults."""
    if not isinstance(saved, dict):
        return ["<checkpoint has no stored configuration>"]
    try:
        old = load_config(overrides=saved)
    except (KeyError, AssertionError, TypeError, ValueError) as e:
        return [f"<stored configuration cannot be loaded: {type(e).__name__}: {e}>"]
    a, b = _flat(old.to_dict()), _flat(cfg.to_dict())
    return sorted(k for k in set(a) | set(b) if k not in RESUME_IGNORED_KEYS
                  and not k.startswith(RESUME_IGNORED_PREFIXES) and a.get(k) != b.get(k))


def build_model(cfg: Config):
    m = cfg.model
    model = official.VeriHGN(cell_feat_dim=m.cell_feat_dim, net_feat_dim=m.net_feat_dim,
                             grid_feat_dim=cfg.grid_feat_dim, pin_feat_dim=m.pin_feat_dim,
                             geom_feat_dim=m.geom_feat_dim, hidden_dim=m.hidden_dim,
                             num_layers=m.num_layers, dropout=m.dropout, use_checkpoint=m.use_checkpoint)
    if not m.cell_head_train_clamp:     # HEADPATH arm U (diagnostic deviation): no new modules, no RNG use
        apply_cell_head_train_unclamped(model)
    return model


def _autocast(precision: str, device: torch.device):
    if precision == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


class _LazyBatches(dict):
    """part -> sequence of single-graph batches, collated on access (no second copy in RAM)."""

    def __init__(self, graphs_by_split, collate):
        super().__init__({p: _CollatedView(gs, collate) for p, gs in graphs_by_split.items()})


class _CollatedView:
    def __init__(self, graphs, collate):
        self.graphs, self.collate = graphs, collate

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.collate([self.graphs[i]])

    def __iter__(self):
        for g in self.graphs:
            yield self.collate([g])


def _tensor_bytes(batch) -> int:
    total = 0
    for store in batch.stores:
        for v in store.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


class Trainer:
    def __init__(self, cfg: Config, graphs_by_split: dict, run_dir, device=None, extra_meta: dict | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt = CheckpointManager(self.run_dir, keep_every=cfg.training.keep_every)
        self.status = RunStatus(self.run_dir)
        self.extra_meta = extra_meta or {}
        self.collate = official.dataset_module().collate_fn
        self.graphs = graphs_by_split
        self.graphs_by_name = {g.placement_name: g for gs in graphs_by_split.values() for g in gs}
        self.batches = _LazyBatches(graphs_by_split, self.collate)
        self._place_batches()

    # ------------------------------------------------------------------ setup
    def _place_batches(self):
        """Batches are collated per step (as the official DataLoader does). On a
        GPU with enough free memory the whole dataset is collated once and kept
        resident, which removes per-step host-to-device copies."""
        mode = self.cfg.training.keep_dataset_on_gpu
        self.resident = False
        self.resident_bytes = sum(_tensor_bytes(g) for gs in self.graphs.values() for g in gs)
        if self.device.type != "cuda" or mode == "false" or mode is False:
            return
        free, _total = torch.cuda.mem_get_info()
        if (mode in ("true", True)) or self.resident_bytes < 0.35 * free:
            self.batches = {p: [self.collate([g]).to(self.device) for g in gs] for p, gs in self.graphs.items()}
            self.resident = True

    def _init_state(self):
        t = self.cfg.training
        torch.backends.cuda.matmul.allow_tf32 = bool(t.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(t.allow_tf32)
        seed_everything(t.seed)
        self.model = build_model(self.cfg).to(self.device)
        self._partition = parameter_partition(self.model) if t.step_log_heads else None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=t.step_size, gamma=t.gamma)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(t.precision == "fp16" and self.device.type == "cuda"))
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(t.seed)
        self.epoch = 0
        self.best = -math.inf
        self.patience_counter = 0

    def _state(self) -> dict:
        return {"epoch": self.epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                "best_avg_spearman": self.best, "patience_counter": self.patience_counter,
                "rng": rng_state(), "shuffle_gen": self.shuffle_gen.get_state(),
                "config": self.cfg.to_dict(), "run_id": self.cfg.run_id()}

    def _restore(self, s: dict) -> dict:
        """`s` comes from load_checkpoint (all tensors on the CPU); load_state_dict
        moves model/optimizer state to the training device."""
        if s["run_id"] != self.cfg.run_id():
            raise CheckpointError(f"checkpoint belongs to run {s['run_id']!r}, not {self.cfg.run_id()!r}")
        diff = resume_config_diff(s.get("config"), self.cfg)
        if diff:
            raise CheckpointError(f"refusing to resume {self.run_dir}: the training configuration differs from the "
                                  f"checkpoint's in {diff}")
        self.model.load_state_dict(s["model"])
        self.optimizer.load_state_dict(s["optimizer"])
        self.scheduler.load_state_dict(s["scheduler"])
        self.scaler.load_state_dict(s["scaler"])
        self.epoch = s["epoch"]
        self.best = s["best_avg_spearman"]
        self.patience_counter = s["patience_counter"]
        restored = set_rng_state(s["rng"])
        self.shuffle_gen.set_state(as_cpu_byte_tensor(s["shuffle_gen"], "shuffle_gen"))
        return restored

    # ------------------------------------------------------------------ epochs
    def _train_epoch(self) -> dict:
        t = self.cfg.training
        lf = official.loss_module().verihgn_loss
        unclamped = not self.cfg.model.cell_head_train_clamp       # HEADPATH arm U: training cell output is raw
        self.model.train()
        order = torch.randperm(len(self.batches["train"]), generator=self.shuffle_gen).tolist()
        n_train = len(self.batches["train"])
        total, n, skipped = 0.0, 0, 0
        recs, gnorms = [], []
        self._step_rows, self._eval_rows, raw, hooks = [], [], {}, []
        if t.step_log:      # read-only observation of the raw head outputs (before the official clamp)
            for name, head in (("cell", self.model.cell_head), ("grid", self.model.grid_head)):
                hooks.append(head.mlp.register_forward_hook(
                    lambda m, inp, o, name=name: raw.__setitem__(name, o.detach().float().squeeze(-1))))

        def after_step(k):  # observation only: eval-mode validation every eval_every_steps global steps
            gstep = (self.epoch - 1) * n_train + k
            if t.eval_every_steps and gstep % t.eval_every_steps == 0:
                self._eval_rows.append(self._step_eval(gstep))
        try:
            for k, i in enumerate(order, 1):
                data = self.batches["train"][i].to(self.device)
                self.optimizer.zero_grad()
                with _autocast(t.precision, self.device):
                    out = self.model(data)
                    cell_pred = out["cell_pred"].float()
                    grid_pred = out["grid_pred"].float()
                    ld = lf(cell_pred, data["cell"].y, grid_pred, data["grid_l0"].y,
                            alpha=t.wmse_alpha, lambda_var=t.lambda_var)
                    loss = ld["total"]
                scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
                if not torch.isfinite(loss):
                    skipped += 1
                    self.optimizer.zero_grad()
                    if t.step_log:
                        self._log_step(i, data, float("nan"), None, scale_before, raw)
                    after_step(k)
                    continue
                pnorms = None
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self._partition is not None:
                        pnorms = self._partition_norms()
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=t.grad_clip)
                    gnorms.append(float(gn))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self._partition is not None:
                        pnorms = self._partition_norms()
                    gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=t.grad_clip)
                    gnorms.append(float(gn))
                    self.optimizer.step()
                total += loss.item()
                n += 1
                if t.step_log:
                    extra = (self._head_terms(cell_pred, grid_pred, data, raw, ld) | pnorms) if t.step_log_heads \
                        else None
                    self._log_step(i, data, float(loss), float(gn), scale_before, raw, extra)
                if t.compute_train_metrics and torch.isfinite(cell_pred).all():
                    rec = {"cell_pred": cell_pred.detach().cpu(), "cell_gt": data["cell"].y.cpu(),
                           "grid_pred": grid_pred.detach().cpu(), "grid_gt": data["grid_l0"].y.cpu()}
                    if unclamped:   # comparable training metrics use clamped predictions; raw kept separately
                        rec["cell_pred_raw"] = rec["cell_pred"]
                        rec["cell_pred"] = rec["cell_pred_raw"].clamp(min=0)
                    recs.append(rec)
                after_step(k)
        finally:
            for h in hooks:
                h.remove()
        res = {"train_loss": total / n if n else float("nan"), "skipped_nonfinite": skipped}
        if gnorms:
            g = torch.tensor(gnorms)
            fin = g[torch.isfinite(g)]
            res.update({"grad_norm_mean": float(fin.mean()) if fin.numel() else float("nan"),
                        "grad_norm_max": float(fin.max()) if fin.numel() else float("nan"),
                        "grad_norm_nonfinite": int((~torch.isfinite(g)).sum()),
                        "grad_clipped_frac": float((g > t.grad_clip).float().mean()),
                        # steps whose whole gradient was exactly zero (all predictions clamped to 0)
                        "grad_zero_steps": int((g == 0).sum()), "train_steps": int(g.numel())})
        if recs:
            m = metrics_of(recs)
            res.update({f"train_cell_{k}": v for k, v in m["cell"].items()})
            res.update({f"train_grid_{k}": v for k, v in m["grid"].items()})
            if unclamped:
                mr = metrics_of([{**r, "cell_pred": r["cell_pred_raw"]} for r in recs])
                res.update({f"train_cell_raw_{k}": v for k, v in mr["cell"].items()})
        return res

    # ------------------------------------------------------------------ HEADPATH observation (read-only)
    def _partition_norms(self) -> dict:
        """Pre-clip gradient L2 norm of the cell-only / grid-only / shared parameters (no state change)."""
        with torch.no_grad():
            parts = {p: [] for p in PARTITIONS}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    parts[self._partition[name]].append(torch.linalg.vector_norm(p.grad.detach().float()))
            return {f"grad_norm_{k}": float(torch.linalg.vector_norm(torch.stack(v))) if v else 0.0
                    for k, v in parts.items()}

    def _head_terms(self, cell_pred, grid_pred, data, raw, ld) -> dict:
        """Per-head loss terms and the three training-time cell streams, recomputed from detached
        tensors with the official loss functions. '_loss' = as entered the training loss (arm U: raw
        cell output); '_clamped' = clamp(cell_pred, min=0), the official prediction; '_raw' = the
        pre-clamp head output; '_zero' = the all-zero predictor on this graph."""
        L, t = official.loss_module(), self.cfg.training
        with torch.no_grad():
            cy, gy = data["cell"].y.float(), data["grid_l0"].y.float()
            cp, gp = cell_pred.detach(), grid_pred.detach()
            cc, craw = cp.clamp(min=0), raw["cell"].to(cp.device)
            zc, zg = torch.zeros_like(cy), torch.zeros_like(gy)
            row = {"cell_wmse_loss": ld["cell_wmse"], "grid_wmse_loss": ld["grid_wmse"], "var_reg_loss": ld["var_reg"],
                   "var_cell_loss": float(L.variance_reg(cp, cy)), "var_grid_loss": float(L.variance_reg(gp, gy)),
                   "cell_wmse_clamped": float(L.weighted_mse(cc, cy, t.wmse_alpha)),
                   "cell_wmse_raw": float(L.weighted_mse(craw, cy, t.wmse_alpha)),
                   "cell_wmse_zero": float(L.weighted_mse(zc, cy, t.wmse_alpha)),
                   "grid_wmse_zero": float(L.weighted_mse(zg, gy, t.wmse_alpha)),
                   "var_cell_zero": float(L.variance_reg(zc, cy)), "var_grid_zero": float(L.variance_reg(zg, gy))}
            for name, s in (("cell_loss", cp), ("cell_clamped", cc), ("cell_raw", craw), ("grid_pred", gp)):
                st = HM.stream_stats(s)
                row.update({f"{name}_std": st["std"], f"{name}_zero_frac": st["zero_frac"],
                            f"{name}_mean": st["mean"]})
        return row

    def _step_eval(self, gstep: int) -> dict:
        """Eval-mode validation at one global step (evals.csv). Uses the official eval path; consumes
        no RNG; always restores the previous train/eval mode."""
        was_training = self.model.training
        try:
            recs = HM.predict_with_raw(self.model, self.batches["val"], self.device)
            row = {"epoch": self.epoch, "global_step": gstep,
                   "val_loss": loss_of(recs, self.cfg.training.wmse_alpha, self.cfg.training.lambda_var),
                   **HM.eval_summary(recs, self.graphs_by_name)}
        finally:
            self.model.train(was_training)
        return row

    def _log_step(self, i, data, loss, gn, scale_before, raw, extra=None):
        """One steps.csv row (training.step_log). Observation only: no RNG, no state change."""
        name = data.placement_name[0] if isinstance(data.placement_name, list) else data.placement_name
        scale_after = self.scaler.get_scale() if self.scaler.is_enabled() else None
        row = {"step_in_epoch": len(self._step_rows) + 1, "graph_index": i, "sample": name, "loss": loss,
               "grad_norm": gn, "grad_exactly_zero": None if gn is None else gn == 0.0,
               "clipped": None if gn is None else gn > self.cfg.training.grad_clip,
               "scale_before": scale_before, "scale_after": scale_after,
               "step_skipped_by_scaler": (scale_after < scale_before) if scale_before is not None else False}
        for k, r in raw.items():
            row[f"{k}_raw_le0"] = float((r <= 0).float().mean())
            row[f"{k}_raw_max"] = float(r.max())
        if extra:
            row.update(extra)
        self._step_rows.append(row)

    def _evaluate(self, part: str):
        recs = predict(self.model, self.batches[part], self.device)
        self.last_eval_health = prediction_health(recs)
        return recs, loss_of(recs, self.cfg.training.wmse_alpha, self.cfg.training.lambda_var), metrics_of(recs)

    # ------------------------------------------------------------------ main
    def _prior_progress(self) -> int:
        """Epochs this run directory records as finished (history.csv rows are written
        only after an epoch completes; status.json is written when one starts)."""
        hist = self.run_dir / "history.csv"
        if not hist.exists():
            return 0
        with open(hist, newline="") as f:
            return max([0] + [int(r["epoch"]) for r in csv.DictReader(f)])

    def resolve_start(self, mode: str = "auto", allow_fallback: bool = False):
        """Decide fresh vs resume without touching anything on disk.
          auto   - resume if the run directory has a checkpoint, else start fresh; a
                   directory that records finished epochs but has no checkpoint is an error
          resume - a checkpoint must exist and load, else raise
          fresh  - the run directory must not contain checkpoints (never overwritten)
        Returns (state_or_None, path_or_None)."""
        if mode not in RESUME_MODES:
            raise ValueError(f"mode must be one of {RESUME_MODES}, got {mode!r}")
        self._archive_prior_attempt = False
        if mode == "fresh":
            if self.ckpt.has_any():
                raise FileExistsError(f"{self.ckpt.ckpt_dir} already holds checkpoints; refusing to start a fresh "
                                      "run over them (use another run directory, or mode='resume')")
            self._archive_prior_attempt = bool(self._prior_attempt_files())
            return None, None
        if mode == "auto" and not self.ckpt.has_any():
            done = self._prior_progress()
            prev = self.status.read() or {}
            # An attempt that logged at most its first epoch but never wrote a checkpoint (interrupted before or
            # between epoch 1's history write and its checkpoint) is archived, not overwritten, and the run starts
            # again. More logged epochs, or a 'completed' status, without a checkpoint mean lost checkpoints.
            if done > 1 or (done == 1 and prev.get("state") in ("completed", "early_stopped")):
                raise CheckpointNotFoundError(
                    f"{self.run_dir} records {done} finished epoch(s) but {self.ckpt.ckpt_dir} has no checkpoint; "
                    "refusing to restart silently from scratch")
            self._archive_prior_attempt = bool(self._prior_attempt_files())
            return None, None
        return self.ckpt.load_for_resume(allow_fallback=allow_fallback)

    PRIOR_ATTEMPT_FILES = ("history.csv", "steps.csv", "evals.csv", "status.json", "config.yaml")

    def _prior_attempt_files(self) -> list:
        return [n for n in self.PRIOR_ATTEMPT_FILES if (self.run_dir / n).exists()]

    def _archive_attempt(self) -> str:
        """Move (never delete) the logs of an attempt that wrote no checkpoint into aborted_attempt_<ts>/."""
        base = self.run_dir / f"aborted_attempt_{time.strftime('%Y%m%d_%H%M%S')}"
        dest, k = base, 1
        while dest.exists():
            dest, k = base.with_name(f"{base.name}_{k}"), k + 1
        dest.mkdir(parents=True)
        for n in self._prior_attempt_files():
            os.replace(self.run_dir / n, dest / n)
        return str(dest)

    def _write_env(self, hw: dict) -> None:
        """env.json is the environment of the FIRST session in this run directory and is never overwritten;
        every session (first included) also gets its own env_session_NNN.json."""
        n = 1 + len(list(self.run_dir.glob("env_session_*.json")))
        atomic_write_json(hw, self.run_dir / f"env_session_{n:03d}.json")
        if not (self.run_dir / "env.json").exists():
            atomic_write_json(hw, self.run_dir / "env.json")

    def run(self, fresh: bool = False, mode: str | None = None, allow_fallback: bool = False) -> dict:
        """mode: 'auto' (default), 'resume' or 'fresh' (see resolve_start);
        `fresh=True` is the old spelling of mode='fresh'."""
        t = self.cfg.training
        mode = mode or ("fresh" if fresh else "auto")
        s, path = self.resolve_start(mode, allow_fallback)
        archived = self._archive_attempt() if s is None and self._archive_prior_attempt else None
        if archived:
            print(f"[run] earlier attempt without a checkpoint moved to {archived}; starting again")
        hw = hardware.env_report({"scatter_impl": official.scatter_impl(),
                                  "official_commit": official.official_commit(),
                                  "dataset_resident_on_gpu": self.resident})
        hw.update(self.extra_meta)
        self._init_state()
        self.resume_info = {"mode": mode, "resumed": s is not None, "checkpoint": str(path) if path else None,
                            "start_epoch": 0}
        if archived:
            self.resume_info["archived_prior_attempt"] = archived
        if s is not None:
            self.resume_info["rng_restored"] = self._restore(s)
            self.resume_info["start_epoch"] = self.epoch
            print(f"[resume] {path} -> continuing after epoch {self.epoch}")
        else:
            print(f"[run] fresh start in {self.run_dir}")
        prev = self.status.read()
        if prev and prev["state"] in ("completed", "early_stopped") and mode != "fresh":
            print(f"[run] already {prev['state']}; nothing to train")
            return prev
        if s is None or not (self.run_dir / "config.yaml").exists():
            save_config(self.cfg, self.run_dir / "config.yaml")   # on resume the config was verified equal (_restore)
        self._write_env(hw)
        hist_path = self.run_dir / "history.csv"
        session_epochs = 0
        state = "running"
        try:
            if t.eval_every_steps and s is None:     # step-0 baseline, fresh start only (never repeated on resume)
                self._append_rows(self.run_dir / "evals.csv", 0, [self._step_eval(0)])
            while self.epoch < t.num_epochs:
                if t.max_epochs_this_session and session_epochs >= t.max_epochs_this_session:
                    break
                self.epoch += 1
                session_epochs += 1
                self.status.write("running", epoch=self.epoch, best_avg_spearman=self.best, run_id=self.cfg.run_id(),
                                  last_session=self.resume_info)
                hardware.cuda_peak_gb(reset=True)
                mon = hardware.GPUMonitor().start()
                t0 = time.time()
                row = {"epoch": self.epoch, "lr": self.optimizer.param_groups[0]["lr"]}
                row.update(self._train_epoch())
                row["train_seconds"] = time.time() - t0
                row.update({f"train_{k}": v for k, v in hardware.cuda_peak_gb(reset=True).items()})
                t1 = time.time()
                _, vloss, vm = self._evaluate("val")
                row["val_seconds"] = time.time() - t1
                row.update({f"val_{k}": v for k, v in hardware.cuda_peak_gb().items()})
                row.update(mon.stop_and_summary())
                self.scheduler.step()
                row["val_loss"] = vloss
                row.update({f"val_{k}": v for k, v in self.last_eval_health.items()})
                if vm:
                    row.update({f"val_cell_{k}": v for k, v in vm["cell"].items()})
                    row.update({f"val_grid_{k}": v for k, v in vm["grid"].items()})
                    row["val_avg_spearman"] = vm["avg_spearman"]
                row["grad_scale"] = self.scaler.get_scale() if self.scaler.is_enabled() else None
                is_best = bool(vm) and vm["avg_spearman"] > self.best
                if is_best:
                    self.best = vm["avg_spearman"]
                    self.patience_counter = 0
                elif vm:
                    self.patience_counter += 1
                row["best_avg_spearman"] = self.best
                self._append_history(hist_path, row)
                if t.step_log:
                    n_train = len(self.batches["train"])
                    self._append_rows(self.run_dir / "steps.csv", self.epoch,
                                      [{"epoch": self.epoch, "global_step": (self.epoch - 1) * n_train + r["step_in_epoch"],
                                        **r} for r in self._step_rows])
                if t.eval_every_steps:               # a resumed epoch replaces its own rows, as steps.csv
                    self._append_rows(self.run_dir / "evals.csv", self.epoch, self._eval_rows)
                self.ckpt.save(self._state(), self.epoch, is_best)
                print(f"epoch {self.epoch:3d} loss {row['train_loss']:.4f} val_loss {vloss:.4f} "
                      f"val_avgSpearman {row.get('val_avg_spearman', float('nan')):.4f} best {self.best:.4f} "
                      f"({row['train_seconds']:.0f}s train, peak {row.get('train_peak_allocated_gb', 0):.1f} GB)")
                if row.get("train_steps") and row.get("grad_zero_steps") == row["train_steps"]:
                    print(f"[warn] epoch {self.epoch}: every training step had an exactly-zero gradient and "
                          f"{row.get('val_grid_pred_zero_frac', float('nan')):.1%} of validation grid predictions are 0 "
                          "(prediction collapse: all head outputs below the official clamp(min=0)); "
                          "see docs/SMOKE_TRAINING_DIAGNOSIS.md")
                if self.patience_counter >= t.patience:
                    state = "early_stopped"
                    break
            else:
                state = "completed"
        except Exception as e:
            self.status.write("failed", epoch=self.epoch, error=f"{type(e).__name__}: {e}")
            raise
        if state == "running":
            # session limit reached: leave as resumable, heartbeat goes stale
            self.status.write("running", epoch=self.epoch, best_avg_spearman=self.best, session_ended=True)
        else:
            self.status.write(state, epoch=self.epoch, best_avg_spearman=self.best)
        return self.status.read()

    def _append_history(self, path: Path, row: dict):
        self._append_rows(path, row["epoch"], [row])

    @staticmethod
    def _append_rows(path: Path, epoch: int, new_rows: list):
        """Rewrite a per-epoch CSV atomically: keep rows of earlier epochs (a resumed
        epoch replaces its own rows), append `new_rows`."""
        rows = []
        if path.exists():
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < epoch]
        rows.extend(new_rows)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)

    # ------------------------------------------------------------------ test
    def test(self, save_preds: bool = True) -> dict:
        """Evaluate best.pt on the test split in FP32 and write eval/ artefacts."""
        s = load_checkpoint(self.ckpt.best_path)
        if not hasattr(self, "model"):
            self._init_state()
        self.model.load_state_dict(s["model"])
        recs, loss, m = self._evaluate("test")
        result = {"run_id": self.cfg.run_id(), "best_epoch": s["epoch"], "test_loss": loss, "overall": m,
                  **breakdown(recs), "precision_eval": "fp32", "provenance": "measured (this run)"}
        eval_dir = self.run_dir / "eval"
        atomic_write_json(result, eval_dir / "test_metrics.json")
        if save_preds:
            save_predictions(recs, self.graphs_by_name, self.run_dir / "predictions")
        return result
