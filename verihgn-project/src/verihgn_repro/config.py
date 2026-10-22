"""Experiment configuration and the deviation registry.

Defaults equal the official configs/ispd2015.yaml wherever that file defines a
value. `deviations(cfg)` lists every way a run departs from the paper and/or
the official code; it is written into each run's config.yaml.
"""
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentCfg:
    name: str = "verihgn_ispd15"
    dataset_scope: str = "official9"          # official9 | all14
    split: str = "official_seed42"            # official_seed42 | pair_aware
    grid_features: str = "paper_eq3"          # paper_eq3 | official_code_available
    label_norm: str = "per_dataset"           # per_dataset (paper) | per_sample (official code)


@dataclass
class ModelCfg:                               # official configs/ispd2015.yaml: model
    hidden_dim: int = 128
    num_layers: int = 3
    cell_feat_dim: int = 13
    net_feat_dim: int = 6
    pin_feat_dim: int = 3
    geom_feat_dim: int = 4
    dropout: float = 0.15
    use_checkpoint: bool = True
    # True = official. False = HEADPATH arm U diagnostic deviation (docs/HEADPATH_PLAN.md): the cell head
    # skips its clamp(min=0) in training-mode forward passes only; every evaluation uses the official forward.
    cell_head_train_clamp: bool = True


@dataclass
class TrainCfg:                               # official configs/ispd2015.yaml: training
    num_epochs: int = 200
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    step_size: int = 50
    gamma: float = 0.5
    patience: int = 20
    wmse_alpha: float = 10.0
    lambda_var: float = 0.5
    precision: str = "fp16"                   # fp16 (official AMP) | fp32 | bf16
    allow_tf32: bool = False
    seed: int = 42
    max_epochs_this_session: int | None = None   # stop early (benchmarks / smoke tests); resumable
    keep_every: int = 10
    keep_dataset_on_gpu: str = "auto"          # auto | true | false
    compute_train_metrics: bool = True         # official computes pooled train metrics each epoch
    step_log: bool = False                     # per-step diagnostics -> steps.csv (observation only)
    step_log_heads: bool = False               # + per-head loss terms, prediction streams, partition grad norms
    eval_every_steps: int = 0                  # > 0: eval-mode validation every k global steps + step 0 -> evals.csv


@dataclass
class Paths:
    processed_root: str = ""
    splits_dir: str = ""
    graph_cache: str = ""
    runs_root: str = ""


@dataclass
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainCfg = field(default_factory=TrainCfg)
    paths: Paths = field(default_factory=Paths)

    @property
    def grid_feat_dim(self) -> int:
        return {"paper_eq3": 5, "official_code_available": 3}[self.experiment.grid_features]

    def to_dict(self) -> dict:
        return asdict(self)

    def run_id(self) -> str:
        e, t = self.experiment, self.training
        rid = f"{e.name}__{e.split}__{e.grid_features}__{e.label_norm}__{t.precision}__seed{t.seed}"
        if not self.model.cell_head_train_clamp:          # diagnostic deviation; official ids are unchanged
            rid += "__DEV-cellhead-train-unclamped"
        return rid


def _merge(dc, d: dict):
    for k, v in d.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key {k} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__"):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(*yaml_paths, overrides: dict | None = None) -> Config:
    cfg = Config()
    for p in yaml_paths:
        _merge(cfg, yaml.safe_load(Path(p).read_text()) or {})
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    e, t = cfg.experiment, cfg.training
    assert e.dataset_scope in ("official9", "all14"), e.dataset_scope
    assert e.split in ("official_seed42", "pair_aware"), e.split
    assert e.grid_features in ("paper_eq3", "official_code_available"), e.grid_features
    assert e.label_norm in ("per_dataset", "per_sample"), e.label_norm
    assert t.precision in ("fp16", "fp32", "bf16"), t.precision
    assert isinstance(cfg.model.cell_head_train_clamp, bool), cfg.model.cell_head_train_clamp
    assert isinstance(t.eval_every_steps, int) and t.eval_every_steps >= 0, t.eval_every_steps
    assert t.step_log or not t.step_log_heads, "step_log_heads needs step_log"


# Always-on adaptations caused by unavailable author preprocessing (docs/RESEARCH_TO_CODE_MAPPING.md)
BASE_DEVIATIONS = [
    "A: preprocessing is ours (author DEF->feature code unpublished); features computed from CircuitNet "
    "instance placement + graph_information",
    "A/D: label = CircuitNet GR overflow maps, (H+V)/2 -> log1p (map type not stated by paper or code)",
    "A: native gcell grid, no resampling (paper's target resolution unknown)",
    "U: geometric edges = symmetric kNN (k from preprocessing config) on cell centres",
    "U: macro pin direction has no source; set by preprocessing config",
    "A: HPWL/net bbox from cell centres (no pin offsets in data)",
    "A: cell->tile assignment from CircuitNet gcell placement (exact) instead of official layout.csv x/y "
    "re-derivation",
    "C: evaluation always in FP32 (official test.py); validation for model selection also FP32",
    "Not claimed: reproduction of paper Tables 2-6",
]


def deviations(cfg: Config) -> list:
    out = list(BASE_DEVIATIONS)
    e, t = cfg.experiment, cfg.training
    if e.label_norm == "per_dataset":
        out.append("P (deviates from official code): labels / c_max of train split (paper); official code "
                   "normalises per sample")
    else:
        out.append("C (deviates from paper): per-sample label normalisation (official code); paper uses per-dataset")
    if e.grid_features == "paper_eq3":
        out.append("P (deviates from official code): grid features = paper Eq. 3; official code uses "
                   "RUDY/PinRUDY/MacroRegion/H,V MacroMargin")
    else:
        out.append("C-ablation: grid features = RUDY, RUDY_pin, macro_map only; H/V MacroMargin unavailable in "
                   "CircuitNet (not fabricated), grid_feat_dim = 3")
    if e.split == "pair_aware":
        out.append("A: pair-aware split (netlist twins kept together); official split leaks twins into test")
    else:
        out.append("C: official seed-42 split; both test designs have a netlist-identical twin in train")
    if e.dataset_scope == "all14":
        out.append("A: includes superblue designs (official code excludes them)")
    if t.precision != "fp16":
        out.append(f"precision {t.precision} (official training uses fp16 autocast + GradScaler)")
    if t.allow_tf32:
        out.append("TF32 matmul enabled (rounding differs from FP32)")
    if not cfg.model.cell_head_train_clamp:
        out.append(HEADPATH_DEVIATION)
    return out


HEADPATH_DEVIATION = (
    "DIAGNOSTIC DEVIATION (model, training only; HEADPATH arm U, docs/HEADPATH_PLAN.md): the cell head's "
    "clamp(min=0) is omitted in training-mode forward passes, so the official loss (weighted MSE and the variance "
    "regulariser) is applied to raw, possibly negative, cell outputs. Every evaluation uses the official clamped "
    "forward. The grid head is unchanged. This run is NOT a reproduction of the paper or of the official code, and "
    "it does not isolate the gradient pathway alone (the loss surface changes too).")


def save_config(cfg: Config, path, extra: dict | None = None) -> None:
    d = cfg.to_dict()
    d["deviations"] = deviations(cfg)
    if extra:
        d.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(json.loads(json.dumps(d, default=str)), sort_keys=False))
