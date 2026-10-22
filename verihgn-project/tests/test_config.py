from pathlib import Path

import yaml

from verihgn_repro.config import deviations, load_config
from verihgn_repro.official import OFFICIAL_DIR

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_matches_official_yaml():
    off = yaml.safe_load((OFFICIAL_DIR / "configs" / "ispd2015.yaml").read_text())
    cfg = load_config(CONFIGS / "base.yaml")
    for k, v in off["model"].items():
        if k == "grid_feat_dim":
            continue                      # derived from the grid feature set
        assert getattr(cfg.model, k) == v, k
    t = off["training"]
    assert cfg.training.num_epochs == t["num_epochs"]
    assert cfg.training.learning_rate == t["learning_rate"]
    assert cfg.training.weight_decay == t["weight_decay"]
    assert cfg.training.grad_clip == t["grad_clip"]
    assert cfg.training.step_size == t["lr_scheduler"]["step_size"]
    assert cfg.training.gamma == t["lr_scheduler"]["gamma"]
    assert cfg.training.patience == t["early_stopping"]["patience"]
    assert cfg.training.wmse_alpha == t["loss"]["wmse_alpha"]
    assert cfg.training.lambda_var == t["loss"]["lambda_var"]
    assert cfg.grid_feat_dim == off["model"]["grid_feat_dim"]


def test_variants_and_deviation_registry():
    base = load_config(CONFIGS / "base.yaml")
    abl = load_config(CONFIGS / "base.yaml", CONFIGS / "ablation_official_code_features.yaml")
    assert abl.grid_feat_dim == 3
    pa = load_config(CONFIGS / "base.yaml", CONFIGS / "split_pair_aware.yaml", CONFIGS / "norm_per_sample.yaml")
    assert pa.experiment.split == "pair_aware" and pa.experiment.label_norm == "per_sample"
    assert len({base.run_id(), abl.run_id(), pa.run_id()}) == 3
    d = " ".join(deviations(abl))
    assert "MacroMargin unavailable" in d and "Not claimed" in d
    fp32 = load_config(CONFIGS / "base.yaml", overrides={"training": {"precision": "fp32"}})
    assert any("precision fp32" in x for x in deviations(fp32))
