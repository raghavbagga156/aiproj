"""Diagnostic deviations applied on top of the unmodified official model (docs/HEADPATH_PLAN.md).

Nothing in third_party/VeriHGN is changed. A deviation is applied to an already-built official
model by re-classing an existing module: no module is created, no parameter is added, no RNG is
consumed, and parameter names, order and state_dict keys are unchanged, so the initial weights
of an official and a deviating model built with the same seed are identical.

Also: the parameter partition by which head can send gradient to a parameter
(docs/POST_COLLAPSE_FORENSIC_REPORT.md, F7), used for per-partition gradient norms.
"""
from functools import lru_cache

from . import official


@lru_cache(maxsize=1)
def _train_unclamped_head_class():
    base = official.model_module().PredictionHead

    class TrainUnclampedPredictionHead(base):
        """HEADPATH arm U. Training mode: the raw head output (the official forward without its
        final clamp(min=0)). Eval mode: exactly the official PredictionHead.forward."""

        def forward(self, h):
            if self.training:
                return self.mlp(h).squeeze(-1)
            return super().forward(h)

    return TrainUnclampedPredictionHead


def apply_cell_head_train_unclamped(model):
    """Turn an official VeriHGN into HEADPATH arm U (in place; returns the model)."""
    base = official.model_module().PredictionHead
    if type(model.cell_head) is not base:
        raise TypeError(f"cell_head is {type(model.cell_head).__name__}, expected the official PredictionHead")
    model.cell_head.__class__ = _train_unclamped_head_class()
    return model


def is_official_head(head) -> bool:
    return type(head) is official.model_module().PredictionHead


# parameters of the LAST message-passing layer that only one head's output depends on
_LAST_LAYER_CELL_PATH = {"mlp_pin", "W_grid_to_net", "W_net_update", "net_norm", "W_net_to_cell", "W_gate",
                         "W_geom_msg", "W_cell_fuse", "cell_norm"}
_LAST_LAYER_GRID_PATH = {"W_cell_to_grid", "W_grid_update", "grid_norm", "phi_fc", "W_l1_spatial", "phi_cf",
                         "W_gate_hier"}
PARTITIONS = ("cell_only", "grid_only", "shared")


def parameter_partition(model) -> dict:
    """name -> 'cell_only' | 'grid_only' | 'shared'. The rule follows from verihgn.py's forward
    (the last layer's grid output does not depend on its cell/net path and vice versa); it is
    checked against autograd in tests/test_headpath.py."""
    last = len(model.layers) - 1
    out = {}
    for name, _ in model.named_parameters():
        p = name.split(".")
        if p[0] == "cell_head" or (p[0] == "layers" and int(p[1]) == last and p[2] in _LAST_LAYER_CELL_PATH):
            out[name] = "cell_only"
        elif p[0] == "grid_head" or (p[0] == "layers" and int(p[1]) == last and p[2] in _LAST_LAYER_GRID_PATH):
            out[name] = "grid_only"
        else:
            out[name] = "shared"
    return out
