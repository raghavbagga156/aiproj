import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_cpu_byte_tensor(x, name: str) -> torch.Tensor:
    """A torch generator state must be a CPU uint8 tensor (torch.ByteTensor), also
    for CUDA generators. A state that was moved to a GPU (e.g. by
    torch.load(map_location='cuda')) is copied back losslessly; anything else is
    rejected with an explicit error instead of torch's generic one."""
    if not torch.is_tensor(x) or x.dtype != torch.uint8:
        raise TypeError(f"RNG state '{name}' must be a uint8 tensor, got "
                        f"{type(x).__name__}{'/' + str(x.dtype) if torch.is_tensor(x) else ''}")
    if x.device.type == "meta":
        raise TypeError(f"RNG state '{name}' is a meta tensor (no data); the checkpoint was loaded "
                        "with an invalid map_location")
    return x.detach().to("cpu").contiguous()


def rng_state() -> dict:
    """Python, NumPy, CPU torch and every CUDA device's generator state.
    Torch states are stored as CPU ByteTensors (what torch returns)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state


def set_rng_state(state: dict) -> dict:
    """Restore a state captured by rng_state(). Returns what was restored.

    Raises if the CUDA states cannot be applied one-to-one (different number of
    devices); a CUDA state saved on a GPU run is ignored only when restoring on a
    machine without CUDA, and that is reported in the return value."""
    for key in ("python", "numpy", "torch"):
        if key not in state:
            raise KeyError(f"RNG state is missing '{key}' (keys: {sorted(state)})")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(as_cpu_byte_tensor(state["torch"], "torch"))
    report = {"python": True, "numpy": True, "torch": True, "cuda_devices_restored": 0,
              "cuda_state_in_checkpoint": "cuda" in state}
    if "cuda" in state and torch.cuda.is_available():
        cuda = [as_cpu_byte_tensor(s, f"cuda[{i}]") for i, s in enumerate(state["cuda"])]
        if len(cuda) != torch.cuda.device_count():
            raise ValueError(f"checkpoint has CUDA RNG states for {len(cuda)} device(s), this runtime has "
                             f"{torch.cuda.device_count()}; resuming would not be reproducible")
        torch.cuda.set_rng_state_all(cuda)
        report["cuda_devices_restored"] = len(cuda)
    return report
