"""GPU detection, hardware profiles and memory/utilisation monitoring.

Profiles only change *how* the official computation is executed (precision
candidates, gradient checkpointing, scatter chunk sizes, whether the dataset
stays resident on the GPU). Batch size is always 1 graph, as in the official
code, because batching changes the optimisation itself.
"""
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

import torch


@dataclass
class HardwareInfo:
    device: str
    name: str = "cpu"
    total_vram_gb: float = 0.0
    compute_capability: str = ""
    bf16_supported: bool = False
    cuda_runtime: str = ""
    driver: str = ""
    torch_version: str = torch.__version__
    python: str = sys.version.split()[0]
    platform: str = platform.platform()


@dataclass
class Profile:
    name: str
    use_gradient_checkpointing: bool = True     # official default
    scatter_chunk_size: int = 500_000           # official constant
    keep_dataset_on_gpu: bool = False
    candidate_precisions: list = field(default_factory=lambda: ["fp32"])
    smoke_only: bool = False


def detect() -> HardwareInfo:
    if not torch.cuda.is_available():
        return HardwareInfo(device="cpu")
    p = torch.cuda.get_device_properties(0)
    driver = ""
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return HardwareInfo(
        device="cuda",
        name=p.name,
        total_vram_gb=round(p.total_memory / 1024 ** 3, 2),
        compute_capability=f"{p.major}.{p.minor}",
        bf16_supported=torch.cuda.is_bf16_supported(),
        cuda_runtime=torch.version.cuda or "",
        driver=driver,
    )


def choose_profile(hw: HardwareInfo) -> Profile:
    if hw.device == "cpu":
        return Profile("cpu", scatter_chunk_size=100_000, smoke_only=True)
    precisions = ["fp32", "fp16"] + (["bf16"] if hw.bf16_supported else [])
    if hw.total_vram_gb >= 35:           # A100-40/80GB, L40S, H100
        return Profile("large_gpu", keep_dataset_on_gpu=True, candidate_precisions=precisions)
    if hw.total_vram_gb >= 20:           # L4, A10
        return Profile("medium_gpu", candidate_precisions=precisions)
    return Profile("small_gpu", scatter_chunk_size=250_000, candidate_precisions=precisions)


def disk_report(paths) -> dict:
    out = {}
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            out[str(p)] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
        except OSError as e:
            out[str(p)] = {"error": str(e)}
    return out


def env_report(extra: dict | None = None) -> dict:
    hw = detect()
    rep = {"hardware": asdict(hw), "profile": asdict(choose_profile(hw))}
    try:
        import torch_geometric
        rep["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        rep["torch_geometric"] = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        rep["system_ram_gb"] = {"total": round(vm.total / 1e9, 1), "available": round(vm.available / 1e9, 1)}
    except ImportError:
        rep["system_ram_gb"] = None
    if extra:
        rep.update(extra)
    return rep


class GPUMonitor:
    """Samples nvidia-smi utilisation/memory in a background thread."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                util, mem = (float(v) for v in out.split(","))
                self.samples.append((time.time(), util, mem))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        if torch.cuda.is_available() and shutil.which("nvidia-smi"):
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop_and_summary(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {}
        utils = [s[1] for s in self.samples]
        mems = [s[2] for s in self.samples]
        self.samples = []
        self._stop.clear()
        return {"gpu_util_mean": sum(utils) / len(utils), "gpu_util_max": max(utils),
                "smi_mem_used_max_mib": max(mems)}


def cuda_peak_gb(reset: bool = False) -> dict:
    if not torch.cuda.is_available():
        return {}
    out = {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return out


if __name__ == "__main__":
    print(json.dumps(env_report(), indent=2))
