"""Detect local hardware and provide typed representations of remote hardware."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GpuInfo:
    name: str
    memory_gb: float
    driver: str = ""


@dataclass
class HardwareProfile:
    kind: str = "local"            # "local", "kaggle_t4x2", "colab_t4", etc.
    cpu_count: int = 0
    ram_gb: float = 0.0
    gpus: list[GpuInfo] = field(default_factory=list)
    os: str = ""
    verified: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.memory_gb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def summary(self) -> str:
        if not self.gpus:
            return f"{self.kind}: CPU only, {self.cpu_count} cores, {self.ram_gb:.1f}GB RAM"
        gpus = ", ".join(f"{g.name} {g.memory_gb:.0f}GB" for g in self.gpus)
        return f"{self.kind}: {self.gpu_count}x GPU ({gpus}), {self.ram_gb:.1f}GB RAM"


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        return 1, str(e)


def _detect_nvidia() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    code, out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem_mb = float(parts[1])
        except ValueError:
            continue
        gpus.append(GpuInfo(name=parts[0], memory_gb=mem_mb / 1024.0,
                            driver=parts[2] if len(parts) > 2 else ""))
    return gpus


def detect_hardware() -> HardwareProfile:
    """Detect the local machine's hardware."""
    profile = HardwareProfile(
        kind="local",
        cpu_count=os.cpu_count() or 0,
        ram_gb=_ram_gb(),
        gpus=_detect_nvidia(),
        os=f"{platform.system()} {platform.release()}",
        verified=True,
    )
    return profile


def _ram_gb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback approximations per platform.
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except FileNotFoundError:
            pass
    return 8.0


# Predefined remote profiles for common providers.

def kaggle_t4x2() -> HardwareProfile:
    return HardwareProfile(
        kind="kaggle_t4x2",
        cpu_count=4,
        ram_gb=32.0,
        gpus=[GpuInfo(name="Tesla T4", memory_gb=15.8), GpuInfo(name="Tesla T4", memory_gb=15.8)],
    )


def kaggle_p100() -> HardwareProfile:
    return HardwareProfile(
        kind="kaggle_p100",
        cpu_count=4,
        ram_gb=13.0,
        gpus=[GpuInfo(name="Tesla P100", memory_gb=16.0)],
    )
