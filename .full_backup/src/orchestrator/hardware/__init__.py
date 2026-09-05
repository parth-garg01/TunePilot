"""Hardware detection and feasibility estimation."""

from .detect import detect_hardware, HardwareProfile, GpuInfo
from .feasibility import (
    estimate_vram_gb,
    feasibility,
    FeasibilityDecision,
    Strategy,
)

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "GpuInfo",
    "estimate_vram_gb",
    "feasibility",
    "FeasibilityDecision",
    "Strategy",
]
