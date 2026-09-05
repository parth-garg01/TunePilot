"""`orchestrator hardware` — detect local and enumerate remote profiles."""

from __future__ import annotations

import typer
from rich import print as rprint

from ..hardware.detect import detect_hardware, kaggle_p100, kaggle_t4x2

app = typer.Typer()


@app.command("detect")
def detect() -> None:
    hw = detect_hardware()
    rprint(hw.summary())
    for g in hw.gpus:
        rprint(f"  {g.name}: {g.memory_gb:.1f}GB, driver={g.driver}")


@app.command("profiles")
def profiles() -> None:
    for hw in (kaggle_t4x2(), kaggle_p100()):
        rprint(hw.summary())
