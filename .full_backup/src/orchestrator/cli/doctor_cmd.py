"""`orchestrator doctor` — environment diagnostics."""

from __future__ import annotations

import importlib
import platform
import shutil

import typer
from rich import print as rprint

from ..hardware.detect import detect_hardware
from ..security import CredentialStore
from ..soup.adapter import SoupAdapter

app = typer.Typer(invoke_without_command=True)


def _check(name: str, ok: bool, detail: str = "") -> None:
    tag = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
    rprint(f"  {tag}  {name}" + (f" — {detail}" if detail else ""))


@app.callback(invoke_without_command=True)
def doctor() -> None:
    rprint(f"Python: {platform.python_version()}  OS: {platform.system()} {platform.release()}")
    for mod in ("pydantic", "yaml", "typer", "rich", "httpx", "dotenv"):
        try:
            importlib.import_module(mod if mod != "yaml" else "yaml")
            _check(f"import {mod}", True)
        except Exception as e:
            _check(f"import {mod}", False, str(e))

    _check("kaggle CLI", shutil.which("kaggle") is not None)
    _check("soup CLI", SoupAdapter().available())

    try:
        CredentialStore().kaggle()
        _check("kaggle credentials", True)
    except Exception as e:
        _check("kaggle credentials", False, str(e))

    hw = detect_hardware()
    rprint(f"  Hardware: {hw.summary()}")
