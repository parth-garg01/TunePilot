from pathlib import Path

from typer.testing import CliRunner

from orchestrator.cli import app


runner = CliRunner()


def test_version_prints():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "soup-orchestrator" in result.stdout


def test_init_creates_project(tmp_path: Path):
    result = runner.invoke(
        app,
        ["init", "demo", "--root", str(tmp_path), "--dataset", str(tmp_path / "d.jsonl")],
    )
    assert result.exit_code == 0
    assert (tmp_path / "demo" / "orchestrator.yaml").exists()


def test_dataset_analyze_json(sample_jsonl: Path):
    result = runner.invoke(app, ["dataset", "analyze", str(sample_jsonl), "--json"])
    assert result.exit_code == 0
    assert '"n_examples"' in result.stdout


def test_hardware_detect_runs():
    result = runner.invoke(app, ["hardware", "detect"])
    assert result.exit_code == 0
    assert "GPU" in result.stdout or "CPU only" in result.stdout
