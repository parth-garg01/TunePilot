from pathlib import Path

import pytest

from orchestrator.errors import ConfigError
from orchestrator.security import CredentialStore, redact, register_secrets, scan_for_secrets
from orchestrator.security.notebook_scan import scan_notebook


def test_redaction_masks_registered_secret():
    register_secrets(["supersecret1234"])
    assert "********" in redact("token=supersecret1234")


def test_credential_store_missing(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    store = CredentialStore(Path("does-not-exist"))
    with pytest.raises(ConfigError):
        store.kaggle()


def test_notebook_scan_finds_leak(tmp_path):
    nb = tmp_path / "leak.ipynb"
    nb.write_text(
        '{"cells":[{"cell_type":"code","source":["KAGGLE_KEY=abcdef1234567890"]}],'
        '"metadata":{},"nbformat":4,"nbformat_minor":5}',
        encoding="utf-8",
    )
    assert scan_notebook(nb)


def test_scan_for_secrets_generic():
    hits = scan_for_secrets("api_key: aaaaBBBBcccc1234")
    assert hits
