"""Credential loading from environment variables and secure local files.

Never returns credentials from `config.yaml`. Reads them from environment
variables or from files under a `credentials/` directory that is excluded from
Git.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from ..errors import ConfigError
from .redaction import register_secrets


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    key: str

    def as_env(self) -> dict[str, str]:
        return {"KAGGLE_USERNAME": self.username, "KAGGLE_KEY": self.key}


class CredentialStore:
    """Locates and loads credentials from env or `credentials/*.env` files."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else Path("credentials")

    def _read_env_file(self, name: str) -> dict[str, str]:
        path = self.root / f"{name}.env"
        if not path.exists():
            return {}
        return {k: v for k, v in dotenv_values(path).items() if v}

    def kaggle(self) -> KaggleCredentials:
        env = {
            **self._read_env_file("kaggle"),
            **dotenv_values(".env"),
            **os.environ,
        }
        user = env.get("KAGGLE_USERNAME")
        key = env.get("KAGGLE_KEY")

        if not user or not key:
            # Check ~/.kaggle/kaggle.json or ./kaggle.json
            import json
            for json_path in [Path.home() / ".kaggle" / "kaggle.json", Path("kaggle.json")]:
                if json_path.exists():
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        user = user or data.get("username")
                        key = key or data.get("key")
                    except Exception:
                        pass

        if not user or not key:
            raise ConfigError(
                "Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY in "
                "environment, .env, credentials/kaggle.env, or ~/.kaggle/kaggle.json."
            )
        register_secrets([key])
        return KaggleCredentials(username=user, key=key)


    def hf_token(self) -> str | None:
        env = {**self._read_env_file("huggingface"), **os.environ}
        tok = env.get("HF_TOKEN") or env.get("HUGGINGFACE_TOKEN")
        if tok:
            register_secrets([tok])
        return tok


def load_kaggle_credentials(root: Path | str | None = None) -> KaggleCredentials:
    return CredentialStore(root).kaggle()
