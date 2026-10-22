"""Reads settings.yaml and the .env file once, so every file uses the same values."""

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def load_settings() -> dict:
    """Returns settings.yaml as a dictionary. Also loads the .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
