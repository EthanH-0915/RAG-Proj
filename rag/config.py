"""Paths, feed URLs, database URL and embedding model, in one place."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

MANIFEST_PATH = RAW_DIR / "manifest.json"
ATTACK_PATH = RAW_DIR / "enterprise-attack.json"
KEV_PATH = RAW_DIR / "known_exploited_vulnerabilities.json"

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data"
    "/master/enterprise-attack/enterprise-attack.json"
)
KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds"
    "/known_exploited_vulnerabilities.json"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _load_env_file(path: Path) -> None:
    """Minimal .env reader, so finding the database needs no third-party import."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env_file(ROOT / ".env")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5433/ragdb"
)
