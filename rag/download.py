"""Download and cache the raw ATT&CK and KEV feeds.

Writes data/raw/manifest.json recording, for each file, the URL, the UTC fetch
time, the size, a SHA-256 digest and the dataset's own version string. That
manifest is what lets the report say which snapshot the results came from, and
it is copied into the `ingest_runs` table on every ingest.

Usage:  python -m rag.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import config

# Some CDNs reject urllib's default User-Agent.
_HEADERS = {"User-Agent": "RAG-Proj/0.1 (coursework; +https://github.com/)"}


def _fetch(url: str, dest: Path, force: bool) -> bool:
    """Download `url` to `dest` unless already cached. True if a fetch happened."""
    if dest.exists() and not force:
        print(f"  cached    {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return False

    print(f"  fetching  {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as fh:
        while block := response.read(1 << 16):
            fh.write(block)
    partial.replace(dest)
    print(f"  saved     {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _attack_version(path: Path) -> dict:
    """Pull the collection version out of the STIX bundle."""
    bundle = json.loads(path.read_text(encoding="utf-8"))
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-collection":
            return {
                "name": obj.get("name"),
                "version": obj.get("x_mitre_version"),
                "modified": obj.get("modified"),
            }
    return {}


def _kev_version(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return {
        "catalog_version": catalog.get("catalogVersion"),
        "date_released": catalog.get("dateReleased"),
        "count": catalog.get("count"),
    }


def _describe(path: Path, url: str, extra: dict) -> dict:
    return {
        "url": url,
        "file": path.name,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if cached"
    )
    args = parser.parse_args()

    print("ATT&CK Enterprise:")
    _fetch(config.ATTACK_URL, config.ATTACK_PATH, args.force)
    print("CISA KEV:")
    _fetch(config.KEV_URL, config.KEV_PATH, args.force)

    print("Reading versions and hashing...")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attack": _describe(
            config.ATTACK_PATH, config.ATTACK_URL, _attack_version(config.ATTACK_PATH)
        ),
        "kev": _describe(
            config.KEV_PATH, config.KEV_URL, _kev_version(config.KEV_PATH)
        ),
    }
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    attack, kev = manifest["attack"], manifest["kev"]
    print(f"\nATT&CK   v{attack.get('version')}  (modified {attack.get('modified')})")
    print(f"KEV      v{kev.get('catalog_version')}  {kev.get('count')} vulnerabilities")
    print(f"Manifest {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
