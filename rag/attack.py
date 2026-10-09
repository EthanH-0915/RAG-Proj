"""Minimal reader for the ATT&CK Enterprise STIX bundle.

mitreattack-python is the conventional choice, but ATT&CK v19 moved detection
guidance off the technique object into `x-mitre-detection-strategy` objects
(DET####) that point at `x-mitre-analytic` objects (AN####) holding the actual
analytic text. Reading the bundle directly keeps every field and relationship
this project depends on visible in one place, rather than resting on whether the
library tracks the current spec.

Relationship directions in the bundle:
    course-of-action            --mitigates--------> attack-pattern
    intrusion-set/malware/tool  --uses-------------> attack-pattern
    x-mitre-detection-strategy  --detects----------> attack-pattern
    attack-pattern (sub)        --subtechnique-of--> attack-pattern (parent)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .text import clean_text


def _is_live(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


class AttackData:
    """Indexed view over one ATT&CK domain bundle."""

    def __init__(self, path: Path):
        bundle = json.loads(Path(path).read_text(encoding="utf-8"))
        self.objects: dict[str, dict] = {}
        self.by_type: dict[str, list[dict]] = defaultdict(list)
        # (relationship_type, target_ref) -> [source_ref], and the reverse.
        self._sources: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._targets: dict[tuple[str, str], list[str]] = defaultdict(list)

        for obj in bundle.get("objects", []):
            kind = obj.get("type")
            if kind == "relationship":
                if not _is_live(obj):
                    continue
                key = obj.get("relationship_type")
                self._sources[(key, obj["target_ref"])].append(obj["source_ref"])
                self._targets[(key, obj["source_ref"])].append(obj["target_ref"])
                continue
            self.objects[obj["id"]] = obj
            self.by_type[kind].append(obj)

        self.collection = next(iter(self.by_type.get("x-mitre-collection", [])), {})
        self.tactic_names = {
            t["x_mitre_shortname"]: t["name"] for t in self.by_type["x-mitre-tactic"]
        }

    # ---------- identifiers ----------

    @staticmethod
    def attack_id(obj: dict) -> str | None:
        """The 'T1566.001' / 'M1017' style ID from the mitre-attack reference."""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                return ref.get("external_id")
        return None

    @staticmethod
    def attack_url(obj: dict) -> str | None:
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                return ref.get("url")
        return None

    # ---------- traversal ----------

    def related(self, relationship: str, stix_id: str, *, incoming: bool) -> list[dict]:
        """Objects joined to `stix_id` by `relationship`, skipping dangling refs."""
        index = self._sources if incoming else self._targets
        found = [self.objects.get(ref) for ref in index.get((relationship, stix_id), [])]
        return [obj for obj in found if obj and _is_live(obj)]

    def related_ids(self, relationship: str, stix_id: str, *, incoming: bool) -> list[str]:
        ids = {
            self.attack_id(obj)
            for obj in self.related(relationship, stix_id, incoming=incoming)
        }
        return sorted(i for i in ids if i)

    # ---------- techniques ----------

    def techniques(self) -> Iterator[dict]:
        """Live techniques and sub-techniques, in ATT&CK ID order."""
        live = [t for t in self.by_type["attack-pattern"] if _is_live(t)]
        yield from sorted(live, key=lambda t: self.attack_id(t) or "")

    def tactics_of(self, technique: dict) -> list[str]:
        names = []
        for phase in technique.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            short = phase.get("phase_name", "")
            names.append(self.tactic_names.get(short, short))
        return names

    def parent_of(self, technique: dict) -> dict | None:
        parents = self.related("subtechnique-of", technique["id"], incoming=False)
        return parents[0] if parents else None

    def detection_strategies(self, technique: dict) -> list[dict]:
        """Detection strategies for a technique, each with its analytic texts.

        Returns [{'id': 'DET0039', 'name': ..., 'analytics': [{'id': 'AN0110',
        'platforms': [...], 'description': ...}]}].
        """
        out = []
        for strategy in self.related("detects", technique["id"], incoming=True):
            analytics = []
            for ref in strategy.get("x_mitre_analytic_refs", []):
                analytic = self.objects.get(ref)
                if not analytic or not _is_live(analytic):
                    continue
                analytics.append(
                    {
                        "id": self.attack_id(analytic),
                        "platforms": analytic.get("x_mitre_platforms", []),
                        "description": clean_text(analytic.get("description")),
                    }
                )
            out.append(
                {
                    "id": self.attack_id(strategy),
                    "name": strategy.get("name", ""),
                    "analytics": analytics,
                }
            )
        return sorted(out, key=lambda s: s["id"] or "")
