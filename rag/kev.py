"""Reader for CISA's Known Exploited Vulnerabilities catalog, joined to ATT&CK.

ATT&CK has no vulnerability objects. The only link between the two feeds is CVE
IDs written in ATT&CK prose ("APT29 has exploited CVE-2019-19781 ..."), so this
module indexes those mentions and hands each KEV entry the groups, software and
campaigns ATT&CK says exploited it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from . import config
from .attack import AttackData, _is_live

# ATT&CK object types whose prose can say they exploited a CVE.
EXPLOITER_TYPES = ("intrusion-set", "malware", "tool", "campaign")


def load(path: Path = config.KEV_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def attack_mentions(data: AttackData) -> dict[str, list[dict]]:
    """CVE ID -> live ATT&CK objects that mention exploiting it, in ID order."""
    index: dict[str, list[dict]] = defaultdict(list)
    for kind in EXPLOITER_TYPES:
        for obj in data.by_type[kind]:
            if not _is_live(obj):
                continue
            for cve in data.cves_exploited_by(obj):
                index[cve].append(obj)
    for objs in index.values():
        objs.sort(key=lambda o: data.attack_id(o) or "")
    return index


def entries(data: AttackData, path: Path = config.KEV_PATH) -> Iterator[dict]:
    """KEV entries in CVE order, each with an added `exploited_by` list of the
    ATT&CK objects that mention it."""
    mentions = attack_mentions(data)
    catalog = load(path)
    for entry in sorted(catalog["vulnerabilities"], key=lambda v: v["cveID"]):
        yield {**entry, "exploited_by": mentions.get(entry["cveID"], [])}
