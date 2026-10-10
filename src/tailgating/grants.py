from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .schemas import Crossing, Grant


def load_grants(path: Path) -> list[Grant]:
    grants: list[Grant] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                grant = Grant(row["grant_id"].strip(), float(row["t_unlock_s"]), float(row["t_relock_s"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid grant row {row_number}: {exc}") from exc
            if grant.grant_id in seen:
                raise ValueError(f"duplicate grant_id at row {row_number}: {grant.grant_id}")
            seen.add(grant.grant_id)
            grants.append(grant)
    return sorted(grants, key=lambda item: item.t_unlock_s)


def find_grant(
    crossing: Crossing,
    grants: Sequence[Grant],
    pre_grace_s: float,
    post_grace_s: float,
) -> Grant | None:
    matches = [
        grant for grant in grants
        if grant.t_unlock_s - pre_grace_s <= crossing.t_cross_s <= grant.t_relock_s + post_grace_s
        and grant.t_unlock_s <= crossing.t_cross_s
    ]
    return max(matches, key=lambda item: item.t_unlock_s) if matches else None


def find_overlaps(grants: Sequence[Grant], pre_grace_s: float, post_grace_s: float) -> list[tuple[str, str]]:
    ordered = sorted(grants, key=lambda item: item.t_unlock_s)
    result = []
    for left, right in zip(ordered, ordered[1:]):
        if right.t_unlock_s - pre_grace_s <= left.t_relock_s + post_grace_s:
            result.append((left.grant_id, right.grant_id))
    return result
