from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .grants import find_grant
from .schemas import Crossing, Event, Grant


class EventAggregator:
    def __init__(self, grants: Sequence[Grant], pre_grace_s: float, post_grace_s: float) -> None:
        self.grants = list(grants)
        self.pre_grace_s = pre_grace_s
        self.post_grace_s = post_grace_s
        self._authorized: dict[str, list[Crossing]] = defaultdict(list)
        self._seen: set[tuple[int, str]] = set()
        self._unauthorized: list[Crossing] = []
        self._unauthorized_tracks: set[int] = set()

    def add(self, crossing: Crossing) -> None:
        if crossing.direction != "in":
            return
        grant = find_grant(crossing, self.grants, self.pre_grace_s, self.post_grace_s)
        if grant is None:
            if crossing.track_id not in self._unauthorized_tracks:
                self._unauthorized.append(crossing)
                self._unauthorized_tracks.add(crossing.track_id)
            return
        key = crossing.track_id, grant.grant_id
        if key not in self._seen:
            self._seen.add(key)
            self._authorized[grant.grant_id].append(crossing)

    def finalize(self) -> list[Event]:
        events: list[Event] = []
        for grant in sorted(self.grants, key=lambda item: item.t_unlock_s):
            crossings = tuple(sorted(self._authorized[grant.grant_id], key=lambda item: item.t_cross_s))
            entries = len(crossings)
            events.append(Event(grant.grant_id, entries, "tailgating" if entries > 1 else "none", crossings, grant.t_unlock_s))
        for crossing in sorted(self._unauthorized, key=lambda item: item.t_cross_s):
            events.append(Event(None, 1, "unauthorised_entry", (crossing,)))
        return events

    def status_at(self, timestamp_s: float) -> tuple[str | None, int, str]:
        active = [
            grant for grant in self.grants
            if grant.t_unlock_s - self.pre_grace_s <= timestamp_s <= grant.t_relock_s + self.post_grace_s
        ]
        if not active:
            return None, 0, "none"
        grant = max(active, key=lambda item: item.t_unlock_s)
        entries = len(self._authorized[grant.grant_id])
        return grant.grant_id, entries, "tailgating" if entries > 1 else "none"
