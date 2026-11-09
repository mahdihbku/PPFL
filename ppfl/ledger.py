"""Per-party accounting of bytes on the wire and CPU time.

Parties are user indices (int) or "S" (server) or "BB" (bulletin board). Every record carries the
role the party played (client, member, server), so per-role costs can be reported separately.
"""

import time
from collections import defaultdict
from contextlib import contextmanager


class Ledger:
    def __init__(self):
        self.bytes = defaultdict(int)  # (party, role, "up"|"down", channel, kind) -> bytes
        self.time = defaultdict(float)  # (party, role, phase) -> seconds

    def send(self, src, src_role, dst, dst_role, channel: str, kind: str, n: int) -> None:
        self.bytes[(src, src_role, "up", channel, kind)] += n
        self.bytes[(dst, dst_role, "down", channel, kind)] += n

    @contextmanager
    def clock(self, party, role: str, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.time[(party, role, phase)] += time.perf_counter() - t0

    def charge(self, parties, role: str, phase: str, dt: float) -> None:
        for p in parties:
            self.time[(p, role, phase)] += dt

    def shared(self, parties, role: str, phase: str, fn, also=()):
        """Run fn once and charge its duration to every party that performs the identical
        computation (e.g., all clients verifying the same committee proofs). `also` is a list
        of (parties, role) pairs charged the same duration under other roles."""
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        self.charge(parties, role, phase, dt)
        for ps, r in also:
            self.charge(ps, r, phase, dt)
        return out

    def party_bytes(self, party, role=None, direction=None, kinds=None, exclude_kinds=()) -> int:
        return sum(
            n
            for (p, r, d, _c, k), n in self.bytes.items()
            if p == party
            and (role is None or r == role)
            and (direction is None or d == direction)
            and (kinds is None or k in kinds)
            and k not in exclude_kinds
        )

    def party_time(self, party, role=None, phases=None) -> float:
        return sum(
            s
            for (p, r, ph), s in self.time.items()
            if p == party and (role is None or r == role) and (phases is None or ph in phases)
        )

    def phases(self, role: str) -> set[str]:
        return {ph for (_p, r, ph) in self.time if r == role}
