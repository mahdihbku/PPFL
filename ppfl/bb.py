"""Append-only bulletin board for the control plane.

Assumption (stated in the paper): every party reads the same log, entries cannot be removed or
altered, and each entry is signed by its author. The board is trusted for availability and
consistency only; it never sees secret material. Consensus/replication cost is not modelled.
"""

from dataclasses import dataclass, field

from .keys import SIG_LEN


@dataclass
class Entry:
    seq: int
    author: object
    kind: str
    t: int
    attempt: int
    payload: bytes
    sig: bytes
    fields: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.payload) + SIG_LEN


class BulletinBoard:
    def __init__(self, ledger):
        self.ledger = ledger
        self.entries: list[Entry] = []

    def post(self, author, role, kind, t, attempt, payload: bytes, sig: bytes, **fields) -> Entry:
        e = Entry(len(self.entries), author, kind, t, attempt, payload, sig, fields)
        self.entries.append(e)
        self.ledger.send(author, role, "BB", "bb", "bb", kind, e.size)
        return e

    def query(self, kind, t, attempt) -> list[Entry]:
        return [e for e in self.entries if e.kind == kind and e.t == t and e.attempt == attempt]

    def charge_read(self, readers, role, entries: list[Entry], kind: str) -> None:
        n = sum(e.size for e in entries)
        for r in readers:
            self.ledger.send("BB", "bb", r, role, "bb", kind, n)
