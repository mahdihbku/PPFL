"""Exact per-party byte counts, derived from the message layouts in protocol.py.

tests/test_costmodel.py checks these functions byte-for-byte against the ledger of end-to-end
runs; the communication figures evaluate them over the full (N, C, P) grid.
"""

from dataclasses import dataclass

from . import lhh, vrf
from .keys import SIG_LEN
from .protocol import HDR_LEN, USER_INFO_LEN

BASELINE_KINDS = {"w_tilde", "model"}  # paid by plain FL as well (masked upload, model download)
OFFLINE_KINDS = {"enroll", "roster"}


@dataclass(frozen=True)
class Sizes:
    election: int
    control: int
    delta: int
    mcontrol: int
    reveal: int
    cosign: int
    w_tilde: int
    subsum: int
    ak: int
    model: int


def sizes(P, fs="ephemeral", lhh_on=True, similarity=False, n_delta=0) -> Sizes:
    epk = 32 if fs == "ephemeral" else 0
    h = lhh.POINT_LEN if lhh_on else 0
    return Sizes(
        election=HDR_LEN + vrf.PROOF_LEN + epk + SIG_LEN,
        control=HDR_LEN + epk + h + (8 if similarity else 0) + 32 + SIG_LEN,
        delta=5 + 32 + 4 + 4 * n_delta + SIG_LEN,
        mcontrol=HDR_LEN + 32 + h + 32 + SIG_LEN,
        reveal=HDR_LEN + 32 + SIG_LEN,
        cosign=HDR_LEN + 32 + SIG_LEN,
        w_tilde=HDR_LEN + 4 * P,
        subsum=HDR_LEN + 4 * P,
        ak=HDR_LEN + 4 * P + (2 * P if lhh_on else 0),
        model=4 * P,
    )


def offline_bytes(n_users, fs="ephemeral") -> dict:
    entry = 4 + 32 + 32 + (32 if fs == "static" else 0) + USER_INFO_LEN
    return {"client": {"up": entry - 4, "down": n_users * entry},
            "server": {"up": n_users * n_users * entry, "down": n_users * (entry - 4)}}


def round_bytes(P, C, n_users, n_submit=None, n_receive=None, n_online=None, fs="ephemeral",
                lhh_on=True, cosign=True, similarity=False, small_attempts=()) -> dict:
    """Bytes of one successful round. `small_attempts` lists the number of election posts in each
    preceding attempt rejected for |committee| < c_min. Members are assumed not to have been
    selected in those attempts (add `election` bytes per earlier selection otherwise)."""
    n_online = n_users if n_online is None else n_online
    n_submit = n_online if n_submit is None else n_submit
    n_receive = n_submit if n_receive is None else n_receive
    s = sizes(P, fs, lhh_on, similarity, n_delta=n_users - n_submit)
    pre = sum(small_attempts) * s.election  # extra election posts every reader downloads
    read_ctrl = C * s.election + s.delta + pre  # every online user, as a client
    cos = C * s.cosign if cosign else 0
    return {
        "sizes": s,
        "client": {"up": s.control, "down": read_ctrl + cos,
                   "up_baseline": s.w_tilde, "down_baseline": s.model},
        "client_dropped_before_submit": {"up": 0, "down": read_ctrl, "up_baseline": 0, "down_baseline": 0},
        "client_dropped_after_submit": {"up": s.control, "down": read_ctrl,
                                        "up_baseline": s.w_tilde, "down_baseline": 0},
        "member": {"up": s.election + s.mcontrol + s.subsum + s.reveal + (s.cosign if cosign else 0),
                   "down": n_submit * s.control + C * s.mcontrol + s.ak + C * s.reveal},
        "server": {"up": s.delta + C * s.ak,
                   "down": C * s.election + pre + n_submit * s.control + C * (s.mcontrol + s.subsum + s.reveal),
                   "up_baseline": n_receive * s.model, "down_baseline": n_submit * s.w_tilde},
    }


def aborted_attempt_bytes(kind, P, C, n_users, n_submit=None, n_online=None, k=1, fs="ephemeral",
                          lhh_on=True, similarity=False) -> dict:
    """Bytes of one attempt aborted because `k` committee members failed.

    kind: "before_subsum" (Case 3: the k members never post their sub-sums) or
          "before_reveal" (Case 4: they post sub-sums but never reveal their masks).
    Clients have already masked and uploaded, so a retry repeats the masked-model upload."""
    n_online = n_users if n_online is None else n_online
    n_submit = n_online if n_submit is None else n_submit
    s = sizes(P, fs, lhh_on, similarity, n_delta=n_users - n_submit)
    client = {"up": s.control, "down": C * s.election + s.delta, "up_baseline": s.w_tilde, "down_baseline": 0}
    if kind == "before_subsum":
        member = {"up": s.election + s.mcontrol + s.subsum, "down": n_submit * s.control}
        server = {"up": s.delta, "down": C * s.election + n_submit * s.control + (C - k) * s.subsum}
    elif kind == "before_reveal":
        member = {"up": s.election + s.mcontrol + s.subsum + s.reveal,
                  "down": n_submit * s.control + C * s.mcontrol + s.ak}
        server = {"up": s.delta + C * s.ak,
                  "down": C * s.election + n_submit * s.control + C * (s.mcontrol + s.subsum)}
    else:
        raise ValueError(kind)
    server.update(up_baseline=0, down_baseline=n_submit * s.w_tilde)
    return {"sizes": s, "client": client, "member": member,
            "client_not_submitting": {"up": 0, "down": C * s.election + s.delta, "up_baseline": 0, "down_baseline": 0},
            "server": server}


def recovery_round_bytes(P, n_elected, n_users, d=0, f3=0, f4=0, n_submit=None, small_attempts=(),
                         lhh_on=True, cosign=True) -> dict:
    """Bytes of a successful round with tau > 0 (ephemeral keys): every elected member shares its
    round secrets; d members drop before sharing, f3 fail before their sub-sum and f4 before their
    reveal, all recovered (d + f3 + f4 <= tau). "member" is a member that survives to the end."""
    from .protocol import RELEASE_MSG_LEN, SHARE_MSG_LEN
    n_submit = n_users if n_submit is None else n_submit
    s = sizes(P, "ephemeral", lhh_on, False, n_delta=n_users - n_submit)
    share, release = SHARE_MSG_LEN + SIG_LEN, RELEASE_MSG_LEN + SIG_LEN
    C = n_elected - d
    active, revealers = C - f3, C - f3 - f4
    pre = sum(small_attempts) * s.election
    cos = revealers * s.cosign if cosign else 0
    rel_reads = f3 * active * release + f4 * revealers * release
    return {
        "sizes": s,
        "client": {"up": s.control, "down": n_elected * s.election + s.delta + pre + cos,
                   "up_baseline": s.w_tilde, "down_baseline": s.model},
        "member": {"up": s.election + (n_elected - 1) * share + s.mcontrol + s.subsum + s.reveal
                   + (s.cosign if cosign else 0) + (f3 + f4) * release,
                   "down": (C - 1) * share + n_submit * s.control + active * s.mcontrol + s.ak
                   + revealers * s.reveal + rel_reads},
        "server": {"up": s.delta + active * s.ak,
                   "down": n_elected * s.election + pre + n_submit * s.control
                   + active * (s.mcontrol + s.subsum) + revealers * s.reveal + rel_reads,
                   "up_baseline": n_submit * s.model, "down_baseline": n_submit * s.w_tilde},
    }
