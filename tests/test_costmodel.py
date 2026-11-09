"""The analytic byte counts must match the ledger of end-to-end runs exactly."""

import numpy as np
import pytest

from ppfl import costmodel as cm
from ppfl.protocol import Behavior, Params, System

N = 14
OVERHEAD_EXCL = cm.BASELINE_KINDS | cm.OFFLINE_KINDS


def measured(L, party, role):
    return {"up": L.party_bytes(party, role, "up", exclude_kinds=OVERHEAD_EXCL),
            "down": L.party_bytes(party, role, "down", exclude_kinds=OVERHEAD_EXCL),
            "up_baseline": L.party_bytes(party, role, "up", kinds=cm.BASELINE_KINDS),
            "down_baseline": L.party_bytes(party, role, "down", kinds=cm.BASELINE_KINDS)}


CONFIGS = [
    dict(P=37, fs="ephemeral", lhh=True, cosign=True, behavior={}),
    dict(P=250, fs="static", lhh=True, cosign=True, behavior={}),
    dict(P=101, fs="ephemeral", lhh=False, cosign=True, behavior={}),
    dict(P=64, fs="ephemeral", lhh=True, cosign=False, behavior={}),
    dict(P=80, fs="ephemeral", lhh=True, cosign=True,
         behavior=dict(drop_before_submit={1, 4}, drop_after_submit={6})),
    dict(P=80, fs="static", lhh=False, cosign=True, behavior=dict(offline={2}, drop_before_submit={9})),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_round_bytes_match_ledger(cfg):
    P = cfg["P"]
    s = System(Params(n_users=N, length=P, L=1, c_min=3, fs=cfg["fs"], lhh=cfg["lhh"],
                      cosign=cfg["cosign"], max_retries=8))
    rng = np.random.default_rng(1)
    beh = Behavior(**cfg["behavior"])
    res = s.run_round(1, {i: rng.normal(0, 1, P) for i in range(N)}, beh)
    assert res.status == "ok"
    online = [i for i in range(N) if i not in beh.offline]
    submit = [i for i in online if i not in beh.drop_before_submit]
    receive = [i for i in submit if i not in beh.drop_after_submit]
    final = res.attempts[-1].committee
    small = [len(a.committee) for a in res.attempts[:-1]]
    assert all(a.outcome == "small_committee" for a in res.attempts[:-1])
    model = cm.round_bytes(P, len(final), N, len(submit), len(receive), len(online), cfg["fs"],
                           cfg["lhh"], cfg["cosign"], small_attempts=small)
    L = s.ledger
    for i in range(N):
        got = measured(L, i, "client")
        if i not in online:
            assert got == {"up": 0, "down": 0, "up_baseline": 0, "down_baseline": 0}
        elif i not in submit:
            assert got == model["client_dropped_before_submit"], i
        elif i not in receive:
            assert got == model["client_dropped_after_submit"], i
        else:
            assert got == model["client"], i
    elections_by = {i: sum(i in a.committee for a in res.attempts[:-1]) for i in range(N)}
    for j in final:
        got = measured(L, j, "member")
        assert got["up"] == model["member"]["up"] + elections_by[j] * model["sizes"].election
        assert got["down"] == model["member"]["down"]
    for i in set(range(N)) - set(final):  # selected only in rejected attempts
        assert measured(L, i, "member")["up"] == elections_by[i] * model["sizes"].election
    assert measured(L, "S", "server") == model["server"]


@pytest.mark.parametrize("fs", ["ephemeral", "static"])
def test_offline_bytes_match_ledger(fs):
    s = System(Params(n_users=N, length=10, L=1, c_min=1, fs=fs))
    model = cm.offline_bytes(N, fs)
    for i in range(N):
        assert s.ledger.party_bytes(i, "client", "up", kinds=cm.OFFLINE_KINDS) == model["client"]["up"]
        assert s.ledger.party_bytes(i, "client", "down", kinds=cm.OFFLINE_KINDS) == model["client"]["down"]
    assert s.ledger.party_bytes("S", "server", "up", kinds=cm.OFFLINE_KINDS) == model["server"]["up"]
    assert s.ledger.party_bytes("S", "server", "down", kinds=cm.OFFLINE_KINDS) == model["server"]["down"]


def test_round_bytes_with_rejected_small_committees():
    """L=3 with c_min=5 at N=40 rejects about half the elections; retry until one is rejected."""
    n, P = 40, 20
    for seed in range(20):
        s = System(Params(n_users=n, length=P, L=3, c_min=5, lhh=True, max_retries=12))
        res = s.run_round(1, {i: np.zeros(P) for i in range(n)})
        if res.status == "ok" and len(res.attempts) > 1:
            break
    else:
        pytest.fail("no run with a rejected election")
    final, small = res.attempts[-1].committee, [len(a.committee) for a in res.attempts[:-1]]
    model = cm.round_bytes(P, len(final), n, small_attempts=small)
    for i in range(n):
        assert measured(s.ledger, i, "client") == model["client"]
    assert measured(s.ledger, "S", "server") == model["server"]


@pytest.mark.parametrize("when", ["before_subsum", "before_reveal"])
def test_aborted_attempt_bytes_match_ledger(when):
    """Round with one aborted attempt (member dropout) followed by a successful retry."""
    n, P = 16, 33
    for _ in range(20):
        s = System(Params(n_users=n, length=P, L=1, c_min=3, max_retries=8))
        first = next((a, c) for a in range(9) if len(c := s.preview_committee(1, a)) >= 3)
        a0, committee = first
        victim = committee[0]
        res = s.run_round(1, {i: np.zeros(P) for i in range(n)}, Behavior(member_drop={victim: when}))
        outcomes = [a.outcome for a in res.attempts]
        if res.status == "ok" and outcomes.count("member_dropout") == 1 and outcomes[-1] == "ok":
            break
    else:
        pytest.fail("no suitable run")
    idx = outcomes.index("member_dropout")
    small = [len(a.committee) for a in res.attempts[:idx]]
    later_small = [len(a.committee) for a in res.attempts[idx + 1:-1]]
    C_abort, C_ok = len(res.attempts[idx].committee), len(res.attempts[-1].committee)
    ab = cm.aborted_attempt_bytes(when, P, C_abort, n)
    ok = cm.round_bytes(P, C_ok, n, n_submit=n - 1, small_attempts=later_small)
    E = ab["sizes"].election
    members = set().union(*[set(a.committee) for a in res.attempts])
    regular = next(i for i in range(n) if i not in members)
    want_client = {key: ab["client"][key] + ok["client"][key] for key in ok["client"]}
    want_client["down"] += sum(small) * E
    assert measured(s.ledger, regular, "client") == want_client
    want_server = {key: ab["server"][key] + ok["server"][key] for key in ok["server"]}
    want_server["down"] += sum(small) * E
    assert measured(s.ledger, "S", "server") == want_server


@pytest.mark.parametrize("drops", [{}, {0: "before_subsum"}, {0: "before_reveal"},
                                   {0: "before_subsum", 1: "before_reveal"},
                                   {0: "before_shares", 1: "before_subsum"}])
def test_recovery_round_bytes_match_ledger(drops):
    n, P = 16, 29
    for _ in range(20):
        s = System(Params(n_users=n, length=P, L=1, c_min=5, tau=2, max_retries=8))
        c = s.preview_committee(1, 0)
        if len(c) >= 5:
            break
    else:
        pytest.fail("no first election reached c_min")
    victims = {c[k]: when for k, when in drops.items()}
    res = s.run_round(1, {i: np.zeros(P) for i in range(n)}, Behavior(member_drop=victims))
    assert res.status == "ok" and len(res.attempts) == 1
    kinds = list(victims.values())
    model = cm.recovery_round_bytes(P, len(c), n, d=kinds.count("before_shares"),
                                    f3=kinds.count("before_subsum"), f4=kinds.count("before_reveal"))
    regular = next(i for i in range(n) if i not in c)
    assert measured(s.ledger, regular, "client") == model["client"]
    survivor = next(j for j in c if j not in victims)
    got = measured(s.ledger, survivor, "member")
    assert (got["up"], got["down"]) == (model["member"]["up"], model["member"]["down"])
    assert measured(s.ledger, "S", "server") == model["server"]
