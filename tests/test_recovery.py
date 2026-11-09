"""Committee-member failures recovered within the same attempt (tau = 2)."""

import numpy as np
import pytest

from ppfl import codec
from ppfl.protocol import Behavior, Params, System

N, P = 16, 47


def make(**kw):
    params = dict(n_users=N, length=P, L=1, c_min=5, tau=2, max_retries=6)
    params.update(kw)
    s = System(Params(**params))
    rng = np.random.default_rng(3)
    return s, {i: rng.normal(0, 1, P) for i in range(N)}


def first_committee(s, t=1):
    for a in range(s.p.max_retries + 1):
        c = s.preview_committee(t, a)
        if len(c) >= s.p.c_min:
            return a, c
    pytest.skip("no committee reaches c_min for this key draw")


def exact(x, ids):
    out = np.zeros(P, dtype=np.uint32)
    for i in ids:
        out += codec.encode(x[i], 16)
    return out


def test_no_failure_with_sharing():
    s, x = make()
    res = s.run_round(1, x)
    assert res.status == "ok" and np.array_equal(res.encoded_sum, exact(x, range(N)))
    assert any(e.kind == "share" for e in s.bb.entries)


@pytest.mark.parametrize("drops", [
    {0: "before_subsum"},
    {0: "before_reveal"},
    {0: "before_subsum", 1: "before_subsum"},
    {0: "before_reveal", 1: "before_reveal"},
    {0: "before_subsum", 1: "before_reveal"},
    {0: "before_shares", 1: "before_subsum"},
])
def test_up_to_tau_failures_recovered_in_same_attempt(drops):
    s, x = make()
    a0, c = first_committee(s)
    victims = {c[k]: when for k, when in drops.items()}
    res = s.run_round(1, x, Behavior(member_drop=victims))
    assert res.status == "ok"
    last = res.attempts[-1]
    assert last.attempt == a0 and last.outcome == "ok" and sorted(last.recovered) == sorted(victims)
    assert res.participants == list(range(N))  # failed members still contributed as clients
    assert np.array_equal(res.encoded_sum, exact(x, range(N)))
    for v in victims:
        assert s.cooldown_until[v] == 1 + s.p.cooldown_rounds
    if any(w != "before_shares" for w in victims.values()):
        assert any(e.kind == "release" for e in s.bb.entries)


@pytest.mark.parametrize("drops", [
    {0: "before_subsum", 1: "before_subsum", 2: "before_subsum"},
    {0: "before_subsum", 1: "before_reveal", 2: "before_reveal"},
    {0: "before_shares", 1: "before_shares", 2: "before_shares"},
    {0: "before_shares", 1: "before_subsum", 2: "before_reveal"},
])
def test_more_than_tau_failures_abort_then_retry(drops):
    s, x = make()
    a0, c = first_committee(s)
    victims = {c[k]: when for k, when in drops.items()}
    res = s.run_round(1, x, Behavior(member_drop=victims))
    assert res.attempts[[a.attempt for a in res.attempts].index(a0)].outcome == "member_dropout"
    assert res.status == "ok"
    ids = [i for i in range(N) if i not in victims]
    assert res.participants == ids and np.array_equal(res.encoded_sum, exact(x, ids))


def test_no_secret_released_when_aborting():
    """Above tau the attempt aborts before any share is released."""
    s, x = make(max_retries=0)
    a0, c = first_committee(s)
    if a0 != 0:
        pytest.skip("first election too small")
    s.run_round(1, x, Behavior(member_drop={c[0]: "before_subsum", c[1]: "before_subsum",
                                              c[2]: "before_subsum"}))
    assert not any(e.kind == "release" for e in s.bb.entries)


def test_recovery_requires_ephemeral_keys():
    with pytest.raises(ValueError):
        Params(n_users=N, length=P, L=1, c_min=5, tau=2, fs="static")


def test_lhh_still_detects_tampering_with_recovery():
    s, x = make()
    a0, c = first_committee(s)
    delta = np.zeros(P, dtype=np.int64)
    delta[1] = 7
    res = s.run_round(1, x, Behavior(member_drop={c[0]: "before_subsum"}, server_delta=delta))
    assert res.status == "detected" and "aggregate_tamper" in res.detections
