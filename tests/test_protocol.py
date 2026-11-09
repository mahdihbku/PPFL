import numpy as np
import pytest

from ppfl import codec
from ppfl.protocol import Behavior, Params, System

N, P = 12, 101


def make(seed=0, **kw):
    params = dict(n_users=N, length=P, L=1, c_min=2, max_retries=5)
    params.update(kw)
    sys_ = System(Params(**params))
    rng = np.random.default_rng(seed)
    inputs = {i: rng.normal(0, 1, P) for i in range(N)}
    return sys_, inputs


def first_good_attempt(sys_, t, online=None):
    """Index and committee of the first attempt whose committee meets c_min."""
    for a in range(sys_.p.max_retries + 1):
        c = sys_.preview_committee(t, a, online)
        if len(c) >= sys_.p.c_min:
            return a, c
    pytest.skip("no attempt reaches c_min for this key draw")


def expected(inputs, ids, f=16):
    s = np.zeros(P, dtype=np.uint32)
    for i in ids:
        s += codec.encode(inputs[i], f)
    return s


@pytest.mark.parametrize("fs", ["ephemeral", "static"])
@pytest.mark.parametrize("use_lhh", [True, False])
def test_honest_round_is_exact(fs, use_lhh):
    s, x = make(fs=fs, lhh=use_lhh)
    res = s.run_round(1, x)
    assert res.status == "ok" and res.participants == list(range(N))
    assert np.array_equal(res.encoded_sum, expected(x, range(N)))
    assert np.allclose(res.aggregate, sum(x.values()), atol=N * 2**-16)
    assert res.detections == []


def test_client_dropout_cases_1_and_2():
    s, x = make()
    res = s.run_round(1, x, Behavior(drop_before_submit={2, 5}, drop_after_submit={7}))
    ids = [i for i in range(N) if i not in (2, 5)]
    assert res.status == "ok" and res.participants == ids
    assert np.array_equal(res.encoded_sum, expected(x, ids))


@pytest.mark.parametrize("when", ["before_subsum", "before_reveal"])
def test_member_dropout_aborts_reelects_and_cools_down(when):
    s, x = make()
    a0, committee = first_good_attempt(s, 1)
    victim = committee[0]
    res = s.run_round(1, x, Behavior(member_drop={victim: when}))
    assert res.status == "ok"
    assert [at.outcome for at in res.attempts if at.committee].count("member_dropout") == 1
    assert victim not in res.participants and victim not in res.attempts[-1].committee
    assert np.array_equal(res.encoded_sum, expected(x, res.participants))
    assert s.cooldown_until[victim] == 1 + s.p.cooldown_rounds
    assert victim not in s.preview_committee(2, 0)  # still cooling down next round


def test_bad_reveal_is_caught_by_commitment():
    s, x = make()
    _, committee = first_good_attempt(s, 1)
    res = s.run_round(1, x, Behavior(member_bad_seed={committee[-1]}))
    assert any(at.outcome == "bad_reveal" and at.culprits == [committee[-1]] for at in res.attempts)
    assert res.status == "ok"
    assert np.array_equal(res.encoded_sum, expected(x, res.participants))


def test_server_tampering_with_aggregate_is_detected():
    s, x = make()
    delta = np.zeros(P, dtype=np.int64)
    delta[3] = 12345
    res = s.run_round(1, x, Behavior(server_delta=delta))
    assert res.status == "detected" and "aggregate_tamper" in res.detections


def test_without_lhh_server_tampering_goes_through():
    """Why the LHH check exists: commit-reveal alone does not bind A to the submitted updates."""
    s, x = make(lhh=False)
    delta = np.zeros(P, dtype=np.int64)
    delta[3] = 12345
    res = s.run_round(1, x, Behavior(server_delta=delta))
    assert res.status == "ok"
    assert not np.array_equal(res.encoded_sum, expected(x, range(N)))


def test_dropout_announcement_equivocation_is_detected():
    s, x = make()
    res = s.run_round(1, x, Behavior(server_equivocate=True))
    assert res.status == "detected" and "equivocation:delta" in res.detections


def test_model_tampering_is_detected_by_cosignatures():
    s, x = make()
    res = s.run_round(1, x, Behavior(server_tamper_model_for={4}))
    assert "model_tamper:4" in res.detections


def test_client_with_wrong_lhh_is_identified_and_excluded():
    s, x = make()
    res = s.run_round(1, x, Behavior(client_bad_lhh={6}))
    assert any(at.outcome == "lhh_dispute" and at.culprits == [6] for at in res.attempts)
    assert res.status == "ok" and 6 not in res.participants
    assert np.array_equal(res.encoded_sum, expected(x, res.participants))


def test_small_committees_are_reelected_until_the_cap():
    s, x = make(c_min=N + 1, max_retries=3)
    res = s.run_round(1, x)
    assert res.status == "failed"
    assert [at.outcome for at in res.attempts] == ["small_committee"] * 4
    assert s.elections == 4


def test_similarity_check_honest():
    s, x = make(similarity=True)
    g = np.random.default_rng(9).normal(0, 1, P)
    res = s.run_round(1, x, global_model=g)
    assert res.status == "ok" and res.detections == []


def test_message_sizes():
    s, x = make()
    res = s.run_round(1, x)
    a = res.attempts[-1].attempt
    sizes = {e.kind: e.size for e in s.bb.entries if e.attempt == a}
    assert sizes["election"] == 9 + 80 + 32 + 64
    assert sizes["control"] == 9 + 32 + 33 + 32 + 64
    assert sizes["mcontrol"] == 9 + 32 + 33 + 32 + 64
    assert sizes["reveal"] == sizes["cosign"] == 9 + 32 + 64
    j = res.attempts[-1].committee[0]
    ak = s.ledger.party_bytes(j, role="member", direction="down", kinds={"AK"})
    assert ak == 9 + 6 * P


def test_multiple_rounds_rotate_committees():
    s, x = make()
    committees = []
    for t in range(1, 4):
        res = s.run_round(t, x)
        assert res.status == "ok"
        committees.append(tuple(res.attempts[-1].committee))
    assert len(set(committees)) > 1


def test_fractional_selection_probability():
    s, x = make(select_p=0.3, c_min=2)
    res = s.run_round(1, x)
    assert res.status == "ok" and np.array_equal(res.encoded_sum, expected(x, range(N)))
    assert s.p.threshold == int(0.3 * (1 << 512))


def test_server_cannot_isolate_a_user_by_declaring_the_others_dropped():
    """Members refuse a dropout announcement larger than d_max, so nothing is revealed."""
    s, x = make(d_max=4)
    victims = set(range(1, N))  # everyone but user 0
    res = s.run_round(1, x, Behavior(server_exclude=victims))
    assert res.status == "failed" and res.aggregate is None
    assert all(at.outcome == "too_many_dropouts" for at in res.attempts if at.outcome != "small")


def test_exclusions_within_d_max_are_visible_and_bounded():
    s, x = make(d_max=4)
    res = s.run_round(1, x, Behavior(server_exclude={3, 8}))
    ids = [i for i in range(N) if i not in (3, 8)]
    assert res.status == "ok" and res.participants == ids
    assert np.array_equal(res.encoded_sum, expected(x, ids))
    assert {"excluded:3", "excluded:8"} <= set(res.detections)


def test_default_dropout_cap_is_half_the_users():
    s, x = make()
    assert s.p.dropout_cap == N // 2
    res = s.run_round(1, x, Behavior(drop_before_submit=set(range(N // 2 + 1))))
    assert res.status == "failed" and res.aggregate is None
