import numpy as np
import pytest

from ppfl import codec
from ppfl import costmodel as cm
from ppfl.baselines import secagg, shamir


def test_shamir_roundtrip_any_t_subset():
    secrets = [bytes(range(32)), b"\xff" * 32]
    xs = list(range(1, 21))
    sh = shamir.share(secrets, xs, t=7)
    assert sh.shape == (20, 2 * shamir.LIMBS)
    for pick in ([0, 1, 2, 3, 4, 5, 6], [3, 5, 7, 11, 13, 17, 19]):
        assert shamir.reconstruct([xs[i] for i in pick], sh[pick], 2) == secrets
    wrong = shamir.reconstruct(xs[:6], sh[:6], 2)  # t-1 shares do not reconstruct
    assert wrong != secrets


def expected(inputs, ids, f=16):
    s = np.zeros(next(iter(inputs.values())).size, dtype=np.uint32)
    for i in ids:
        s += codec.encode(inputs[i], f)
    return s


@pytest.mark.parametrize("k", [None, 6])
def test_secagg_exact_without_dropouts(k):
    n, P = 16, 53
    rng = np.random.default_rng(0)
    x = {i: rng.normal(0, 1, P) for i in range(n)}
    s = secagg.SecAggSystem(n, P, k=k, seed=1)
    z, survivors = s.run_round(x)
    assert survivors == list(range(n)) and np.array_equal(z, expected(x, range(n)))


@pytest.mark.parametrize("k", [None, 8])
def test_secagg_dropouts_in_every_round(k):
    n, P = 20, 40
    rng = np.random.default_rng(2)
    x = {i: rng.normal(0, 1, P) for i in range(n)}
    s = secagg.SecAggSystem(n, P, k=k, seed=3)
    z, survivors = s.run_round(x, drop_before_share={0}, drop_after_share={5, 9}, drop_after_mask={12})
    ids = [i for i in range(n) if i not in (0, 5, 9)]
    assert survivors == ids and np.array_equal(z, expected(x, ids))


def test_harary_graph_is_k_regular_and_symmetric():
    g = secagg.harary_graph(30, 6, np.random.default_rng(0))
    assert all(len(nb) == 6 and u not in nb for u, nb in enumerate(g))
    assert all(u in g[v] for u, nb in enumerate(g) for v in nb)


@pytest.mark.parametrize("k", [None, 6])
def test_secagg_round_bytes_match_ledger(k):
    n, P = 14, 31
    x = {i: np.zeros(P) for i in range(n)}
    s = secagg.SecAggSystem(n, P, k=k, seed=4)
    s.run_round(x)
    deg = n - 1 if k is None else k
    model = secagg.round_bytes(P, n, deg)
    L = s.ledger
    for i in range(n):
        got = {"up": L.party_bytes(i, "client", "up", exclude_kinds=cm.BASELINE_KINDS),
               "down": L.party_bytes(i, "client", "down", exclude_kinds=cm.BASELINE_KINDS),
               "up_baseline": L.party_bytes(i, "client", "up", kinds=cm.BASELINE_KINDS),
               "down_baseline": L.party_bytes(i, "client", "down", kinds=cm.BASELINE_KINDS)}
        assert got == model["client"]
    got_s = {"up": L.party_bytes("S", "server", "up", exclude_kinds=cm.BASELINE_KINDS),
             "down": L.party_bytes("S", "server", "down", exclude_kinds=cm.BASELINE_KINDS),
             "up_baseline": L.party_bytes("S", "server", "up", kinds=cm.BASELINE_KINDS),
             "down_baseline": L.party_bytes("S", "server", "down", kinds=cm.BASELINE_KINDS)}
    assert got_s == model["server"]


def test_secagg_plus_parameters_match_bell_et_al_anchors():
    import math
    from ppfl.baselines import secagg_params as sp
    # Fig. 1: n=1e4, k=200, gamma=1/5, delta=1/10, t=100 gives P[X>=t] < 2^-40 and P[Y<t] < 2^-30
    n, k, t = 10**4, 200, 100
    assert sp._tail_ge(sp.hyper_log_pmf(n - 1, n // 5, k), t) < 2.0 ** -40
    assert sp._tail_lt(sp.hyper_log_pmf(n - 1, int(0.9 * n), k), t) < 2.0 ** -30
    # Table 2: gamma=1/20, delta=1/3, sigma=40, eta=30 needs "roughly 100" neighbours
    for n in (10**3, 10**4, 10**5):
        k, _, _ = sp.choose_k_t(n, 1 / 20, 1 / 3)
        assert 70 <= k <= 130
