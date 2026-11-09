import math
import random

import pytest

from ppfl import security as sec


def test_binomial_tails_sum_to_one():
    for n, p, k in ((50, 0.1, 5), (3000, 2**-7, 16), (300, 0.5, 150)):
        assert math.isclose(sec.binom_sf(k, n, p) + sec.binom_cdf_below(k, n, p), 1.0, rel_tol=1e-9)


def test_exact_eps_below_bound_and_monte_carlo():
    N, T, L, c = 30, 20, 2, 3  # eps around 0.05, so sampling can check the closed form
    assert sec.eps_exact(N, T, L, c) <= sec.eps_bound(N, T, c)
    rng, hits, trials, p = random.Random(0), 0, 200_000, 2.0 ** -L
    for _ in range(trials):
        honest_selected = sum(rng.random() < p for _ in range(N - T))
        corrupt_selected = sum(rng.random() < p for _ in range(T))
        hits += honest_selected == 0 and corrupt_selected >= c
    assert sec.eps_exact(N, T, L, c) == pytest.approx(hits / trials, rel=0.05)


def test_cmin_matches_paper_single_election():
    assert sec.c_min_for(40, 0.1) == 13  # paper: C >= 12.04
    assert sec.c_min_for(40, 0.1, elections=4000) == 16
    assert math.isclose(sec.max_t_ratio(40, 10) * 100, 6.25)  # paper: N=100, C=10 -> T <= 6


def test_choose_L_meets_rho():
    L = sec.choose_L(3000, 300, 16, rho=0.01)
    assert sec.reelection_prob(2700, L, 16) <= 0.01 < sec.reelection_prob(2700, L + 1, 16)


def test_tau_zero_reduces_to_previous_analysis():
    for r in (0.05, 0.1, 0.3):
        for E in (1, 4000):
            assert sec.c_min_for_tau(40, r, 0, E) == sec.c_min_for(40, r, E)
    assert sec.eps_exact_tau(1000, 100, 5, 16, 0) == pytest.approx(sec.eps_exact(1000, 100, 5, 16))


def test_hypergeometric_tail_below_binomial():
    for N, T, c, tau in ((1000, 100, 21, 2), (3000, 300, 25, 2), (500, 150, 30, 1)):
        assert sec.tail_tau_hyper(c, N, T, tau) <= sec.tail_tau_binom(c, T / N, tau)


def test_eps_tau_monte_carlo():
    N, T, L, c, tau = 30, 18, 2, 4, 1
    rng, hits, trials, p = random.Random(1), 0, 200_000, 2.0 ** -L
    for _ in range(trials):
        h = sum(rng.random() < p for _ in range(N - T))
        k = sum(rng.random() < p for _ in range(T))
        hits += h <= tau and h + k >= c
    assert sec.eps_exact_tau(N, T, L, c, tau) == pytest.approx(hits / trials, rel=0.05)


def test_choose_p_meets_rho_and_is_minimal():
    p = sec.choose_p(3000, 300, 20, rho=0.01)
    assert sec.binom_cdf_below(20, 2700, p) <= 0.01 < sec.binom_cdf_below(20, 2700, p * 0.99)
    assert 2700 * p < 3000 * 2.0 ** -6  # smaller committee than the power-of-two rule (E[C] = 47)
