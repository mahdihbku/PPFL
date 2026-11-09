"""SecAgg+ graph parameters following Bell et al. (CCS 2020), Sec. 3.

Each client has k neighbours sampled from the other n - 1 clients. With X ~ HyperGeom(n - 1, gamma n, k)
the number of corrupted neighbours and Y ~ HyperGeom(n - 1, (1 - delta) n, k) the number of surviving
neighbours, the threshold t must satisfy (union bound over clients, events E1 and E3)
    n * P(X >= t) <= 2^-sigma     (security)
    n * P(Y <  t) <= 2^-eta       (correctness)
We return the smallest even k for which such a t exists. The connectivity event E2 is not checked,
so k is a lower bound on what Bell et al. would choose (favourable to the baseline).
"""

import math


def _log_comb(a, b):
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def hyper_log_pmf(pop, succ, draws):
    """log P(X = x) for x = 0..draws, X ~ HyperGeom(pop, succ, draws); -inf where impossible."""
    total = _log_comb(pop, draws)
    out = []
    for x in range(draws + 1):
        if x > succ or draws - x > pop - succ:
            out.append(-math.inf)
        else:
            out.append(_log_comb(succ, x) + _log_comb(pop - succ, draws - x) - total)
    return out


def _sum_exp(vals):
    vals = [v for v in vals if v > -math.inf]
    if not vals:
        return 0.0
    m = max(vals)
    return math.exp(m) * sum(math.exp(v - m) for v in vals)


def _tail_ge(logpmf, t):
    return _sum_exp(logpmf[t:])


def _tail_lt(logpmf, t):
    return _sum_exp(logpmf[:t])


def feasible_t(n, k, gamma, delta, sigma=40, eta=30):
    """Thresholds t that satisfy both conditions for degree k (empty if none)."""
    pop = n - 1
    lx = hyper_log_pmf(pop, int(math.floor(gamma * n)), k)
    ly = hyper_log_pmf(pop, int(math.ceil((1 - delta) * n)), k)
    sec_ok = [n * _tail_ge(lx, t) <= 2.0 ** -sigma for t in range(k + 2)]
    out = []
    for t in range(1, k + 1):
        if sec_ok[t] and n * _tail_lt(ly, t) <= 2.0 ** -eta:
            out.append(t)
    return out


def choose_k_t(n, gamma, delta, sigma=40, eta=30, k_max=None):
    """Smallest even k (< n) with a feasible t; t is the middle of the feasible range."""
    k_max = n - 1 if k_max is None else min(k_max, n - 1)
    for k in range(2, k_max + 1, 2):
        ts = feasible_t(n, k, gamma, delta, sigma, eta)
        if ts:
            return k, ts[len(ts) // 2], (ts[0], ts[-1])
    return None
