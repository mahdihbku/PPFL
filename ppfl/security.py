"""Committee-size analysis (corrected Sec. VI of the paper).

Model. N_e users are eligible at election time (N_e >= N - D), at most T of them are corrupted
(static corruption). Each eligible user is selected independently with p = 2^-L (VRF output with
at least L leading zero bits). A committee is accepted iff it has at least C_min valid proofs;
otherwise the election is repeated with a fresh input.

Per accepted election, privacy fails iff no honest user was selected:
    eps = P(c_h = 0 and c_t >= C_min) = (1 - p)^(N_e - T) * P(Bin(T, p) >= C_min)
        <= (T / N_e)^C_min                     (given |committee| = c, the committee is a uniform
                                                c-subset, so P(all corrupted) <= (T / N_e)^c)
Corrupted users withholding their proofs cannot raise eps, since the event requires c_h = 0.
Over a run with E elections (rounds plus re-elections) P(any failure) <= E * eps (union bound),
so sigma bits for the whole run need C_min >= (sigma + log2 E) / log2(N_e / T).
The old "at least two honest participants" condition is deterministic: the number of honest
participants is at least N_p - T >= N - D - T, so it holds whenever N - D >= T + 2.
"""

import math


def _log_binom_pmf(i, n, p):
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def _logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def binom_sf(k, n, p) -> float:
    """P(Bin(n, p) >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return math.exp(_logsumexp([_log_binom_pmf(i, n, p) for i in range(k, n + 1)]))


def binom_cdf_below(k, n, p) -> float:
    """P(Bin(n, p) < k)."""
    if k <= 0:
        return 0.0
    return math.exp(_logsumexp([_log_binom_pmf(i, n, p) for i in range(0, min(k, n + 1))]))


def eps_exact(n_eligible, T, L, c_min) -> float:
    p = 2.0 ** -L
    return (1.0 - p) ** (n_eligible - T) * binom_sf(c_min, T, p)


def eps_bound(n_eligible, T, c_min) -> float:
    return (T / n_eligible) ** c_min


def c_min_for(sigma, t_ratio, elections=1) -> int:
    return math.ceil((sigma + math.log2(elections)) / -math.log2(t_ratio))


def max_t_ratio(sigma, c, elections=1) -> float:
    return 2.0 ** (-(sigma + math.log2(elections)) / c)


def reelection_prob(n_selectable, L, c_min) -> float:
    """P(fewer than c_min proofs) when n_selectable users post their proofs."""
    return binom_cdf_below(c_min, n_selectable, 2.0 ** -L)


def choose_L(n_eligible, T, c_min, rho=0.01, honest_only=True) -> int:
    """Largest L whose re-election probability is at most rho. With honest_only, corrupted users
    are assumed to withhold their proofs (worst case for liveness)."""
    n_sel = n_eligible - T if honest_only else n_eligible
    L = 0
    while reelection_prob(n_sel, L + 1, c_min) <= rho:
        L += 1
    return L


def old_paper_L(N, c_theoretical) -> int:
    """Rule in the submitted paper: L = floor(log2(N / C))."""
    return math.floor(math.log2(N / c_theoretical))


# ----- recovery of up to tau failed members (privacy needs at least tau + 1 honest members) -----

def tail_tau_binom(c, t_ratio, tau) -> float:
    """P(at most tau honest among c members), members drawn with replacement (corruption ratio r)."""
    return sum(math.comb(c, h) * (1 - t_ratio) ** h * t_ratio ** (c - h) for h in range(tau + 1))


def tail_tau_hyper(c, n_eligible, T, tau) -> float:
    """Same, exact for a uniform c-subset of n_eligible users with T corrupted."""
    total = math.comb(n_eligible, c)
    return sum(math.comb(n_eligible - T, h) * math.comb(T, c - h) for h in range(tau + 1)) / total


def c_min_for_tau(sigma, t_ratio, tau, elections=1, horizon=400) -> int:
    """Smallest C_min with E * P(at most tau honest | |committee| = c) <= 2^-sigma for every c >= C_min
    (binomial form; the hypergeometric tail for finite N is smaller, see tests)."""
    target = 2.0 ** -sigma / elections
    c = tau + 1
    while True:
        if all(tail_tau_binom(cc, t_ratio, tau) <= target for cc in range(c, c + horizon)):
            return c
        c += 1


def eps_exact_tau(n_eligible, T, L, c_min, tau) -> float:
    """Per accepted election: P(c_h <= tau and c_h + c_t >= C_min) under Bernoulli sortition."""
    p = 2.0 ** -L
    return sum(math.exp(_log_binom_pmf(h, n_eligible - T, p)) * binom_sf(c_min - h, T, p)
               for h in range(tau + 1))


# ----- arbitrary selection probability p (threshold on the VRF output) --------------------------

def choose_p(n_eligible, T, c_min, rho=0.01, honest_only=True, tol=1e-6) -> float:
    """Smallest p whose re-election probability P(fewer than C_min proofs) is at most rho."""
    n_sel = n_eligible - T if honest_only else n_eligible
    lo, hi = 1e-9, 1.0
    if binom_cdf_below(c_min, n_sel, hi) > rho:
        return hi
    while hi - lo > tol * hi:
        mid = math.sqrt(lo * hi)
        if binom_cdf_below(c_min, n_sel, mid) <= rho:
            hi = mid
        else:
            lo = mid
    return hi


def eps_exact_tau_p(n_eligible, T, p, c_min, tau) -> float:
    return sum(math.exp(_log_binom_pmf(h, n_eligible - T, p)) * binom_sf(c_min - h, T, p)
               for h in range(tau + 1))
