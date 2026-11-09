"""Fig. 4 data for the threshold selection rule (selected iff int(beta) < p * 2^512), real ECVRF outputs.

For each N and tau: C_min for sigma = 40 over E = 4000 elections, p = smallest probability with
re-election <= rho when corrupted users withhold (security.choose_p), simulated committee sizes.

  .venv/bin/python bench/fig4_threshold.py --out results/<dir> [--trials 20]
"""

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np

from ppfl import security as sec
from ppfl import vrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--elections", type=int, default=4000)
    ap.add_argument("--taus", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[500] + list(range(1000, 10001, 1000)))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for N in args.ns:
        T = int(args.ratio * N)
        keys = [os.urandom(32) for _ in range(N)]
        for tau in args.taus:
            c_min = sec.c_min_for_tau(40, args.ratio, tau, args.elections)
            p = sec.choose_p(N, T, c_min, args.rho)
            thr = vrf.threshold(p)
            counts = []
            for trial in range(args.trials):
                alpha = f"fig4thr/{N}/{tau}/{trial}".encode()
                counts.append(sum(vrf.below(vrf.output(sk, alpha), thr) for sk in keys))
            c = np.array(counts)
            rows.append([N, tau, c_min, p, N * p, c.mean(), 1.96 * c.std(ddof=1) / math.sqrt(len(c)),
                         (c < c_min).mean(), sec.binom_cdf_below(c_min, N, p),
                         sec.binom_cdf_below(c_min, N - T, p), sec.eps_exact_tau_p(N, T, p, c_min, tau),
                         " ".join(map(str, counts))])
            print(f"N={N} tau={tau} C_min={c_min} EC={N * p:.1f} sim={c.mean():.1f} below={(c < c_min).mean():.2f}",
                  flush=True)
    with open(args.out / "fig4_threshold.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "tau", "C_min", "p", "EC", "sim_mean", "sim_ci95", "sim_frac_below_cmin",
                    "theory_reelect_all", "theory_reelect_withholding", "eps_exact", "counts"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
