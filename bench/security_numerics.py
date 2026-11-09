"""Phase 7: corrected committee-size analysis (R5.2, R6.3) and the Fig. 3 / Fig. 4 data.

  .venv/bin/python bench/security_numerics.py --out results/<dir> [--trials 20]
Writes cmin_table.csv, fig3_tmax.csv, params_table.csv and fig4_vrf_sim.csv (real ECVRF outputs).
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np

from ppfl import security as sec
from ppfl import vrf

SIGMA = 40
RATIOS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
ELECTIONS = [1, 1000, 4000, 10000, 40000]  # 4000 = 1000 rounds x (1 + 3 retries)


def write(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--sim-ns", type=int, nargs="+", default=[500] + list(range(1000, 10001, 1000)))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    write(args.out / "cmin_table.csv", ["T_over_N"] + [f"E={e}" for e in ELECTIONS],
          [[r] + [sec.c_min_for(SIGMA, r, e) for e in ELECTIONS] for r in RATIOS])
    write(args.out / "fig3_tmax.csv", ["C"] + [f"E={e}" for e in ELECTIONS],
          [[c] + [sec.max_t_ratio(SIGMA, c, e) for e in ELECTIONS] for c in range(5, 41)])

    rows = []
    for E in (1, 4000):
        c_theo = (SIGMA + math.log2(E)) / -math.log2(args.ratio)
        c_min = math.ceil(c_theo)
        for N in (100, 500, 1000, 2000, 3000, 5000, 10000):
            T = int(args.ratio * N)
            L_old = sec.old_paper_L(N, c_theo)
            L_new = sec.choose_L(N, T, c_min, args.rho, honest_only=True)
            rows.append([E, N, T, round(c_theo, 2), c_min,
                         L_old, N * 2.0 ** -L_old, sec.reelection_prob(N, L_old, c_min),
                         sec.reelection_prob(N - T, L_old, c_min),
                         L_new, N * 2.0 ** -L_new, sec.reelection_prob(N, L_new, c_min),
                         sec.reelection_prob(N - T, L_new, c_min),
                         sec.eps_exact(N, T, L_new, c_min), sec.eps_bound(N, T, c_min),
                         math.log2(E * sec.eps_bound(N, T, c_min))])
    write(args.out / "params_table.csv",
          ["elections", "N", "T", "C_theoretical", "C_min", "L_old", "EC_old", "reelect_old_all",
           "reelect_old_honest_only", "L_new", "EC_new", "reelect_new_all", "reelect_new_honest_only",
           "eps_exact_new", "eps_bound", "log2_run_failure_bound"], rows)
    print("tables written", flush=True)

    # Fig. 4: committee sizes from real ECVRF outputs (E = 4000 elections, sigma = 40)
    c_theo = (SIGMA + math.log2(4000)) / -math.log2(args.ratio)
    c_min = math.ceil(c_theo)
    rows = []
    for N in args.sim_ns:
        T = int(args.ratio * N)
        keys = [os.urandom(32) for _ in range(N)]
        for rule, L in (("new", sec.choose_L(N, T, c_min, args.rho)), ("old", sec.old_paper_L(N, c_theo))):
            t0 = time.perf_counter()
            counts = []
            for trial in range(args.trials):
                alpha = f"fig4/{N}/{trial}".encode()
                counts.append(sum(vrf.selected(vrf.output(sk, alpha), L) for sk in keys))
            c = np.array(counts)
            ci = 1.96 * c.std(ddof=1) / math.sqrt(len(c))
            rows.append([N, rule, L, c_min, N * 2.0 ** -L, c.mean(), ci, (c < c_min).mean(),
                         sec.reelection_prob(N, L, c_min), " ".join(map(str, counts))])
            print(f"N={N} rule={rule} L={L} mean={c.mean():.1f} below_cmin={(c < c_min).mean():.2f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    write(args.out / "fig4_vrf_sim.csv",
          ["N", "rule", "L", "C_min", "EC", "sim_mean", "sim_ci95", "sim_frac_below_cmin",
           "theory_reelect_all", "counts"], rows)


if __name__ == "__main__":
    main()
