"""Per-party communication over the (N, C, P) grid from the validated byte model (costmodel.py).

  .venv/bin/python bench/comm_grid.py --out results/<dir> [--tau 2]
With --tau > 0 the online rows are successful rounds with committee-dropout recovery and no
failures (costmodel.recovery_round_bytes: every member also shares its round secrets); tau > 0
requires ephemeral keys, so only the ephemeral variants are written.
Writes online.csv (per role: overhead up/down, baseline up/down, overhead bytes per weight),
baselines.csv (SecAgg / SecAgg+ from their validated byte model) and offline.csv.
"""

import argparse
import csv
import itertools
from pathlib import Path

from ppfl import costmodel as cm
from ppfl.baselines import secagg

NS = [10, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]
CS = [3, 5, 10, 13, 15, 16, 20, 25, 30]
PS = [10000, 64811, 100000, 183429, 237506, 1000000, 10000000, 11511784, 25557032, 100000000, 138357544]
VARIANTS = {  # name -> (fs, lhh, cosign)
    "full": ("ephemeral", True, True),
    "static_keys": ("static", True, True),
    "no_lhh": ("ephemeral", False, True),
    "minimal": ("static", False, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 100], help="SecAgg+ neighbour counts")
    ap.add_argument("--tau", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "online.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "tau", "N", "C", "P", "role", "overhead_up_B", "overhead_down_B",
                    "baseline_up_B", "baseline_down_B", "overhead_B_per_weight"])
        for (name, (fs, use_lhh, cosign)), N, C, P in itertools.product(VARIANTS.items(), NS, CS, PS):
            if C > N or (args.tau and fs != "ephemeral"):
                continue
            if args.tau:
                m = cm.recovery_round_bytes(P, C, N, lhh_on=use_lhh, cosign=cosign)
            else:
                m = cm.round_bytes(P, C, N, fs=fs, lhh_on=use_lhh, cosign=cosign)
            for role in ("client", "member", "server"):
                r = m[role]
                tot = r["up"] + r["down"]
                w.writerow([name, args.tau, N, C, P, role, r["up"], r["down"], r.get("up_baseline", 0),
                            r.get("down_baseline", 0), tot / P])
    with open(args.out / "baselines.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "N", "k", "P", "role", "overhead_up_B", "overhead_down_B",
                    "baseline_up_B", "baseline_down_B"])
        for N, P in itertools.product(NS, PS):
            for scheme, k in [("secagg", N - 1)] + [("secagg+", k) for k in args.ks if k < N]:
                m = secagg.round_bytes(P, N, k)
                for role in ("client", "server"):
                    r = m[role]
                    w.writerow([scheme, N, k, P, role, r["up"], r["down"], r["up_baseline"], r["down_baseline"]])
    with open(args.out / "offline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fs", "N", "client_up_B", "client_down_B", "server_up_B", "server_down_B"])
        for fs, N in itertools.product(("ephemeral", "static"), NS):
            o = cm.offline_bytes(N, fs)
            w.writerow([fs, N, o["client"]["up"], o["client"]["down"], o["server"]["up"], o["server"]["down"]])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
