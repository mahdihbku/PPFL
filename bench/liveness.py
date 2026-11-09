"""Phase 6: cost of committee dropouts and adversarial aborts over a training run (R6.2, R5.4).

Monte Carlo over R rounds with per-user state (corruption, cooldown). Per attempt: every eligible
user is selected with p = 2^-L; |committee| < C_min triggers re-election; each honest member drops
with probability q (half before its sub-sum, half before its reveal); in the adversarial scenario,
one corrupted member aborts (before its reveal, the costliest point for the others) whenever the
committee contains corrupted and honest members. Failed members are put on cooldown for K rounds.
A round fails after 1 + A_max unsuccessful attempts. Bytes come from the validated byte model
(costmodel.py); times from the per-role benchmark plus a stated network model.

  .venv/bin/python bench/liveness.py --timing results/<role_timing_dir> --out results/<dir>
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ppfl import costmodel as cm
from ppfl import security as sec


def nearest(rows, **key):
    return min(rows, key=lambda r: sum(abs(float(r[k]) - v) / v for k, v in key.items()))


class Timing:
    """Per-attempt latency from per-role CPU times plus transfer times on the critical path."""

    def __init__(self, timing_dir, N, C, P, client_bps, server_bps, deadline):
        load = lambda f: list(csv.DictReader(open(timing_dir / f)))
        tot = lambda r, skip: sum(float(v) for k, v in r.items() if k not in skip)
        cl = nearest(load("client.csv"), C=C, P=P)
        me = nearest(load("member.csv"), N=N, C=C, P=P)
        sv = nearest(load("server.csv"), N=N, C=C, P=P)
        self.elect = float(cl["vrf_prove"]) + float(cl["verify_committee"])
        self.client_crypto = float(cl["mask"]) + float(cl.get("lhh", 0)) + float(cl["control_post"])
        self.member_cpu = tot(me, ("N", "C", "P"))
        self.recover_subsum = float(me["derive_submasks"]) + float(me.get("lhh", 0))
        self.server_cpu = tot(sv, ("N", "C", "P", "upload_pool"))
        s = cm.sizes(P)
        up = 8 * s.w_tilde / client_bps
        self.upload = max(up, 8 * N * s.w_tilde / server_bps)  # client uplink or server ingress
        self.member_io = 8 * (N * s.control + s.subsum + s.ak) / client_bps
        self.download = max(8 * s.model / client_bps, 8 * N * s.model / server_bps)
        self.deadline = deadline

    def success(self):
        return (self.elect + self.client_crypto + self.upload + self.member_io + self.member_cpu
                + self.server_cpu + self.download)

    def aborted(self, kind):
        base = self.elect + self.client_crypto + self.upload
        if kind == "before_reveal":
            base += self.member_io + self.member_cpu + self.server_cpu
        return base + self.deadline

    def small(self):
        return self.elect + self.deadline

    def recovery(self, f3, f4):
        """Extra latency of recovering failures: one deadline per failing stage, and for each
        Case 3 failure the recomputation of r^j and its LHH."""
        return (self.deadline * ((f3 > 0) + (f4 > 0)) + f3 * self.recover_subsum)


STAGES = ("before_shares", "before_subsum", "before_reveal")


def simulate(N, t_ratio, P, c_min, sel_p, q, adversary, K, R, A_max, rng, timing, tau=0):
    """tau = 0: any member failure aborts the attempt. tau > 0: up to tau failures are recovered in
    the same attempt (shares step, Case 3/4 recovery); more abort. Honest members fail with
    probability q at a uniformly chosen stage. The adversary sees honest failures first and makes
    its corrupted members fail (before reveal) whenever that pushes the total above tau and the
    committee is mixed."""
    T = int(round(t_ratio * N))
    corrupt = np.zeros(N, dtype=bool)
    corrupt[rng.choice(N, T, replace=False)] = True
    cooldown = np.full(N, -1)
    p = sel_p
    E_sz = cm.sizes(P).election
    stats = dict(rounds=R, failed_rounds=0, elections=0, privacy_failures=0, attempts=[], aborts=0,
                 adversarial_aborts=0, recovered=0, small=0, client_bytes=0.0, server_bytes=0.0, time=0.0)
    for t in range(1, R + 1):
        for a in range(A_max + 1):
            stats["elections"] += 1
            sel = np.flatnonzero((cooldown < t) & (rng.random(N) < p))
            C = sel.size
            if C < c_min:
                stats["small"] += 1
                stats["client_bytes"] += C * E_sz
                stats["server_bytes"] += C * E_sz
                stats["time"] += timing.small()
                continue
            bad = corrupt[sel]
            honest = sel[~bad]
            fail_mask = rng.random(honest.size) < q
            failed = list(honest[fail_mask])
            stages = list(rng.choice(STAGES, size=fail_mask.sum())) if fail_mask.any() else []
            adv_fail = []
            if adversary and bad.any() and not bad.all() and len(failed) + bad.sum() > tau:
                k = max(0, tau + 1 - len(failed))  # fewest corrupted failures that force an abort
                adv_fail = list(rng.choice(sel[bad], size=min(k, bad.sum()), replace=False))
            n_fail = len(failed) + len(adv_fail)
            if n_fail > tau:
                kind = "before_reveal" if adv_fail or "before_reveal" in stages else "before_subsum"
                cooldown[failed] = t + K
                if adv_fail:
                    cooldown[adv_fail] = t + K
                    stats["adversarial_aborts"] += 1
                ab = cm.aborted_attempt_bytes(kind, P, C, N)
                stats["client_bytes"] += sum(ab["client"].values())
                stats["server_bytes"] += sum(ab["server"].values())
                stats["time"] += timing.aborted(kind)
                stats["aborts"] += 1
                continue
            if bad.all():
                stats["privacy_failures"] += 1
            if tau:
                d, f3, f4 = (stages.count(x) for x in STAGES)
                ok = cm.recovery_round_bytes(P, C, N, d=d, f3=f3, f4=f4)
                stats["recovered"] += n_fail
                cooldown[failed] = t + K
                stats["time"] += timing.success() + timing.recovery(f3, f4)
            else:
                ok = cm.round_bytes(P, C, N)
                stats["time"] += timing.success()
            stats["client_bytes"] += sum(ok["client"].values())
            stats["server_bytes"] += sum(ok["server"].values())
            stats["attempts"].append(a + 1)
            break
        else:
            stats["failed_rounds"] += 1
            stats["attempts"].append(A_max + 1)
    att = stats.pop("attempts")
    stats["mean_attempts"] = float(np.mean(att))
    for k in ("client_bytes", "server_bytes", "time"):
        stats[k + "_per_round"] = stats.pop(k) / R
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--Ns", type=int, nargs="+", default=[1000, 3000])
    ap.add_argument("--P", type=int, default=64811)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--A-max", type=int, default=3)
    ap.add_argument("--qs", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--Ks", type=int, nargs="+", default=[0, 10, 100, 1000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--client-mbps", type=float, default=10.0)
    ap.add_argument("--server-gbps", type=float, default=1.0)
    ap.add_argument("--deadline", type=float, default=2.0)
    ap.add_argument("--L-rules", nargs="+", default=["pow2_rho1", "p_rho1", "p_rho10"],
                    help="pow2_rho1: p = 2^-L, largest L with re-election <= 1%% if corrupted users withhold; "
                         "p_rho1 / p_rho10: smallest real p with re-election <= 1%% / 10%% (same worst case)")
    ap.add_argument("--taus", type=int, nargs="+", default=[0, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    E = args.rounds * (1 + args.A_max)
    rows = []
    for N, tau in ((N, tau) for N in args.Ns for tau in args.taus):
        T = int(round(args.ratio * N))
        c_min = sec.c_min_for_tau(40, args.ratio, tau, E)
        Ls = {"pow2_rho1": 2.0 ** -sec.choose_L(N, T, c_min, 0.01),
              "p_rho1": sec.choose_p(N, T, c_min, 0.01),
              "p_rho10": sec.choose_p(N, T, c_min, 0.10)}
        timing = Timing(args.timing, N, 20, args.P, args.client_mbps * 1e6, args.server_gbps * 1e9, args.deadline)
        base_time = timing.success()
        scen = [("honest", q, False, 10) for q in args.qs] + [("adversary", 0.01, True, K) for K in args.Ks]
        for (rule, L), (name, q, adv, K) in ((rl, sc) for rl in Ls.items() if rl[0] in args.L_rules for sc in scen):
            runs = [simulate(N, args.ratio, args.P, c_min, L, q, adv, K, args.rounds, args.A_max,
                             np.random.default_rng(1000 * s + N), timing, tau) for s in range(args.seeds)]
            avg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            rows.append(dict(N=N, T=T, tau=tau, rule=rule, p=L, C_min=c_min, EC=N * L, scenario=name,
                             q=q, cooldown=K, ideal_round_time_s=base_time, **avg))
            r = rows[-1]
            print(f"N={N} tau={tau} Cmin={c_min} {rule}(EC={r['EC']:.0f}) {name:9s} q={q:<5} K={K:<4} "
                  f"attempts={r['mean_attempts']:.3f} failed={r['failed_rounds']:.1f} rec={r['recovered']:.0f} "
                  f"adv_aborts={r['adversarial_aborts']:.0f} priv={r['privacy_failures']:.0f} "
                  f"client={r['client_bytes_per_round'] / 1e6:.3f}MB server={r['server_bytes_per_round'] / 1e9:.3f}GB "
                  f"time={r['time_per_round']:.2f}s (ideal {base_time:.2f}s)", flush=True)
    with open(args.out / "liveness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "params.json").write_text(json.dumps(vars(args) | {"elections_budget": E},
                                                     default=str, indent=2))


if __name__ == "__main__":
    main()
