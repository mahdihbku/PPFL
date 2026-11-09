"""Per-party cost of one aggregation round (bytes on the wire and CPU time per phase).

Usage: .venv/bin/python bench/round_cost.py --n 100 --p 64811 --L 3 --c-min 5 --out results/<dir>
Baseline FL traffic (masked-model upload, model download) is reported separately from overhead.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ppfl.protocol import Params, System

BASELINE_KINDS = {"w_tilde", "model"}


def summarise(sys_, res):
    L = sys_.ledger
    committee = res.attempts[-1].committee
    clients = res.participants
    out = {"committee_size": len(committee), "participants": len(clients),
           "attempts": [a.outcome for a in res.attempts], "status": res.status}

    def role_stats(parties, role):
        up = [L.party_bytes(q, role, "up", exclude_kinds=BASELINE_KINDS) for q in parties]
        down = [L.party_bytes(q, role, "down", exclude_kinds=BASELINE_KINDS) for q in parties]
        phases = sorted(L.phases(role))
        t = {ph: statistics.median(L.party_time(q, role, {ph}) for q in parties) for ph in phases}
        return {"overhead_up_B": statistics.median(up), "overhead_down_B": statistics.median(down),
                "time_s": t, "time_total_s": sum(t.values())}

    out["client"] = role_stats(clients, "client")
    out["client"]["baseline_B"] = statistics.median(
        L.party_bytes(q, "client", kinds=BASELINE_KINDS) for q in clients)
    out["member"] = role_stats(committee, "member")
    out["server"] = role_stats(["S"], "server")
    out["server"]["baseline_B"] = L.party_bytes("S", "server", kinds=BASELINE_KINDS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--c-min", type=int, required=True)
    ap.add_argument("--fs", default="ephemeral")
    ap.add_argument("--no-lhh", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    params = Params(n_users=args.n, length=args.p, L=args.L, c_min=args.c_min, fs=args.fs,
                    lhh=not args.no_lhh, max_retries=10)
    t0 = time.perf_counter()
    sys_ = System(params)
    rng = np.random.default_rng(args.seed)
    inputs = {i: rng.normal(0, 0.1, args.p) for i in range(args.n)}
    res = sys_.run_round(1, inputs)
    summary = summarise(sys_, res)
    summary["params"] = vars(args) | {"out": str(args.out)}
    summary["wall_s"] = time.perf_counter() - t0
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"round_n{args.n}_p{args.p}_L{args.L}_{args.fs}{'_nolhh' if args.no_lhh else ''}.json"
    (args.out / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
