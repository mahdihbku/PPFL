"""End-to-end rounds used to validate the byte model and the per-role timings.

  .venv/bin/python bench/e2e_validate.py --n 500 --L 5 --c-min 10 --workers 10 \
      --timing results/<role_timing_dir> --out results/<dir>
Checks every party's bytes against costmodel.round_bytes (exact) and reports measured member and
server times next to the per-role benchmark. Client work runs in a process pool, so client
times from this script are not used (pool workers share cores, including efficiency cores).
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from ppfl import costmodel as cm
from ppfl.protocol import Params, System

EXCL = cm.BASELINE_KINDS | cm.OFFLINE_KINDS


def nearest(rows, **key):
    def dist(r):
        return sum(abs(float(r[k]) - v) / v for k, v in key.items())
    return min(rows, key=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, default=64811)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--c-min", type=int, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    s = System(Params(n_users=args.n, length=args.p, L=args.L, c_min=args.c_min, max_retries=10),
               workers=args.workers)
    rng = np.random.default_rng(0)
    res = s.run_round(1, {i: rng.normal(0, 0.1, args.p) for i in range(args.n)})
    wall = time.perf_counter() - t0
    assert res.status == "ok", res
    final = res.attempts[-1].committee
    small = [len(a.committee) for a in res.attempts[:-1]]
    model = cm.round_bytes(args.p, len(final), args.n, small_attempts=small)
    L = s.ledger
    mismatches = []
    for i in range(args.n):
        got = (L.party_bytes(i, "client", "up", exclude_kinds=EXCL), L.party_bytes(i, "client", "down", exclude_kinds=EXCL))
        if got != (model["client"]["up"], model["client"]["down"]):
            mismatches.append(("client", i, got))
    prior = {j: sum(j in a.committee for a in res.attempts[:-1]) for j in final}
    for j in final:
        got = (L.party_bytes(j, "member", "up", exclude_kinds=EXCL), L.party_bytes(j, "member", "down", exclude_kinds=EXCL))
        want = (model["member"]["up"] + prior[j] * model["sizes"].election, model["member"]["down"])
        if got != want:
            mismatches.append(("member", j, got, want))
    got_s = (L.party_bytes("S", "server", "up", exclude_kinds=EXCL), L.party_bytes("S", "server", "down", exclude_kinds=EXCL))
    if got_s != (model["server"]["up"], model["server"]["down"]):
        mismatches.append(("server", got_s))

    member_t = {ph: statistics.median(L.party_time(j, "member", {ph}) for j in final) for ph in L.phases("member")}
    server_t = {ph: L.party_time("S", "server", {ph}) for ph in L.phases("server")}
    out = {"n": args.n, "p": args.p, "L": args.L, "committee_size": len(final),
           "attempts": [a.outcome for a in res.attempts], "wall_s": wall,
           "byte_mismatches": [str(m) for m in mismatches],
           "member_time_s": member_t, "server_time_s": server_t}
    if args.timing:
        with open(args.timing / "member.csv") as fh:
            mrow = nearest(list(csv.DictReader(fh)), N=args.n, C=len(final), P=args.p)
        with open(args.timing / "server.csv") as fh:
            srow = nearest(list(csv.DictReader(fh)), N=args.n, C=len(final), P=args.p)
        out["role_bench_member_row"] = mrow
        out["role_bench_server_row"] = srow
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"e2e_n{args.n}_p{args.p}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if mismatches:
        raise SystemExit("byte model mismatch")


if __name__ == "__main__":
    main()
