"""Per-role CPU time of SecAgg / SecAgg+ (semi-honest) with the same primitives as our protocol.

Client: all four rounds with `deg` neighbours (real keys, real ciphertexts). Server: share
reconstruction for every user plus self-mask removal, plus pairwise-mask removal for a fraction
`drop` of users that dropped between rounds 1 and 2. Synthetic but well-formed inputs; single
process; median over --reps (client) and a single run for large servers.

  .venv/bin/python bench/baseline_timing.py --out results/<dir> [--ks 32 64 100]
"""

import argparse
import csv
import itertools
import os
import statistics
import time
from pathlib import Path

import numpy as np

from ppfl.baselines import secagg, shamir
from ppfl.keys import KAKey

FRAC = 16


def timeit(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def bench_client(deg, t, P, reps):
    u = 0
    nbrs = list(range(1, deg + 1))
    c_nb = {v: KAKey() for v in nbrs}
    s_nb = {v: KAKey() for v in nbrs}
    x = np.random.default_rng(0).normal(0, 0.1, P)
    row = {"r0_keygen": timeit(lambda: (KAKey(), KAKey()), reps)}
    c_key, s_key, b = KAKey(), KAKey(), os.urandom(32)
    cpk = {v: k.pk for v, k in c_nb.items()}
    row["r1_share"] = timeit(lambda: secagg.client_share_keys(u, c_key, s_key.raw(), b, cpk, t), reps)
    spk = {v: k.pk for v, k in s_nb.items()}
    row["r2_mask"] = timeit(lambda: secagg.client_masked_input(u, x, s_key, b, spk, P, FRAC), reps)
    # ciphertexts addressed to u, produced by its neighbours
    keys, cts = {}, {}
    for v in nbrs:
        cts_v, keys_v = secagg.client_share_keys(v, c_nb[v], s_nb[v].raw(), os.urandom(32), {u: c_key.pk}, 1)
        cts[v] = cts_v[u]
        keys[v] = keys_v[u]
    survivors = set(nbrs)
    row["r3_unmask"] = timeit(lambda: secagg.client_unmask(u, keys, cts, survivors), reps)
    return row


def bench_server(N, k, t, P, drop, reps, max_bytes=3 << 30):
    rng = np.random.default_rng(1)
    graph = secagg.complete_graph(N) if k is None else secagg.harary_graph(N, k, rng)
    complete = k is None
    n_drop = int(round(drop * N))
    dropped = set(range(N - n_drop, N))
    survivors = set(range(N - n_drop))
    pool = max(1, min(len(survivors), max_bytes // (4 * P)))
    bufs = [rng.integers(0, 2**32, P, dtype=np.int64).astype(np.uint32) for _ in range(pool)]
    ys = {u: bufs[u % pool] for u in survivors}
    owners = survivors | dropped
    holders = sorted(survivors)[:t + 1] if complete else sorted(survivors)
    released = {}
    for h in holders:
        mine = [o for o in graph[h] if o in owners]
        blob = os.urandom(shamir.SHARE_LEN * len(mine))
        released[h] = {o: blob[i * shamir.SHARE_LEN:(i + 1) * shamir.SHARE_LEN] for i, o in enumerate(mine)}
    spk = {u: KAKey().pk for u in range(N)}
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        secagg.server_unmask(ys, released, graph, survivors, dropped, spk, t, P, complete)
        times.append(time.perf_counter() - t0)
    return {"r3_reconstruct_unmask": statistics.median(times), "upload_pool": pool}


def write(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scheme", "N", "k", "t", "P", "drop"), k))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[100, 500, 1000, 2000, 3000])
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 100])
    ap.add_argument("--Ps", type=int, nargs="+", default=[64811])
    ap.add_argument("--drops", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--roles", nargs="+", default=["client", "server"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bell-gamma", type=float, default=None,
                    help="use Bell et al.'s (k, t) per N for this corruption ratio instead of --ks")
    ap.add_argument("--bell-deltas", type=float, nargs="+", default=[0.05, 0.1])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.txt").write_text(f"{os.popen('sysctl -n machdep.cpu.brand_string').read().strip()}\n{vars(args)}\n")
    if args.bell_gamma is not None:
        from ppfl.baselines.secagg_params import choose_k_t
        rows_c, rows_s = [], []
        for N, P, delta in itertools.product(args.Ns, args.Ps, args.bell_deltas):
            k, t, _ = choose_k_t(N, args.bell_gamma, delta)
            base = {"scheme": "secagg+", "N": N, "k": k, "t": t, "P": P, "gamma": args.bell_gamma, "delta": delta}
            if "client" in args.roles:
                rows_c.append({**base, **bench_client(k, t, P, args.reps)})
                print("client", rows_c[-1], flush=True)
            if "server" in args.roles:
                for drop in (0.0, delta):
                    rows_s.append({**base, "drop": drop, **bench_server(N, k, t, P, drop, args.reps)})
                    print("server", rows_s[-1], flush=True)
        if rows_c:
            write(args.out / "client_bell.csv", rows_c)
        if rows_s:
            write(args.out / "server_bell.csv", rows_s)
        return
    schemes = [("secagg", None)] + [("secagg+", k) for k in args.ks]
    if "client" in args.roles:
        rows = []
        for (name, k), N, P in itertools.product(schemes, args.Ns, args.Ps):
            deg = N - 1 if k is None else k
            if deg >= N:
                continue
            t = deg // 2 + 1
            rows.append({"scheme": name, "N": N, "k": deg, "t": t, "P": P, **bench_client(deg, t, P, args.reps)})
            print("client", name, N, deg, P, f"{sum(v for kk, v in rows[-1].items() if kk.startswith('r')):.3f}s", flush=True)
        write(args.out / "client.csv", rows)
    if "server" in args.roles:
        rows = []
        for (name, k), N, P, drop in itertools.product(schemes, args.Ns, args.Ps, args.drops):
            deg = N - 1 if k is None else k
            if deg >= N:
                continue
            t = deg // 2 + 1
            reps = 1 if (k is None and N >= 2000) else args.reps
            rows.append({"scheme": name, "N": N, "k": deg, "t": t, "P": P, "drop": drop,
                         **bench_server(N, k, t, P, drop, reps)})
            print("server", name, N, deg, P, drop, f"{rows[-1]['r3_reconstruct_unmask']:.3f}s", flush=True)
        write(args.out / "server.csv", rows)


if __name__ == "__main__":
    main()
