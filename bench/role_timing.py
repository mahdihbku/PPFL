"""Per-role CPU time of one aggregation round, timing the same functions System.run_round uses.

Inputs are synthetic but well-formed (real keys, real VRF proofs, real signatures and
commitments; uniform masked vectors; arbitrary valid curve points as the clients' LHH values,
which does not change the cost of the check). Single process, median over --reps.

  .venv/bin/python bench/role_timing.py --roles vrf client member server --out results/<dir>
"""

import argparse
import csv
import itertools
import os
import statistics
import struct
import time
from pathlib import Path

import numpy as np
from coincurve import PrivateKey

from ppfl import lhh, vrf
from ppfl.bb import Entry
from ppfl.keys import KAKey, SigningKey, sha256, verify
from ppfl.protocol import (HDR, client_job, lhh_check, member_commit, member_derive, model_hash,
                           open_commitments, server_combine, verify_committee)

D = sha256(b"bench-digest")
T, A_ATT = 1, 0
FRAC = 16


def timeit(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def make_committee(C, alpha, L=0):
    """C members with real VRF proofs (L=0 so every proof qualifies), signed election posts."""
    sig_pks, vrf_pks, posts, member_kas = {}, {}, [], {}
    for j in range(C):
        s = SigningKey()
        sk, pk = vrf.keygen()
        ka = KAKey()
        payload = HDR.pack(j, T, A_ATT) + vrf.prove(sk, alpha) + ka.pk
        posts.append(Entry(j, j, "election", T, A_ATT, payload, s.sign(payload)))
        sig_pks[j], vrf_pks[j], member_kas[j] = s.pk, pk, ka
    return sig_pks, vrf_pks, posts, member_kas


def make_reveals(C, P):
    reveals, commitments, sig_pks = [], {}, {}
    for j in range(C):
        s = SigningKey()
        _, seed, cj = member_commit(np.zeros(P, dtype=np.uint32), j, T, A_ATT, P)
        payload = HDR.pack(j, T, A_ATT) + seed
        reveals.append(Entry(j, j, "reveal", T, A_ATT, payload, s.sign(payload)))
        commitments[j], sig_pks[j] = cj, s.pk
    return reveals, commitments, sig_pks


def bench_vrf(reps, Cs):
    rows = []
    sk, pk = vrf.keygen()
    alpha = sha256(b"alpha")
    pi = vrf.prove(sk, alpha)
    rows.append({"op": "keygen", "C": 1, "s": timeit(lambda: vrf.keygen(), reps)})
    rows.append({"op": "prove", "C": 1, "s": timeit(lambda: vrf.prove(sk, alpha), reps)})
    rows.append({"op": "verify", "C": 1, "s": timeit(lambda: vrf.verify(pk, alpha, pi), reps)})
    for C in Cs:
        sig_pks, vrf_pks, posts, _ = make_committee(C, alpha)
        elig = set(range(C))
        rows.append({"op": "verify_committee", "C": C,
                     "s": timeit(lambda: verify_committee(posts, sig_pks, vrf_pks, elig, alpha, vrf.threshold_for_L(0)), reps)})
    return rows


def bench_client(C, P, reps, use_lhh=True):
    lhh.generators(-(-P // lhh.LANES))  # public parameters, precomputed once (not per round)
    alpha = sha256(b"alpha")
    sig_pks, vrf_pks, posts, member_kas = make_committee(C, alpha)
    member_pk = {j: ka.pk for j, ka in member_kas.items()}
    user, (vsk, _) = SigningKey(), vrf.keygen()
    x = np.random.default_rng(0).normal(0, 0.1, P)
    job = (0, user.raw(), None, x, member_pk, D, T, A_ATT, P, FRAC, True, use_lhh, False, None)
    phases = {k: [] for k in ("mask", "lhh", "control_post")}
    for _ in range(reps):
        _, w, _, _, _, times = client_job(job)
        for k, v in times.items():
            phases[k].append(v)
    row = {k: statistics.median(v) for k, v in phases.items() if v}
    row["vrf_prove"] = timeit(lambda: vrf.selected(vrf.proof_to_hash(vrf.prove(vsk, alpha)), 5), reps)
    elig = set(range(C))
    row["verify_committee"] = timeit(lambda: verify_committee(posts, sig_pks, vrf_pks, elig, alpha, vrf.threshold_for_L(0)), reps)
    srv = SigningKey()
    body = struct.pack(">IB", T, A_ATT) + D + struct.pack(">I", 0)
    dsig = srv.sign(body)
    row["read_delta"] = timeit(lambda: verify(srv.pk, body, dsig), reps)
    enc = w  # any P-length uint32 vector stands in for the received model
    hm = model_hash(T, A_ATT, enc.tobytes())
    cos = []
    signers = [SigningKey() for _ in range(C)]
    for j, s in enumerate(signers):
        payload = HDR.pack(j, T, A_ATT) + hm
        cos.append((s.pk, payload, s.sign(payload)))
    row["check_cosign"] = timeit(lambda: model_hash(T, A_ATT, enc.tobytes()) == hm and all(
        verify(pk, pl, sg) for pk, pl, sg in cos), reps)
    return row


def bench_member(N, C, P, reps, use_lhh=True):
    lhh.generators(-(-P // lhh.LANES))
    rng = np.random.default_rng(1)
    signers = [SigningKey() for _ in range(N)]
    client_kas = [KAKey() for _ in range(N)]
    ctrl = []
    for i in range(N):
        payload = HDR.pack(i, T, A_ATT) + client_kas[i].pk + bytes(33 + 32)
        ctrl.append(Entry(i, i, "control", T, A_ATT, payload, signers[i].sign(payload)))
    sig_pks = {i: s.pk for i, s in enumerate(signers)}
    client_pk = {i: k.pk for i, k in enumerate(client_kas)}
    ka = KAKey()
    row = {"verify_controls": timeit(lambda: all(verify(sig_pks[e.author], e.payload, e.sig) for e in ctrl), reps),
           "derive_submasks": timeit(lambda: member_derive(ka, 0, client_pk, D, T, A_ATT, P), reps)}
    acc = member_derive(ka, 0, client_pk, D, T, A_ATT, P)
    row["commit"] = timeit(lambda: member_commit(acc, 0, T, A_ATT, P), reps)
    rt = member_commit(acc, 0, T, A_ATT, P)[0]
    me = SigningKey()
    row["posts"] = timeit(lambda: [me.sign(HDR.pack(0, T, A_ATT) + bytes(97)) for _ in range(3)], reps)
    if use_lhh:
        row["lhh"] = timeit(lambda: lhh.hash_vector(rt.astype(np.int64)), reps)
        A = rng.integers(0, 2**32, P, dtype=np.int64).astype(np.uint32)
        K = rng.integers(-C, N, P).astype(np.int16)
        pts = [PrivateKey().public_key.format() for _ in range(N + C)]
        row["lhh_check"] = timeit(lambda: lhh_check(A, K, pts[:C], pts[C:]), reps)
    reveals, commitments, rpks = make_reveals(C, P)
    row["open_commitments"] = timeit(lambda: open_commitments(reveals, commitments, rpks, T, A_ATT, P), reps)
    return row


def bench_server(N, C, P, reps, max_bytes=4 << 30):
    rng = np.random.default_rng(2)
    pool = max(1, min(N, max_bytes // (4 * P)))  # distinct buffers, cycled if N exceeds memory
    uploads = [rng.integers(0, 2**32, P, dtype=np.int64).astype(np.uint32) for _ in range(pool)]
    subsums = [rng.integers(0, 2**32, P, dtype=np.int64).astype(np.uint32) for _ in range(C)]
    hashes = [sha256(u.tobytes()) for u in uploads]
    seq = [k % pool for k in range(N)]
    row = {"check_uploads": timeit(lambda: [sha256(uploads[k].tobytes()) == hashes[k] for k in seq], reps),
           "aggregate": timeit(lambda: server_combine((uploads[k] for k in seq), subsums, P), reps)}
    reveals, commitments, rpks = make_reveals(C, P)
    row["open_commitments"] = timeit(lambda: open_commitments(reveals, commitments, rpks, T, A_ATT, P), reps)
    alpha = sha256(b"alpha")
    sig_pks, vrf_pks, posts, _ = make_committee(C, alpha)
    elig = set(range(C))
    row["verify_committee"] = timeit(lambda: verify_committee(posts, sig_pks, vrf_pks, elig, alpha, vrf.threshold_for_L(0)), reps)
    srv = SigningKey()
    row["delta_post"] = timeit(lambda: srv.sign(bytes(105)), reps)
    row["upload_pool"] = pool
    return row


def write(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("N", "C", "P"), k))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="+", default=["vrf", "client", "member", "server"])
    ap.add_argument("--Ns", type=int, nargs="+", default=[100, 500, 1000, 2000, 3000])
    ap.add_argument("--Cs", type=int, nargs="+", default=[5, 10, 13, 16, 20, 30])
    ap.add_argument("--Cs-nodes", type=int, nargs="+", default=[5, 20, 30])
    ap.add_argument("--Ps", type=int, nargs="+", default=[64811, 183429, 237506, 1000000])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"cpu": os.popen("sysctl -n machdep.cpu.brand_string").read().strip(), "reps": args.reps}
    with (args.out / "meta.txt").open("a") as fh:  # one line per invocation (roles may run separately)
        fh.write(f"{meta}\n{vars(args)}\n")
    if "vrf" in args.roles:
        write(args.out / "vrf.csv", bench_vrf(max(args.reps, 20), args.Cs))
        print("vrf done", flush=True)
    if "client" in args.roles:
        rows = []
        for C, P in itertools.product(args.Cs, args.Ps):
            rows.append({"C": C, "P": P, **bench_client(C, P, args.reps)})
            print("client", C, P, f"{sum(v for k, v in rows[-1].items() if k not in ('C', 'P')):.3f}s", flush=True)
        write(args.out / "client.csv", rows)
    if "member" in args.roles:
        rows = []
        for N, C, P in itertools.product(args.Ns, args.Cs_nodes, args.Ps):
            rows.append({"N": N, "C": C, "P": P, **bench_member(N, C, P, args.reps)})
            print("member", N, C, P, flush=True)
        write(args.out / "member.csv", rows)
    if "server" in args.roles:
        rows = []
        for N, C, P in itertools.product(args.Ns, args.Cs_nodes, args.Ps):
            rows.append({"N": N, "C": C, "P": P, **bench_server(N, C, P, args.reps)})
            print("server", N, C, P, flush=True)
        write(args.out / "server.csv", rows)


if __name__ == "__main__":
    main()
