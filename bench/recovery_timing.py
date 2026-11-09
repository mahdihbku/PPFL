"""CPU time of the tau-recovery steps: a member sharing its round secrets with the other C-1
members, a holder releasing one share, and reconstructing a secret from t_s = C - tau shares.
(Recomputing a failed member's sub-sum costs one sub-mask derivation plus one LHH, which
role_timing.py measures as derive_submasks and lhh.)

  .venv/bin/python bench/recovery_timing.py --out results/<dir>
"""

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

from ppfl.keys import KAKey, SigningKey, sha256
from ppfl.protocol import HDR, member_open_share, member_share_secrets, reconstruct_secret

D, T, A = sha256(b"bench"), 1, 0


def timeit(fn, reps=20):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Cs", type=int, nargs="+", default=[5, 10, 16, 20, 25, 30, 40, 60])
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for C in args.Cs:
        kas = [KAKey() for _ in range(C)]
        signer = SigningKey()
        t_s = C - args.tau
        holder_pk = {h: kas[h].pk for h in range(1, C)}
        seed = os.urandom(32)

        def share():
            bundle = member_share_secrets(0, kas[0], seed, holder_pk, t_s, D, T, A)
            return {h: signer.sign(pl) for h, pl in bundle.items()}

        bundle = member_share_secrets(0, kas[0], seed, holder_pk, t_s, D, T, A)
        release = timeit(lambda: signer.sign(HDR.pack(1, T, A) + member_open_share(
            1, kas[1], kas[0].pk, bundle[1], D, T, A)[0]))
        shares = {h: member_open_share(h, kas[h], kas[0].pk, bundle[h], D, T, A)[0] for h in range(1, C)}
        rec = timeit(lambda: KAKey(reconstruct_secret(shares, t_s)).pk == kas[0].pk)
        rows.append({"C": C, "tau": args.tau, "share_secrets_s": timeit(share), "release_one_s": release,
                     "reconstruct_and_check_s": rec})
        print(rows[-1], flush=True)
    with open(args.out / "recovery_timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
