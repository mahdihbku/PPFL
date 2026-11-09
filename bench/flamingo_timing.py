"""Flamingo (Ma et al., S&P'23) measured with the authors' code on this machine.

The authors' prototype (github.com/cslyc/flamingo, ABIDES simulator) is cloned in
../flamingo-ref with its own Python 3.9 venv. The only changes (instrumentation.diff, saved in the
output dir) are measurement hooks and parameter plumbing: per-agent rows from their own step
timers, the first report step timed like the later ones, message sizes counted as in their debug
logging (dill, EC points as integers) in separate runs, vector length P, and simulated waiting
times long enough that every client and decryptor participates (no dropout, as for the other
schemes). Server runs single-threaded (-p ''), like ours and SecAgg/SecAgg+. Their defaults are
kept for the protocol: 60 decryptors, threshold 1/3, neighbourhood factor 1 (degree about
2 log2 N).

  .venv/bin/python bench/flamingo_timing.py --out results/<dir> --ns 128 256 --runs 3
Per (N, run): timing_n{N}_r{k}.csv and bytes_n{N}.csv (one byte run per N).
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

FREF = Path(os.environ.get("FLAMINGO_REF", Path(__file__).resolve().parents[2] / "flamingo-ref"))


def run(n, csv_path, log_path, P, count_bytes):
    env = dict(os.environ, FLAMINGO_VECTOR_LEN=str(P), FLAMINGO_WAIT_S="600", FLAMINGO_STATS_CSV=str(csv_path.resolve()))
    if count_bytes:
        env["FLAMINGO_COUNT_BYTES"] = "1"
    csv_path.unlink(missing_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        rc = subprocess.run([str(FREF / ".venv/bin/python"), "abides.py", "-c", "flamingo", "-n", str(n), "-i", "1",
                             "-p", ""], cwd=FREF, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    text = log_path.read_text()
    ok = rc == 0 and f"number of collected vectors: {n}" in text and "collected shares from decryptors: 60" in text
    print(f"N={n} {'bytes' if count_bytes else 'timing'} rc={rc} ok={ok} {time.perf_counter() - t0:.0f}s", flush=True)
    if not ok:
        raise RuntimeError(f"Flamingo run failed or had dropouts: {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--first-run", type=int, default=1)
    ap.add_argument("--P", type=int, default=64811)
    ap.add_argument("--bytes", action="store_true", help="also do the byte-counting run")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "instrumentation.diff").exists():
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=FREF, capture_output=True, text=True).stdout
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=FREF, capture_output=True, text=True).stdout
        (args.out / "instrumentation.diff").write_text(f"# base commit {head}{diff}")
    for n in args.ns:
        if not args.no_timing:
            for k in range(args.first_run, args.first_run + args.runs):
                run(n, args.out / f"timing_n{n}_r{k}.csv", args.out / f"timing_n{n}_r{k}.log", args.P, False)
        if args.bytes:
            run(n, args.out / f"bytes_n{n}.csv", args.out / f"bytes_n{n}.log", args.P, True)


if __name__ == "__main__":
    main()
