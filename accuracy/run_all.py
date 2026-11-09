"""Run Fig. 6 configurations sequentially, skipping completed runs.

  .venv/bin/python accuracy/run_all.py --out results/<YYYYMMDD>_accuracy
  .venv/bin/python accuracy/run_all.py --out results/<dir> --datasets sg --splits dir03 dir10 --modes codec
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [("nsl_multiclass", 100), ("nsl_binary", 200), ("sg", 100)]  # fastest first
SPLITS = [("iid", None), ("dir03", 0.3), ("dir10", 1.0)]
MODES = ["fedavg", "secure"]  # "codec" = fixed-point codec only, see run_fl.py


def done(run_dir: Path, rounds: int) -> bool:
    """Complete if metrics.csv (or a filled copy, metrics_filled.csv) has all rounds."""
    return any((run_dir / f).exists() and sum(1 for _ in (run_dir / f).open()) == rounds + 1
               for f in ("metrics.csv", "metrics_filled.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    ap.add_argument("--splits", nargs="+", default=[s for s, _ in SPLITS])
    ap.add_argument("--modes", nargs="+", default=MODES, choices=["fedavg", "secure", "codec"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    datasets = [d for d in DATASETS if d[0] in args.datasets]
    splits = [s for s in SPLITS if s[0] in args.splits]
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "run_all.log").open("a")
    for (ds, rounds), (tag, alpha), mode in ((d, s, m) for d in datasets for s in splits for m in args.modes):
        name = f"{ds}_{tag}_{mode}"
        run_dir = args.out / name
        if done(run_dir, rounds):
            print(f"skip {name}", file=log, flush=True)
            continue
        cmd = [sys.executable, str(ROOT / "accuracy/run_fl.py"), "--dataset", ds, "--mode", mode,
               "--rounds", str(rounds), "--out", str(run_dir)]
        cmd += ["--split", "iid"] if alpha is None else ["--split", "dirichlet", "--alpha", str(alpha)]
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {name}: {' '.join(cmd)}", file=log, flush=True)
        if args.dry_run:
            continue
        t0 = time.perf_counter()
        with (run_dir.parent / f"{name}.stdout").open("w") as out:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT).returncode
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end {name}: rc={rc} {time.perf_counter() - t0:.0f}s",
              file=log, flush=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} all done", file=log, flush=True)


if __name__ == "__main__":
    main()
