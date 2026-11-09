"""Combine repeated timing runs: per configuration, the minimum over runs of each timing column.

Each run is an independent process that already reports the median over its repetitions, so the
result is the min-over-processes of per-process medians. This removes slowdowns from the OS
placing a process on efficiency cores or next to background load (observed on the M3 Pro: the
same ECVRF prove measured 0.26-0.82 ms in different processes). Applied identically to every
scheme.

  .venv/bin/python bench/combine_runs.py --runs results/<dir>/role_run1 results/<dir>/role_run2 ... \
      --out results/<dir>/role
"""

import argparse
from pathlib import Path

import pandas as pd

KEYS = {"C", "N", "P", "op", "scheme", "k", "t", "drop", "delta", "gamma", "upload_pool", "tau"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted({f.name for r in args.runs for f in r.glob("*.csv")})
    with (args.out / "combined_from.txt").open("w") as fh:
        fh.write("min over runs of per-run medians\n" + "\n".join(str(r) for r in args.runs) + "\n")
    for name in names:
        frames = [pd.read_csv(r / name).assign(_run=i) for i, r in enumerate(args.runs) if (r / name).exists()]
        d = pd.concat(frames)
        keys = [c for c in d.columns if c in KEYS]
        vals = [c for c in d.columns if c not in KEYS and c != "_run"]
        order = frames[0][keys].drop_duplicates()
        out = order.merge(d.groupby(keys, as_index=False)[vals].min(), on=keys, how="left")
        out = out[[c for c in frames[0].columns if c != "_run"]]
        out.to_csv(args.out / name, index=False)
        spread = (d.groupby(keys)[vals].max() / d.groupby(keys)[vals].min()).max()
        print(f"{name}: {len(frames)} runs, {len(out)} rows, worst max/min ratio per column: "
              + ", ".join(f"{c} {v:.2f}" for c, v in spread.items()))


if __name__ == "__main__":
    main()
