"""Summarise bench/flamingo_timing.py output per N and role (writes summary.csv in the same dir).

Time: per run, the median over agents of the role; then the minimum over runs (same rule as
bench/combine_runs.py). Roles: regular client (not a decryptor), decryptor (extra work beyond its
own client report: cross-check + reconstruction), server (all three steps).
Bytes (byte-counting run): overhead excludes the masked vector upload, measured as the serialized
size of the vector alone, so it is comparable with the overhead columns of the other schemes.

  .venv/bin/python bench/flamingo_summary.py results/<dir>
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

FREF = Path(os.environ.get("FLAMINGO_REF", Path(__file__).resolve().parents[2] / "flamingo-ref"))


def vector_bytes(P):
    code = f"import dill, numpy as np; print(len(dill.dumps(np.ones({P}, dtype='uint32'))))"
    return int(subprocess.run([str(FREF / ".venv/bin/python"), "-c", code], capture_output=True, text=True).stdout)


def main(d: Path):
    rows = []
    for bf in sorted(d.glob("bytes_n*.csv"), key=lambda p: int(p.stem.split("_n")[1])):
        n = int(bf.stem.split("_n")[1])
        b = pd.read_csv(bf)
        P = int(b.vector_len.iloc[0])
        vb = vector_bytes(P)
        cl, de, sv = b[(b.role == "client") & (b.decryptor == 0)], b[b.decryptor == 1], b[b.role == "server"].iloc[0]
        t = {"client": [], "decryptor": [], "server": []}
        for tf in sorted(d.glob(f"timing_n{n}_r*.csv")):
            x = pd.read_csv(tf)
            c = x[(x.role == "client") & (x.decryptor == 0)]
            dd = x[x.decryptor == 1]
            s = x[x.role == "server"].iloc[0]
            t["client"].append(c.report_s.median())
            t["decryptor"].append((dd.crosscheck_s + dd.reconstruction_s).median())
            t["server"].append(s.report_s + s.crosscheck_s + s.reconstruction_s)
        rows.append({
            "N": n, "P": P, "decryptors": len(de), "runs": len(t["client"]),
            "client_s": min(t["client"]), "decryptor_extra_s": min(t["decryptor"]), "server_s": min(t["server"]),
            "client_report_s_runs": " ".join(f"{v:.4f}" for v in t["client"]),
            "server_s_runs": " ".join(f"{v:.4f}" for v in t["server"]),
            "vector_B": vb,
            "client_up_B": int(cl.bytes_sent.median()) - vb, "client_down_B": int(cl.bytes_recv.median()),
            "decryptor_extra_B": int((de.bytes_sent + de.bytes_recv).median() - (cl.bytes_sent + cl.bytes_recv).median()),
            "server_up_B": int(sv.bytes_sent), "server_down_B": int(sv.bytes_recv) - n * vb,
        })
    out = pd.DataFrame(rows)
    out.to_csv(d / "summary.csv", index=False)
    print(out.drop(columns=["client_report_s_runs", "server_s_runs"]).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
