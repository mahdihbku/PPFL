"""All evaluation figures of the paper, from results/ CSVs and the validated cost model.

  .venv/bin/python figures/make_figures.py --out results/<YYYYMMDD>_figures

Adopted configuration: tau = 2, threshold selection with rho = 10% (C_min = 20, E[C] about 29 at
T/N = 10%, E = 4000 elections), ephemeral keys, LHH, co-signatures, f = 16, native libsodium.
Figures are drawn at print size for single-column IEEE figures (\\linewidth = 3.5 in): subfloats
at 0.49\\linewidth, single figures at 0.8\\linewidth; include them at exactly those widths.
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from ppfl import costmodel as cm  # noqa: E402
from ppfl import security as sec  # noqa: E402
from ppfl.baselines import secagg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# Timings: foreground, min over independent processes of per-process medians (bench/combine_runs.py)
TFG = RES / "20261103_timing_fg"
OLD_ACC = Path(os.environ.get("PPFL_DATA", ROOT / "data"))  # distances_working.csv (Fig. 5, from the original FedAvg code)

# Validated categorical slots (dataviz reference palette, light mode), fixed order.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ROLE = {  # entity -> (label, color, marker, linestyle)
    "client": ("Client", S1, "o", "-"),
    "server": ("Server", S2, "s", "--"),
    "member": ("Committee member", S3, "^", "-."),
}
SUB = (1.72, 1.34)  # 0.49\linewidth
SINGLE = (2.8, 1.85)  # 0.8\linewidth
P_BASIC, N_MAX = 64811, 3000
C_PANELS = (20, 30)
SIGMA, T_RATIO, ELECTIONS, TAU = 40, 0.10, 4000, 2

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "axes.titlesize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.4, "lines.markeredgewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 2.0, "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.25, "legend.handletextpad": 0.4,
    "svg.fonttype": "path", "pdf.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})


def fmt_bytes(v, _pos=None):
    for unit, name in ((1e9, "GB"), (1e6, "MB"), (1e3, "KB"), (1, "B")):
        if v >= unit:
            x = v / unit
            return f"{x:.0f} {name}" if abs(x - round(x)) < 1e-6 else f"{x:g} {name}"
    return f"{v:g} B"


def fmt_pow10(v, _pos=None):
    if v <= 0:
        return ""
    e = math.log10(v)
    if abs(e - round(e)) > 1e-6:
        return ""
    e = int(round(e))
    return {0: "1", 1: "10", 2: "100"}.get(e, f"$10^{{{e}}}$")


def log_bytes_axis(ax, lo, hi):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_bytes(v) if _is_decade(v) else ""))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _is_decade(v):
    return v > 0 and abs(math.log10(v) - round(math.log10(v))) < 1e-6


def thin_decades(ax, keep_every=2):
    """Label every other decade on a crowded log axis."""
    locs = [v for v in ax.get_yticks() if _is_decade(v)]
    lo, hi = ax.get_ylim()
    locs = [v for v in locs if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(locs[::keep_every]))


def role_line(ax, x, y, role, **kw):
    label, color, marker, ls = ROLE[role]
    ax.plot(x, y, color=color, marker=marker, linestyle=ls, markeredgecolor="white", label=label, **kw)


def save(fig, out, name):
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig. 3: max T/N vs C
def max_t_ratio_tau(c, tau, elections):
    target = 2.0 ** -SIGMA / elections
    if c <= tau:
        return 0.0
    lo, hi = 0.0, 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        if sec.tail_tau_binom(c, mid, tau) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def fig_sec_t_over_n(out):
    cs = np.arange(3, 41)
    curves = [
        ("Single election, $\\tau=0$", S1, "-", [sec.max_t_ratio(SIGMA, c, 1) for c in cs]),
        (f"$E={ELECTIONS}$ elections, $\\tau=0$", S2, "--", [sec.max_t_ratio(SIGMA, c, ELECTIONS) for c in cs]),
        (f"$E={ELECTIONS}$ elections, $\\tau=2$", S3, "-.", [max_t_ratio_tau(c, TAU, ELECTIONS) for c in cs]),
    ]
    c_min = sec.c_min_for_tau(SIGMA, T_RATIO, TAU, ELECTIONS)
    assert max_t_ratio_tau(c_min, TAU, ELECTIONS) >= T_RATIO > max_t_ratio_tau(c_min - 1, TAU, ELECTIONS)
    fig, ax = plt.subplots(figsize=SINGLE)
    for label, color, ls, y in curves:
        ax.plot(cs, np.array(y) * 100, color=color, linestyle=ls, label=label)
    ax.axhline(T_RATIO * 100, color=MUTED, linewidth=0.6, linestyle=":")
    ax.plot([c_min], [T_RATIO * 100], marker="o", color=S3, markeredgecolor="white", markersize=4.5, zorder=5)
    ax.annotate(f"$C_{{\\min}}={c_min}$ at $T/N=10\\%$", (c_min, T_RATIO * 100), xytext=(c_min + 1.2, 3.2),
                fontsize=6.2, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xlabel("Committee size, $C$")
    ax.set_ylabel("Max. corrupted ratio $T/N$ (%)")
    ax.set_xlim(3, 40)
    ax.set_ylim(0, 50)
    ax.legend(loc="upper left")
    save(fig, out, "secToverN")
    return c_min


# ---------------------------------------------------------------- Fig. 4: committee size vs N
def fig_expected_c(out):
    d = pd.read_csv(RES / "20261103_fig4_threshold_rho10/fig4_threshold.csv")
    fig, ax = plt.subplots(figsize=SINGLE)
    for tau, color, marker in ((0, S1, "o"), (2, S2, "s")):
        x = d[d.tau == tau].sort_values("N")
        ax.plot(x.N, x.EC, color=color, linestyle="-", linewidth=1.0)
        ax.errorbar(x.N, x.sim_mean, yerr=x.sim_ci95, fmt=marker, color=color, markeredgecolor="white",
                    ecolor=color, elinewidth=0.6, capsize=1.5, markersize=3.4)
        ax.axhline(x.C_min.iloc[0], color=color, linestyle=":", linewidth=0.8)
        ax.text(10150, x.C_min.iloc[0], f"$C_{{\\min}}={x.C_min.iloc[0]}$", va="center", fontsize=6, color=INK2)
    handles = [Line2D([], [], color=S1, marker="o", markeredgecolor="white", label="$\\tau=0$"),
               Line2D([], [], color=S2, marker="s", markeredgecolor="white", label="$\\tau=2$"),
               Line2D([], [], color=INK2, linestyle="-", label="Expected $C=Np$"),
               Line2D([], [], color=INK2, marker="o", linestyle="none", label="Simulated mean (95% CI)"),
               Line2D([], [], color=INK2, linestyle=":", label="$C_{\\min}$")]
    ax.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.9)
    ax.set_xlabel("Number of clients, $N$")
    ax.set_ylabel("Committee size, $C$")
    ax.set_xlim(0, 10000)
    ax.set_ylim(10, 42)
    ax.set_xticks([500, 2000, 4000, 6000, 8000, 10000])
    save(fig, out, "expected_C_with_L")
    return d


# ---------------------------------------------------------------- Fig. 5: distances (old data)
def fig_distances(out):
    if not (OLD_ACC / "distances_working.csv").exists():
        print("skip distances (no distances_working.csv)")
        return
    d = pd.read_csv(OLD_ACC / "distances_working.csv", header=None)
    r = np.arange(1, len(d) + 1)
    fig, ax = plt.subplots(figsize=SINGLE)
    for i in range(1, d.shape[1] - 1):
        ax.plot(r, d.iloc[:, i], color=S1, linewidth=0.9, label="Honest clients (5)" if i == 1 else None)
    ax.plot(r, d.iloc[:, 0], color=S2, linestyle="--", linewidth=1.0, label="Malicious client")
    ax.set_xlabel("Round")
    ax.set_ylabel("Distance $d_i=\\mathbf{w}^{(t)}\\cdot\\mathbf{w}_i^{(t)}$")
    ax.set_xlim(1, len(d))
    ax.set_ylim(0, 1200)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0))
    lo, hi = 93, 99
    ins = ax.inset_axes([0.47, 0.38, 0.48, 0.5])
    sl = slice(lo - 1, hi)
    for i in range(1, d.shape[1] - 1):
        ins.plot(r[sl], d.iloc[sl, i], color=S1, linewidth=0.9)
    ins.plot(r[sl], d.iloc[sl, 0], color=S2, linestyle="--", linewidth=1.0)
    ymin, ymax = d.iloc[sl, :-1].values.min(), d.iloc[sl, :-1].values.max()
    pad = 0.15 * (ymax - ymin)
    ins.set_xlim(lo, hi)
    ins.set_ylim(ymin - pad, ymax + pad)
    ins.tick_params(labelsize=5.5)
    ins.set_xticks(range(lo, hi + 1, 2))
    ins.set_title(f"Rounds {lo}-{hi}", fontsize=5.8, color=INK2, pad=2)
    ins.spines[["top", "right"]].set_visible(True)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, linewidth=0.5)
    save(fig, out, "distances")


# ---------------------------------------------------------------- Fig. 6: accuracy
ACC_DIR = RES / "20261026_accuracy"
CODEC_DIR = RES / "20261103_accuracy_codec"
PART = {"iid": ("i.i.d.", S1, "o"), "dir03": ("$\\alpha=0.3$", S2, "s"), "dir10": ("$\\alpha=1.0$", S3, "^")}


def acc_curves(ds, split):
    """(rounds, fedavg, proposed) accuracy arrays."""
    if ds == "sg" and split != "iid":
        d = CODEC_DIR / f"sg_{split}_codec"
        m = pd.read_csv(d / "metrics_filled.csv" if (d / "metrics_filled.csv").exists() else d / "metrics.csv")
        return m["round"].values, m["accuracy_fedavg"].values, m["accuracy"].values
    f = pd.read_csv(ACC_DIR / f"{ds}_{split}_fedavg/metrics.csv")
    s = pd.read_csv(ACC_DIR / f"{ds}_{split}_secure/metrics.csv")
    assert (f["round"].values == s["round"].values).all()
    return f["round"].values, f["accuracy"].values, s["accuracy"].values


def fig_accuracy(out):
    summary = []
    dash = (0, (3, 1.6))
    for ds in ("sg", "nsl_binary", "nsl_multiclass"):
        for panel, splits in (("iid", ["iid"]), ("non_iid", ["dir03", "dir10"])):
            fig, ax = plt.subplots(figsize=SUB)
            lo_all, hi_all = 1.0, 0.0
            for split in splits:
                label, color, marker = PART[split]
                r, fa, pr = acc_curves(ds, split)
                every = max(1, len(r) // 10)
                ax.plot(r, fa, color=color, linewidth=1.0)
                ax.plot(r, pr, color=color, linewidth=0.7, linestyle=dash, marker=marker,
                        markevery=(every // 2, every), markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=0.8, markersize=3.6)
                lo_all, hi_all = min(lo_all, fa.min(), pr.min()), max(hi_all, fa.max(), pr.max())
                summary.append((ds, split, fa[-1], pr[-1], np.abs(fa - pr).max()))
            lo = math.floor(lo_all * 20) / 20
            hi = min(1.0, max(math.ceil(hi_all * 20 - 1e-9) / 20, lo + 0.1))
            ax.set_ylim(lo, hi)
            ax.set_xlim(0, r[-1])
            ax.set_xlabel("Round")
            ax.set_ylabel("Test accuracy")
            key = S1 if panel == "iid" else INK2
            handles = [Line2D([], [], color=key, linewidth=1.0, label="FedAvg"),
                       Line2D([], [], color=key, linewidth=0.7, linestyle=dash, marker="o", markerfacecolor="white",
                              markeredgecolor=key, markeredgewidth=0.8, label="Proposed")]
            if panel != "iid":
                handles += [Line2D([], [], color=PART[sp][1], marker=PART[sp][2], markeredgecolor="white",
                                   label=PART[sp][0]) for sp in splits]
            ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncol=1 if panel == "iid" else 2, columnspacing=1.0, handlelength=2.2)
            save(fig, out, f"accuracy_{ds}_{panel}")
    return summary


# ---------------------------------------------------------------- Fig. 7: VRF
def fig_vrf(out):
    v = pd.read_csv(TFG / "vrf/vrf.csv")
    ops = v[v.C == 1].set_index("op").s * 1e3
    fig, ax = plt.subplots(figsize=SUB)
    names = [("keygen", "Gen"), ("prove", "Eval"), ("verify", "Verify")]
    xs = np.arange(len(names))
    vals = [ops[k] for k, _ in names]
    ax.bar(xs, vals, width=0.55, color=S1, edgecolor="white", linewidth=0.8)
    for x, y in zip(xs, vals):
        ax.text(x, y + max(vals) * 0.03, f"{y:.3f}" if y < 0.1 else f"{y:.2f}", ha="center", va="bottom",
                fontsize=6.2, color=INK)
    ax.set_xticks(xs, [n for _, n in names])
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="x", visible=False)
    save(fig, out, "vrf1")
    c = v[v.op == "verify_committee"].sort_values("C")
    fig, ax = plt.subplots(figsize=SUB)
    ax.plot(c.C, c.s * 1e3, color=S1, marker="o", markeredgecolor="white")
    ax.set_xlabel("Committee size, $C$")
    ax.set_ylabel("Committee verification (ms)")
    ax.set_xlim(3, 32)
    ax.set_ylim(0, None)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    save(fig, out, "vrf2")
    return ops, c


# ---------------------------------------------------------------- Figs. 8-10: communication
GRID_DIR = RES / "20261103_comm_grid_tau2"
MODELS = {64811: "Basic", 25557032: "ResNet-50", 138357544: "VGG-16"}


def online():
    d = pd.read_csv(GRID_DIR / "online.csv")
    d = d[d.variant == "full"].copy()
    d["total"] = d.overhead_up_B + d.overhead_down_B
    return d


def model_top_axis(ax):
    top = ax.secondary_xaxis("top")
    top.set_xticks(list(MODELS), list(MODELS.values()), fontsize=5.6)
    top.tick_params(length=1.5, pad=1, colors=MUTED, labelcolor=INK2)
    top.xaxis.set_minor_locator(FixedLocator([]))
    for lab in top.get_xticklabels():
        lab.set_ha({"ResNet-50": "right", "VGG-16": "left"}.get(lab.get_text(), "center"))


def fig_offline(out):
    o = pd.read_csv(GRID_DIR / "offline.csv")
    o = o[o.fs == "ephemeral"].sort_values("N")
    for size, name in ((SINGLE, "off_comm"), (SUB, "off_comm_sub")):  # single figure / panel of a figure*
        fig, ax = plt.subplots(figsize=size)
        role_line(ax, o.N, o.client_up_B + o.client_down_B, "client")
        role_line(ax, o.N, o.server_up_B + o.server_down_B, "server")
        log_bytes_axis(ax, 100, 1e10)
        thin_decades(ax)
        ax.set_xlabel("Number of clients, $N$")
        ax.set_ylabel("Offline communication")
        ax.set_xlim(0, N_MAX + 80)
        ax.set_xticks([100, 1000, 2000, 3000])
        ax.legend(loc="lower right")
        save(fig, out, name)
    return o


def fig_online(out):
    d = online()
    res = {}
    for C in C_PANELS:  # vs N
        x = d[(d.C == C) & (d.P == P_BASIC)]
        fig, ax = plt.subplots(figsize=SUB)
        for role in ("server", "member", "client"):
            y = x[x.role == role].sort_values("N")
            role_line(ax, y.N, y.total, role)
        log_bytes_axis(ax, 100, 1e10)
        thin_decades(ax)
        ax.set_xlabel("Number of clients, $N$")
        ax.set_ylabel("Online overhead")
        ax.set_xlim(0, N_MAX + 80)
        ax.set_xticks([100, 1000, 2000, 3000])
        ax.legend(loc="upper right")
        save(fig, out, f"online_comm_c{C}_p{P_BASIC}")
        res[("N", C)] = x
    for C in C_PANELS:  # vs P
        x = d[(d.C == C) & (d.N == N_MAX)]
        fig, ax = plt.subplots(figsize=SUB)
        for role in ("server", "member", "client"):
            y = x[x.role == role].sort_values("P")
            role_line(ax, y.P, y.total, role)
        ax.set_xscale("log")
        log_bytes_axis(ax, 100, 1e12)
        thin_decades(ax, 3)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_pow10))
        ax.set_xlabel("Model size, $P$")
        ax.set_ylabel("Online overhead")
        model_top_axis(ax)
        ax.legend(loc="upper left")
        save(fig, out, f"online_comm_c{C}_n{N_MAX}")
        res[("P", C)] = x
        fig, ax = plt.subplots(figsize=SUB)  # per weight
        for role in ("server", "member", "client"):
            y = x[x.role == role].sort_values("P")
            role_line(ax, y.P, y.overhead_B_per_weight, role)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e4)
        ax.yaxis.set_major_locator(FixedLocator([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4]))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_pow10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_pow10))
        ax.set_xlabel("Model size, $P$")
        ax.set_ylabel("Overhead (bytes per weight)")
        model_top_axis(ax)
        ax.legend(loc="lower left")
        save(fig, out, f"online_comm_c{C}_n{N_MAX}_per_weight")
    x = d[(d.N == N_MAX) & (d.P == P_BASIC)]  # vs C
    fig, ax = plt.subplots(figsize=SUB)
    for role in ("server", "member", "client"):
        y = x[x.role == role].sort_values("C")
        role_line(ax, y.C, y.total, role)
    log_bytes_axis(ax, 100, 1e10)
    thin_decades(ax)
    c_min = sec.c_min_for_tau(SIGMA, T_RATIO, TAU, ELECTIONS)
    ax.axvline(c_min, color=MUTED, linestyle=":", linewidth=0.7)
    ax.text(c_min + 0.6, 250, "$C_{\\min}$", ha="left", fontsize=6, color=INK2)
    ax.set_xlabel("Committee size, $C$")
    ax.set_ylabel("Online overhead")
    ax.set_xlim(0, 32)
    ax.set_xticks([3, 10, 20, 30])
    ax.legend(loc="upper left")
    save(fig, out, f"online_comm_p{P_BASIC}")
    res["C"] = x
    return res


# ---------------------------------------------------------------- Fig. 11: computation
TIMING = TFG / "role"
CLIENT_COLS = ["check_cosign", "control_post", "lhh", "mask", "read_delta", "verify_committee", "vrf_prove"]
MEMBER_COLS = ["commit", "derive_submasks", "lhh", "lhh_check", "open_commitments", "posts", "verify_controls"]
SERVER_COLS = ["aggregate", "check_uploads", "delta_post", "open_commitments", "verify_committee"]


def role_times(C, P=P_BASIC):
    """Seconds per role vs N (tau = 2: members also share their round secrets)."""
    cl = pd.read_csv(TIMING / "client.csv")
    me = pd.read_csv(TIMING / "member.csv")
    se = pd.read_csv(TIMING / "server.csv")
    rec = pd.read_csv(TFG / "recovery/recovery_timing.csv").set_index("C")
    c = float(cl[(cl.C == C) & (cl.P == P)][CLIENT_COLS].sum(axis=1).iloc[0])
    m = me[(me.C == C) & (me.P == P)].sort_values("N")
    s = se[(se.C == C) & (se.P == P)].sort_values("N")
    member = m[MEMBER_COLS].sum(axis=1).values + rec.loc[C, "share_secrets_s"]
    return m.N.values, c, member, s[SERVER_COLS].sum(axis=1).values


def fig_online_comp(out):
    res = {}
    for C in C_PANELS:
        ns, c, member, server = role_times(C)
        fig, ax = plt.subplots(figsize=SUB)
        role_line(ax, ns, server * 1e3, "server")
        role_line(ax, ns, member * 1e3, "member")
        role_line(ax, ns, np.full(len(ns), c * 1e3), "client")
        ax.set_yscale("log")
        ax.set_ylim(10, 6000)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}" if _is_decade(v) else ""))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of clients, $N$")
        ax.set_ylabel("Computation (ms)")
        ax.set_xlim(0, N_MAX + 80)
        ax.set_xticks([100, 1000, 2000, 3000])
        ax.legend(loc="upper left")
        save(fig, out, f"online_comp_C{C}")
        res[C] = (ns, c, member, server)
    return res


# ---------------------------------------------------------------- new: baselines
BELL = TFG / "bell"
SECAGG_T = TFG / "secagg"
SCHEME = {"ours": ("Proposed ($C=30$)", S1, "o", "-"), "secagg": ("SecAgg", S2, "s", "--"),
          "secagg+": ("SecAgg+", S3, "^", "-."), "flamingo": ("Flamingo", S4, "D", ":")}
BELL_DELTA, C_OURS = 0.05, 30
FLAMINGO = RES / "20261103_flamingo/summary.csv"  # authors' code, N = 128..2048 (powers of 2)


def baseline_data():
    cb = pd.read_csv(BELL / "client_bell.csv")
    sb = pd.read_csv(BELL / "server_bell.csv")
    cs = pd.read_csv(SECAGG_T / "client.csv")
    ss = pd.read_csv(SECAGG_T / "server.csv")
    cb = cb[(cb.delta == BELL_DELTA) & (cb.P == P_BASIC)].sort_values("N")
    sb = sb[(sb.delta == BELL_DELTA) & (sb["drop"] == 0.0) & (sb.P == P_BASIC)].sort_values("N")
    cs = cs[(cs.scheme == "secagg") & (cs.P == P_BASIC)].sort_values("N")
    ss = ss[(ss.scheme == "secagg") & (ss["drop"] == 0.0) & (ss.P == P_BASIC)].sort_values("N")
    ns = cb.N.values
    assert (cs.N.values == ns).all() and (sb.N.values == ns).all() and (ss.N.values == ns).all()
    rounds = ["r0_keygen", "r1_share", "r2_mask", "r3_unmask"]
    tn, tc, _, tsrv = role_times(C_OURS)
    assert (tn == ns).all()
    out = {"N": ns, "k_bell": cb.k.values}
    out["comm"] = {
        "ours": [cm.recovery_round_bytes(P_BASIC, C_OURS, n) for n in ns],
        "secagg": [secagg.round_bytes(P_BASIC, n, n - 1) for n in ns],
        "secagg+": [secagg.round_bytes(P_BASIC, n, k) for n, k in zip(ns, cb.k.values)],
    }
    out["time"] = {
        "ours": (np.full(len(ns), tc), tsrv),
        "secagg": (cs[rounds].sum(axis=1).values, ss.r3_reconstruct_unmask.values),
        "secagg+": (cb[rounds].sum(axis=1).values, sb.r3_reconstruct_unmask.values),
    }
    out["flamingo"] = pd.read_csv(FLAMINGO).sort_values("N")
    return out


def fig_baselines(out):
    b = baseline_data()
    ns = b["N"]
    panels = [("client", "comm", "Client overhead"), ("server", "comm", "Server overhead"),
              ("client", "time", "Client computation (ms)"), ("server", "time", "Server computation (ms)")]
    for role, kind, ylabel in panels:
        fig, ax = plt.subplots(figsize=SUB)
        for scheme, (label, color, marker, ls) in SCHEME.items():
            x = ns
            if scheme == "flamingo":
                f = b["flamingo"]
                x = f.N.values
                y = (f[f"{role}_up_B"] + f[f"{role}_down_B"]).values if kind == "comm" else f[f"{role}_s"].values * 1e3
            elif kind == "comm":
                y = [r[role]["up"] + r[role]["down"] for r in b["comm"][scheme]]
            else:
                y = b["time"][scheme][0 if role == "client" else 1] * 1e3
            ax.plot(x, y, color=color, marker=marker, linestyle=ls, markeredgecolor="white", label=label,
                    markersize=3.0 if scheme == "flamingo" else None)
        ax.set_yscale("log")
        if kind == "comm":
            lo, hi = (10, 1e7) if role == "client" else (1e5, 1e10)
            log_bytes_axis(ax, lo, hi)
            thin_decades(ax)
        else:
            ax.set_ylim((1, 1000) if role == "client" else (5, 2000))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}" if _is_decade(v) else ""))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of clients, $N$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, N_MAX + 80)
        ax.set_xticks([100, 1000, 2000, 3000])
        if role == "client" and kind == "comm":
            ax.legend(loc="lower right")
        save(fig, out, f"baselines_{kind}_{role}")
    return b


# ---------------------------------------------------------------- new: liveness
LIVE = RES / "20261021_liveness_threshold/liveness.csv"
RULE = {"pow2_rho1": ("$p=2^{-L}$, $\\rho=1\\%$", S1, "o", "-"),
        "p_rho1": ("Threshold, $\\rho=1\\%$", S2, "s", "--"),
        "p_rho10": ("Threshold, $\\rho=10\\%$", S3, "^", "-.")}


def fig_liveness(out):
    lv = pd.read_csv(LIVE)
    x = lv[(lv.N == N_MAX) & (lv.tau == TAU)]
    h = x[x.scenario == "honest"]
    qs = sorted(h.q.unique())
    fig, ax = plt.subplots(figsize=SUB)
    for rule, (label, color, marker, ls) in RULE.items():
        y = h[h.rule == rule].sort_values("q")
        ax.plot(range(len(qs)), 100 * y.failed_rounds / y.rounds, color=color, marker=marker, linestyle=ls,
                markeredgecolor="white", label=label)
    ax.set_xticks(range(len(qs)), [f"{100 * q:g}" for q in qs])
    ax.set_xlabel("Member dropout rate $q$ (%)")
    ax.set_ylabel("Failed rounds (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    save(fig, out, "liveness_dropout")
    a = x[x.scenario == "adversary"]
    ks = sorted(a.cooldown.unique())
    fig, ax = plt.subplots(figsize=SUB)
    for rule, (label, color, marker, ls) in RULE.items():
        y = a[a.rule == rule].sort_values("cooldown")
        ax.plot(range(len(ks)), 100 * y.failed_rounds / y.rounds, color=color, marker=marker, linestyle=ls,
                markeredgecolor="white", label=label)
    ax.set_xticks(range(len(ks)), [f"{int(k)}" for k in ks])
    ax.set_xlabel("Cooldown $K$ (rounds)")
    ax.set_ylabel("Failed rounds (%)")
    ax.set_ylim(0, None)
    save(fig, out, "liveness_adversary")
    return h, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    c_min = fig_sec_t_over_n(args.out)
    fig_expected_c(args.out)
    fig_distances(args.out)
    acc = fig_accuracy(args.out)
    ops, vc = fig_vrf(args.out)
    fig_offline(args.out)
    on = fig_online(args.out)
    comp = fig_online_comp(args.out)
    b = fig_baselines(args.out)
    fig_liveness(args.out)
    with open(args.out / "numbers.txt", "w") as f:
        f.write(f"C_min(tau=2, T/N=10%, E={ELECTIONS}) = {c_min}\n")
        for ds, split, fa, pr, dev in acc:
            f.write(f"accuracy {ds} {split}: final FedAvg {fa:.4f} proposed {pr:.4f} max|diff| {dev:.4f}\n")
        f.write("VRF ms: " + ", ".join(f"{k} {v:.4f}" for k, v in ops.items()) + "\n")
        f.write("committee verify ms: " + ", ".join(f"C={c} {s * 1e3:.2f}" for c, s in zip(vc.C, vc.s)) + "\n")
        for C in C_PANELS:
            ns, c, member, server = comp[C]
            f.write(f"comp C={C}: client {c * 1e3:.1f} ms; member " + ", ".join(
                f"N={n} {m * 1e3:.0f}" for n, m in zip(ns, member)) + " ms; server " + ", ".join(
                f"N={n} {s * 1e3:.0f}" for n, s in zip(ns, server)) + " ms\n")
            x = on[("N", C)]
            for role in ("client", "member", "server"):
                y = x[(x.role == role) & (x.N == N_MAX)].iloc[0]
                f.write(f"comm C={C} N={N_MAX} P={P_BASIC} {role}: up {y.overhead_up_B} B, down {y.overhead_down_B} B\n")
        f.write(f"SecAgg+ k (Bell, gamma=0.1, delta={BELL_DELTA}): " + ", ".join(
            f"N={n} k={k}" for n, k in zip(b["N"], b["k_bell"])) + "\n")
        for scheme in ("ours", "secagg", "secagg+"):
            r = b["comm"][scheme][-1]
            tcl, tsv = b["time"][scheme]
            f.write(f"baseline N={N_MAX} {scheme}: client {r['client']['up'] + r['client']['down']} B, "
                    f"server {r['server']['up'] + r['server']['down']} B, client {tcl[-1] * 1e3:.1f} ms, "
                    f"server {tsv[-1] * 1e3:.0f} ms\n")
    with open(args.out / "numbers.txt", "a") as f:
        for r in b["flamingo"].itertuples():
            f.write(f"flamingo N={r.N}: client {r.client_up_B + r.client_down_B} B {r.client_s * 1e3:.1f} ms; "
                    f"decryptor extra {r.decryptor_extra_B} B {r.decryptor_extra_s * 1e3:.1f} ms; "
                    f"server {r.server_up_B + r.server_down_B} B {r.server_s * 1e3:.0f} ms\n")
    print("wrote", args.out / "numbers.txt")


if __name__ == "__main__":
    main()
