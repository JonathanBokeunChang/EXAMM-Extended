#!/usr/bin/env python3
"""Figure: where each model's cross-sectional ranking skill lives, as a function of horizon.

    python3 scripts/stock_run/plot_horizon.py            # -> results/paper/figures/horizon.pdf

WHAT THE FIGURE ARGUES. Every model here emits the same one-day-ahead prediction; nothing is
retrained. Only the realised return the prediction is scored against changes, from the next day's
return to the forward h-day compounded return. So the x-axis is not "how far ahead each model was
asked to forecast" -- it is "over what span the ranking it produced turns out to be right". A model
whose curve decays has information about tomorrow specifically; a model whose curve rises carries
information that only resolves over several days.

That distinction is the paper's bridge between two tables that otherwise contradict each other.
EXAMM is mid-field at h=1 -- the horizon the forecasting literature reports and the only one its
IC column has ever been scored at -- yet leads the net-return column by a wide margin. The
resolution is that Algorithm 2 re-enters the book on only ~15% of test days and HOLDS on the rest,
a 6.9-day mean holding period against 1.1-1.9 days for every baseline. It is graded at h=1 and
paid at h~7.

COLOUR CARRIES THE SECOND FINDING SO NO SECOND PANEL IS NEEDED. Lines are coloured by log
parameter count rather than model identity, because the horizon lift is a CAPACITY property and
not an EXAMM one: the h=5/h=1 ratio falls with size across the eight models (Spearman rho = -0.79)
and exceeds 1.0 only for the two smallest. Drawn this way the gradient is legible in the curves
themselves -- dark lines above, bright ones below. The correlation is printed on stdout for prose.

NO SHADING, BY DECISION. An earlier version carried a bootstrap band on EXAMM and a grey span
marking the holding period; both were removed for legibility. The uncertainty they showed is real
and still belongs in the text: bootstrapped over the twelve cells, EXAMM's interval spans roughly
+0.003 to +0.021 and contains most of the baseline curves, so the separation here is a
point-estimate ordering and not a resolved one. Do not let the clean lines imply otherwise.

WHAT IT DOES NOT CLAIM, and the CI band is here to keep that visible: bootstrapped over the twelve
validation cells, EXAMM's advantage clears zero only against the weakest arms, and DLinear (194
parameters) tracks it closely throughout. The supported reading is "low-capacity models retain the
multi-day component", not "neuroevolution found something unavailable to the others".

Deliberately no h=20: dropped by decision, so the curve stops at 10.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results/paper/figures")
HS = list(range(1, 11))
VAL = [(s, y) for s in ["set1", "set2", "set3", "set4"] for y in [2022, 2023, 2024]]
# (display name, parameter count). Order here is only the legend's; colour comes from the count.
MODELS = [("EXAMM", 66), ("DLinear", 194), ("LSTM-i", 295), ("GRU-i", 295),
          ("PatchTST", 16836), ("DeformT.", 67449), ("TimeMixer", 72997), ("iTransf.", 817153)]
HOLD = 6.9          # EXAMM's mean holding period, = 100 / 14.5% turnover (tab:portfolio)
N_BOOT = 10000


def load():
    recs = [r for r in json.load(open(os.path.join(REPO, "results/forecasting_cells.json")))
            if r.get("n_rep")]
    F = {(r["panel"], r["yr"], r["model"]): r for r in recs}
    # per-cell matrix so the bootstrap can resample CELLS, the only independent unit here:
    # forward windows overlap, so resampling days would badly understate variance at h=10.
    out = {}
    for m, _ in MODELS:
        out[m] = np.array([[F[(p, y, m)].get(f"ic{h}", np.nan) for h in HS] for p, y in VAL])
    return out


def boot_ci(cells, rng, q=(2.5, 97.5)):
    idx = rng.integers(0, len(cells), (N_BOOT, len(cells)))
    draws = np.nanmean(cells[idx], axis=1)
    return np.percentile(draws, q[0], axis=0), np.percentile(draws, q[1], axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load()
    rng = np.random.default_rng(11)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed real glyphs, not Type-3
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    # SINGLE COLUMN. Generating at 3.4in means the 8pt type here is 8pt on the page; generating
    # wide and letting \includegraphics shrink it to \columnwidth would land the labels near 4pt,
    # which is the usual way a good figure becomes unreadable in print.
    fig, ax = plt.subplots(figsize=(3.4, 3.05))
    fig.subplots_adjust(bottom=0.34, left=0.20, right=0.985, top=0.90)

    counts = np.array([p for _, p in MODELS], float)
    norm = matplotlib.colors.LogNorm(vmin=counts.min() * 0.75, vmax=counts.max() * 1.3)
    cmap = plt.get_cmap("viridis")
    col = {m: cmap(norm(p)) for m, p in MODELS}
    mk = {"EXAMM": "o", "DLinear": "s", "LSTM-i": "^", "GRU-i": "v",
          "PatchTST": "D", "DeformT.": "P", "TimeMixer": "X", "iTransf.": "*"}

    means = {m: np.nanmean(data[m], axis=0) for m, _ in MODELS}

    # Zero kept as the one reference line: several models cross it, and "is this skill or is this
    # nothing" is not readable without it.
    ax.axhline(0, color="0.65", lw=0.7, zorder=1)

    for m, p in MODELS:
        exam = (m == "EXAMM")
        ax.plot(HS, means[m], color=col[m], lw=2.1 if exam else 0.95,
                marker=mk[m], ms=3.6 if exam else 2.5, mew=0,
                markevery=1 if exam else 3,
                zorder=7 if exam else 3, alpha=1.0 if exam else 0.85,
                solid_capstyle="round")

    ax.set_xlabel("Scoring horizon $h$ (trading days)")
    ax.set_ylabel("Rank IC")
    ax.set_xlim(0.6, 10.4)
    ax.set_xticks(HS)
    ax.set_title("Ranking skill by horizon", loc="left", pad=4, fontweight="semibold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend ordered by size, which is the figure's organising variable.
    handles = [Line2D([], [], color=col[m], lw=2.1 if m == "EXAMM" else 1.1, marker=mk[m],
                      ms=3.6 if m == "EXAMM" else 2.8, mew=0, label=f"{m} {p:,}")
               for m, p in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, -0.012),
               frameon=False, ncol=2, handlelength=1.7, columnspacing=1.2,
               labelspacing=0.32, handletextpad=0.45, fontsize=6.4)

    # Reported on stdout, not plotted: the capacity correlation the colouring encodes.
    ratio = np.array([means[m][4] / means[m][0] for m, _ in MODELS])
    rk = lambda v: np.argsort(np.argsort(v))                                  # noqa: E731
    rho = np.corrcoef(rk(counts), rk(ratio))[0, 1]

    for f in ("horizon.pdf", "horizon.png"):
        fig.savefig(os.path.join(OUT, f), dpi=400)
    print("wrote", os.path.join(OUT, "horizon.pdf"))
    print(f"  EXAMM h=1 {means['EXAMM'][0]:+.4f} -> peak h={HS[int(np.argmax(means['EXAMM']))]} "
          f"{means['EXAMM'].max():+.4f}   Spearman(params, lift) = {rho:+.3f}")


if __name__ == "__main__":
    main()
