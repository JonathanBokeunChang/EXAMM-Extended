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

PANEL B IS THE PART THAT GENERALISES. The horizon lift is not an EXAMM property, it is a CAPACITY
property: the h5/h1 ratio falls monotonically enough with parameter count to give Spearman
rho = -0.79 across the eight models, and only the two smallest exceed 1.0. Daily cross-sectional
returns are nearly all noise, so the component that survives to day five is the slow one, and a
model with 66 parameters cannot fit anything else. This is why the figure is drawn with colour
mapped to log parameter count rather than to model identity -- the ordering IS the finding, and a
categorical palette would hide it.

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

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.15),
                                   gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.30})
    fig.subplots_adjust(bottom=0.30)

    # Colour encodes log parameter count -- see module docstring: the size ordering is the finding,
    # so a categorical palette would hide it. Markers disambiguate the two ~70k models, whose
    # counts are close enough that their colours necessarily are too.
    counts = np.array([p for _, p in MODELS], float)
    norm = matplotlib.colors.LogNorm(vmin=counts.min() * 0.75, vmax=counts.max() * 1.3)
    cmap = plt.get_cmap("viridis")
    col = {m: cmap(norm(p)) for m, p in MODELS}
    mk = {"EXAMM": "o", "DLinear": "s", "LSTM-i": "^", "GRU-i": "v",
          "PatchTST": "D", "DeformT.": "P", "TimeMixer": "X", "iTransf.": "*"}

    # ---- Panel A: IC vs horizon -----------------------------------------------------------
    means = {m: np.nanmean(data[m], axis=0) for m, _ in MODELS}
    lo, hi = boot_ci(data["EXAMM"], rng)

    # The band marks where the gated book is actually held, so a reader can see EXAMM's rise
    # coincides with it rather than reading a maximum off the curve after the fact.
    axA.axvspan(5, 8, color="0.945", zorder=0, lw=0)
    axA.axhline(0, color="0.6", lw=0.7, zorder=1)
    axA.fill_between(HS, lo, hi, color=col["EXAMM"], alpha=0.10, lw=0, zorder=2)

    for m, p in MODELS:
        exam = (m == "EXAMM")
        axA.plot(HS, means[m], color=col[m], lw=2.3 if exam else 1.0,
                 marker=mk[m], ms=4.2 if exam else 2.9, mew=0,
                 markevery=1 if exam else 2,
                 zorder=7 if exam else 3, alpha=1.0 if exam else 0.8,
                 solid_capstyle="round")

    axA.set_xlabel("Scoring horizon $h$ (trading days)")
    axA.set_ylabel("Rank IC")
    axA.set_xlim(0.6, 10.4)
    axA.set_xticks(HS)
    axA.set_title("Ranking skill by horizon", loc="left", pad=5, fontweight="semibold")
    ylo, yhi = axA.get_ylim()
    axA.annotate("book held $\\approx$7 days", xy=(6.5, ylo + 0.035 * (yhi - ylo)),
                 ha="center", va="bottom", fontsize=6.5, color="0.45")
    for side in ("top", "right"):
        axA.spines[side].set_visible(False)
    axA.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    axA.set_axisbelow(True)

    # ---- Panel B: horizon lift vs capacity ------------------------------------------------
    ratio = np.array([means[m][4] / means[m][0] for m, _ in MODELS])   # h=5 over h=1
    axB.axhline(1.0, color="0.6", lw=0.7, ls=(0, (4, 2)), zorder=1)
    for (m, p), r in zip(MODELS, ratio):
        axB.scatter(p, r, s=64 if m == "EXAMM" else 34, color=col[m], marker=mk[m],
                    zorder=5, linewidths=0, alpha=0.95)
    # Spearman is REPORTED, not fitted through: n=8, and a regression line would imply more
    # precision than eight points carry.
    rk = lambda v: np.argsort(np.argsort(v))                                  # noqa: E731
    rho = np.corrcoef(rk(counts), rk(ratio))[0, 1]
    axB.set_xscale("log")
    axB.set_xlim(25, 6e6)
    # Explicit headroom: the labels are placed beside their markers, so the default margins clip
    # the lowest one (DeformTime, the only negative lift).
    axB.set_ylim(float(ratio.min()) - 0.30, float(ratio.max()) + 0.26)
    axB.set_xlabel("Parameters")
    axB.set_ylabel("Horizon lift  IC$_{h=5}$ / IC$_{h=1}$")
    axB.set_title("Lift falls with capacity", loc="left", pad=5, fontweight="semibold")
    axB.text(0.97, 0.96, f"Spearman $\\rho$ = {rho:+.2f}", transform=axB.transAxes,
             ha="right", va="top", fontsize=7.5, color="0.25")
    axB.text(0.97, 0.575, "no lift", transform=axB.transAxes, ha="right", va="bottom",
             fontsize=6.4, color="0.6")
    # Hand-placed so the two ~70k models (whose x positions nearly coincide) do not collide.
    LAB = {"EXAMM": (2.3, 0.00, "left"), "DLinear": (2.3, 0.00, "left"),
           "LSTM-i": (2.3, 0.00, "left"), "GRU-i": (2.3, 0.00, "left"),
           "PatchTST": (2.3, 0.00, "left"), "DeformT.": (0.34, -0.03, "right"),
           "TimeMixer": (2.6, 0.10, "left"), "iTransf.": (0.34, 0.00, "right")}
    for (m, p), r in zip(MODELS, ratio):
        fx, dy, ha = LAB[m]
        axB.annotate(m, (p, r), xytext=(p * fx, r + dy), fontsize=6.4, color="0.3",
                     ha=ha, va="center")
    for side in ("top", "right"):
        axB.spines[side].set_visible(False)
    axB.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    axB.set_axisbelow(True)

    # ---- shared legend below both panels, ordered by size (the figure's organising variable)
    handles = [Line2D([], [], color=col[m], lw=2.3 if m == "EXAMM" else 1.2, marker=mk[m],
                      ms=4.2 if m == "EXAMM" else 3.2, mew=0, label=f"{m}  {p:,}")
               for m, p in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               frameon=False, ncol=4, handlelength=1.9, columnspacing=1.6,
               labelspacing=0.4, handletextpad=0.5)

    for f in ("horizon.pdf", "horizon.png"):
        fig.savefig(os.path.join(OUT, f), dpi=400)
    print("wrote", os.path.join(OUT, "horizon.pdf"))
    print(f"  EXAMM h=1 {means['EXAMM'][0]:+.4f} -> peak h={HS[int(np.argmax(means['EXAMM']))]} "
          f"{means['EXAMM'].max():+.4f}   Spearman(params, lift) = {rho:+.3f}")


if __name__ == "__main__":
    main()
