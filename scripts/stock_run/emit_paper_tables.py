#!/usr/bin/env python3
"""Emit the paper's two headline tables from the scored JSON panels.

    python3 scripts/stock_run/emit_paper_tables.py --which forecasting
    python3 scripts/stock_run/emit_paper_tables.py --which portfolio

Reads results/forecasting_cells.json (score_forecasting.py) and results/portfolio_cells.json
(score_portfolio.py). Nothing is transcribed by hand: the previous single table accumulated
values from five different scorers over three weeks and needed a provenance paragraph to explain
which came from where.

MODELS ARE ROWS, METRICS ARE COLUMNS. The table this replaces put the fourteen cells down the
side and eight models x four metrics across, which is 36 columns inside a \\resizebox -- legible
only at a magnification no reader applies. Eight models against six horizons plus MSE/MAE cannot
be done that way at all. Per-cell detail moves to the appendix; what a reader needs here is the
ordering, and the ordering lives in the means.

BOTH SUMMARY ROWS ARE KEPT. 'Replication' is the twelve homogeneous portfolio cells; 'All panels'
adds the two Dataset A cells, which are a different universe and move several models noticeably.
Reporting only one invites the question of what the other showed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = [("EXAMM", "$55$--$77$"), ("DLinear", "$194$"), ("PatchTST", "$16{,}836$"),
          ("iTransf.", "$817{,}153$"), ("DeformT.", "$67{,}449$"), ("TimeMixer", "$72{,}997$"),
          ("LSTM-i", "$295$"), ("GRU-i", "$295$")]
HS = [1, 2, 3, 5, 10, 20]


def groups(recs):
    """(label, predicate) for the two summary blocks."""
    return [("Replication (12 cells)", lambda r: r["panel"] != "Dataset A"),
            ("All panels (14 cells)", lambda _r: True)]


def agg(recs, flt, model, key):
    v = [r[key] for r in recs if flt(r) and r["model"] == model and key in r
         and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def mark(vals, i, lower_is_better=False):
    """Bold best, underline second, on finite entries only."""
    fin = [(v, j) for j, v in enumerate(vals) if np.isfinite(v)]
    if len(fin) < 2:
        return "{:+.4f}"
    order = sorted(fin, key=lambda t: (t[0] if lower_is_better else -t[0]))
    if i == order[0][1]:
        return "\\textbf{{{}}}"
    if i == order[1][1]:
        return "\\underline{{{}}}"
    return "{}"


def fmt(v, dec=4, pct=False):
    if not np.isfinite(v):
        return "---"
    return f"{v:+.{dec}f}" if not pct else f"{v:+.{dec}f}"


def forecasting(recs):
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Forecasting accuracy. Rank IC is the daily cross-sectional Spearman "
             r"correlation between each model's one-day-ahead prediction and the forward "
             r"$h$-day compounded return; no model is retrained, only the realised return it is "
             r"scored against changes. $h{=}1$ is the horizon the literature reports and $h{=}3$ "
             r"and $h{=}5$ are the horizons Algorithm~2 actually holds a position (Table~"
             r"\ref{tab:portfolio}, turnover column); $h{=}10$ and $h{=}20$ are shown so the "
             r"whole arc is visible, including where it closes. MSE and MAE are on the same "
             r"seed-mean ensembles. EXAMM's row is the mean over four independent reproductions "
             r"of the identical configuration (five on Dataset~A); every baseline is a single "
             r"10-run ensemble. Bold/underline is best/second per column.}")
    L.append(r"\label{tab:forecasting}")
    L.append(r"\begin{tabular}{lr" + "r" * (len(HS) + 2) + "}")
    L.append(r"\toprule")
    L.append(r"Model & Params & " + " & ".join(rf"$h{{=}}{h}$" for h in HS)
             + r" & MSE ($\times10^{-4}$) & MAE ($\times10^{-2}$) \\")
    L.append(r"\midrule")
    for gi, (label, flt) in enumerate(groups(recs)):
        if gi:
            L.append(r"\midrule")
        L.append(rf"\multicolumn{{{len(HS) + 4}}}{{l}}{{\textit{{{label}}}}} \\")
        cols = {k: [agg(recs, flt, m, k) for m, _ in MODELS]
                for k in [f"ic{h}" for h in HS] + ["mse", "mae"]}
        for i, (m, p) in enumerate(MODELS):
            row = [m, p]
            for h in HS:
                v = cols[f"ic{h}"][i]
                row.append(mark(cols[f"ic{h}"], i).format(fmt(v)))
            for k, sc in (("mse", 1e4), ("mae", 1e2)):
                v = cols[k][i]
                s = "---" if not np.isfinite(v) else f"{v * sc:.3f}"
                row.append(mark(cols[k], i, lower_is_better=True).format(s))
            L.append(" & ".join(row) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def portfolio(recs):
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Portfolio performance at the pre-committed long/short~10 book under "
             r"Algorithm~2. Return is net of per-share transaction costs; Sharpe ($r_f{=}0$) and "
             r"maximum drawdown are gross, because the equity-curve script refuses to emit a "
             r"curve under transaction costs and a net Sharpe would require a second "
             r"implementation of the strategy. Turnover is the share of test days on which the "
             r"gate opens and the book is re-entered; on the remaining days positions are held, "
             r"not flattened, so its reciprocal is the mean holding period. EXAMM's row is the "
             r"mean over four independent reproductions (five on Dataset~A); baselines are single "
             r"10-run ensembles. Bold/underline is best/second per column.}")
    L.append(r"\label{tab:portfolio}")
    L.append(r"\begin{tabular}{lrrrrrr}")
    L.append(r"\toprule")
    L.append(r"Model & Params & Net (\%) & Gross (\%) & Sharpe & MDD (\%) & Turnover (\%) \\")
    L.append(r"\midrule")
    for gi, (label, flt) in enumerate(groups(recs)):
        if gi:
            L.append(r"\midrule")
        L.append(r"\multicolumn{7}{l}{\textit{" + label + r"}} \\")
        cols = {k: [agg(recs, flt, m, k) for m, _ in MODELS]
                for k in ["net", "gross", "sharpe", "mdd", "turnover"]}
        for i, (m, p) in enumerate(MODELS):
            row = [m, p]
            for k, dec, low in (("net", 2, False), ("gross", 2, False), ("sharpe", 2, False),
                                ("mdd", 2, False), ("turnover", 1, True)):
                v = cols[k][i]
                s = "---" if not np.isfinite(v) else (f"{v:+.{dec}f}" if k != "turnover"
                                                      else f"{v:.{dec}f}")
                row.append(mark(cols[k], i, lower_is_better=low).format(s))
            L.append(" & ".join(row) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["forecasting", "portfolio"], required=True)
    a = ap.parse_args()
    path = os.path.join(REPO, f"results/{a.which}_cells.json")
    recs = [r for r in json.load(open(path)) if r.get("n_rep")]
    print(forecasting(recs) if a.which == "forecasting" else portfolio(recs))


if __name__ == "__main__":
    main()
