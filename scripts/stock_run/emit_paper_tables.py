#!/usr/bin/env python3
"""Emit the paper's two headline tables from the scored JSON panels.

    python3 scripts/stock_run/emit_paper_tables.py --which forecasting
    python3 scripts/stock_run/emit_paper_tables.py --which portfolio

Reads results/forecasting_cells.json (score_forecasting.py) and results/portfolio_cells.json
(score_portfolio.py). Nothing is transcribed by hand: the single table these replace accumulated
values from five different scorers over three weeks and needed a provenance paragraph to explain
which came from where. The two benchmark columns are the exception and are declared below.

LAYOUT MATCHES THE TABLE THIS SPLITS -- panels and trade years down the side, models across, four
metrics per model, best/second marked per metric per row. What changes is that FORECASTING and
PORTFOLIO are now two tables rather than one 36-column object, so a reader can see that the two
orderings disagree instead of having to extract that from a single grid.

FOUR METRICS PER MODEL IS THE WIDTH LIMIT, and it is why each table carries what it does:
  forecasting -- IC at h=1, h=3, h=5, and MSE.
  portfolio   -- net return, Sharpe, max drawdown, turnover.
MAE is omitted from the per-cell grid for width, not because it disagrees: it tracks MSE closely
enough across these cells that a column pair would spend a ninth of the table restating it. Both
live in the JSON, and the aggregate is one command away.

TURNOVER REPLACES IC IN THE PORTFOLIO TABLE and that swap is the point of the split. Algorithm 2
re-enters only on gate-open days and does NOT flatten otherwise (Financial_toolbox
portfolio.py:451 updates the held book only inside `if do_trade`), so turnover IS the holding
period -- roughly 5 days for EXAMM against 1 for the baselines. Reporting net return without it
lets a turnover advantage read as a forecasting advantage, which is the likeliest misreading of
this paper. IC moves to the forecasting table where it belongs.

BENCHMARK COLUMNS ARE TRANSCRIBED, uniquely in these tables, from the table this replaces
(git e73352c9). They use no predictions at all -- buy-and-hold and daily equal-weight rebalancing
via Financial_toolbox @4bab989 -- so nothing in the rescoring could have moved them, and rerunning
them would only risk a transcription-equivalent error in the other direction. Unlike every model
column they are ungated and trade every day.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = ["EXAMM", "DLinear", "PatchTST", "iTransf.", "DeformT.", "TimeMixer", "LSTM-i", "GRU-i"]
PARAMS = {"EXAMM": "$55$--$77$", "DLinear": "$194$", "PatchTST": "$16{,}836$",
          "iTransf.": "$817{,}153$", "DeformT.": "$67{,}449$", "TimeMixer": "$72{,}997$",
          "LSTM-i": "$295$", "GRU-i": "$295$"}
# (panel, trade year) -> (buy-and-hold, equal-weight), net %. See module docstring.
BENCH = {("Dataset A", 2022): (-14.03, -22.28), ("Dataset A", 2023): (+7.87, +0.97),
         ("set1", 2022): (-7.16, -14.26), ("set1", 2023): (+5.18, -1.67),
         ("set1", 2024): (-4.73, -9.92), ("set2", 2022): (-11.42, -18.91),
         ("set2", 2023): (+15.37, +7.40), ("set2", 2024): (+12.13, +3.46),
         ("set3", 2022): (-11.39, -19.08), ("set3", 2023): (+14.77, +7.05),
         ("set3", 2024): (+13.11, +3.89), ("set4", 2022): (-10.25, -17.33),
         ("set4", 2023): (+9.41, +3.16), ("set4", 2024): (+6.69, -1.14)}

# key, header, decimals, scale, lower-is-better
FORECAST_METRICS = [("ic1", r"$h1$", 4, 1, False), ("ic3", r"$h3$", 4, 1, False),
                    ("ic5", r"$h5$", 4, 1, False), ("mse", r"MSE", 2, 1e4, True)]
PORTFOLIO_METRICS = [("net", r"Net", 2, 1, False), ("sharpe", r"Sh.", 2, 1, False),
                     ("mdd", r"MDD", 1, 1, False), ("turnover", r"Turn.", 0, 1, True)]


def order(recs):
    """The paper's fourteen rows, Dataset A first, with a rule between panels."""
    seen, out = set(), []
    for r in recs:
        k = (r["panel"], r["yr"])
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out, key=lambda k: (k[0] != "Dataset A", k[0], k[1]))


def marked(vals, i, dec, scale, low):
    v = vals[i]
    if not np.isfinite(v):
        return "---"
    s = f"{v * scale:.{dec}f}" if low and scale != 1 else (
        f"{v:.{dec}f}" if low else f"{v * scale:+.{dec}f}")
    fin = [(x, j) for j, x in enumerate(vals) if np.isfinite(x)]
    if len(fin) >= 2:
        o = sorted(fin, key=lambda t: (t[0] if low else -t[0]))
        if i == o[0][1]:
            return rf"\textbf{{{s}}}"
        if i == o[1][1]:
            return rf"\underline{{{s}}}"
    return s


def build(recs, metrics, caption, label, benchmarks):
    idx = {(r["panel"], r["yr"], r["model"]): r for r in recs}
    nmet = len(metrics)
    ncol = 2 + len(MODELS) * nmet + (2 if benchmarks else 0)

    L = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{ll" + "r" * (len(MODELS) * nmet + (2 if benchmarks else 0)) + "}",
         r"\toprule"]
    head = "Panel & Trade yr & " + " & ".join(
        rf"\multicolumn{{{nmet}}}{{c}}{{{m} {PARAMS[m]}}}" for m in MODELS)
    if benchmarks:
        head += r" & \multicolumn{2}{c}{Benchmarks, net (\%)}"
    L.append(head + r" \\")
    rules = []
    for j in range(len(MODELS)):
        a = 3 + j * nmet
        rules.append(rf"\cmidrule(lr){{{a}-{a + nmet - 1}}}")
    if benchmarks:
        a = 3 + len(MODELS) * nmet
        rules.append(rf"\cmidrule(lr){{{a}-{a + 1}}}")
    L.append(" ".join(rules))
    sub = " & ".join(h for _ in MODELS for _, h, _, _, _ in metrics)
    L.append(" & & " + sub + (r" & B\&H & EW" if benchmarks else "") + r" \\")
    L.append(r"\midrule")

    rows = order(recs)
    for n, (panel, yr) in enumerate(rows):
        if n and rows[n - 1][0] != panel:
            L.append(r"\midrule" if panel != rows[n - 1][0] and n == 2 else r"\addlinespace")
        cells = [panel, str(yr)]
        for k, _, dec, scale, low in metrics:
            vals = [idx.get((panel, yr, m), {}).get(k, float("nan")) for m in MODELS]
            for i in range(len(MODELS)):
                cells.append((i, k, marked(vals, i, dec, scale, low)))
        # re-interleave: the loop above is metric-major, the table is model-major
        body = [cells[0], cells[1]]
        got = {(i, k): s for i, k, s in cells[2:]}
        for i in range(len(MODELS)):
            for k, _, _, _, _ in metrics:
                body.append(got[(i, k)])
        if benchmarks:
            b, e = BENCH[(panel, yr)]
            body += [f"{b:+.2f}", f"{e:+.2f}"]
        L.append(" & ".join(body) + r" \\")

    L.append(r"\midrule")
    for lbl, flt in (("Mean, replication", lambda p: p != "Dataset A"),
                     ("Mean, all panels", lambda _p: True)):
        cells = {}
        for k, _, dec, scale, low in metrics:
            vals = [float(np.mean([idx[(p, y, m)][k] for p, y in rows
                                   if flt(p) and k in idx.get((p, y, m), {})]))
                    for m in MODELS]
            for i in range(len(MODELS)):
                cells[(i, k)] = marked(vals, i, dec, scale, low)
        body = [rf"\textit{{{lbl}}}", ""]
        for i in range(len(MODELS)):
            for k, _, _, _, _ in metrics:
                body.append(cells[(i, k)])
        if benchmarks:
            bs = [BENCH[(p, y)] for p, y in rows if flt(p)]
            body += [f"{np.mean([x[0] for x in bs]):+.2f}", f"{np.mean([x[1] for x in bs]):+.2f}"]
        L.append(" & ".join(body) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}%", r"}\end{table*}"]
    # Count only UNESCAPED ampersands: "B\\&H" is a literal character, not a column break.
    def ncells(line):
        return len(re.findall(r"(?<!\\)&", line)) + 1
    bad = [l for l in L if l.rstrip().endswith(r"\\") and "\\multicolumn" not in l
           and ncells(l) != ncol]
    assert not bad, f"column mismatch (want {ncol}): " + " || ".join(f"{ncells(b)}: {b[:70]}" for b in bad[:2])
    return "\n".join(L)


FORECAST_CAP = (
    r"Forecasting accuracy per panel and trade year. Rank IC is the daily cross-sectional "
    r"Spearman correlation between each model's one-day-ahead prediction and the forward "
    r"$h$-day compounded return; no model is retrained, only the horizon it is scored against "
    r"changes. $h1$ is the horizon the literature reports; $h3$ and $h5$ are the horizons "
    r"Algorithm~\ref{alg:long-short} actually holds a position, per the turnover column of "
    r"Table~\ref{tab:portfolio}. MSE is $\times10^{-4}$ on the same seed-mean ensembles. Every "
    r"cell is a 10-run ensemble. Bold/underline is best/second per metric per row.")

PORTFOLIO_CAP = (
    r"Portfolio performance per panel and trade year, at the pre-committed long/short~10 book "
    r"under Algorithm~\ref{alg:long-short}. Net is after per-share transaction costs; Sharpe "
    r"($r_f{=}0$) and maximum drawdown are gross, because the equity-curve script refuses to "
    r"emit a curve under transaction costs and a net Sharpe would need a second implementation "
    r"of the strategy. Turnover is the percentage of test days on which the gate opens and the "
    r"book is re-entered; on the remaining days positions are held rather than flattened, so its "
    r"reciprocal is the mean holding period. The two benchmark columns use no predictions at "
    r"all -- buy-and-hold and daily equal-weight rebalancing (Financial\_toolbox @4bab989) -- "
    r"and unlike every model column they are ungated and trade every day. Bold/underline is "
    r"best/second per metric per row.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["forecasting", "portfolio"], required=True)
    a = ap.parse_args()
    recs = [r for r in json.load(open(os.path.join(REPO, f"results/{a.which}_cells.json")))
            if r.get("n_rep")]
    if a.which == "forecasting":
        print(build(recs, FORECAST_METRICS, FORECAST_CAP, "tab:forecasting", benchmarks=False))
    else:
        print(build(recs, PORTFOLIO_METRICS, PORTFOLIO_CAP, "tab:portfolio", benchmarks=True))


if __name__ == "__main__":
    main()
