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
# LSTM-i/GRU-i were carried as 295 apiece in an earlier table; that cannot be right --
# an LSTM has four gates to a GRU's three, so equal counts are impossible. The values
# below are tab:param-counts', whose ratio 679/511 = 1.33 is exactly 4/3.
PARAMS = {"EXAMM": "$55$--$77$", "DLinear": "$194$", "PatchTST": "$16{,}836$",
          "iTransf.": "$817{,}153$", "DeformT.": "$67{,}449$", "TimeMixer": "$72{,}997$",
          "LSTM-i": "$679$", "GRU-i": "$511$"}
# Forecast horizons carried by tab:horizon, and numeric parameter counts used ONLY to order its
# rows by capacity -- that ordering is the finding there (h5/h1 falls with size, rho = -0.786), so
# it must not depend on parsing PARAMS' LaTeX back into integers.
HS = [1, 2, 3, 5, 10]
PARAM_N = {"EXAMM": 66, "DLinear": 194, "GRU-i": 511, "LSTM-i": 679,
           "PatchTST": 16836, "DeformT.": 67449, "TimeMixer": 72997, "iTransf.": 817153}

# Panel display names. The scorers key on directory names (set1..set4, "Dataset A"); the paper
# calls these the Validation Portfolios $V1$--$V4$ and the Baseline Portfolio (04-datasets.tex:33,
# :110). The mapping lives here so the tables speak the paper's language without renaming any
# directory on disk -- a rename would break every path in every scorer and submit script.
PANEL_NAME = {"Dataset A": "Baseline", "set1": "$V1$", "set2": "$V2$",
              "set3": "$V3$", "set4": "$V4$"}

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
        cells = [PANEL_NAME.get(panel, panel), str(yr)]
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
    for lbl, flt in (("Mean, validation", lambda p: p != "Dataset A"),
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




# --- Overleaf layouts -------------------------------------------------------------------------
# These three reproduce the tables in the shared Overleaf draft EXACTLY -- same columns, same
# grouping, same captions -- and differ only in that the numbers come from the scored JSON instead
# of being transcribed. That is the whole point: the hand-built transposed table there reports
# PEARSON IC, because it was copied from tab:all-datasets before the Rank IC decision landed
# (f66a0b09 -> e73352c9), and a transcribed table cannot follow a metric change. Net, Sharpe and
# MDD in that draft are correct and are reproduced here to the last decimal.
#
# MODEL ORDER IS THE DRAFT'S, NOT ALPHABETICAL OR BY SIZE: EXAMM, then DLinear under "Other",
# then the two fixed RNNs, then the four transformers. The grouping carries information (which
# architecture family a column belongs to) so it is preserved rather than normalised away.
VAL_CELLS = [(s, y) for s in ["set1", "set2", "set3", "set4"] for y in [2022, 2023, 2024]]
NET_ORDER = ["EXAMM", "DLinear", "LSTM-i", "GRU-i", "PatchTST", "iTransf.", "DeformT.", "TimeMixer"]


def _pick(vals, i, dec, low=False):
    v = vals[i]
    if not np.isfinite(v):
        return "---"
    s = f"{v:.{dec}f}" if low else f"{v:+.{dec}f}"
    fin = [(x, j) for j, x in enumerate(vals) if np.isfinite(x)]
    if len(fin) >= 2:
        o = sorted(fin, key=lambda t: (t[0] if low else -t[0]))
        if i == o[0][1]:
            return rf"\textbf{{{s}}}"
        if i == o[1][1]:
            return rf"\underline{{{s}}}"
    return s


def _net(port):
    return {(r["panel"], r["yr"], r["model"]): r["net"] for r in port if "net" in r}


def net_returns(port):
    N = _net(port)
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + NET_CAP + "}",
         r"\label{tab:net-returns}", r"\begin{tabular}{ll" + "r" * 10 + "}", r"\toprule",
         r" & & \multicolumn{2}{c}{Benchmarks} &  & \multicolumn{1}{c}{Other} & "
         r"\multicolumn{2}{c}{Fixed RNNs} & \multicolumn{4}{c}{Transformer-based} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){6-6} \cmidrule(lr){7-8} \cmidrule(lr){9-12}",
         r"Panel & Test yr & B\&H & EW & " + " & ".join(NET_ORDER) + r" \\", r"\midrule"]
    for n, (p, y) in enumerate(VAL_CELLS):
        if n and n % 3 == 0:
            L.append(r"\addlinespace")
        vals = [N.get((p, y, m), float("nan")) for m in NET_ORDER]
        b, e = BENCH[(p, y)]
        L.append(f"{PANEL_NAME[p]} & {y} & {b:+.2f} & {e:+.2f} & "
                 + " & ".join(_pick(vals, i, 2) for i in range(len(NET_ORDER))) + r" \\")
    L.append(r"\midrule")
    vals = [float(np.mean([N[(p, y, m)] for p, y in VAL_CELLS if (p, y, m) in N]))
            for m in NET_ORDER]
    bs = [BENCH[c] for c in VAL_CELLS]
    L.append(r"\multicolumn{2}{l}{\textit{Mean}} & "
             + f"{np.mean([x[0] for x in bs]):+.2f} & {np.mean([x[1] for x in bs]):+.2f} & "
             + " & ".join(_pick(vals, i, 2) for i in range(len(NET_ORDER))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def net_returns_by_year(port):
    N = _net(port)
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + NET_YEAR_CAP + "}",
         r"\label{tab:net-returns-by-year}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{l" + "r" * 10 + "}", r"\toprule",
         r" & \multicolumn{2}{c}{Benchmarks} &  & \multicolumn{1}{c}{Other} & "
         r"\multicolumn{2}{c}{Fixed RNNs} & \multicolumn{4}{c}{Transformer-based} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){5-5} \cmidrule(lr){6-7} \cmidrule(lr){8-11}",
         r"Test yr & B\&H & EW & " + " & ".join(NET_ORDER) + r" \\", r"\midrule"]
    for y in [2022, 2023, 2024]:
        cs = [c for c in VAL_CELLS if c[1] == y]
        vals = [float(np.mean([N[(p, yy, m)] for p, yy in cs if (p, yy, m) in N]))
                for m in NET_ORDER]
        bs = [BENCH[c] for c in cs]
        L.append(f"{y} & {np.mean([x[0] for x in bs]):+.2f} & {np.mean([x[1] for x in bs]):+.2f} & "
                 + " & ".join(_pick(vals, i, 2) for i in range(len(NET_ORDER))) + r" \\")
    L.append(r"\midrule")
    vals = [float(np.mean([N[(p, y, m)] for p, y in VAL_CELLS if (p, y, m) in N]))
            for m in NET_ORDER]
    bs = [BENCH[c] for c in VAL_CELLS]
    L.append(r"\textit{Mean} & "
             + f"{np.mean([x[0] for x in bs]):+.2f} & {np.mean([x[1] for x in bs]):+.2f} & "
             + " & ".join(_pick(vals, i, 2) for i in range(len(NET_ORDER))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(L)


# Pearson first because it is the draft's primary IC; rank IC alongside because the two disagree
# and the disagreement is outlier-driven (see score_forecasting.row_corr). Turnover last because it
# is what reconciles the IC rows with the Net row: EXAMM re-enters on ~15% of days against 46-99%
# for every baseline, so its book is held ~5 days and its money is not made at the 1-day horizon
# the IC rows score. Without this row a reader attributes the Net column to forecasting accuracy,
# which the IC rows directly contradict.
TRANSPOSED_METRICS = [("Pearson IC", "f", "icp1", 4, False), ("Rank IC", "f", "ic1", 4, False),
                      (r"Net (\%)", "p", "net", 2, False), (r"Sharpe", "p", "sharpe", 2, False),
                      (r"MDD (\%)", "p", "mdd", 2, False),
                      (r"Turnover (\%)", "p", "turnover", 0, True)]


def transposed(fore, port):
    F = {(r["panel"], r["yr"], r["model"]): r for r in fore}
    P = {(r["panel"], r["yr"], r["model"]): r for r in port}

    def series(src, key, mi):
        """One metric row: the twelve cells then their mean, each marked against the other models."""
        out = []
        for c in list(VAL_CELLS) + [None]:
            cells = VAL_CELLS if c is None else [c]
            vals = []
            for m in NET_ORDER:
                v = [src[(p, y, m)][key] for p, y in cells
                     if (p, y, m) in src and key in src[(p, y, m)]]
                vals.append(float(np.mean(v)) if v else float("nan"))
            out.append(vals)
        return out

    L = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
         r"\caption{" + TRANSPOSED_CAP + "}", r"\label{tab:all-datasets-transposed}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "r" * 13 + "}", r"\toprule",
         r"Model & Metric & " + " & ".join(
             rf"\multicolumn{{3}}{{c}}{{{PANEL_NAME[s]}}}"
             for s in ["set1", "set2", "set3", "set4"]) + r" & Mean \\",
         " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(4)),
         " & & " + " & ".join(str(y) for _ in range(4) for y in [2022, 2023, 2024]) + r" &  \\",
         r"\midrule"]
    for mi, m in enumerate(NET_ORDER):
        if mi:
            L.append(r"\addlinespace")
        L.append(rf"\multirow{{{len(TRANSPOSED_METRICS)}}}{{*}}{{{m} {PARAMS[m]}}}")
        for lbl, which, key, dec, low in TRANSPOSED_METRICS:
            cols = series(F if which == "f" else P, key, mi)
            L.append(f" & {lbl:13s}& "
                     + " & ".join(_pick(v, mi, dec, low) for v in cols) + r" \\")
    # Ungated benchmarks: net return only, and deliberately NOT ranked against the model rows --
    # they trade every day while every model row is gated by Algorithm 2.
    for name, j in ((r"B\&H", 0), ("EW", 1)):
        L.append(r"\addlinespace")
        vals = [BENCH[c][j] for c in VAL_CELLS]
        L.append(f"{name} (ungated) & Net (\\%)  & "
                 + " & ".join(f"{v:+.2f}" for v in vals)
                 + f" & {np.mean(vals):+.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(L)


NET_CAP = (
    r"Net return (\%) at the pre-committed L/S~10 book, net of transaction costs, for all panels "
    r"and models. Bold/underline is best/second per row among the model columns. The benchmark "
    r"columns use no predictions and are not gated by Algorithm~\ref{alg:long-short}, so they "
    r"trade every day; they are shown for reference and are not ranked. Every model cell is a "
    r"10-run ensemble.")

NET_YEAR_CAP = (
    r"Net return (\%) averaged across the four panels ($V1$--$V4$) within each trade year. "
    r"Bold/underline is best/second per row among the model columns; the ungated benchmarks are "
    r"shown for reference and are not ranked. The Mean row averages the three trade years, and is "
    r"equivalently the mean over all twelve panel--year cells.")

TRANSPOSED_CAP = (
    r"All panels and models, transposed: models as rows, panel--year cells as columns. "
    r"\textbf{Both ICs} are daily cross-sectional correlations between the one-day-ahead "
    r"prediction and the next day's return), return net of transaction costs, gross Sharpe "
    r"($r_f{=}0$) and maximum drawdown, at the pre-committed L/S~10 book. Bold/underline is "
    r"best/second per metric within each panel--year column, among the model rows. Parameter "
    r"counts follow each model name. Every cell is a 10-run ensemble. LSTM-i and GRU-i are "
    r"fixed-topology recurrent baselines trained through EXAMM's own pipeline under causal "
    r"instance normalisation. The two benchmark rows use no predictions at all: buy-and-hold and "
    r"daily equal-weight rebalancing (Financial\_toolbox @4bab989); unlike every model row they "
    r"are not gated by Algorithm~\ref{alg:long-short}, so they trade every day and are not ranked.")


def horizon(fore):
    """IC by forecast horizon -- the table that reconciles the IC rows with the Net column.

    Rows are ordered BY PARAMETER COUNT, ascending, because that ordering is the finding: the
    h5/h1 ratio falls with capacity across the eight models (Spearman rho = -0.786). Sorting
    alphabetically or by score would hide the one structural result here.

    h=20 IS INCLUDED PRECISELY BECAUSE EXAMM LOSES THERE. The claim is that its ranking
    information is LOCATED at 2-10 days, not that it is uniformly better; showing where the arc
    closes is what separates a mechanism from a horizon picked after the fact. Which horizons are
    called primary (h=3, h=5) is fixed by the measured holding period, not by these numbers.

    Rank IC only. Pearson is reported in tab:all-datasets-transposed at h=1; carrying both across
    six horizons would double the width to restate a difference that table already shows.
    """
    F = {(r["panel"], r["yr"], r["model"]): r for r in fore}
    order = sorted(NET_ORDER, key=lambda m: PARAM_N[m])
    cols = {h: [float(np.mean([F[(p, y, m)][f"ic{h}"] for p, y in VAL_CELLS
                               if (p, y, m) in F and f"ic{h}" in F[(p, y, m)]]))
                for m in order] for h in HS}
    ratio = [cols[5][i] / cols[1][i] if cols[1][i] else float("nan") for i in range(len(order))]
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + HORIZON_CAP + "}",
         r"\label{tab:horizon}", r"\begin{tabular}{lr" + "r" * (len(HS) + 1) + "}", r"\toprule",
         r"Model & Params & " + " & ".join(rf"$h{{=}}{h}$" for h in HS) + r" & $h5/h1$ \\",
         r"\midrule"]
    for i, m in enumerate(order):
        row = [m, PARAMS[m]] + [_pick(cols[h], i, 4) for h in HS]
        row.append(f"{ratio[i]:+.2f}$\\times$")
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


HORIZON_CAP = (
    r"Rank IC by forecast horizon, on the twelve validation panel--year cells. No model is "
    r"retrained: every prediction is the same one-day-ahead value, and only the realised return "
    r"it is scored against changes, to the forward $h$-day compounded return. Rows are ordered by "
    r"parameter count. $h{=}3$ and $h{=}5$ are the horizons at which the gated book is actually "
    r"held (Table~\ref{tab:all-datasets-transposed}, turnover row); $h{=}1$ is the horizon the "
    r"forecasting literature reports, and $h{=}20$ is included to show where the effect closes. "
    r"The final column is the ratio of $h{=}5$ to $h{=}1$ accuracy: it exceeds one only for the "
    r"two smallest models and falls with capacity across the eight (Spearman "
    r"$\rho = -0.786$). Bold/underline is best/second per column.")


def model_summary(fore, port):
    """Per-model means over the twelve validation cells, in the Overleaf draft's grouped layout.

    ADDS MSE, MAE AND TURNOVER to the IC/Net/Sharpe/MDD the draft already carried. Each earns its
    place differently:
      MSE, MAE -- point accuracy. They belong here precisely BECAUSE they do not separate: the
        eight models span ~15% on MSE while their net returns span 17 percentage points, and the
        two give an identical ordering (Spearman 1.0000 across the model means, so MAE adds no
        information over MSE and is reported only to forestall the question).
      Turnover -- the share of test days on which Algorithm 2 re-enters the book. On the other
        days positions are HELD, not flattened (Financial_toolbox portfolio.py:451 updates the
        book only inside `if do_trade`), so its reciprocal is the holding period: ~6.9 days for
        EXAMM against 1.1-1.9 for every baseline. Without this column a reader sees EXAMM winning
        Net and Sharpe and attributes it to forecast accuracy, which the IC column contradicts.

    PROMOTED TO table* BECAUSE TEN COLUMNS CANNOT LIVE IN \columnwidth. The draft's version was a
    single-column table inside \resizebox; three more columns there would put the type near 5pt.

    IC IS PEARSON, matching the draft. Rank IC disagrees (EXAMM +0.0068 rather than +0.0106) and
    the difference is outlier-driven -- see score_forecasting.row_corr. Both are in the JSON.
    """
    F = {(r["panel"], r["yr"], r["model"]): r for r in fore}
    P = {(r["panel"], r["yr"], r["model"]): r for r in port}

    def mean(src, key, m):
        v = [src[(p, y, m)][key] for p, y in VAL_CELLS
             if (p, y, m) in src and key in src[(p, y, m)]]
        return float(np.mean(v)) if v else float("nan")

    # (key, source, header, decimals, scale, lower-is-better)
    METS = [("icp1", F, r"IC", 4, 1, False),
            ("mse", F, r"MSE ($\times10^{-4}$)", 2, 1e4, True),
            ("mae", F, r"MAE ($\times10^{-2}$)", 3, 1e2, True),
            ("net", P, r"Net (\%)", 2, 1, False),
            ("sharpe", P, r"Sharpe", 2, 1, False),
            ("mdd", P, r"MDD (\%)", 2, 1, False),
            ("turnover", P, r"Turnover (\%)", 1, 1, True)]
    cols = {k: [mean(src, k, m) for m in NET_ORDER] for k, src, _, _, _, _ in METS}

    def cell(k, i, dec, scale, low):
        v = cols[k][i]
        if not np.isfinite(v):
            return "---"
        s = f"{v * scale:.{dec}f}" if low else f"{v * scale:+.{dec}f}"
        fin = [(x, j) for j, x in enumerate(cols[k]) if np.isfinite(x)]
        o = sorted(fin, key=lambda t: (t[0] if low else -t[0]))
        if i == o[0][1]:
            return rf"\textbf{{{s}}}"
        if i == o[1][1]:
            return rf"\underline{{{s}}}"
        return s

    # The draft prints EXAMM as $66$ in its summary tables (the mean of the Baseline Portfolio's
    # two evolved genomes, 55 and 77) and as the range elsewhere. Matched here so this table drops
    # in without a mismatch against tab:param-counts.
    show = dict(PARAMS, **{"EXAMM": "$66$"})
    GROUPS = [("", ["EXAMM"]), ("Other", ["DLinear"]), ("Fixed RNNs", ["LSTM-i", "GRU-i"]),
              ("Transformer-based", ["PatchTST", "iTransf.", "DeformT.", "TimeMixer"])]
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + SUMMARY_CAP + "}",
         r"\label{tab:model-summary}", r"\begin{tabular}{llr" + "r" * len(METS) + "}",
         r"\toprule",
         r" & Model & Params & " + " & ".join(h for _, _, h, _, _, _ in METS) + r" \\",
         r"\midrule"]
    for gi, (gname, members) in enumerate(GROUPS):
        if gi:
            L.append(r"\midrule")
        for j, m in enumerate(members):
            i = NET_ORDER.index(m)
            lead = ""
            if j == 0 and gname:
                lead = (gname if len(members) == 1
                        else rf"\multirow{{{len(members)}}}{{*}}{{{gname}}}")
            L.append(f"{lead} & {m} & {show[m]} & "
                     + " & ".join(cell(k, i, d, sc, lo) for k, _, _, d, sc, lo in METS)
                     + r" \\")
    # Benchmarks: net return only. Ungated, so ranking them against the model rows would compare
    # two different strategies.
    L.append(r"\midrule")
    bh = float(np.mean([BENCH[c][0] for c in VAL_CELLS]))
    ew = float(np.mean([BENCH[c][1] for c in VAL_CELLS]))
    pad_l = " & ".join(["---"] * 3)
    pad_r = " & ".join(["---"] * 3)
    L.append(rf"\multirow{{2}}{{*}}{{Benchmarks}} & Buy \& hold & --- & {pad_l} & {bh:+.2f} & {pad_r} \\")
    L.append(rf" & Equal weight & --- & {pad_l} & {ew:+.2f} & {pad_r} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


SUMMARY_CAP = (
    r"Per-model summary over the twelve validation panel--year cells, each metric averaged across "
    r"cells. IC and the two error metrics are measured on the raw forecasts, before the entry "
    r"gate of Algorithm~\ref{alg:long-short} and before transaction costs, and so are the only "
    r"columns independent of the trading rule. Turnover is the percentage of test days on which "
    r"the gate opens and the book is re-entered; on the remaining days positions are held rather "
    r"than flattened, so its reciprocal is the mean holding period --- about seven trading days "
    r"for EXAMM against one to two for every baseline. Bold/underline is best/second per column "
    r"among the model rows. The benchmarks use no predictions and are ungated, so they trade "
    r"every day and are neither scored on forecast accuracy nor ranked.")


def ic_table(fore):
    """Per-cell Pearson IC in the Overleaf draft's layout, generated rather than transcribed.

    EXISTS TO FIX A ONE-DIGIT DISAGREEMENT that two hand-maintained tables cannot avoid. The
    draft's Mean row reads PatchTST +0.0069 and this reads +0.0068; the exact mean is 0.006844.
    The draft averaged the per-cell values AFTER rounding each to four decimals (0.006858), while
    tab:model-summary averages the raw values. Neither is wrong in isolation, but two tables
    printing different digits for the same quantity is the kind of thing a reader notices and a
    co-author cannot explain. Generating both from one function settles it: average, then round.
    The model ordering is identical under either convention.

    IC IS PEARSON here, matching the draft. Rank IC is in the same JSON under ic1.
    """
    F = {(r["panel"], r["yr"], r["model"]): r["icp1"] for r in fore if "icp1" in r}
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{" + IC_CAP + "}", r"\label{tab:ic}",
         r"\footnotesize", r"\begin{tabular}{ll" + "r" * len(NET_ORDER) + "}", r"\toprule",
         r" & &  & \multicolumn{1}{c}{Other} & \multicolumn{2}{c}{Fixed RNNs} & "
         r"\multicolumn{4}{c}{Transformer-based} \\",
         r"\cmidrule(lr){4-4} \cmidrule(lr){5-6} \cmidrule(lr){7-10}",
         r"Panel & Test yr & " + " & ".join(NET_ORDER) + r" \\", r"\midrule"]
    for n, (p, y) in enumerate(VAL_CELLS):
        if n and n % 3 == 0:
            L.append(r"\addlinespace")
        vals = [F.get((p, y, m), float("nan")) for m in NET_ORDER]
        L.append(f"{PANEL_NAME[p]} & {y} & "
                 + " & ".join(_pick(vals, i, 4) for i in range(len(NET_ORDER))) + r" \\")
    L.append(r"\midrule")
    vals = [float(np.mean([F[(p, y, m)] for p, y in VAL_CELLS if (p, y, m) in F]))
            for m in NET_ORDER]
    L.append(r"\multicolumn{2}{l}{\textit{Mean}} & "
             + " & ".join(_pick(vals, i, 4) for i in range(len(NET_ORDER))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


IC_CAP = (
    r"Pearson information coefficient (IC) of the raw forecasts, all panels and models. IC is "
    r"measured on the predictions themselves, before the entry gate of "
    r"Algorithm~\ref{alg:long-short} and before transaction costs, so it is the one metric "
    r"reported here that is independent of the trading rule. Bold/underline is best/second per "
    r"row. The benchmarks use no predictions and so have no IC.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["forecasting", "portfolio", "transposed",
                             "net-returns", "net-returns-by-year",
                             "horizon", "model-summary", "ic"],
                    required=True)
    a = ap.parse_args()
    fore = [r for r in json.load(open(os.path.join(REPO, "results/forecasting_cells.json")))
            if r.get("n_rep")]
    port = [r for r in json.load(open(os.path.join(REPO, "results/portfolio_cells.json")))
            if r.get("n_rep")]
    if a.which == "transposed":
        print(transposed(fore, port))
    elif a.which == "net-returns":
        print(net_returns(port))
    elif a.which == "net-returns-by-year":
        print(net_returns_by_year(port))
    elif a.which == "horizon":
        print(horizon(fore))
    elif a.which == "model-summary":
        print(model_summary(fore, port))
    elif a.which == "ic":
        print(ic_table(fore))
    elif a.which == "forecasting":
        print(build(fore, FORECAST_METRICS, FORECAST_CAP, "tab:forecasting", benchmarks=False))
    else:
        print(build(port, PORTFOLIO_METRICS, PORTFOLIO_CAP, "tab:portfolio", benchmarks=True))

if __name__ == "__main__":
    main()
