#!/usr/bin/env python3
"""Forecasting-accuracy panel for the paper: rank IC at several horizons, plus MSE/MAE.

    python3 scripts/stock_run/score_forecasting.py            # table to stderr, JSON to stdout
    python3 scripts/stock_run/score_forecasting.py --latex    # emit tab:forecasting

WHY MORE THAN ONE HORIZON. Every model here predicts one day ahead and the literature scores
h=1, so h=1 is the primary column. But Algorithm 2 holds a position through gate-shut days
(Financial_toolbox portfolio.py:451 updates the held book only inside `if do_trade`), and EXAMM's
gate opens on 18% of days against the baselines' 52-94%, so its realised holding period is 3-12
days while theirs is ~1. Scoring only h=1 therefore grades every model at a horizon only some of
them actually trade. h=3 and h=5 report the horizon the book is held. This is NOT a search over
horizons for a favourable one: the holding period is measured first and fixes which columns
appear, and the supporting figure carries h=20 as well, where EXAMM loses.

HORIZON MEANS THE TARGET, NOT THE MODEL. No model is retrained. The prediction is the same
one-day-ahead value throughout; only the realised return it is scored against changes, to the
forward h-day compounded return prod(1+RET[d..d+h-1])-1.

EXAMM IS REPLICATE-AVERAGED, THE BASELINES ARE NOT, AND THAT IS DELIBERATE. The EXAMM-minus-
baseline return difference on this study swings t=+2.26 to +0.85 depending purely on which
reproduction is scored, so a single tree is not a measurement. Every EXAMM number below is the
mean of the per-cell metric over the independent reproductions on disk (base/rep2/rep3/rep4 for
the replication cells, base/rep3-6 for Dataset A). The METRIC is averaged, not the predictions:
pooling predictions into a 40-run ensemble would compare a 40-run object against the baselines'
10-run ones. Each replicate stays a valid 10-run ensemble and the average estimates what the
protocol produces, which is the quantity a reader cares about. The `n_rep` column records how
many went into each row so a thin cell is visible rather than silent.

DATASET A's 2022 ROW IS DATE-SLICED. cohort_2020's test split spans 2022 AND 2023 (501 rows);
scoring it whole returns a two-year blend, which results/hp_grid_orig records as +0.0157 against
a true +0.0030. IC can be sliced because it is a per-day statistic. (Returns cannot -- they
compound -- which is why the portfolio scorer uses the trader's --window-start/--window-end
instead. That asymmetry is real and is why the two panels are computed by two scripts.)

MSE/MAE ARE REPORTED AS A NEGATIVE RESULT, not as a metric anyone wins. Across these models they
span ~15% while returns span 17 percentage points, and the best MSE in the study belongs to the
model ranked last on IC and second-worst on return. The column exists so a reader does not have
to wonder whether it was hidden.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/transformer_bench"))
from score_cells import ensemble  # noqa: E402

MID = os.path.join(REPO, "datasets/walkforward/mid_highmid_price")
DSA = os.path.join(REPO, "datasets/walkforward")
HORIZONS = [1, 2, 3, 5, 10, 20]
PRIMARY = [1, 3, 5]

# (display name, parameter count) in the order the tables use.
MODELS = [("EXAMM", "55--77"), ("DLinear", "194"), ("PatchTST", "16{,}836"),
          ("iTransf.", "817{,}153"), ("DeformT.", "67{,}449"), ("TimeMixer", "72{,}997"),
          ("LSTM-i", "295"), ("GRU-i", "295")]
TB_NAME = {"DLinear": "DLinearOfficial", "PatchTST": "PatchTSTOfficial",
           "iTransf.": "ITransformerOfficial", "DeformT.": "DeformTime",
           "TimeMixer": "TimeMixer"}
RNN_NAME = {"LSTM-i": "lstm", "GRU-i": "gru"}
# Independent reproductions of the identical config. '' is the untagged original.
REPS_MID = ["", "_rep2", "_rep3", "_rep4"]
REPS_DSA = ["", "_rep3", "_rep4", "_rep5", "_rep6"]


def cells():
    """The paper's fourteen rows: Dataset A first, then the four portfolio draws."""
    out = [("Dataset A", 2022, "dsA", "cohort_2020_aligned"),
           ("Dataset A", 2023, "dsA", "cohort_2021_aligned")]
    for s in ["set1", "set2", "set3", "set4"]:
        for c in [2020, 2021, 2022]:
            out.append((s, c + 2, s, f"cohort_{c}_aligned"))
    return out


def data_dir(panel, cohort):
    return os.path.join(DSA, cohort) if panel == "dsA" else os.path.join(MID, panel, cohort)


def from_per_ticker(d, dd):
    """EXAMM / train_rnn layout: one <TICKER>_test_predictions.csv per stock, no date column.

    Dates are rebuilt from the matching *_test.csv and the pairing is ASSERTED, not assumed:
    prediction row i targets test row i+1. A silent misalignment here would shift every horizon.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*_test_predictions.csv"))):
        tic = os.path.basename(f).split("_test_")[0]
        p = pd.read_csv(f)
        t = pd.read_csv(os.path.join(dd, f"{tic}_test.csv"))
        if len(p) != len(t) - 1:
            raise ValueError(f"{tic}: {len(p)} predictions vs {len(t)} test rows (want n-1)")
        rows.append(pd.DataFrame({"ticker": tic,
                                  "date": t["date"].astype(str).to_numpy()[1:],
                                  "expected_RET": t["RET"].astype(float).to_numpy()[1:],
                                  "predicted_RET": p["predicted_RET"].to_numpy(float)}))
    if not rows:
        raise FileNotFoundError(d)
    return pd.concat(rows, ignore_index=True)


def load(model, panel, cohort, rep=""):
    dd = data_dir(panel, cohort)
    if model == "EXAMM":
        d = (os.path.join(REPO, f"test_output/mse_{cohort}{rep}/ensemble_test") if panel == "dsA"
             else os.path.join(REPO, "test_output/mse_pooled_mid_highmid",
                               f"{panel}_{cohort[:11]}{rep}/ensemble_test"))
        return from_per_ticker(d, dd)
    if model in RNN_NAME:
        tag = f"dsA_{cohort}" if panel == "dsA" else f"{panel}_{cohort}"
        return from_per_ticker(
            os.path.join(REPO, f"test_output/rnn_{RNN_NAME[model]}_inst_{tag}/ensemble_test"), dd)
    root = (os.path.join(REPO, "results/transformer_bench/author", cohort) if panel == "dsA"
            else os.path.join(REPO, "results/transformer_bench/mid_highmid", panel, "author", cohort))
    fs = sorted(glob.glob(os.path.join(root, TB_NAME[model], "L96", "seed_*/predictions.csv")))
    if len(fs) < 5:
        raise FileNotFoundError(f"{root}/{TB_NAME[model]} has {len(fs)} seeds")
    return ensemble(fs, dd)


def row_spearman(A, B):
    """Per-row cross-sectional Spearman (== Pearson on ranks), NaN-aware."""
    ra = pd.DataFrame(A).rank(axis=1).to_numpy()
    rb = pd.DataFrame(B).rank(axis=1).to_numpy()
    ok = np.isfinite(A) & np.isfinite(B)
    out = np.full(len(A), np.nan)
    for i in range(len(A)):
        k = ok[i]
        if k.sum() < 5:
            continue
        x, y = ra[i, k] - ra[i, k].mean(), rb[i, k] - rb[i, k].mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            out[i] = (x * y).sum() / den
    return out


def metrics(df, year=None):
    R = df.pivot(index="date", columns="ticker", values="expected_RET").sort_index()
    P = df.pivot(index="date", columns="ticker", values="predicted_RET").reindex_like(R)
    keep = np.array([str(d)[:4] == str(year) for d in R.index]) if year else np.ones(len(R), bool)
    Rv, Pv, n = R.to_numpy(float), P.to_numpy(float), len(R)
    # Forward compounding uses the FULL series so an h-day window starting in December still
    # completes; only the day being scored is restricted to the trade year.
    g = np.cumprod(1 + np.nan_to_num(Rv), axis=0)
    base = np.vstack([np.ones((1, Rv.shape[1])), g[:-1]])
    out = {}
    for h in HORIZONS:
        if n <= h:
            continue
        fwd = np.full_like(Rv, np.nan)
        fwd[: n - h + 1] = g[h - 1:] / base[: n - h + 1] - 1
        ic = row_spearman(Pv, fwd)[keep]
        if np.isfinite(ic).any():
            out[f"ic{h}"] = float(np.nanmean(ic))
    err = (Pv - Rv)[keep]
    err = err[np.isfinite(err)]
    out["mse"] = float(np.mean(err ** 2))
    out["mae"] = float(np.mean(np.abs(err)))
    out["days"] = int(keep.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "results/forecasting_cells.json"))
    args = ap.parse_args()

    recs = []
    for panel, yr, pk, cohort in cells():
        # Dataset A's cohort_2020 test split spans two calendar years; slice IC to the trade year.
        year = yr if pk == "dsA" else None
        for model, _ in MODELS:
            reps = (REPS_DSA if pk == "dsA" else REPS_MID) if model == "EXAMM" else [""]
            got = []
            for rep in reps:
                try:
                    got.append(metrics(load(model, pk, cohort, rep), year))
                except Exception as exc:                                   # noqa: BLE001
                    if model != "EXAMM" or rep == "":
                        last = f"{type(exc).__name__}: {exc}"[:90]
            if not got:
                recs.append({"panel": panel, "yr": yr, "model": model, "n_rep": 0,
                             "err": locals().get("last", "no data")})
                continue
            keys = set().union(*[set(g) for g in got])
            rec = {"panel": panel, "yr": yr, "model": model, "n_rep": len(got)}
            for k in keys:
                v = [g[k] for g in got if k in g]
                rec[k] = float(np.mean(v))
            recs.append(rec)

    json.dump(recs, open(args.out, "w"), indent=1)

    hdr = " ".join(f"{'IC h='+str(h):>9s}" for h in PRIMARY)
    print(f"{'panel':10s} {'yr':>4s} {'model':10s} {'rep':>3s} {hdr} {'MSE':>10s} {'MAE':>9s}",
          file=sys.stderr)
    for r in recs:
        if not r.get("n_rep"):
            print(f"{r['panel']:10s} {r['yr']:4d} {r['model']:10s}   -  {r.get('err','')}",
                  file=sys.stderr)
            continue
        ics = " ".join(f"{r.get('ic'+str(h), float('nan')):+9.4f}" for h in PRIMARY)
        print(f"{r['panel']:10s} {r['yr']:4d} {r['model']:10s} {r['n_rep']:3d} {ics} "
              f"{r['mse']:10.6f} {r['mae']:9.6f}", file=sys.stderr)

    print(f"\nwrote {args.out} ({len(recs)} rows)", file=sys.stderr)
    json.dump(recs, sys.stdout)


if __name__ == "__main__":
    main()
