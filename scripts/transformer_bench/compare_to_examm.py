#!/usr/bin/env python3
"""Score EXAMM against the transformers on the IDENTICAL test rows, and report both aggregations.

THE ALIGNMENT PROBLEM THIS SOLVES
---------------------------------
EXAMM is an RNN and emits a prediction at every timestep: 500 per stock on a 501-row test file,
covering targets t[2]..t[501]. A transformer with L=20 cannot predict before t[21], so it emits 481,
covering t[21]..t[501]. Scoring them on their own row sets would compare different test periods.

So EXAMM's first 19 predictions are dropped. Verified against the cached files: EXAMM's prediction
row 19 (0-indexed) has expected_RET == the transformer's first target == raw RET[20] of the test csv,
and 500 - 19 = 481. That truncation is applied here, in one place, rather than trusted to each
downstream script.

EXAMM's cached predictions carry NO date column -- they are positional, in test-file row order. The
date for prediction row j is therefore test_csv.date[j+1], because --time_offset 1 means row j
predicts the value at row j+1. Getting this off by one would shift every IC by a day and is exactly
the kind of error that still produces plausible-looking numbers, so it is asserted below against
expected_RET rather than assumed.

BOTH AGGREGATIONS ARE REPORTED, ALWAYS
--------------------------------------
  mean-over-runs : average of each run's daily-IC series. Supports claims about the METHOD, and
                   carries the seed spread that this venue has repeatedly shown to dominate.
  ensemble       : average the PREDICTIONS across runs first, then score once. This is the
                   deployable artifact, and on this project it has historically beaten the mean run
                   by a wide margin.
Reporting only whichever favours one side after seeing the numbers is the failure mode; printing
both unconditionally removes the choice. Both sides use 10 runs, so the comparison is symmetric.

Usage:
    python3 scripts/transformer_bench/compare_to_examm.py \
        --cohort cohort_2020_aligned \
        --examm-root test_output/mse_cohort_2020_aligned \
        --tf-root results/transformer_bench/cohort_2020_aligned
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "vol_baselines"))
from ic_stats import daily_ic, paired, report  # noqa: E402

SEQ_LEN = 20          # must match the transformer protocol
EXAMM_DROP = SEQ_LEN - 1


def load_examm_run(run_dir, data_dir):
    """One EXAMM run -> tidy (ticker, date, pred, exp), truncated to the transformer's row set."""
    frames = []
    for f in sorted(glob.glob(f"{run_dir}/eval_test/*_test_predictions.csv")):
        ticker = os.path.basename(f).split("_test_predictions")[0]
        src = f"{data_dir}/{ticker}_test.csv"
        if not os.path.exists(src):
            continue
        p = pd.read_csv(f)
        raw = pd.read_csv(src, usecols=["date", "RET"])
        pcol = next((c for c in p.columns if c.startswith("predicted_")), None)
        ecol = next((c for c in p.columns if c.startswith("expected_")), None)
        if pcol is None or ecol is None:
            raise ValueError(f"{f}: no predicted_/expected_ column")

        # positional: EXAMM row j predicts the value at test-file row j+1
        n = len(p)
        dates = raw["date"].values[1:n + 1]
        truth = raw["RET"].values[1:n + 1]
        if not np.allclose(p[ecol].values[:len(truth)], truth, atol=1e-9, equal_nan=True):
            raise AssertionError(
                f"{ticker}: EXAMM expected_{ecol} does not match test csv RET shifted by one. "
                f"The positional date mapping is wrong -- do not proceed.")

        d = pd.DataFrame({"ticker": ticker, "date": dates,
                          "pred": p[pcol].values, "exp": truth})
        frames.append(d.iloc[EXAMM_DROP:].reset_index(drop=True))
    if not frames:
        raise FileNotFoundError(f"no EXAMM predictions under {run_dir}/eval_test/")
    return pd.concat(frames, ignore_index=True)


def load_tf_run(seed_dir):
    f = os.path.join(seed_dir, "predictions.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).rename(columns={"predicted_RET": "pred", "expected_RET": "exp"})
    return d[["ticker", "date", "pred", "exp"]]


def score(runs, hac):
    """(per-run daily-IC arrays, ensemble daily-IC array) for a list of tidy frames."""
    per_run = [daily_ic(r, pred_col="pred", label_col="exp", date_col="date") for r in runs]
    ens = (pd.concat(runs).groupby(["ticker", "date"], as_index=False)
           .agg(pred=("pred", "mean"), exp=("exp", "first")))
    return per_run, daily_ic(ens, pred_col="pred", label_col="exp", date_col="date"), ens


def err(df):
    e = df["pred"].values - df["exp"].values
    return float(np.mean(e ** 2)), float(np.mean(np.abs(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--examm-root", required=True)
    ap.add_argument("--tf-root", required=True)
    ap.add_argument("--hac-lag", type=int, default=1, help="1 for a 1-day forecast horizon")
    ap.add_argument("--year", default=None,
                    help="restrict scoring to one calendar year (e.g. 2022). The cohort-2020 test "
                         "window spans 2022-2023, and the two years behave very differently, so a "
                         "combined number hides more than it shows.")
    a = ap.parse_args()

    data_dir = os.path.join(REPO, "datasets", "walkforward", a.cohort)
    hac = a.hac_lag

    # ---- EXAMM
    def yr(df):
        return df if a.year is None else df[df.date.astype(str).str.startswith(a.year)].reset_index(drop=True)

    examm_runs = [yr(load_examm_run(d, data_dir))
                  for d in sorted(glob.glob(os.path.join(REPO, a.examm_root, "run_*")))
                  if os.path.isdir(os.path.join(d, "eval_test"))]
    if not examm_runs or len(examm_runs[0]) == 0:
        sys.exit(f"ERROR: no EXAMM rows under {a.examm_root}"
                 + (f" for year {a.year}" if a.year else ""))
    n_rows = len(examm_runs[0])
    print(f"[EXAMM]  {len(examm_runs)} runs x {n_rows} rows "
          f"(first {EXAMM_DROP} predictions dropped to match L={SEQ_LEN})")

    # ---- transformers, grouped by model
    models = {}
    for mdir in sorted(glob.glob(os.path.join(REPO, a.tf_root, "*"))):
        if not os.path.isdir(mdir):
            continue
        runs = [yr(r) for d in sorted(glob.glob(f"{mdir}/seed_*"))
                if (r := load_tf_run(d)) is not None]
        runs = [r for r in runs if len(r)]
        if runs:
            models[os.path.basename(mdir)] = runs
            print(f"[{os.path.basename(mdir)}]  {len(runs)} seeds x {len(runs[0])} rows")
    if not models:
        print(f"WARNING: no transformer predictions under {a.tf_root} -- EXAMM-only report")

    # ---- identical-row gate. Without this the comparison is meaningless, so it is fatal.
    key = set(map(tuple, examm_runs[0][["ticker", "date"]].values))
    for name, runs in models.items():
        k = set(map(tuple, runs[0][["ticker", "date"]].values))
        if k != key:
            sys.exit(f"ERROR: {name} scores {len(k)} rows vs EXAMM's {len(key)}; "
                     f"{len(key - k)} EXAMM-only, {len(k - key)} {name}-only. "
                     f"These are different test sets -- fix before reporting anything.")
    print(f"[gate]   identical row sets confirmed across all models ({len(key)} rows)\n")

    # ---- report
    all_models = {"EXAMM": examm_runs, **models}
    scored, rows = {}, []
    for name, runs in all_models.items():
        per_run, ens_ic, ens_df = score(runs, hac)
        scored[name] = (per_run, ens_ic)
        means = np.array([m.mean() for m in per_run])
        mse, mae = err(ens_df)
        rows.append({"model": name, "n_runs": len(runs),
                     "mean_IC": means.mean(), "seed_sd": means.std(ddof=1) if len(means) > 1 else np.nan,
                     "ens_IC": float(np.mean(ens_ic)), "ens_MSE": mse, "ens_MAE": mae})

    print("=== daily cross-sectional rank IC, identical rows, HAC(%d) ===" % hac)
    for name, (per_run, ens_ic) in scored.items():
        means = np.array([m.mean() for m in per_run])
        sd = means.std(ddof=1) if len(means) > 1 else float("nan")
        print(f"\n  {name}")
        print(f"    mean over runs : {means.mean():+.5f}   seed sd {sd:.5f}   "
              f"range [{means.min():+.5f}, {means.max():+.5f}]")
        report(f"    ensemble", ens_ic, hac)

    if "EXAMM" in scored and len(scored) > 1:
        print(f"\n=== paired vs EXAMM (ensemble daily-IC difference, HAC {hac}) ===")
        base = np.asarray(scored["EXAMM"][1], float)
        for name, (_, ens_ic) in scored.items():
            if name == "EXAMM":
                continue
            e = np.asarray(ens_ic, float)
            n = min(len(base), len(e))
            p = paired(e[:n], base[:n], hac)
            print(f"  {name:14s} - EXAMM: {p['delta']:+.5f}   t_HAC {p['t_hac']:+.2f}")

    out = os.path.join(REPO, a.tf_root, f"comparison_{a.cohort}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("NOTE: mean-over-runs and ensemble are both printed by design. Pick which the paper leads "
          "with BEFORE looking at them, not after.")


if __name__ == "__main__":
    main()
