#!/usr/bin/env python3
"""Score the fixed-topology LSTM/GRU cells the same way every other model in the paper is scored.

    python3 scripts/stock_run/score_rnn_cells.py > rnn_cells.json

Input is test_output/rnn_<type>_<cell>/ensemble_test/, the 10-run prediction-mean ensembles built on
Anvil by eval_rnn_campaign.sh. Output is IC / ICIR / net / gross / Sharpe per cell.

WHY THE IC IS RECOMPUTED HERE rather than taken from eval_rnn_campaign.sh's printout. Two reasons,
both about matching the rest of the table:

  1. dsA_cohort_2020's test split spans 2022 AND 2023 (500 rows). eval_ensemble_ic.py scores whatever
     it is given, so its Dataset A 2022 figure is a two-year blend -- the same trap tab:all-datasets
     documents, where reading the blend gave +0.0157 against a true +0.0030. Sliced here.
  2. eval_ensemble_ic.py's "IC info ratio" is mean/sd*sqrt(n), which is a t-STATISTIC, not the ICIR
     the paper reports. ICIR is mean/sd. Taking that column as ICIR would inflate it by sqrt(250)
     -- about 16x.

The ensemble CSVs carry no date column (they are row-indexed against the split), so dates are
rebuilt from the matching *_test.csv. Prediction row i targets test row i+1, which is asserted
rather than assumed.
"""
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/transformer_bench"))
from score_cells import sharpe as curve_sharpe          # noqa: E402  (identical Sharpe path)

TRADER = os.path.join(REPO, "scripts/stock_run/trade_portfolio.py")


def data_dir_for(cell):
    if cell.startswith("dsA_"):
        return os.path.join(REPO, "datasets/walkforward", cell[4:]), None
    set_, coh = cell.split("_", 1)
    return os.path.join(REPO, "datasets/walkforward/mid_highmid_price", set_, coh), None


def year_for(cell):
    """Trade year. cohort_YYYY's test split is the two years after YYYY; this study reports the
    first of them, and cohort_YYYY+1 covers the second."""
    coh = cell.split("cohort_")[1][:4]
    return int(coh) + 2


def load(cell_dir, data_dir):
    """Ensemble -> tidy frame with real dates, verified against the source test files."""
    rows = []
    for f in sorted(glob.glob(os.path.join(cell_dir, "*_test_predictions.csv"))):
        tic = os.path.basename(f).split("_test_")[0]
        p = pd.read_csv(f)
        t = pd.read_csv(os.path.join(data_dir, f"{tic}_test.csv"))
        if len(p) != len(t) - 1:
            raise SystemExit(f"{tic}: {len(p)} predictions vs {len(t)} test rows (want n-1)")
        # The trader makes this same check at 1e-10; failing here means the ensemble was built
        # against a different split, which would misalign every date downstream.
        err = np.abs(p["expected_RET"].to_numpy(float) - t["RET"].to_numpy(float)[1:]).max()
        if not err <= 1e-9:
            raise SystemExit(f"{tic}: expected_RET vs RET[t+1] differ by {err:.3e}")
        rows.append(pd.DataFrame({"ticker": tic,
                                  "date": t["date"].astype(str).to_numpy()[1:],
                                  "expected_RET": p["expected_RET"].to_numpy(float),
                                  "predicted_RET": p["predicted_RET"].to_numpy(float)}))
    return pd.concat(rows, ignore_index=True)


def daily_ic(df):
    ics = []
    for _, g in df.groupby("date"):
        if len(g) > 2 and g["predicted_RET"].nunique() > 1:
            c = g["predicted_RET"].corr(g["expected_RET"], method="spearman")
            if np.isfinite(c):
                ics.append(c)
    a = np.asarray(ics)
    sd = a.std(ddof=1)
    return float(a.mean()), (float(a.mean() / sd) if sd > 0 else float("nan")), len(a)


def gate_days(df, long=10, short=10):
    n = 0
    for _, g in df.groupby("date"):
        p = g["predicted_RET"].sort_values()
        if len(p) >= long + short and p.iloc[-long:].min() > 0 and p.iloc[:short].max() < 0:
            n += 1
    return n


def trade(pred_dir, data_dir, window, tc):
    cmd = [sys.executable, TRADER, "--pred-dir", pred_dir, "--data-dir", data_dir,
           "--align", "suffix", "--expect-n", "50", "--long", "10", "--short", "10",
           "--strategy", "daily_hybrid_long_short_return"]
    if window:
        cmd += ["--window-start", window[0], "--window-end", window[1]]
    if tc:
        cmd += ["--use-tc"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "strategy return" in line:
            for tok in line.replace("%", " ").split():
                try:
                    return float(tok), ""
                except ValueError:
                    pass
    return float("nan"), (f"rc={r.returncode} " + r.stderr.strip().splitlines()[-1][:80]
                          if r.stderr.strip() else f"rc={r.returncode}")


def main():
    out = []
    for d in sorted(glob.glob(os.path.join(REPO, "test_output/rnn_*/ensemble_test"))):
        cell = os.path.basename(os.path.dirname(d))[len("rnn_"):]
        typ, rest = cell.split("_", 1)
        # The instance-normalisation arm writes rnn_<type>_inst_<cell>. Without stripping the tag
        # here, rest keeps a leading "inst_" and data_dir_for() reads it as the SET name -- pointing
        # at mid_highmid_price/inst/..., which does not exist, so every instance cell would be
        # skipped or mis-paired rather than failing outright.
        norm = "avg_std_dev"
        if rest.startswith("inst_"):
            rest = rest[len("inst_"):]
            norm = "instance"
        data_dir, _ = data_dir_for(rest)
        yr = year_for(rest)
        try:
            df = load(d, data_dir)
            full_days = df["date"].nunique()
            sub = df[df["date"].str[:4] == str(yr)]
            ic, icir, ndays = daily_ic(sub)
            window = (f"{yr}-01-01", f"{yr}-12-31") if full_days > ndays + 2 else None
            gross, e1 = trade(d, data_dir, window, tc=False)
            net, e2 = trade(d, data_dir, window, tc=True)
            sh = curve_sharpe(d, data_dir, window=window)
            out.append({"model": typ.upper(), "norm": norm, "cell": rest, "yr": yr,
                        "days": ndays, "days_full": full_days,
                        "ic": round(ic, 5), "icir": round(icir, 3),
                        "gross": None if not np.isfinite(gross) else round(gross, 2),
                        "net": None if not np.isfinite(net) else round(net, 2),
                        "sharpe": None if not np.isfinite(sh) else round(sh, 3),
                        "gate": gate_days(sub), "err": (e1 + " " + e2).strip()})
        except Exception as exc:                                   # noqa: BLE001
            out.append({"model": typ.upper(), "norm": norm, "cell": rest, "yr": yr,
                        "ic": None, "icir": None, "gross": None, "net": None, "sharpe": None,
                        "err": f"{type(exc).__name__}: {exc}"[:120]})

    print(f"  {'model':5s} {'norm':11s} {'cell':26s} {'yr':>4s} {'IC':>9s} {'ICIR':>7s} "
          f"{'net%':>8s} {'gross%':>8s} {'Sharpe':>7s} {'gate':>5s}", file=sys.stderr)
    for r in out:
        f = lambda v, p=4: ("---" if v is None else f"{v:+.{p}f}")                 # noqa: E731
        print(f"  {r['model']:5s} {r.get('norm','?'):11s} {r['cell']:26s} {r['yr']:4d} "
              f"{f(r['ic']):>9s} {f(r.get('icir'),3):>7s} {f(r.get('net'),2):>8s} "
              f"{f(r.get('gross'),2):>8s} {f(r.get('sharpe'),2):>7s} {str(r.get('gate','')):>5s}"
              + (f"  {r['err']}" if r.get("err") else ""), file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
