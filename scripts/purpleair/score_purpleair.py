#!/usr/bin/env python3
"""Score an evolved PurpleAir corrector on the held-out site, against the deployed incumbent.

    python3 scripts/purpleair/score_purpleair.py --eval-dir test_output/<run>/eval_test \
        --data-dir datasets/purpleair --site az

THE COMPARISON MUST BE ROW-MATCHED. The incumbent is evaluated on exactly the rows the
evolved model produced a prediction for, and nothing else. Scoring the two on different
row sets is the easiest way to manufacture a win, because the sites differ in mean
concentration by more than the effect being measured.

THE INCUMBENT. Barkjohn, Gantt & Clements, AMT 14, 4617-4637 (2021), Eq. 10:
    PM2.5 = 0.524 * PA_cf_1 - 0.0862 * RH + 5.75
using the PurpleAir's OWN onboard RH, as fitted. This implementation was validated
against the authors' released data: RMSE 2.507 against their precomputed column's 2.508,
max row difference 0.012 ug/m3.

BOTH HOURLY AND 24-HOUR ARE REPORTED. Eq. 10 was fitted on 24-hour averages, so the 24-h
comparison is the apples-to-apples one and the hourly figure flatters neither side by
accident. A paper that quotes only the hourly number is comparing against an incumbent
outside the regime it was built for.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

BARKJOHN = dict(slope=0.524, rh=-0.0862, intercept=5.75)


def metrics(pred, ref):
    e = np.asarray(pred, float) - np.asarray(ref, float)
    ok = np.isfinite(e)
    e = e[ok]
    ref = np.asarray(ref, float)[ok]
    ss = ((ref - ref.mean()) ** 2).sum()
    return dict(n=int(e.size), rmse=float(np.sqrt((e ** 2).mean())),
                mae=float(np.abs(e).mean()), bias=float(e.mean()),
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="dir of *_predictions.csv from evaluate_rnn")
    ap.add_argument("--data-dir", default="datasets/purpleair")
    ap.add_argument("--site", required=True)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.eval_dir, "*_predictions.csv"))):
        p = pd.read_csv(f)
        stem = os.path.basename(f).replace("_predictions.csv", "")
        src = os.path.join(a.data_dir, stem if stem.endswith(".csv") else stem + ".csv")
        if not os.path.exists(src):
            print(f"! no source segment for {os.path.basename(f)}; skipped")
            continue
        s = pd.read_csv(src)
        # evaluate_rnn emits one prediction per input row at --time_offset 0.
        n = min(len(p), len(s))
        rows.append(pd.DataFrame({
            "DateTime": s["DateTime"].to_numpy()[:n],
            "pa_cf1": s["pa_cf1"].to_numpy(float)[:n],
            "pa_rh": s["pa_rh"].to_numpy(float)[:n],
            "ref": s["ref_pm25"].to_numpy(float)[:n],
            "evolved": p["predicted_ref_pm25"].to_numpy(float)[:n]
            if "predicted_ref_pm25" in p.columns else p.iloc[:, -1].to_numpy(float)[:n],
        }))
    if not rows:
        raise SystemExit(f"no scoreable predictions in {a.eval_dir}")
    d = pd.concat(rows, ignore_index=True).dropna()
    d["incumbent"] = (BARKJOHN["slope"] * d["pa_cf1"]
                      + BARKJOHN["rh"] * d["pa_rh"] + BARKJOHN["intercept"])

    print(f"\n=== held-out site {a.site}: {len(d)} row-matched hours ===\n")
    print(f"{'model':<26} {'n':>7} {'RMSE':>8} {'MAE':>8} {'bias':>8} {'R2':>8}")
    for label, col in (("Barkjohn Eq.10 (incumbent)", "incumbent"),
                       ("raw pa_cf1 (uncorrected)", "pa_cf1"),
                       ("evolved", "evolved")):
        m = metrics(d[col], d["ref"])
        print(f"{label:<26} {m['n']:>7} {m['rmse']:>8.3f} {m['mae']:>8.3f} "
              f"{m['bias']:>+8.3f} {m['r2']:>8.3f}")

    # 24-hour averages: the regime Eq. 10 was actually fitted in.
    d["day"] = pd.to_datetime(d["DateTime"], errors="coerce").dt.floor("D")
    g = d.dropna(subset=["day"]).groupby("day").agg(
        {"ref": "mean", "incumbent": "mean", "evolved": "mean", "pa_cf1": "count"})
    g = g[g["pa_cf1"] >= 18]  # EPA's completeness rule for a valid 24-h average
    if len(g) >= 5:
        print(f"\n--- 24-hour averages ({len(g)} site-days, >=18 valid hours) ---")
        for label, col in (("Barkjohn Eq.10", "incumbent"), ("evolved", "evolved")):
            m = metrics(g[col], g["ref"])
            print(f"{label:<26} {m['n']:>7} {m['rmse']:>8.3f} {m['mae']:>8.3f} "
                  f"{m['bias']:>+8.3f} {m['r2']:>8.3f}")
    else:
        print(f"\n--- 24-hour comparison skipped: only {len(g)} complete site-days ---")


if __name__ == "__main__":
    main()
