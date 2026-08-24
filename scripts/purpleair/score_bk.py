#!/usr/bin/env python3
"""Score a Barkjohn-panel LORO ensemble on its held-out region, against the incumbent.

Rows here are already 24-hour averages -- the regime Eq. 10 was fitted in -- so unlike the
hourly LTPP panel there is no aggregation step and the comparison is apples to apples by
construction.

The incumbent is recomputed on exactly the rows the evolved ensemble predicted, never
quoted from the paper. Its own precomputed column was reproduced to 0.012 ug/m3 on this
same dataset earlier, so the implementation is known good.

Per-run spread is reported next to every ensemble figure: a single run on this problem has
been measured spanning 2x, so an ensemble number without its members' spread is not
interpretable.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARKJOHN = dict(slope=0.524, rh=-0.0862, intercept=5.75)


def metrics(pred, ref):
    e = np.asarray(pred, float) - np.asarray(ref, float)
    ok = np.isfinite(e)
    e, ref = e[ok], np.asarray(ref, float)[ok]
    ss = ((ref - ref.mean()) ** 2).sum()
    return dict(n=int(e.size), rmse=float(np.sqrt((e ** 2).mean())),
                mae=float(np.abs(e).mean()), bias=float(e.mean()),
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    src_dir = os.path.join(REPO, "datasets/purpleair_bk", a.variant, a.region)
    runs = []
    for r in range(1, a.repeats + 1):
        d = os.path.join(REPO, "test_output",
                         f"purpleair_{a.tag}_{a.arm}_r{r}_holdout_{a.region}", "eval_test")
        if not os.path.isdir(d):
            continue
        preds = {}
        for f in sorted(glob.glob(os.path.join(d, "*_predictions.csv"))):
            stem = os.path.basename(f).replace("_predictions.csv", "").replace(".csv", "")
            p = pd.read_csv(f)
            col = "predicted_ref_pm25" if "predicted_ref_pm25" in p.columns else p.columns[-1]
            preds[stem] = p[col].to_numpy(float)
        if preds:
            runs.append(preds)
    if not runs:
        print(f"  ! no completed repeats for {a.arm}/{a.region}")
        return

    stems = sorted(set.intersection(*[set(r) for r in runs]))
    rows = []
    for stem in stems:
        src = os.path.join(src_dir, stem + ".csv")
        if not os.path.exists(src):
            continue
        s = pd.read_csv(src)
        preds = [r[stem] for r in runs]
        n = min([len(s)] + [len(p) for p in preds])
        rows.append(pd.DataFrame({
            "pa_cf1": s["pa_cf1"].to_numpy(float)[:n],
            "pa_rh": s["pa_rh"].to_numpy(float)[:n],
            "ref": s["ref_pm25"].to_numpy(float)[:n],
            "ens": np.mean([p[:n] for p in preds], axis=0),
            **{f"r{i}": p[:n] for i, p in enumerate(preds)},
        }))
    if not rows:
        print(f"  ! no scoreable sensors for {a.arm}/{a.region}")
        return
    d = pd.concat(rows, ignore_index=True).dropna()
    d["inc"] = (BARKJOHN["slope"] * d["pa_cf1"] + BARKJOHN["rh"] * d["pa_rh"]
                + BARKJOHN["intercept"])

    mi, me = metrics(d["inc"], d["ref"]), metrics(d["ens"], d["ref"])
    pr = np.array([metrics(d[f"r{i}"], d["ref"])["rmse"] for i in range(len(runs))])
    delta = 100.0 * (me["rmse"] - mi["rmse"]) / mi["rmse"]
    print(f"  {a.arm:<9} {a.variant:<8} {a.region:<18} runs={len(runs)} n={mi['n']:<5} "
          f"| inc {mi['rmse']:.3f}  ens {me['rmse']:.3f} ({delta:+.1f}%) "
          f"| per-run {pr.mean():.3f}+/-{pr.std(ddof=1) if len(pr) > 1 else 0:.3f} "
          f"| ens R2 {me['r2']:+.3f} bias {me['bias']:+.2f} (inc {mi['bias']:+.2f})")


if __name__ == "__main__":
    main()
