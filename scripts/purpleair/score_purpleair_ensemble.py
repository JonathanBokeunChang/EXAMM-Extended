#!/usr/bin/env python3
"""Score a multi-run PurpleAir ensemble on its held-out site, against the incumbent.

    python3 scripts/purpleair/score_purpleair_ensemble.py --tag ens200 --arm exagpmin \
        --site az --repeats 10

Averages predicted_ref_pm25 across the repeat runs BEFORE scoring, which is the deployed
object -- a mean of individually-scored runs is a different and less relevant quantity.
The per-run spread is reported alongside, because with EXAMM's nondeterminism (same config,
24-hour RMSE 1.857 vs 2.470 on two runs of held-out az) an ensemble figure quoted without
its members' spread invites reading a coin flip as a result.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARKJOHN = dict(slope=0.524, rh=-0.0862, intercept=5.75)
DATA = os.path.join(REPO, "datasets/purpleair")


def metrics(pred, ref):
    e = np.asarray(pred, float) - np.asarray(ref, float)
    ok = np.isfinite(e)
    e, ref = e[ok], np.asarray(ref, float)[ok]
    ss = ((ref - ref.mean()) ** 2).sum()
    return dict(n=int(e.size), rmse=float(np.sqrt((e ** 2).mean())),
                mae=float(np.abs(e).mean()), bias=float(e.mean()),
                r2=float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"))


def load_run(run_dir):
    """{segment_stem: predicted series} for one repeat."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "eval_test", "*_predictions.csv"))):
        stem = os.path.basename(f).replace("_predictions.csv", "")
        p = pd.read_csv(f)
        col = "predicted_ref_pm25" if "predicted_ref_pm25" in p.columns else p.columns[-1]
        out[stem] = p[col].to_numpy(float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()

    runs = []
    for r in range(1, a.repeats + 1):
        d = os.path.join(REPO, "test_output",
                         f"purpleair_{a.tag}_{a.arm}_r{r}_holdout_{a.site}")
        if os.path.isdir(os.path.join(d, "eval_test")):
            runs.append(load_run(d))
    if not runs:
        print(f"  ! no completed repeats for {a.arm}/{a.site}")
        return

    rows, per_run_rmse = [], []
    stems = sorted(set.intersection(*[set(r) for r in runs]))
    for stem in stems:
        src = os.path.join(DATA, stem if stem.endswith(".csv") else stem + ".csv")
        if not os.path.exists(src):
            continue
        s = pd.read_csv(src)
        preds = [r[stem] for r in runs]
        n = min([len(s)] + [len(p) for p in preds])
        rows.append(pd.DataFrame({
            "DateTime": s["DateTime"].to_numpy()[:n],
            "pa_cf1": s["pa_cf1"].to_numpy(float)[:n],
            "pa_rh": s["pa_rh"].to_numpy(float)[:n],
            "ref": s["ref_pm25"].to_numpy(float)[:n],
            "ens": np.mean([p[:n] for p in preds], axis=0),
            **{f"r{i}": p[:n] for i, p in enumerate(preds)},
        }))
    if not rows:
        print(f"  ! no scoreable segments for {a.arm}/{a.site}")
        return
    d = pd.concat(rows, ignore_index=True).dropna()
    d["inc"] = (BARKJOHN["slope"] * d["pa_cf1"] + BARKJOHN["rh"] * d["pa_rh"]
                + BARKJOHN["intercept"])

    d["day"] = pd.to_datetime(d["DateTime"], errors="coerce").dt.floor("D")
    g = d.dropna(subset=["day"]).groupby("day").agg(
        {c: "mean" for c in d.columns if c not in ("DateTime", "day")} | {"pa_cf1": "count"})
    g = g[g["pa_cf1"] >= 18]

    mi, me = metrics(g["inc"], g["ref"]), metrics(g["ens"], g["ref"])
    for i in range(len(runs)):
        per_run_rmse.append(metrics(g[f"r{i}"], g["ref"])["rmse"])
    pr = np.array(per_run_rmse)
    delta = 100.0 * (me["rmse"] - mi["rmse"]) / mi["rmse"]
    print(f"  {a.arm:<9} {a.site:<3} runs={len(runs):<3} days={mi['n']:<4} "
          f"| 24h RMSE inc {mi['rmse']:.3f}  ens {me['rmse']:.3f} ({delta:+.1f}%)  "
          f"| per-run {pr.mean():.3f} +/- {pr.std(ddof=1) if len(pr) > 1 else 0:.3f} "
          f"[{pr.min():.3f}, {pr.max():.3f}]  | ens R2 {me['r2']:+.3f}")


if __name__ == "__main__":
    main()
