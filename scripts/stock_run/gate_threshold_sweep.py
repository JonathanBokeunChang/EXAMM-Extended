#!/usr/bin/env python3
"""Algorithm 2 gate-threshold sensitivity, in MODEL-RELATIVE units.

Algorithm 2 gates on `all L longs > 0 and all S shorts < 0`. That zero is an
ABSOLUTE constant compared against raw model output, so the gate's firing rate is
governed by (prediction mean)/(prediction sd) -- i.e. by how much the model shrank
toward the unconditional mean -- and not by forecast skill. A heavily shrunk model
(Crossformer: ~54x) has its whole prediction cloud sitting several sd above zero,
so its short leg never clears and the gate never fires: 0% return by construction.

This writes, for each model and each threshold tau, a copy of the predictions
shifted by -tau. A constant shift is RANK-PRESERVING WITHIN EVERY DAY, so the
long/short SELECTION (and therefore IC, ICIR, hit rate) is bit-identical; only the
gate moves. That lets the existing, already-verified trade_portfolio.py run
unmodified.

tau is set as a percentile of the model's OWN pooled prediction distribution, so
"q" means the same thing across models with wildly different output scales.
q = frac(pred <= 0) reproduces that model's published Algorithm 2 number exactly.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

SUF = "_test_predictions.csv"


def load(pred_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*" + SUF))):
        out[os.path.basename(f)[: -len(SUF)]] = pd.read_csv(f)
    if not out:
        sys.exit(f"no *{SUF} under {pred_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--q", type=float, required=True, help="percentile in [0,100]")
    a = ap.parse_args()

    frames = load(a.pred_dir)
    pool = np.concatenate([d["predicted_RET"].to_numpy() for d in frames.values()])
    tau = float(np.percentile(pool, a.q))

    shutil.rmtree(a.out_dir, ignore_errors=True)
    os.makedirs(a.out_dir)
    for tic, d in frames.items():
        d = d.copy()
        d["predicted_RET"] = d["predicted_RET"] - tau
        d.to_csv(os.path.join(a.out_dir, tic + SUF), index=False)
    print(f"q={a.q:5.1f}  tau={tau:+.6e}  natural_q0={100*np.mean(pool <= 0):5.1f}%")


if __name__ == "__main__":
    main()
