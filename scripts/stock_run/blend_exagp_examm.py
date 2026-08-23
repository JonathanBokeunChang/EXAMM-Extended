#!/usr/bin/env python3
"""Does blending EXA-GP-MIN with EXAMM beat either alone?

    python3 scripts/stock_run/blend_exagp_examm.py --cohort cohort_2021_aligned \
        --a "EXA-GP-MIN=test_output/exagpmin_cohort_2021_aligned_g500/ensemble_test" \
        --b "EXAMM=test_output/mse_cohort_2021_aligned/ensemble_test"

WHY IT MIGHT WORK. Ensemble gain comes from decorrelated errors. Averaging ten seeds of ONE
method decorrelates only the seed noise; the two arms here are different model CLASSES --
gated memory cells against closed-form GP expressions over explicit lags -- so their errors
have a reason to be less correlated than two draws of either. The per-cell correlation of
the two prediction streams is printed, because a blend that helps while the streams are
correlated at 0.9 is a fluke and one that helps at 0.2 is a mechanism.

BLENDING RAW PREDICTIONS WOULD BE MEANINGLESS HERE. The two arms live on wildly different
scales -- EXA-GP-MIN's predictions on this panel span ~1/45th the dispersion of the realised
returns, EXAMM's do not -- so a plain mean is ~EXAMM plus a rounding error, and would report
"the blend equals the better arm" no matter what. Each stream is therefore standardised
cross-sectionally WITHIN each day before averaging, which is how alpha signals are normally
combined and which leaves each arm's own rank IC untouched.

RANK IC IS UNCHANGED BY THE STANDARDISATION, so the single-arm numbers this prints are
directly comparable to compare_exagp.py's.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_forecasting import PRIMARY, data_dir, from_per_ticker, metrics  # noqa: E402

TRADE_YEAR = {"cohort_2020_aligned": 2022, "cohort_2021_aligned": 2023}


def zscore_by_day(df):
    """Cross-sectional standardisation within each date. Rank IC is invariant to it."""
    out = df.copy()
    g = out.groupby("date")["predicted_RET"]
    mu, sd = g.transform("mean"), g.transform("std")
    out["predicted_RET"] = np.where(sd > 0, (out["predicted_RET"] - mu) / sd, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(TRADE_YEAR))
    ap.add_argument("--a", required=True, metavar="LABEL=ENSEMBLE_DIR")
    ap.add_argument("--b", required=True, metavar="LABEL=ENSEMBLE_DIR")
    ap.add_argument("--weights", default="0.5",
                    help="comma-separated weights on A to sweep (default 0.5)")
    args = ap.parse_args()

    year = TRADE_YEAR[args.cohort]
    dd = data_dir("dsA", args.cohort)
    la, pa = args.a.split("=", 1)
    lb, pb = args.b.split("=", 1)

    A = zscore_by_day(from_per_ticker(pa, dd))
    B = zscore_by_day(from_per_ticker(pb, dd))

    key = ["ticker", "date"]
    M = A.merge(B[key + ["predicted_RET"]], on=key, suffixes=("_a", "_b"))
    if len(M) != len(A) or len(M) != len(B):
        print(f"! partial overlap: A={len(A)} B={len(B)} joined={len(M)}", file=sys.stderr)
    rho = float(np.corrcoef(M["predicted_RET_a"], M["predicted_RET_b"])[0, 1])

    def score(series):
        d = M[key + ["expected_RET"]].copy()
        d["predicted_RET"] = series
        return metrics(d, year)

    rows = [(la, score(M["predicted_RET_a"])), (lb, score(M["predicted_RET_b"]))]
    for w in [float(x) for x in args.weights.split(",")]:
        blend = w * M["predicted_RET_a"] + (1 - w) * M["predicted_RET_b"]
        rows.append((f"blend {w:.2f}*{la} + {1-w:.2f}*{lb}", score(blend)))

    print(f"\n=== {args.cohort} -> trade {year} ===")
    print(f"correlation of the two prediction streams: {rho:+.4f}\n")
    hdr = ["model"] + [f"IC h={h}" for h in PRIMARY]
    print(" | ".join(hdr))
    print("-|-".join("-" * len(h) for h in hdr))
    for label, m in rows:
        print(" | ".join([label] + [f"{m.get(f'ic{h}', float('nan')):+.4f}" for h in PRIMARY]))


if __name__ == "__main__":
    main()
