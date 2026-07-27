#!/usr/bin/env python3
"""Linear-ceiling gate on the MASTER-replica CSI300 dataset -- the number a learned model must
clear on THIS data to be worth training.

v2 protocol changes (see the dataset builder's header for why each was necessary):

  * Scores ONLY the rows in eval_index.csv -- the dataset's frozen (date, instrument) set. In v1
    this script scored 430 stocks / 290 per date while the LSTM/GRU script scored 240 / 189, so
    the two were never comparable. Every model now scores the identical row set.
  * Standard errors come from ic_stats (Newey-West HAC, lag 4). The 4-day overlapping label makes
    daily ICs strongly autocorrelated (rho_1 = 0.74), so the naive SE understates by ~1.5x. v1's
    reported t = 13.50 is really 8.59.
  * Trains on LABEL_CSRANK (MASTER's actual learn_processors target) as the headline, with the
    raw-return target reported alongside as an ablation. Evaluation always uses raw LABEL.

Ridge (not OLS) because the features are heavily collinear; the penalty is chosen on VALIDATION
only, never on test.

Note on the market-block ablation: MASTER's market features are INDEX-level series, identical for
every stock on a given date (verified: max within-date std across stocks = 0.0). A linear model
therefore adds the same scalar to every prediction that day, which cannot change the within-date
ordering -- so their contribution to rank IC is exactly zero BY CONSTRUCTION, and the measured
delta here is only fit noise from re-selecting the ridge penalty. This ablation is reported to
document that fact, not because a linear model could ever show a market-block effect. Testing
whether market information helps requires a NONLINEAR model that can use it to modulate
stock-specific features -- which is precisely MASTER's own market-guided-gating thesis.

Usage:
    python3 scripts/vol_baselines/csi300_master_replica_gate.py --data datasets/csi300_master_replica_invdata
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ic_stats import DEFAULT_HAC_LAG, daily_ic, paired, report, summarize  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


def load(data_dir, split):
    frames = []
    for f in sorted(glob.glob(f"{data_dir}/*_{split}.csv")):
        s = os.path.basename(f)[: -len(f"_{split}.csv")]
        d = pd.read_csv(f)
        d["instrument"] = s
        frames.append(d)
    if not frames:
        sys.exit(f"ERROR: no *_{split}.csv under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def restrict_to_eval_index(te, data_dir):
    """Inner-join test rows onto the frozen eval index. Hard-fails if any eval row is missing --
    a silently short join would reintroduce exactly the population mismatch this fixes."""
    ev = pd.read_csv(f"{data_dir}/eval_index.csv")
    n_before = len(te)
    te = ev.merge(te, on=["date", "instrument"], how="left", suffixes=("", "_dup"))
    missing = te["LABEL"].isna().sum()
    if missing:
        sys.exit(f"ERROR: {missing} eval_index rows have no matching data row -- dataset and "
                 f"eval_index.csv are out of sync; rebuild the dataset.")
    print(f"  eval index: {len(ev)} rows ({n_before} test rows available, "
          f"{n_before - len(ev)} outside the index)")
    return te


def cs_ic(df, pred_col):
    return daily_ic(df, pred_col=pred_col, label_col="LABEL", date_col="date")


def fit_ridge(tr, va, te, feats, label):
    """Ridge with the penalty selected on VALIDATION cross-sectional IC (never on test)."""
    X = tr[feats].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    y = tr[label].to_numpy(float)
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    def design(d):
        return np.column_stack([np.ones(len(d)), (d[feats].to_numpy(float) - mu) / sd])

    best = None
    for alpha in ALPHAS:
        R = np.eye(Xs.shape[1]) * alpha
        R[0, 0] = 0.0
        b = np.linalg.solve(Xs.T @ Xs + R, Xs.T @ y)
        va_ic = cs_ic(va.assign(_p=design(va) @ b), "_p").mean()
        if best is None or va_ic > best[1]:
            best = (alpha, va_ic, b)
    alpha, va_ic, beta = best
    return alpha, va_ic, beta, cs_ic(te.assign(_p=design(te) @ beta), "_p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata")
    ap.add_argument("--hac-lag", type=int, default=None,
                     help="default: label_horizon-1 from MANIFEST, so a non-overlapping "
                          "(next-day) label gets lag 1 and MASTER's 5-day label gets lag 4")
    a = ap.parse_args()

    mf = f"{a.data}/MANIFEST.json"
    if os.path.exists(mf):
        m = json.load(open(mf))
        print(f"[dataset] {m['dataset']}")
        print(f"          bundle={m['bundle'].get('bundle')} "
              f"tag={m['bundle'].get('release_tag')} content={m['content_sha256'][:16]}...")
        if a.hac_lag is None:
            a.hac_lag = max(1, m.get("label_horizon_trading_days", DEFAULT_HAC_LAG + 1) - 1)
    if a.hac_lag is None:
        a.hac_lag = DEFAULT_HAC_LAG

    tr, va, te = (load(a.data, s) for s in ("train", "val", "test"))
    te = restrict_to_eval_index(te, a.data)

    drop = {"date", "instrument", "segment_id", "LABEL", "LABEL_CSRANK"}
    all_cols = [c for c in tr.columns if c not in drop]
    mkt_cols = [c for c in all_cols if c.startswith("MKT_")]
    a158_cols = [c for c in all_cols if not c.startswith("MKT_")]
    print(f"[gate] train {len(tr):,} | val {len(va):,} | eval {len(te):,} rows | "
          f"{te['instrument'].nunique()} stocks | {te['date'].nunique()} dates | "
          f"{len(a158_cols)} stock + {len(mkt_cols)} market\n")
    print(f"HAC lag = {a.hac_lag}\n")

    print("=== headline: trained on LABEL_CSRANK (MASTER's own training target) ===")
    al, vic, beta, ics_full = fit_ridge(tr, va, te, all_cols, "LABEL_CSRANK")
    print(f"   ridge alpha={al:.0f} chosen on val (val IC {vic:+.4f})")
    s_full = report(f"full {len(all_cols)}-feature ridge", ics_full, a.hac_lag)

    # Only meaningful when a market block exists; without one the two feature sets are identical
    # and the "ablation" would compare a model against itself (delta 0, t = nan).
    s_a158, p = None, None
    if mkt_cols:
        print("\n=== ablation: stock features only, same target ===")
        al2, vic2, _, ics_a158 = fit_ridge(tr, va, te, a158_cols, "LABEL_CSRANK")
        print(f"   ridge alpha={al2:.0f} chosen on val (val IC {vic2:+.4f})")
        s_a158 = report("stock-features-only ridge", ics_a158, a.hac_lag)

        p = paired(ics_full, ics_a158, a.hac_lag)
        print(f"\n   market block adds: {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}  "
              f"win-rate {p['win_rate']:.1%}")
    else:
        print("\n=== ablation: skipped (no market block in this dataset) ===")

    print("\n=== ablation: raw-return training target (what v1 used) ===")
    al3, vic3, _, ics_raw = fit_ridge(tr, va, te, all_cols, "LABEL")
    print(f"   ridge alpha={al3:.0f} chosen on val (val IC {vic3:+.4f})")
    s_raw = report(f"full {len(all_cols)} ridge (raw target)", ics_raw, a.hac_lag)
    p2 = paired(ics_full, ics_raw, a.hac_lag)
    print(f"\n   CSRankNorm target adds: {p2['delta']:+.4f}  HAC t {p2['t_hac']:+.2f}  "
          f"win-rate {p2['win_rate']:.1%}")

    print("\n=== top |standardised coefficient| (headline model) ===")
    for f, b in sorted(zip(all_cols, beta[1:]), key=lambda x: -abs(x[1]))[:12]:
        print(f"   {'[MKT]' if f.startswith('MKT_') else '     '} {f:<22} {b:+.6f}")

    print(f"\n=== why HAC matters here ===")
    print(f"   naive SE {s_full['se_naive']:.5f} -> HAC SE {s_full['se_hac']:.5f} "
          f"({s_full['se_hac']/s_full['se_naive']:.2f}x)")
    print(f"   naive t  {s_full['t_naive']:+.2f}    -> HAC t  {s_full['t_hac']:+.2f}")
    print("   The naive SE assumes independent daily ICs; the 4-day overlapping label makes them")
    print("   strongly autocorrelated, so only the HAC number is reportable.")

    print("\n=== context (different data/protocol -- NOT targets) ===")
    print("   MASTER paper's own GRU column (their protocol) : +0.0520  RankIC")
    print("   MASTER (published, own confidential data)       : +0.0760  RankIC")
    print("   NOTE: the '+0.0584 Qlib GRU' quoted elsewhere in this project is a DIFFERENT")
    print("   benchmark (20 filtered features, next-day label, step_len 20) -- not comparable.")

    out = {"eval_rows": len(te), "stocks": int(te["instrument"].nunique()),
           "dates": int(te["date"].nunique()), "hac_lag": a.hac_lag,
           "ridge_csrank": s_full, "ridge_stock_only": s_a158, "ridge_raw_target": s_raw,
           "market_block_delta": p, "csrank_target_delta": p2}
    with open(f"{a.data}/_gate_result.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {a.data}/_gate_result.json")


if __name__ == "__main__":
    main()
