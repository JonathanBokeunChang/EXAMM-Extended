#!/usr/bin/env python3
"""Score every completed EXA-GP-MIN vs EXAMM cell, both metrics, both trading conventions.

    python3 scripts/stock_run/score_all_cells.py --budget g500
    python3 scripts/stock_run/score_all_cells.py --budget g500 --json results/exagp_cells.json

A cell is (panel, cohort). It is scored only when BOTH arms have all ten seeds done, so a
half-finished sweep can never contribute a lopsided comparison. Everything is cached: the
expensive step is eval_ic_run.sh (one evaluate_rnn call per stock per seed), and it is
skipped when ensemble_test/ already exists.

WHY BOTH TRADING CONVENTIONS ARE REPORTED, ALWAYS. Algorithm 2 trades only when all L
candidate longs are predicted positive and all S candidate shorts negative, so its return
depends on the LEVEL of the predictions -- a quantity that carries no cross-sectional
information and that no dollar-neutral book acts on. Measured: on the 2022 cell the raw
gate opened 29/250 days for EXAMM and 1/250 for EXA-GP-MIN, giving +36.71% and +1.86%;
after cross-sectional demeaning both gates open every day and both return about -18%. On
set1/cohort_2022 the raw gate never opened at all and the strategy reported exactly
+0.00%, which reads as flat and means "never traded". Quoting one convention alone
therefore reports an artifact half the time, in whichever direction the bias happens to
fall. `gate_raw` is printed next to the raw return so a near-zero open count is visible
rather than inferred.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts/stock_run"))
from score_forecasting import PRIMARY, data_dir, from_per_ticker, metrics  # noqa: E402

TRADE_YEAR = {"cohort_2020_aligned": 2022, "cohort_2021_aligned": 2023,
              "cohort_2022_aligned": 2024}
ARMS = ["exagpmin", "examm"]
L = S = 10


def tree(arm, panel, cohort, budget):
    mid = f"{panel}_" if panel != "dsA" else ""
    return os.path.join(REPO, "test_output", f"{arm}_{mid}{cohort}_{budget}")


def n_done(root):
    return len(glob.glob(os.path.join(root, "run_*", ".done")))


def ensure_ensemble(root):
    if os.path.isdir(os.path.join(root, "ensemble_test")):
        return True
    rel = os.path.relpath(root, REPO)
    r = subprocess.run(["bash", "scripts/stock_run/eval_ic_run.sh", rel, "test"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! eval failed for {rel}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def ensure_demeaned(root):
    out = os.path.join(root, "ensemble_test_demeaned")
    if os.path.isdir(out):
        return out
    r = subprocess.run(["python3", "scripts/stock_run/demean_predictions.py",
                        "--in-dir", os.path.join(root, "ensemble_test"), "--out-dir", out],
                       cwd=REPO, capture_output=True, text=True)
    return out if r.returncode == 0 else None


def gate_opens(pred_dir, panel, cohort, year):
    df = from_per_ticker(pred_dir, data_dir(panel, cohort))
    df = df[df["date"].astype(str).str[:4] == str(year)]
    P = df.pivot(index="date", columns="ticker", values="predicted_RET").sort_index()
    n = 0
    for _, row in P.iterrows():
        v = row.to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < L + S:
            continue
        o = np.argsort(-v)
        if (v[o[:L]] > 0).all() and (v[o[-S:]] < 0).all():
            n += 1
    return n, len(P)


def trade(pred_dir, panel, cohort, year, strategy="daily_hybrid_long_short_return"):
    cmd = ["python3", "scripts/stock_run/trade_portfolio.py",
           "--pred-dir", os.path.relpath(pred_dir, REPO),
           "--data-dir", os.path.relpath(data_dir(panel, cohort), REPO),
           "--align", "suffix", "--expect-n", "50", "--long", str(L), "--short", str(S),
           "--strategy", strategy]
    # cohort_2020's test split spans two years; restrict to the trade year, as the
    # forecasting scorer does with its year slice.
    cmd += ["--window-start", f"{year}-01-01", "--window-end", f"{year}-12-31"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    m = re.search(r"strategy return\s*:\s*([+-]?[\d.]+)%", r.stdout)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="g500")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    panels = ["dsA", "set1", "set2", "set3", "set4"]
    cells, rows = [], []
    for panel in panels:
        for cohort in sorted(TRADE_YEAR):
            roots = {arm: tree(arm, panel, cohort, a.budget) for arm in ARMS}
            if not all(n_done(r) == 10 for r in roots.values()):
                continue
            cells.append((panel, cohort))
            year = TRADE_YEAR[cohort]
            rec = {"panel": panel, "cohort": cohort, "trade_year": year}
            for arm in ARMS:
                root = roots[arm]
                if not ensure_ensemble(root):
                    continue
                ens = os.path.join(root, "ensemble_test")
                m = metrics(from_per_ticker(ens, data_dir(panel, cohort)), year)
                dm = ensure_demeaned(root)
                g, nd = gate_opens(ens, panel, cohort, year)
                rec[arm] = {
                    **{f"ic{h}": m.get(f"ic{h}") for h in PRIMARY},
                    "mse": m["mse"],
                    "ret_raw": trade(ens, panel, cohort, year),
                    "ret_demeaned": trade(dm, panel, cohort, year) if dm else float("nan"),
                    # The ungated strategy ranks by prediction and trades every day. Ranking is
                    # invariant to a per-day level shift, so this is immune to the prediction-bias
                    # artifact BY CONSTRUCTION rather than by a demeaning preprocessing step --
                    # which is why it, not ret_demeaned, is the headline trading column.
                    "ret_ungated": trade(ens, panel, cohort, year, "daily_long_short_return"),
                    "gate_raw": g, "gate_days": nd,
                }
            rows.append(rec)

    if not rows:
        print(f"no complete paired cells at budget {a.budget} yet")
        return

    hdr = f"{'cell':<18} {'yr':>5} | {'IC1 gp':>8} {'IC1 ex':>8} | {'ungated gp':>11} {'ungated ex':>11} | {'gate gp':>8} {'gate ex':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        gp, ex = r.get("exagpmin", {}), r.get("examm", {})
        print(f"{r['panel']+'/'+r['cohort'][7:11]:<18} {r['trade_year']:>5} | "
              f"{gp.get('ic1', float('nan')):>+8.4f} {ex.get('ic1', float('nan')):>+8.4f} | "
              f"{gp.get('ret_ungated', float('nan')):>+11.2f} {ex.get('ret_ungated', float('nan')):>+11.2f} | "
              f"{gp.get('gate_raw', -1):>4}/{gp.get('gate_days', 0):<3} {ex.get('gate_raw', -1):>4}/{ex.get('gate_days', 0):<3}")

    # Paired summary -- the form the claim should take, per-cell wins rather than a mean
    # dominated by whichever cell happened to be extreme.
    for key, label in (("ic1", "rank IC h=1"), ("ret_ungated", "return (ungated)"),
                       ("ret_demeaned", "return (gated+demeaned)")):
        w = sum(1 for r in rows
                if np.isfinite(r.get("exagpmin", {}).get(key, np.nan))
                and np.isfinite(r.get("examm", {}).get(key, np.nan))
                and r["exagpmin"][key] > r["examm"][key])
        n = sum(1 for r in rows
                if np.isfinite(r.get("exagpmin", {}).get(key, np.nan))
                and np.isfinite(r.get("examm", {}).get(key, np.nan)))
        print(f"\nEXA-GP-MIN beats EXAMM on {label}: {w}/{n} cells")

    if a.json:
        os.makedirs(os.path.dirname(os.path.join(REPO, a.json)), exist_ok=True)
        with open(os.path.join(REPO, a.json), "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
