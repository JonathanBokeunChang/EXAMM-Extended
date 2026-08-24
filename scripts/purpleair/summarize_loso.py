#!/usr/bin/env python3
"""Tabulate a PurpleAir LOSO sweep: evolved vs the deployed incumbent, per held-out site.

    python3 scripts/purpleair/summarize_loso.py --tag loso

Reads each run's test_score.txt (written by purpleair_run.sh) rather than the sweep log, so
a partially-written log or an interleaved second sweep cannot corrupt the table.

BOTH TIMESCALES, ALWAYS. Barkjohn Eq. 10 was fitted on 24-hour averages, so the 24-h row is
the apples-to-apples comparison and the hourly row is the harder, out-of-regime one. A
result quoted at one timescale only is not interpretable.

RMSE AND R2 ARE NOT ENOUGH ON THEIR OWN. Several cells win on squared error while losing on
MAE and bias, which matters for an air-quality correction: a forecaster that is right on
average but badly biased will systematically mis-call the threshold that triggers a public
health advisory. All four are printed, and the win column counts RMSE only so the reader
can see for themselves where the other three disagree.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROW = re.compile(r"^\s*(Barkjohn Eq\.10(?: \(incumbent\))?|evolved|raw pa_cf1[^\d]*)\s+"
                 r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-]?[\d.]+)\s*$")


def parse(path):
    """Return {'hourly': {...}, 'daily': {...}} of model -> metrics."""
    out, section = {"hourly": {}, "daily": {}}, "hourly"
    for line in open(path):
        if "24-hour averages" in line:
            section = "daily"
            continue
        m = ROW.match(line)
        if not m:
            continue
        name = "incumbent" if m.group(1).startswith("Barkjohn") else (
            "evolved" if m.group(1).startswith("evolved") else "raw")
        out[section][name] = dict(n=int(m.group(2)), rmse=float(m.group(3)),
                                  mae=float(m.group(4)), bias=float(m.group(5)),
                                  r2=float(m.group(6)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="loso")
    a = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "test_output", f"purpleair_{a.tag}_*_holdout_*"))):
        base = os.path.basename(d)
        arm = base.split(f"purpleair_{a.tag}_")[1].split("_holdout_")[0]
        site = base.split("_holdout_")[1]
        f = os.path.join(d, "test_score.txt")
        if not os.path.exists(f):
            continue
        p = parse(f)
        if "evolved" not in p["daily"] or "incumbent" not in p["daily"]:
            continue
        rows.append((arm, site, p))

    for arm in ("exagpmin", "examm"):
        sub = [r for r in rows if r[0] == arm]
        if not sub:
            continue
        print(f"\n=== {arm} : evolved vs Barkjohn Eq.10, per held-out site ===")
        print(f"{'site':<6} | {'24h RMSE inc':>12} {'24h RMSE evo':>12} {'delta%':>8} "
              f"| {'24h R2 inc':>10} {'24h R2 evo':>10} | {'hr RMSE inc':>11} {'hr RMSE evo':>11}")
        wins = 0
        deltas = []
        for _, site, p in sorted(sub, key=lambda x: x[1]):
            di, de = p["daily"]["incumbent"], p["daily"]["evolved"]
            hi, he = p["hourly"]["incumbent"], p["hourly"]["evolved"]
            d_pct = 100.0 * (de["rmse"] - di["rmse"]) / di["rmse"]
            deltas.append(d_pct)
            wins += de["rmse"] < di["rmse"]
            print(f"{site:<6} | {di['rmse']:>12.3f} {de['rmse']:>12.3f} {d_pct:>+8.1f} "
                  f"| {di['r2']:>10.3f} {de['r2']:>10.3f} | {hi['rmse']:>11.3f} {he['rmse']:>11.3f}")
        print(f"{'':<6} | evolved beats incumbent on 24-h RMSE in {wins}/{len(sub)} sites; "
              f"median delta {np.median(deltas):+.1f}%, mean {np.mean(deltas):+.1f}%")


if __name__ == "__main__":
    main()
