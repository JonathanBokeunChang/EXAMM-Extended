#!/usr/bin/env python3
"""Score the Dataset A cells (the original 50-stock universe) exactly as score_cells.py scores the
four portfolio draws.

    python3 scripts/transformer_bench/score_dataset_a.py            # every model found
    python3 scripts/transformer_bench/score_dataset_a.py DLinearOfficial DeformTime

WHY A SEPARATE ENTRY POINT AND NOT A FLAG ON score_cells.py. Two things differ, and both are about
the data rather than the metric:

  1. There is no SET axis. score_cells.py builds data_dir as <mid_highmid_price>/<set>/<cohort>;
     Dataset A lives at datasets/walkforward/<cohort> and already carries PRC and TRAN_COST.
  2. cohort_2020_aligned's TEST WINDOW SPANS TWO CALENDAR YEARS -- 501 rows, 2022-01-03 to
     2023-12-29, against cohort_2021_aligned's 250. Scoring it whole returns a two-year blend.
     results/hp_grid_orig records what that costs: +0.0157 against a true 2022 figure of +0.0030,
     a 5x overstatement that looks entirely plausible. Every 2022 number here is date-filtered.

EVERYTHING ELSE IS IMPORTED from score_cells.py rather than reimplemented -- the seed-mean ensemble,
the daily rank IC, Algorithm 2 via the real trader, the net-of-cost pass and the Sharpe curve. A
second implementation of any of those would be a second thing to keep in agreement with the paper.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_cells import REPO, ensemble, daily_ic, trade, gate_days   # noqa: E402

ROOT = os.path.join(REPO, "results/transformer_bench/author")
DATA = os.path.join(REPO, "datasets/walkforward")

# cohort -> the calendar year this study reports for it. cohort_2020's test split also contains
# 2023, which cohort_2021 covers on its own; reporting both would double-count the year.
YEAR = {"cohort_2020_aligned": 2022, "cohort_2021_aligned": 2023}


def cells(models=None):
    found = {}
    for f in glob.glob(os.path.join(ROOT, "*/*/L*/seed_*/predictions.csv")):
        d = os.path.dirname(f)
        if not os.path.exists(os.path.join(d, ".done")):
            continue
        parts = os.path.relpath(f, ROOT).split(os.sep)
        coh, model = parts[0], parts[1]
        if coh not in YEAR or (models and model not in models):
            continue
        found.setdefault((model, coh), []).append(f)
    return found


def main():
    models = set(sys.argv[1:]) or None
    out = []
    for (model, coh), files in sorted(cells(models).items()):
        files = sorted(files)
        data_dir = os.path.join(DATA, coh)
        yr = YEAR[coh]
        try:
            full = ensemble(files, data_dir)
            before = full["date"].nunique()
            # IC, MSE and MAE are per-day averages, so slicing the frame is the right restriction.
            df = full[full["date"].astype(str).str[:4] == str(yr)].reset_index(drop=True)
            after = df["date"].nunique()
            ic, tstat, _ = daily_ic(df)
            icir = float(tstat) / math.sqrt(after) if after else float("nan")
            err = df["predicted_RET"].to_numpy(float) - df["expected_RET"].to_numpy(float)
            # RETURNS COMPOUND, so they cannot be sliced after the fact -- and handing the trader a
            # sliced frame fails outright, because --align suffix trims to the trailing window and
            # 2022 is the PREFIX of cohort_2020's 501-day file (verified: rc=1). Pass the whole
            # frame and let the trader restrict trading dates after alignment, which is what
            # --window-start/--window-end exist for.
            win = (f"{yr}-01-01", f"{yr}-12-31")
            ret, n_tic, e1, sh = trade(full, data_dir, window=win)
            net, _, e2, _ = trade(full, data_dir, tc=True, window=win)
            # A friction cost cannot raise a return. If it does, the short-sale credit fix is
            # missing from the local Financial_toolbox and every net figure is suspect.
            flag = "NET>GROSS" if np.isfinite(ret) and np.isfinite(net) and net > ret + 1e-6 else ""
            out.append({"model": model, "cohort": coh, "yr": yr, "n": len(files),
                        "days_before": before, "days": after,
                        "ic": round(float(ic), 5), "t": round(float(tstat), 2),
                        "icir": round(icir, 3),
                        "mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
                        "ret": None if not np.isfinite(ret) else round(float(ret), 2),
                        "net": None if not np.isfinite(net) else round(float(net), 2),
                        "sharpe": None if not np.isfinite(sh) else round(float(sh), 3),
                        "gate": gate_days(df), "tickers": n_tic,
                        "err": " ".join(x for x in (e1, e2, flag) if x)})
        except Exception as exc:                                  # noqa: BLE001
            out.append({"model": model, "cohort": coh, "yr": yr, "n": len(files),
                        "ic": None, "icir": None, "ret": None, "net": None, "sharpe": None,
                        "gate": 0, "days": 0, "err": f"{type(exc).__name__}: {exc}"[:120]})

    hdr = f"  {'model':22s} {'yr':>4s} {'n':>3s} {'days':>9s} {'IC':>9s} {'ICIR':>7s} {'net%':>8s} {'gross%':>8s} {'Sharpe':>7s} {'gate':>5s}"
    print(hdr, file=sys.stderr)
    for r in out:
        f = lambda v, p=4: ("---" if v is None else f"{v:+.{p}f}")                  # noqa: E731
        days = f"{r.get('days','?')}/{r.get('days_before','?')}"
        print(f"  {r['model']:22s} {r['yr']:4d} {r['n']:3d} {days:>9s} {f(r['ic']):>9s} "
              f"{f(r.get('icir'), 3):>7s} {f(r.get('net'), 2):>8s} {f(r.get('ret'), 2):>8s} "
              f"{f(r.get('sharpe'), 2):>7s} {str(r.get('gate','')):>5s}"
              + (f"   {r['err']}" if r.get("err") else ""), file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
