"""Canonical volatility dataset builder for the ICAIF paper (E1, E1b, E2).

Supersedes scratchpad/qlib_vol_dataset_big.py (fixed split) and build_vol_cohorts.py
(walk-forward). Both are reproduced here from ONE shared feature/target function, so the
three experiments differ only along the axis each is meant to test:

    E1   qlib_vol_big_v2   --mode fixed   --tickers TICKERS.txt   (120 pinned)
    E1b  qlib_vol_full     --mode fixed   --universe all          (universe varies)
    E2   qlib_vol_cohorts  --mode cohorts --tickers TICKERS.txt   (test year varies)

Per stock, per day t (inputs backward-looking; TARGET is the forecast):
  LV     = log Parkinson range vol from high/low
  MA5/22 = moving averages of LV (the HAR information set)
  RET    = log close-to-close return (leverage)
  LOGVOL = log volume
  TARGET = mean(LV[t+1..t+H]), H=5     -> EXAMM --time_offset 0

EXAMM: --input_parameter_names LV MA5 MA22 RET LOGVOL --output_parameter_names TARGET

BOUNDARY LEAK FIX (the reason this file exists)
-----------------------------------------------
TARGET is computed on the full per-stock panel BEFORE slicing, so row t's label aggregates
LV[t+1..t+H]. The last H rows of a split therefore carry labels built from the NEXT split's
volatility: train's last H rows peek into val, val's last H rows peek into test. Val drives
genome selection, so that second one leaks test information into model selection.

Both legacy builders had this. Fix: after slicing, drop the trailing H rows of train and of
val. Test needs no trim -- its labels reach past the panel end and were already removed by
the pre-slice dropna, and nothing is selected on test anyway.

PRESERVED VERBATIM from the legacy builders (do not "clean up" -- changing any of these
would make v2 differ from the banked results in more than the leak fix):
  * the Parkinson/LV/MA/RET/LOGVOL formulas and their epsilons (1e-9, 1e-4)
  * the shift(-1).rolling(H).mean().shift(-(H-1)) target construction order
  * dropna BEFORE slicing
  * the `ok` mask is used ONLY as a per-stock keep threshold, never as a row filter

Run locally under the qlib venv:
  scratchpad/qlibenv/bin/python scripts/stock_run/build_vol_dataset.py --mode fixed ...
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

REPO = "/Users/jonathanchang/EXAMM-Extended"
DEFAULT_PROVIDER = ("/private/tmp/claude-501/-Users-jonathanchang-EXAMM-Extended/"
                    "215ee96d-8804-494b-9850-55d94ff513a6/scratchpad/qlib_data/cn_data")

#: written column order; EXAMM and the harness both resolve by NAME, not position
COLS = ["date", "LV", "MA5", "MA22", "RET", "LOGVOL", "TARGET"]


# ======================================================================================
# shared feature engineering -- the single source of truth for both modes
# ======================================================================================
def engineer(g: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame | None, int]:
    """Per-stock panel. Returns (frame, n_valid_ohlcv_rows); frame is None if unusable."""
    H_, L_, C, V = g["high"].values, g["low"].values, g["close"].values, g["volume"].values
    ok = (H_ > 0) & (L_ > 0) & (H_ >= L_) & (C > 0)

    park = np.sqrt((np.log(np.clip(H_ / np.maximum(L_, 1e-9), 1e-9, None)) ** 2) / (4 * np.log(2)))
    lv = np.log(park + 1e-4)
    lvs = pd.Series(lv)
    d = pd.DataFrame({
        "date": g.index.get_level_values(1),
        "LV": lv,
        "MA5": lvs.rolling(5).mean().values,
        "MA22": lvs.rolling(22).mean().values,
        "RET": np.concatenate([[0.0], np.diff(np.log(np.maximum(C, 1e-9)))]),
        "LOGVOL": np.log(np.maximum(V, 1.0)),
    })
    # forward H-day mean log vol, aligned to row t (the forecast target)
    d["TARGET"] = lvs.shift(-1).rolling(horizon).mean().shift(-(horizon - 1)).values
    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    return d, int(ok.sum())


def trim_boundary(split: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Drop the trailing `horizon` rows, whose labels reach into the following split."""
    return split.iloc[:-horizon] if len(split) > horizon else split.iloc[0:0]


def write_split(cdir: str, ticker: str, splits: dict[str, pd.DataFrame]) -> None:
    safe = ticker.replace("/", "_")
    for k, v in splits.items():
        v[COLS].to_csv(f"{cdir}/{safe}_{k}.csv", index=False)


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "cohorts"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    # universe control
    ap.add_argument("--tickers", default=None,
                    help="file with one ticker per line -- pins the universe exactly")
    ap.add_argument("--universe", choices=["topn", "all"], default="all")
    ap.add_argument("--n-stocks", type=int, default=120, help="only used by --universe topn")
    # date ranges
    ap.add_argument("--start", default=None, help="default: 2015-01-01 fixed / 2008-01-01 cohorts")
    ap.add_argument("--end", default="2020-12-31")
    # fixed-mode split boundaries (legacy E1 values)
    ap.add_argument("--train-end", default="2017-12-31")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-start", default="2019-01-01")
    # cohorts-mode
    ap.add_argument("--years", nargs="+", type=int, default=[2016, 2017, 2018, 2019, 2020])
    # retention thresholds (mode-dependent defaults applied below)
    ap.add_argument("--min-valid", type=int, default=None, help="min valid OHLCV rows per stock")
    ap.add_argument("--min-train", type=int, default=None, help="min TRAIN rows after trimming")
    ap.add_argument("--min-split", type=int, default=60, help="min val/test rows after trimming")
    a = ap.parse_args()

    if a.start is None:
        a.start = "2015-01-01" if a.mode == "fixed" else "2008-01-01"
    if a.min_valid is None:
        a.min_valid = 600 if a.mode == "fixed" else 400
    if a.min_train is None:
        a.min_train = 60 if a.mode == "fixed" else 250

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=a.provider, region=REG_CN, kernels=1)
    from qlib.data import D

    print(f"loading CSI300 OHLCV {a.start}..{a.end} ...", flush=True)
    df = D.features(D.instruments(market="csi300"),
                    ["$open", "$high", "$low", "$close", "$volume"],
                    start_time=a.start, end_time=a.end, freq="day").dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    print(f"{len(df):,} rows, {df.index.get_level_values(0).nunique()} stocks", flush=True)

    # ---- universe selection -------------------------------------------------------
    pinned = None
    if a.tickers:
        with open(a.tickers) as fh:
            pinned = [ln.strip() for ln in fh if ln.strip()]
        print(f"universe: PINNED to {len(pinned)} tickers from {a.tickers}", flush=True)
    elif a.universe == "topn":
        counts = df.groupby(level=0).size().sort_values(ascending=False)
        pinned = list(counts.index[: a.n_stocks])
        print(f"universe: top {len(pinned)} by row count", flush=True)
    else:
        print("universe: ALL stocks clearing the retention thresholds", flush=True)

    # ---- engineer once; cohorts are date slices of the same panels ----------------
    panels, missing = {}, []
    wanted = pinned if pinned is not None else None
    for t, g in df.groupby(level=0):
        if wanted is not None and t not in wanted:
            continue
        d, n_ok = engineer(g.sort_index(), a.horizon)
        if n_ok < a.min_valid:
            continue
        panels[t] = d
    if wanted is not None:
        missing = [t for t in wanted if t not in panels]
        if missing:
            print(f"WARNING: {len(missing)} pinned tickers unusable "
                  f"(<{a.min_valid} valid rows or absent): {missing[:8]}"
                  f"{' ...' if len(missing) > 8 else ''}", flush=True)
    print(f"engineered {len(panels)} stocks", flush=True)

    os.makedirs(a.out, exist_ok=True)
    manifest, H = [], a.horizon

    def emit(cdir: str, label, tr, va, te) -> tuple[int, int]:
        """Trim the leak, apply thresholds, write. Returns (kept, rows_trimmed)."""
        os.makedirs(cdir, exist_ok=True)
        kept, trimmed = 0, 0
        for t in sorted(panels):
            a_tr, a_va, a_te = tr(panels[t]), va(panels[t]), te(panels[t])
            n_before = len(a_tr) + len(a_va)
            a_tr, a_va = trim_boundary(a_tr, H), trim_boundary(a_va, H)   # <-- LEAK FIX
            trimmed += n_before - len(a_tr) - len(a_va)
            if len(a_tr) < a.min_train or len(a_va) < a.min_split or len(a_te) < a.min_split:
                continue
            write_split(cdir, t, {"train": a_tr, "val": a_va, "test": a_te})
            kept += 1
        return kept, trimmed

    if a.mode == "fixed":
        VY = a.val_year
        kept, trimmed = emit(
            a.out, "fixed",
            lambda d: d[d.date <= a.train_end],
            lambda d: d[(d.date >= f"{VY}-01-01") & (d.date <= f"{VY}-12-31")],
            lambda d: d[d.date >= a.test_start],
        )
        rows = sum(len(pd.read_csv(f"{a.out}/{f}"))
                   for f in os.listdir(a.out) if f.endswith("_train.csv"))
        manifest.append(dict(split="fixed", stocks=kept, train_end=a.train_end, val=VY,
                             test_start=a.test_start, pooled_train_rows=rows,
                             boundary_rows_trimmed=trimmed))
        print(f"fixed: {kept} stocks, train<= {a.train_end}, val {VY}, test >= {a.test_start}, "
              f"{rows:,} pooled train rows, {trimmed:,} boundary rows trimmed", flush=True)
    else:
        for Y in a.years:
            tr_end = f"{Y-2}-12-31"
            kept, trimmed = emit(
                f"{a.out}/cohort_{Y}", Y,
                lambda d, e=tr_end: d[d.date <= e],
                lambda d, y=Y - 1: d[(d.date >= f"{y}-01-01") & (d.date <= f"{y}-12-31")],
                lambda d, y=Y: d[(d.date >= f"{y}-01-01") & (d.date <= f"{y}-12-31")],
            )
            cdir = f"{a.out}/cohort_{Y}"
            rows = sum(len(pd.read_csv(f"{cdir}/{f}"))
                       for f in os.listdir(cdir) if f.endswith("_train.csv"))
            manifest.append(dict(split=f"cohort_{Y}", stocks=kept, train_end=tr_end,
                                 val=Y - 1, test_start=f"{Y}-01-01", pooled_train_rows=rows,
                                 boundary_rows_trimmed=trimmed))
            flag = "  <-- LOW" if (pinned and kept < 90) else ""
            print(f"cohort_{Y}: {kept} stocks, train<= {tr_end}, val {Y-1}, test {Y}, "
                  f"{rows:,} pooled train rows, {trimmed:,} trimmed{flag}", flush=True)

    pd.DataFrame(manifest).to_csv(f"{a.out}/MANIFEST.csv", index=False)
    print(f"\nwrote -> {a.out}   (MANIFEST.csv written)")
    print("EXAMM: --input_parameter_names LV MA5 MA22 RET LOGVOL "
          "--output_parameter_names TARGET --time_offset 0")


if __name__ == "__main__":
    main()
