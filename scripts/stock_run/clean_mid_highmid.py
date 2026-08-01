#!/usr/bin/env python3
"""Clean blank fields out of the mid_highmid_20yr_portfolios dataset before it ever
reaches EXAMM.

WHY THIS EXISTS: EXAMM's C++ CSV parser (time_series/time_series.cxx) has no
date-awareness -- each stock's file becomes a purely positional per-column array, and on
an unparseable (blank) field it Log::errors and silently skips appending that ONE value to
that ONE column's array, permanently misaligning it against every other column from that
row forward. Not a crash -- silent corruption of feature/label alignment for the rest of
the file. This script resolves every blank before that can happen, or refuses to proceed.

Methodology independently verified across two rounds of expert review plus a design pass
that re-audited the raw data under this exact rule ordering (see the ICAIF paper-review
conversation this was built for -- ordering matters: rule 1 MUST run before rules 3/4, or
real blank-RET rows get misread as "mixed" runs that don't actually exist once rule 1 has
already dropped them).

Per-ticker pipeline, strictly ordered (see clean_ticker()):
  1. Drop rows where RET (the label) is blank -- no fill can substitute for a missing
     target. ~20 rows total, all pre-2021, none in this project's val/test years.
  2. Trim each file's full LEADING blank run (not just row 0) -- a pure prefix trim, no
     interior gap created either way. One file (KEY) has a 1,016-row leading BA_SPREAD gap.
  3. TURNOVER==0 (halt-day) domain-aware fix, corpus-wide:
       - same-day ILLIQUIDITY blank -> per-ticker sentinel, -10 * that ticker's own
         max(|ILLIQUIDITY|) elsewhere in its file (real values are tiny/signed, ~1e-9..
         1e-11; this stays proportionate to that stock's own scale and is contained to
         that one stock/column since --normalize avg_std_dev statistics are computed
         per-file).
       - VOL_CHANGE blank -> recomputed against the last NON-ZERO-turnover baseline
         instead of the halt day itself (verified: VOL_CHANGE already equals this exact
         ratio formula everywhere it's populated in the source data).
  4. BA_SPREAD forward-fill (LOCF), guarded: auto-clear a contiguous blank run only if
     BA_SPREAD is the sole blank column in that range, AND (the run is entirely dated
     before 2004-01-01 [the known vendor-outage pattern, up to 109 rows, ~121/200 files]
     OR the run is exactly 1 row long, any date [confirmed low-risk shape, same as the
     reviewed pattern]). Anything else hard-fails loudly for a human to look at.
  5. Final assertion: zero remaining blanks anywhere in RET/VOL_CHANGE/BA_SPREAD/
     ILLIQUIDITY/sprtrn/TURNOVER. No permissive fallback -- an unexplained blank stops the
     run rather than getting silently patched by an unreviewed generic rule.

Usage:
    python3 scripts/stock_run/clean_mid_highmid.py \
        --in datasets/mid_highmid_20yr_portfolios \
        --out datasets/mid_highmid_20yr_portfolios_clean
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

COLUMNS = ["date", "RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER"]
CHECK_COLS = ["RET", "VOL_CHANGE", "BA_SPREAD", "ILLIQUIDITY", "sprtrn", "TURNOVER"]
BA_SPREAD_GUARD_CUTOFF = "2004-01-01"

# OPTIONAL trading columns. Present only in the price-bearing copy of the dataset
# (mid_highmid_20yr_portfolios 2); absent from the training-only copy. When present they
# are carried through cleaning so the same test rows can feed trade_portfolio.py.
#
# VERIFIED on all 200 tickers: TRAN_COST == BA_SPREAD * PRC / 2 exactly (max abs deviation
# 4.7e-15 -- pure floating-point noise), i.e. the standard half-spread cost model. Two
# consequences drive the handling below:
#   - TRAN_COST is blank exactly where BA_SPREAD is blank (16,808 rows, identical count).
#     Rather than fill it independently, RECOMPUTE it from the already-LOCF-filled
#     BA_SPREAD, which keeps the identity exact instead of letting the two drift apart.
#   - TRAN_COST < 0 (8,945 rows) wherever BA_SPREAD < 0, i.e. a CROSSED QUOTE (bid > ask),
#     a known CRSP artifact rather than a real negative spread. Left as-is it becomes a
#     trading SUBSIDY in Financial_toolbox (buy_stock does cash/(price + tc), so tc<0 buys
#     MORE shares). Floored at zero: a data error must not pay the strategy. Only 18 of
#     these fall inside any trading window (all 2024), so this is a correctness guard, not
#     a results-moving choice.
PRICE_COLS = ["PRC", "TRAN_COST"]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def find_runs(mask: np.ndarray):
    """Contiguous (start, end) index pairs (inclusive) where mask is True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def clean_ticker(df: pd.DataFrame, ticker: str, log_rows: list):
    """Returns (cleaned_df, ret_blank_dates) -- the latter feeds apply_set_alignment()."""
    def log(rule, column, d0, d1, n, detail):
        log_rows.append({
            "ticker": ticker, "rule": rule, "column": column,
            "date_start": d0, "date_end": d1, "n_rows": n, "detail": detail,
        })

    # ---- rule 1: drop RET-blank rows (must run first -- see module docstring) ----
    ret_blank_mask = df["RET"].isna()
    ret_blank_dates = set()
    if ret_blank_mask.any():
        dropped = df.loc[ret_blank_mask, "date"]
        ret_blank_dates = set(dropped)
        log("drop_ret_blank", "RET", dropped.iloc[0], dropped.iloc[-1],
            int(ret_blank_mask.sum()), "label missing, no fill possible")
        df = df.loc[~ret_blank_mask].reset_index(drop=True)
    if df["TURNOVER"].isna().any():
        fail(f"{ticker}: TURNOVER still has blanks after dropping RET-blank rows -- "
             f"the two were expected to coincide exactly; investigate before proceeding")

    # ---- rule 2: trim the full leading blank run (not just row 0) ----
    valid = df[CHECK_COLS].notna().all(axis=1).to_numpy()
    if len(valid) == 0:
        fail(f"{ticker}: no rows left after dropping RET-blank rows")
    if not valid[0]:
        first_valid = int(np.argmax(valid)) if valid.any() else len(df)
        if first_valid == 0:
            fail(f"{ticker}: row 0 blank but argmax found no leading run -- internal error")
        dropped = df.iloc[:first_valid]
        log("trim_leading_run", "multiple", dropped["date"].iloc[0], dropped["date"].iloc[-1],
            first_valid, "prefix trim, no interior gap")
        df = df.iloc[first_valid:].reset_index(drop=True)

    # ---- rule 3: TURNOVER==0 (halt-day) domain-aware fix, corpus-wide ----
    # NOTE: the ILLIQUIDITY sentinel is specifically scoped to TURNOVER==0 rows (that is
    # the actual undefined-ratio cause). The VOL_CHANGE recompute is NOT scoped that way
    # on purpose -- it must run for ANY remaining blank VOL_CHANGE, not just ones adjacent
    # to a recorded TURNOVER==0 halt. A second, distinct cause produces the same shape of
    # blank: rule 1 can delete a contiguous RET-blank block (e.g. a multi-day blackout with
    # TURNOVER genuinely NaN, not 0.0, for every row in the block -- TTWO's 2002-01-23..
    # 02-14 blackout is exactly this), and the first row after the deleted block then has a
    # real TURNOVER but a VOL_CHANGE that was computed (in the source file) against a
    # now-deleted, invalid prior row. The fix is identical either way -- recompute against
    # the most recent row with a real, non-zero TURNOVER -- so it must not be gated behind
    # "does this file have a TURNOVER==0 row anywhere"; TTWO has none, yet still needs this.
    turnover = df["TURNOVER"].to_numpy(dtype=float)
    illiq_fix = (turnover == 0.0) & df["ILLIQUIDITY"].isna().to_numpy()
    if illiq_fix.any():
        max_abs_illiq = df["ILLIQUIDITY"].abs().max()
        if not np.isfinite(max_abs_illiq) or max_abs_illiq == 0:
            fail(f"{ticker}: cannot derive an ILLIQUIDITY sentinel -- "
                 f"no finite non-zero real ILLIQUIDITY value anywhere in this file")
        sentinel = -10.0 * max_abs_illiq
        for idx in np.flatnonzero(illiq_fix):
            log("turnover0_illiquidity_sentinel", "ILLIQUIDITY",
                df.at[idx, "date"], df.at[idx, "date"], 1,
                f"turnover=0 halt day; sentinel={sentinel:.6e} "
                f"(-10x max|ILLIQUIDITY|={max_abs_illiq:.6e} in this file)")
        df.loc[illiq_fix, "ILLIQUIDITY"] = sentinel

    vol_blank = df["VOL_CHANGE"].isna().to_numpy()
    for pos in np.flatnonzero(vol_blank):
        t0 = pos - 1
        while t0 >= 0 and turnover[t0] == 0.0:
            t0 -= 1
        if t0 < 0:
            continue  # no valid baseline -- rule 5's assertion will catch this
        baseline = turnover[t0]
        new_val = (turnover[pos] - baseline) / baseline
        df.iat[pos, df.columns.get_loc("VOL_CHANGE")] = new_val
        log("volchange_recompute", "VOL_CHANGE",
            df.at[pos, "date"], df.at[pos, "date"], 1,
            f"baseline_date={df.at[t0, 'date']} baseline_turnover={baseline:.6f} "
            f"-> new_value={new_val:.6f}")

    # ---- rule 4: guarded BA_SPREAD LOCF ----
    ba_blank = df["BA_SPREAD"].isna().to_numpy()
    other_cols = [c for c in CHECK_COLS if c != "BA_SPREAD"]
    for start, end in find_runs(ba_blank):
        run_len = end - start + 1
        dates = df.loc[start:end, "date"]
        if df.loc[start:end, other_cols].isna().any().any():
            fail(f"{ticker}: BA_SPREAD blank run {dates.iloc[0]}..{dates.iloc[-1]} "
                 f"({run_len} rows) has another blank column in the same range -- "
                 f"refusing to auto-clear, needs human review")
        all_pre_cutoff = (dates < BA_SPREAD_GUARD_CUTOFF).all()
        if not (all_pre_cutoff or run_len == 1):
            fail(f"{ticker}: BA_SPREAD blank run {dates.iloc[0]}..{dates.iloc[-1]} "
                 f"({run_len} rows) does not satisfy the auto-clear guard "
                 f"(entirely pre-{BA_SPREAD_GUARD_CUTOFF} OR isolated 1-row gap) -- "
                 f"refusing to auto-clear, needs human review")
        if start == 0:
            fail(f"{ticker}: BA_SPREAD run starts at row 0 -- should have been removed "
                 f"by the leading-run trim; internal error")
        fill_value = df.at[start - 1, "BA_SPREAD"]
        df.loc[start:end, "BA_SPREAD"] = fill_value
        log("ba_spread_locf", "BA_SPREAD", dates.iloc[0], dates.iloc[-1], run_len,
            f"forward-filled from {df.at[start - 1, 'date']}={fill_value}")

    # ---- rule 6 (price-bearing copy only): PRC / TRAN_COST ----
    # Runs AFTER the BA_SPREAD LOCF above so the recompute uses filled spreads.
    have_price = all(c in df.columns for c in PRICE_COLS)
    if have_price:
        tc_blank = df["TRAN_COST"].isna().to_numpy()
        if tc_blank.any():
            recomputed = df["BA_SPREAD"] * df["PRC"] / 2.0
            n_bad = int((tc_blank & recomputed.isna().to_numpy()).sum())
            if n_bad:
                fail(f"{ticker}: {n_bad} blank TRAN_COST rows cannot be recomputed "
                     f"(BA_SPREAD or PRC still blank there) -- refusing to guess")
            dates = df.loc[tc_blank, "date"]
            df.loc[tc_blank, "TRAN_COST"] = recomputed[tc_blank]
            log("tran_cost_recompute", "TRAN_COST", dates.iloc[0], dates.iloc[-1],
                int(tc_blank.sum()), "= BA_SPREAD * PRC / 2 (verified identity), "
                                     "using the LOCF-filled BA_SPREAD")

        neg = (df["TRAN_COST"] < 0).to_numpy()
        if neg.any():
            dates = df.loc[neg, "date"]
            log("tran_cost_floor_zero", "TRAN_COST", dates.iloc[0], dates.iloc[-1],
                int(neg.sum()), "crossed quote (BA_SPREAD<0) -- floored at 0 so a data "
                                "artifact cannot pay the strategy")
            df.loc[neg, "TRAN_COST"] = 0.0

        if (df["PRC"] <= 0).any():
            n = int((df["PRC"] <= 0).sum())
            fail(f"{ticker}: {n} rows with PRC <= 0 -- trade_portfolio.py rejects these")

    # ---- rule 5: final assertion, no silent fallback ----
    assert_cols = CHECK_COLS + (PRICE_COLS if have_price else [])
    remaining = df[assert_cols].isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        fail(f"{ticker}: unexplained blanks remain after cleaning: "
             f"{remaining.to_dict()} -- refusing to write output")

    return df, ret_blank_dates


def apply_set_alignment(cleaned: dict, ret_blank_dates: set, log_rows: list) -> dict:
    """Cross-ticker reconciliation, run once per SET after every ticker is otherwise
    cleaned. Rule 1 (drop RET-blank rows) can remove interior dates from just the one
    affected ticker -- align_cohort.py requires byte-identical calendars across all 50
    stocks in a set once truncated to a common start date, and it was designed for
    different LISTING dates (a leading-trim difference, which it already handles), not an
    interior gap in just one stock's series. Standard fix for an aligned cross-sectional
    panel: exclude the same calendar dates from every stock in the set, not just the one
    that was missing data that day. Confirmed cheap: only 2 tickers ever trigger rule 1
    (TTWO, BIIB), ~20 total dates, against a ~5,000-row training window per stock.
    """
    if not ret_blank_dates:
        return cleaned
    out = {}
    for ticker, df in cleaned.items():
        hit = df["date"].isin(ret_blank_dates)
        if hit.any():
            dates = df.loc[hit, "date"]
            log_rows.append({
                "ticker": ticker, "rule": "set_alignment_drop", "column": "multiple",
                "date_start": dates.iloc[0], "date_end": dates.iloc[-1], "n_rows": int(hit.sum()),
                "detail": "date(s) dropped elsewhere in this set for a blank RET row; "
                          "excluded here too so every stock in the set shares one calendar",
            })
            df = df.loc[~hit].reset_index(drop=True)
        out[ticker] = df
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, type=Path)
    ap.add_argument("--out", dest="outdir", required=True, type=Path)
    ap.add_argument("--sets", nargs="+", default=["set1", "set2", "set3", "set4"])
    args = ap.parse_args()

    indir = args.indir if args.indir.is_absolute() else REPO / args.indir
    outdir = args.outdir if args.outdir.is_absolute() else REPO / args.outdir
    if not indir.is_dir():
        fail(f"input dir not found: {indir}")

    log_rows = []
    print(f"{'set':6s} {'ticker':7s} {'rows_in':>8s} {'rows_out':>8s}  fixes applied")
    total_in = total_out = 0
    n_tickers = 0

    for set_name in args.sets:
        set_in = indir / set_name
        if not set_in.is_dir():
            fail(f"{set_in} not found")
        set_out = outdir / set_name
        set_out.mkdir(parents=True, exist_ok=True)

        # Pass 1: clean every ticker in this set independently, and collect the union of
        # RET-blank dates across all 50 -- needed for the cross-ticker alignment pass.
        rows_in = {}
        cleaned = {}
        ret_blank_union = set()
        for f in sorted(set_in.glob("*.csv")):
            ticker = f.stem
            df = pd.read_csv(f)
            missing = [c for c in COLUMNS if c not in df.columns]
            # keep PRC/TRAN_COST when the source has them (price-bearing copy)
            keep = COLUMNS + [c for c in PRICE_COLS if c in df.columns]
            if missing:
                fail(f"{set_name}/{ticker}: missing columns {missing}")
            df = df[keep]
            rows_in[ticker] = len(df)
            cleaned[ticker], ret_blank_dates = clean_ticker(df, ticker, log_rows)
            ret_blank_union |= ret_blank_dates

        # Pass 2: cross-ticker alignment -- see apply_set_alignment() docstring.
        cleaned = apply_set_alignment(cleaned, ret_blank_union, log_rows)

        for ticker, df in cleaned.items():
            remaining = df[CHECK_COLS].isna().sum()
            remaining = remaining[remaining > 0]
            if len(remaining):
                fail(f"{set_name}/{ticker}: unexplained blanks after set-alignment: "
                     f"{remaining.to_dict()} -- refusing to write output")
            df.to_csv(set_out / f"{ticker}.csv", index=False)
            n_in, n_out = rows_in[ticker], len(df)
            n_fixes = sum(1 for r in log_rows if r["ticker"] == ticker)
            print(f"{set_name:6s} {ticker:7s} {n_in:8d} {n_out:8d}  {n_fixes} log entries")
            total_in += n_in
            total_out += n_out
            n_tickers += 1

    log_df = pd.DataFrame(log_rows, columns=["ticker", "rule", "column", "date_start",
                                              "date_end", "n_rows", "detail"])
    log_path = outdir / "clean_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"\n{n_tickers} tickers cleaned, {total_in} rows in -> {total_out} rows out "
          f"({total_in - total_out} dropped)")
    print(f"wrote {len(log_df)} fix-log entries -> {log_path}")
    print(f"wrote cleaned dataset -> {outdir}")


if __name__ == "__main__":
    main()
