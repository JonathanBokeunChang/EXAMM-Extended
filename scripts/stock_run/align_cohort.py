#!/usr/bin/env python3
"""Produce a calendar-ALIGNED copy of a walk-forward cohort for cross-sectional
training (the IC loss needs date-index j to be the same date for every stock).

The train files in a cohort start on different dates (stocks IPO at different
times), so their row index j does NOT correspond to a common date. This script:

  1. finds the common start = max over stocks of each train file's first date,
  2. truncates every stock's train to rows with date >= common start,
  3. VERIFIES that the truncated train calendars are byte-identical across all
     stocks (same dates, same order, same count) -- fails loudly otherwise,
  4. VERIFIES val and test calendars are already identical across stocks,
  5. writes <out>/ with truncated train + copied val/test.

Val/test are copied byte-for-byte (they are already aligned in the walk-forward
cohorts). Pure stdlib; data rows pass through byte-faithfully.

Usage:
    python3 scripts/stock_run/align_cohort.py \
        --in datasets/walkforward/cohort_2021 \
        --out datasets/walkforward/cohort_2021_aligned
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def read_lines(path):
    with open(path, "r", newline="") as f:
        return f.read().splitlines(keepends=True)


def header_and_rows(lines):
    return lines[0], lines[1:]


def date_of(row_line):
    return row_line.split(",", 1)[0]


def tickers_in(indir, split):
    return sorted(p.name[: -len(f"_{split}.csv")] for p in indir.glob(f"*_{split}.csv"))


def calendar(rows):
    return [date_of(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, type=Path)
    ap.add_argument("--out", dest="outdir", required=True, type=Path)
    args = ap.parse_args()

    indir = args.indir if args.indir.is_absolute() else REPO / args.indir
    outdir = args.outdir if args.outdir.is_absolute() else REPO / args.outdir
    if not indir.is_dir():
        fail(f"input dir not found: {indir}")

    tickers = tickers_in(indir, "train")
    if not tickers:
        fail(f"no *_train.csv found in {indir}")
    # every ticker must have all three splits
    for split in ("train", "val", "test"):
        have = set(tickers_in(indir, split))
        missing = [t for t in tickers if t not in have]
        if missing:
            fail(f"tickers missing {split} files: {missing}")

    print(f"{len(tickers)} tickers in {indir.name}")

    # load train, compute common start
    train = {}
    first_dates = {}
    header0 = None
    for t in tickers:
        lines = read_lines(indir / f"{t}_train.csv")
        h, rows = header_and_rows(lines)
        if header0 is None:
            header0 = h
        elif h != header0:
            fail(f"{t}: train header differs from {tickers[0]}")
        if not rows:
            fail(f"{t}: empty train file")
        train[t] = (h, rows)
        first_dates[t] = date_of(rows[0])

    common_start = max(first_dates.values())
    print(f"common train start date = {common_start}  "
          f"(latest first-date; from {max(first_dates, key=first_dates.get)})")

    outdir.mkdir(parents=True, exist_ok=True)

    # truncate train, collect calendars for verification
    ref_train_cal = None
    for t in tickers:
        h, rows = train[t]
        kept = [r for r in rows if date_of(r) >= common_start]
        cal = calendar(kept)
        if ref_train_cal is None:
            ref_train_cal = cal
            ref_ticker = t
        elif cal != ref_train_cal:
            # find first mismatch for a useful message
            n = min(len(cal), len(ref_train_cal))
            where = next((i for i in range(n) if cal[i] != ref_train_cal[i]), n)
            fail(f"{t}: truncated train calendar != {ref_ticker}'s "
                 f"(len {len(cal)} vs {len(ref_train_cal)}; first diff at row {where})")
        with open(outdir / f"{t}_train.csv", "w", newline="") as f:
            f.write(h)
            f.writelines(kept)

    print(f"train aligned: {len(ref_train_cal)} rows/stock, "
          f"{ref_train_cal[0]} .. {ref_train_cal[-1]}  (identical across all {len(tickers)} stocks)")

    # verify + copy val and test
    for split in ("val", "test"):
        ref_cal = None
        for t in tickers:
            lines = read_lines(indir / f"{t}_{split}.csv")
            h, rows = header_and_rows(lines)
            if h != header0:
                fail(f"{t}: {split} header differs from train header")
            cal = calendar(rows)
            if ref_cal is None:
                ref_cal = cal
                ref_t = t
            elif cal != ref_cal:
                n = min(len(cal), len(ref_cal))
                where = next((i for i in range(n) if cal[i] != ref_cal[i]), n)
                fail(f"{t}: {split} calendar != {ref_t}'s "
                     f"(len {len(cal)} vs {len(ref_cal)}; first diff at row {where}). "
                     f"{split} must be aligned for cross-sectional IC.")
            with open(outdir / f"{t}_{split}.csv", "w", newline="") as f:
                f.write(h)
                f.writelines(rows)
        print(f"{split} aligned: {len(ref_cal)} rows/stock, "
              f"{ref_cal[0]} .. {ref_cal[-1]}  (identical across all {len(tickers)} stocks)")

    print(f"\nwrote aligned cohort -> {outdir}")


if __name__ == "__main__":
    main()
