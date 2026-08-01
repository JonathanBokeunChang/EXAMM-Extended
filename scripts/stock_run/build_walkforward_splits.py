#!/usr/bin/env python3
"""Build calendar-based walk-forward splits from either of two source shapes.

SPLIT3 mode (default): reconstructs each stock's full chronological series by
concatenating its existing train/val/test files, verifies the calendar is strictly
increasing (no duplicated or out-of-order dates -- 9 of the 50 stocks in the original
701515_split dataset fail this and must not be used), then writes calendar splits.

CONTINUOUS mode (--continuous): reads a single already-continuous {ticker}.csv per stock
(no pre-existing train/val/test files to concatenate -- e.g. the mid_highmid_20yr_portfolios
dataset, already cleaned by clean_mid_highmid.py). Everything downstream of "get (header,
rows) for one ticker" -- dedup guard, strictly-increasing check, cutoff/val/test
partitioning, the sanity checks, write_rows -- is identical between the two modes; only the
loader differs.

Both modes write:
    train = everything <= --cutoff
    val   = --val-year
    test  = --test-year            (or --test-year .. --test-end-year inclusive)

This mirrors the professor's paper (arXiv 2410.17212) split style: train to
year N-1, validate on year N, trade year N+1.

--dedup-dates enables the 9 dup-calendar stocks (SPLIT3 mode only, so far): their
duplicated date rows are ADJACENT and byte-identical in every column except VOL_CHANGE,
where the second copy is a 0.0 artifact of the duplication itself (verified across all 31
dup pairs, 2026-07-18). Keep-first is therefore exact; anything violating that pattern
fails loudly rather than being silently dropped.

Pure stdlib (no pandas -- cluster login nodes don't have it), and data rows
are passed through byte-faithfully rather than re-parsed.

Usage (pilot, SPLIT3):
    python3 scripts/stock_run/build_walkforward_splits.py \
        --cutoff 2021-12-31 --val-year 2022 --test-year 2023 \
        --out datasets/walkforward/pilot_2023 \
        --tickers AKAM ATO CAG COO DECK HBAN JBHT NTRS PKG TRV

Usage (CONTINUOUS, tickers auto-derived from every *.csv in --source-dir):
    python3 scripts/stock_run/build_walkforward_splits.py --continuous \
        --source-dir datasets/mid_highmid_20yr_portfolios_clean/set1 \
        --cutoff 2020-12-31 --val-year 2021 --test-year 2022 \
        --out datasets/walkforward/mid_highmid/set1/cohort_2020
"""
import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("SOURCE", REPO / "datasets/701515_split"))


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            fail(f"{path}: empty file")
        return header, [r for r in reader if r]


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_split3(ticker, source):
    """Concatenate {ticker}_train/val/test.csv into one chronological (header, rows)."""
    header = None
    rows = []
    for s in ("train", "val", "test"):
        f = source / f"{ticker}_{s}.csv"
        if not f.is_file():
            fail(f"{ticker}: missing source file {f}")
        h, r = read_rows(f)
        if header is None:
            header = h
        elif h != header:
            fail(f"{ticker}: column mismatch between splits ({s})")
        rows.extend(r)
    return header, rows


def load_continuous(ticker, source):
    """Read an already-continuous {ticker}.csv directly -- nothing to concatenate."""
    f = source / f"{ticker}.csv"
    if not f.is_file():
        fail(f"{ticker}: missing source file {f}")
    return read_rows(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, metavar="YYYY-MM-DD",
                    help="last training date (inclusive)")
    ap.add_argument("--val-year", required=True)
    ap.add_argument("--test-year", required=True)
    ap.add_argument("--test-end-year", default=None,
                    help="optional: extend test split through this year "
                         "(inclusive); absent = single test year (byte-identical "
                         "to previous behavior)")
    ap.add_argument("--dedup-dates", action="store_true",
                    help="keep-first dedup of adjacent duplicated date rows "
                         "(required for the 9 dup-calendar stocks); guarded: "
                         "dup rows must be identical except VOL_CHANGE")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="required in SPLIT3 mode; optional in --continuous mode "
                         "(auto-derived from every *.csv in --source-dir if omitted)")
    ap.add_argument("--min-train-rows", type=int, default=1000)
    ap.add_argument("--continuous", action="store_true",
                    help="source is one already-continuous {ticker}.csv per stock "
                         "(e.g. mid_highmid_20yr_portfolios_clean), not pre-split "
                         "{ticker}_train/val/test.csv files to concatenate")
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="overrides SOURCE env var / the 701515_split default; "
                         "required in practice for --continuous, since that dataset "
                         "isn't the default SOURCE")
    args = ap.parse_args()

    test_end_year = args.test_end_year or args.test_year
    if not (args.cutoff < f"{args.val_year}-01-01" < f"{args.test_year}-01-01"):
        fail("expected cutoff < val-year < test-year (chronological, no overlap)")
    if int(test_end_year) < int(args.test_year):
        fail("--test-end-year must be >= --test-year")

    source = args.source_dir if args.source_dir is not None else SOURCE
    source = source if source.is_absolute() else REPO / source
    if not source.is_dir():
        fail(f"source dir not found: {source}")

    tickers = args.tickers
    if tickers is None:
        if not args.continuous:
            fail("--tickers is required unless --continuous (SPLIT3 mode cannot "
                 "auto-derive tickers from pre-split filenames unambiguously)")
        tickers = sorted(p.stem for p in source.glob("*.csv"))
        if not tickers:
            fail(f"--continuous: no *.csv files found in {source}")

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'ticker':7s} {'train':>6s} {'val':>5s} {'test':>5s}   train range")
    for t in tickers:
        if args.continuous:
            header, rows = load_continuous(t, source)
        else:
            header, rows = load_split3(t, source)
        if header[0] != "date":
            fail(f"{t}: first column is '{header[0]}', expected 'date'")

        n_dups = 0
        if args.dedup_dates:
            try:
                vol_idx = header.index("VOL_CHANGE")
            except ValueError:
                fail(f"{t}: no VOL_CHANGE column -- dedup guard cannot apply")
            deduped = []
            for r in rows:
                if deduped and r[0] == deduped[-1][0]:
                    # guarded keep-first: the dup must be ADJACENT (checked by
                    # construction here) and identical except VOL_CHANGE
                    prev = deduped[-1]
                    diff_cols = [header[k] for k in range(len(header))
                                 if k != vol_idx and prev[k] != r[k]]
                    if diff_cols:
                        fail(f"{t}: duplicated date {r[0]} differs in "
                             f"{diff_cols} -- not the known VOL_CHANGE-only "
                             f"pattern; refusing to dedup")
                    n_dups += 1
                    continue
                deduped.append(r)
            rows = deduped

        d = [r[0] for r in rows]
        bad = [i for i in range(len(d) - 1) if not (d[i] < d[i + 1])]
        if bad:
            i = bad[0]
            fail(f"{t}: calendar not strictly increasing at row {i} "
                 f"({d[i]} -> {d[i+1]}) -- duplicated/out-of-order dates; "
                 f"use --dedup-dates or pick a different ticker")

        test_years = [str(y) for y in range(int(args.test_year),
                                            int(test_end_year) + 1)]
        train = [r for r in rows if r[0] <= args.cutoff]
        val = [r for r in rows if r[0].startswith(f"{args.val_year}-")]
        test = [r for r in rows if r[0][:4] in test_years]

        if len(train) < args.min_train_rows:
            fail(f"{t}: only {len(train)} training rows (min {args.min_train_rows})")
        if not (240 <= len(val) <= 260):
            fail(f"{t}: val has {len(val)} rows -- expected a full "
                 f"trading year (~250); wrong year or missing data?")
        for y in test_years:
            n_y = sum(1 for r in test if r[0].startswith(f"{y}-"))
            if not (240 <= n_y <= 260):
                fail(f"{t}: test year {y} has {n_y} rows -- expected a full "
                     f"trading year (~250); wrong year or missing data?")
        if len(train) + len(val) + len(test) > len(rows):
            fail(f"{t}: splits overlap -- internal error")

        write_rows(out / f"{t}_train.csv", header, train)
        write_rows(out / f"{t}_val.csv", header, val)
        write_rows(out / f"{t}_test.csv", header, test)
        dup_note = f"   (deduped {n_dups})" if n_dups else ""
        print(f"{t:7s} {len(train):6d} {len(val):5d} {len(test):5d}   "
              f"{train[0][0]} .. {train[-1][0]}{dup_note}")

    print(f"\nwrote {len(tickers)} stocks x 3 splits -> {out}")
    print("unused rows between cutoff and val-year (if any) are dropped by design")


if __name__ == "__main__":
    main()
