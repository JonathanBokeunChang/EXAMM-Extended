#!/usr/bin/env python3
"""Add a cross-sectional z-scored TARGET column (Qlib's CSZScoreNorm) to an ALIGNED
walk-forward cohort, so `--loss mse` trains on the cross-section instead of the raw
next-day return.

For each split (train/val/test) and each date j, the new column RET_CS is the
per-date cross-sectional z-score of RET across the pooled stocks:

    RET_CS[i][j] = (RET[i][j] - mean_i RET[.][j]) / std_i RET[.][j]

This is a *contemporaneous* cross-sectional transform (only date-j returns of the
50 stocks) -- NO temporal look-ahead, so it is leak-free across the train/val/test
boundary (exactly Qlib's per-date CSZScoreNorm). MSE against a unit-variance target
is monotonically equivalent to maximizing the cross-sectional (Pearson) correlation,
i.e. it is an IC objective without the scale-invariance/collapse pathology of the
pure IC loss.

RET stays an untouched input feature; RET_CS is added as a NEW last column so the 6
raw inputs are byte-identical to the source cohort (a controlled objective comparison:
raw-MSE / z-score-MSE / IC arms all share the same inputs and search). Train with
`--output_parameter_names RET_CS`.

Requires an ALIGNED cohort (row index j = the same date for every stock; build it with
scripts/stock_run/align_cohort.py). std uses the sample (n-1) denominator to match
pandas/Qlib; a date with std < 1e-12 is guarded to RET_CS = 0. All original columns
pass through byte-for-byte; the value is written at full double precision.

Usage:
    python3 scripts/stock_run/build_cs_target.py \
        --in datasets/walkforward/cohort_2021_aligned \
        --out datasets/walkforward/cohort_2021_zscore
"""
import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
SOURCE_COL = "RET"
TARGET_COL = "RET_CS"
STD_FLOOR = 1e-12


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def split_terminator(line):
    """Return (content, line_terminator) preserving the exact terminator bytes."""
    for t in ("\r\n", "\n", "\r"):
        if line.endswith(t):
            return line[: -len(t)], t
    return line, ""


def read_raw(path):
    with open(path, "r", newline="") as f:
        return f.read().splitlines(keepends=True)


def parse_row(content):
    # csv.reader is robust to quoted commas in fields (e.g. company names); we still
    # write RET_CS by appending to the raw source line, so pass-through stays byte-exact.
    return next(csv.reader([content]))


def tickers_in(indir, split):
    return sorted(p.name[: -len(f"_{split}.csv")] for p in indir.glob(f"*_{split}.csv"))


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
    for split in SPLITS:
        have = set(tickers_in(indir, split))
        missing = [t for t in tickers if t not in have]
        if missing:
            fail(f"tickers missing {split} files: {missing}")
    print(f"{len(tickers)} tickers in {indir.name}")

    outdir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        # ---- load every stock's file (header identical; data row-index aligned) ----
        header_line = None
        ret_idx = date_idx = None
        rows = {}  # ticker -> list of (content, terminator) for data rows
        for t in tickers:
            lines = read_raw(indir / f"{t}_{split}.csv")
            if len(lines) < 2:
                fail(f"{t}: {split} file has no data rows")
            if header_line is None:
                header_line = lines[0]
                hcontent, _ = split_terminator(header_line)
                header_fields = parse_row(hcontent)
                if SOURCE_COL not in header_fields:
                    fail(f"{SOURCE_COL} column absent from {t}_{split}.csv header")
                if TARGET_COL in header_fields:
                    fail(f"{TARGET_COL} already present in {t}_{split}.csv header")
                ret_idx = header_fields.index(SOURCE_COL)
                date_idx = header_fields.index("date") if "date" in header_fields else 0
            elif lines[0] != header_line:
                fail(f"{t}: {split} header differs from {tickers[0]}")
            rows[t] = [split_terminator(l) for l in lines[1:]]

        n_rows = len(rows[tickers[0]])
        for t in tickers:
            if len(rows[t]) != n_rows:
                fail(f"{t}: {split} has {len(rows[t])} rows != {n_rows} ({tickers[0]}) -- not aligned "
                     f"(rebuild with align_cohort.py)")

        parsed = {t: [parse_row(c) for (c, _term) in rows[t]] for t in tickers}

        # ---- verify calendar alignment: row j is the same date for every stock ----
        ref_dates = [parsed[tickers[0]][j][date_idx] for j in range(n_rows)]
        for t in tickers:
            for j in range(n_rows):
                if parsed[t][j][date_idx] != ref_dates[j]:
                    fail(f"{t}: {split} date at row {j} = {parsed[t][j][date_idx]} != "
                         f"{ref_dates[j]} ({tickers[0]}) -- not aligned; rebuild with align_cohort.py")

        # ---- per-date cross-sectional z-score ----
        z = {t: [0.0] * n_rows for t in tickers}
        n_guarded = 0
        worst_abs_mean = 0.0
        worst_std_err = 0.0
        for j in range(n_rows):
            vals = []
            for t in tickers:
                s = parsed[t][j][ret_idx]
                try:
                    v = float(s)
                except ValueError:
                    fail(f"{t}: {split} row {j} ({ref_dates[j]}) has non-numeric {SOURCE_COL}='{s}'")
                if not math.isfinite(v):
                    fail(f"{t}: {split} row {j} ({ref_dates[j]}) has non-finite {SOURCE_COL}={v}")
                vals.append(v)
            n = len(vals)
            mean = sum(vals) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
            if std < STD_FLOOR:
                n_guarded += 1
                continue  # all-zero z for this degenerate date
            for i, t in enumerate(tickers):
                z[t][j] = (vals[i] - mean) / std
            # accumulate self-check stats (mean~0, sample std~1 by construction)
            zs = [z[t][j] for t in tickers]
            zm = sum(zs) / n
            zstd = math.sqrt(sum((zz - zm) ** 2 for zz in zs) / (n - 1))
            worst_abs_mean = max(worst_abs_mean, abs(zm))
            worst_std_err = max(worst_std_err, abs(zstd - 1.0))

        # ---- write: append ,RET_CS to header + each row (original bytes preserved) ----
        hcontent, hterm = split_terminator(header_line)
        for t in tickers:
            with open(outdir / f"{t}_{split}.csv", "w", newline="") as f:
                f.write(hcontent + "," + TARGET_COL + hterm)
                for j, (content, term) in enumerate(rows[t]):
                    f.write(content + "," + f"{z[t][j]:.17g}" + term)

        # ---- verify: stripping the appended column reproduces the source byte-for-byte ----
        for t in tickers:
            src = read_raw(indir / f"{t}_{split}.csv")
            out = read_raw(outdir / f"{t}_{split}.csv")
            if len(src) != len(out):
                fail(f"{t}: {split} line count changed ({len(src)} -> {len(out)})")
            for k in range(len(src)):
                oc, ot = split_terminator(out[k])
                if oc.rsplit(",", 1)[0] + ot != src[k]:
                    fail(f"{t}: {split} row {k} original columns not byte-identical to source")

        if worst_abs_mean > 1e-9 or worst_std_err > 1e-9:
            fail(f"{split}: z-score self-check failed "
                 f"(max|mean|={worst_abs_mean:.2e}, max|std-1|={worst_std_err:.2e})")

        print(f"{split}: {n_rows} rows/stock x {len(tickers)} stocks | "
              f"per-date RET_CS max|mean|={worst_abs_mean:.1e} max|std-1|={worst_std_err:.1e} "
              f"guarded(std~0)={n_guarded} | inputs byte-identical to source")

    print(f"\nwrote z-scored cohort -> {outdir}")


if __name__ == "__main__":
    main()
