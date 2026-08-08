#!/usr/bin/env python3
"""Build the COMBINED (wide) panel: 50 stocks x 6 predictors in one 300-input, 50-output series.

    # production: the two ablation cohorts
    python3 scripts/stock_run/build_combined_panel.py \
        --src datasets/walkforward/cohort_2021_aligned \
        --out datasets/walkforward/cohort_2021_aligned_combined

    # verification: reproduce the orphan reference files byte-for-byte
    python3 scripts/stock_run/build_combined_panel.py \
        --src datasets/701515_split --out /tmp/ref_rebuild \
        --ticker-order datasets/701515_split/combined_predictors_train.csv

WHY THIS EXISTS. datasets/701515_split/combined_predictors_{train,val,test}.csv were the only wide
panel in the repo and NOTHING built them -- no script anywhere produces that shape, so they were
unreproducible orphans. Worse, their calendar is a 70/15/15 proportional split whose TEST window
(2021-08-05 ->) opens inside the ablation cohorts' TRAINING span, so they cannot fill
tab:pooling-ablation, whose entire purpose is identical windows across the three constructions.

THE INNER JOIN IS THE POINT, NOT A BUG. Combined can only use days on which ALL 50 stocks trade,
because one row must carry every stock's predictors at once. Pooled has no such constraint -- each
stock contributes its own windows independently -- which is exactly why tab:constructions reports
~4k training examples for combined against 164,450 for pooled. The row loss is reported rather than
silently absorbed, because it IS the mechanism behind that difference.

COLUMN ORDER IS REPRODUCED, NOT INVENTED. The reference file's ticker order is arbitrary (NDSN,
HOLX, ATO, ...) and its per-ticker field order is BA_SPREAD, ILLIQUIDITY, TURNOVER, VOL_CHANGE, RET,
sprtrn -- neither is alphabetical. EXAMM addresses columns by name via --input_parameter_names, so
order cannot change a result; but --ticker-order lets us rebuild the reference exactly and prove the
builder correct, while new panels default to alphabetical so they are self-documenting.

sprtrn IS DUPLICATED 50 TIMES AND THAT IS DELIBERATE. Every <TICKER>_sprtrn column carries the same
S&P 500 return, so 49 of the 300 inputs are exact duplicates. This reproduces the construction
tab:constructions publishes as "300 inputs"; deduplicating would be a different experiment and would
make the table's own figure wrong.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

# Order taken from the reference panel's header, not alphabetised -- see the docstring.
FIELDS = ["BA_SPREAD", "ILLIQUIDITY", "TURNOVER", "VOL_CHANGE", "RET", "sprtrn"]
SPLITS = ["train", "val", "test"]


def tickers_in(src):
    t = sorted({os.path.basename(f).rsplit("_", 1)[0]
                for f in glob.glob(os.path.join(src, "*_train.csv"))
                if "combined" not in os.path.basename(f)})
    return t


def order_from(reference_csv):
    """Ticker order as it appears in an existing wide panel, via its _RET columns."""
    hdr = pd.read_csv(reference_csv, nrows=0).columns
    seen, out = set(), []
    for c in hdr:
        if c.endswith("_RET"):
            t = c[: -len("_RET")]
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def build_split(src, split, order, dedupe=False):
    frames = []
    per_rows = {}
    dupes_found = {}
    for tic in order:
        f = os.path.join(src, f"{tic}_{split}.csv")
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
        df = pd.read_csv(f)
        missing = [c for c in FIELDS if c not in df.columns]
        if missing:
            sys.exit(f"{f}: missing columns {missing}")
        per_rows[tic] = len(df)
        d = df[["date"] + FIELDS].copy()
        d.columns = ["date"] + [f"{tic}_{c}" for c in FIELDS]
        # DUPLICATE DATES ARE A HARD FAILURE BY DEFAULT. A repeated date makes the join a partial
        # cross-product -- pandas raises InvalidIndexError rather than silently widening, which is
        # the correct behaviour and must not be papered over. datasets/701515_split carries them in
        # 10 files (STLD 8, WRB 5, BXP 4); the aligned cohorts this study actually uses have none.
        # --dedupe exists only to rebuild that legacy panel for comparison, and says so loudly.
        dup = int(d["date"].duplicated().sum())
        if dup:
            if not dedupe:
                sys.exit(f"{f}: {dup} duplicate date(s). The aligned cohorts have none -- if this "
                         f"is the legacy 701515_split, pass --dedupe; otherwise fix the source.")
            dupes_found[tic] = dup
            d = d.drop_duplicates(subset="date", keep="first")
        frames.append(d.set_index("date"))

    # INNER join: a row survives only if every stock traded that day.
    wide = pd.concat(frames, axis=1, join="inner").reset_index()
    return wide, per_rows, dupes_found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of per-ticker {TICKER}_{split}.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticker-order", default=None,
                    help="existing wide CSV whose ticker order to reproduce (default: alphabetical)")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--dedupe", action="store_true",
                    help="drop duplicate dates (legacy 701515_split only; keeps first)")
    a = ap.parse_args()

    order = order_from(a.ticker_order) if a.ticker_order else tickers_in(a.src)
    have = set(tickers_in(a.src))
    if set(order) != have:
        sys.exit(f"ticker mismatch: order has {len(order)}, src has {len(have)}; "
                 f"only in order={sorted(set(order)-have)[:5]} only in src={sorted(have-set(order))[:5]}")
    if len(order) != 50:
        sys.exit(f"expected 50 tickers, got {len(order)}")

    os.makedirs(a.out, exist_ok=True)
    for split in a.splits:
        wide, per_rows, dupes = build_split(a.src, split, order, a.dedupe)
        n_in = sum(1 for c in wide.columns if c != "date")
        n_out = sum(1 for c in wide.columns if c.endswith("_RET"))
        if len(wide.columns) != 301 or n_in != 300 or n_out != 50:
            sys.exit(f"{split}: expected 301 cols / 300 inputs / 50 outputs, "
                     f"got {len(wide.columns)} / {n_in} / {n_out}")
        if wide.isna().any().any():
            bad = wide.columns[wide.isna().any()].tolist()[:5]
            sys.exit(f"{split}: NaNs present in {bad}")

        dst = os.path.join(a.out, f"combined_predictors_{split}.csv")
        wide.to_csv(dst, index=False)
        # Row loss is the construction's defining constraint -- report it, do not bury it.
        if dupes:
            print(f"  {split:5s} WARNING dropped duplicate dates: "
                  f"{', '.join(f'{k}x{v}' for k, v in sorted(dupes.items()))}")
        shortest = min(per_rows.values())
        longest = max(per_rows.values())
        print(f"  {split:5s} {len(wide):5d} rows x {len(wide.columns)} cols  "
              f"(per-stock {shortest}-{longest}; inner join dropped {longest - len(wide)})  "
              f"{wide['date'].iloc[0]} -> {wide['date'].iloc[-1]}  -> {dst}")


if __name__ == "__main__":
    main()
