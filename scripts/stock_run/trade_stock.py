#!/usr/bin/env python3
"""Trade one stock's EXAMM predictions through Financial_toolbox strategies.

Usage:
    python3 scripts/stock_run/trade_stock.py AKAM
    python3 scripts/stock_run/trade_stock.py AKAM --debug      # watch every trade
    python3 scripts/stock_run/trade_stock.py AKAM --oracle     # perfect-model sanity check

Requires evaluate_stock.sh <TICKER> to have been run first (needs the
denormalized predictions CSV).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# --- locate the toolbox (respects her env-var convention, sane default) -----
TOOLBOX = Path(os.getenv("FIN_TOOLBOX_DIR", Path.home() / "Financial_toolbox"))
if not TOOLBOX.is_dir():
    sys.exit(f"Financial_toolbox not found at {TOOLBOX} (set FIN_TOOLBOX_DIR)")
sys.path.insert(0, str(TOOLBOX))
from stock import Stock          # noqa: E402  (import after path setup)
from Logger import Logger        # noqa: E402

REPO = Path(__file__).resolve().parents[2]   # EXAMM-Extended root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--debug", action="store_true", help="print every buy/sell")
    ap.add_argument("--oracle", action="store_true",
                    help="trade on expected_RET instead of predictions (upper bound)")
    args = ap.parse_args()
    s = args.ticker

    pred_file = REPO / f"test_output/stock_{s}_eval/{s}_test_predictions.csv"
    test_file = REPO / f"datasets/701515_split/{s}_test.csv"
    for f, fix in [(pred_file, f"run: sh scripts/stock_run/evaluate_stock.sh {s}"),
                   (test_file, "check the ticker name")]:
        if not f.is_file():
            sys.exit(f"missing {f}\n  -> {fix}")

    stock = Stock(s)
    stock.set_logger(Logger("DEBUG" if args.debug else "NONE"))
    stock.read_return_prediction(str(pred_file))
    stock.read_stock_price(str(test_file))

    # alignment guard: day i's prediction trades at day i's price; price has 1 extra row
    n_pred, n_price = len(stock.return_prediction), len(stock.stock_price)
    if n_price != n_pred + 1:
        sys.exit(f"alignment problem: {n_pred} predictions vs {n_price} prices "
                 f"(expected prices = predictions + 1)")

    if args.oracle:  # swap in ground truth WITHOUT touching her code
        stock.return_prediction = pd.read_csv(
            pred_file, usecols=["expected_RET"]).to_numpy().flatten()

    final = stock.simple_return(args.capital)
    strat = (final - args.capital) / args.capital * 100

    p = stock.stock_price                       # buy-and-hold, her do_nothing convention
    hold = (p[-2] - p[0]) / p[0] * 100

    label = "oracle (expected_RET)" if args.oracle else "model (predicted_RET)"
    print(f"{s}: {n_pred} trading days, ${args.capital:.0f} initial capital")
    print(f"  {label:26s} simple_return: {strat:+8.2f}%   (final ${final:,.2f})")
    print(f"  {'buy-and-hold baseline':26s}              : {hold:+8.2f}%")


if __name__ == "__main__":
    main()