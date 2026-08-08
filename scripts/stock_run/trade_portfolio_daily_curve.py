#!/usr/bin/env python3
"""Daily equity curve for Sharpe/MDD -- instruments the REAL toolbox strategy method,
does not reimplement it.

WHY THIS EXISTS
---------------
trade_portfolio.py reports only the final cumulative return; Sharpe and max drawdown need the
full daily equity path. Reimplementing the strategy separately would reintroduce exactly the
"is this really the same algorithm" risk already raised and checked once this session. Instead:
this pulls the ACTUAL Portfolio.daily_hybrid_long_short_return source via inspect.getsource,
inserts ONE line at the end of the day loop that records CurrentCash + sum(share*price[t]) (the
same formula Stock.get_current_liquid uses), and exec()s that patched source as a method on the
SAME Portfolio class. The logic is byte-identical to the real method except for that one
recording line -- there is no separate reimplementation to diverge from the real one.

VALIDATION: after building the curve, this reruns the REAL (unpatched) strategy on a fresh,
identically-built Portfolio and asserts the final cumulative return matches the instrumented
run's last equity point to high precision. If they don't match, something about the patch
diverged from the real method and this script refuses to report anything.

Data loading (load_stock / align_suffix / verify_market_agreement / apply_window / Portfolio
construction) is imported directly from trade_portfolio.py, not reimplemented, for the same
reason -- these are the identical functions used everywhere else in this project.

Usage:
    python3 scripts/stock_run/trade_portfolio_daily_curve.py \
        --pred-dir test_output/mse_cohort_2020_aligned/ensemble_test \
        --data-dir datasets/walkforward/cohort_2020_aligned \
        --long 10 --short 10 --window-start 2022-01-03 --window-end 2022-12-30 --expect-n 50
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import textwrap
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "stock_run"))
import trade_portfolio as tp  # noqa: E402

TOOLBOX = os.environ.get("FIN_TOOLBOX_DIR", os.path.expanduser("~/Financial_toolbox"))
sys.path.insert(0, TOOLBOX)
from portfolio import Portfolio  # noqa: E402
from Logger import Logger  # noqa: E402


def build_instrumented_method():
    """Patch Portfolio.daily_hybrid_long_short_return's source with one equity-recording line,
    exec it, return the new function object. Fails loudly if the source doesn't match the exact
    shape expected (so an upstream toolbox change can't silently go unrecorded).

    Ground truth for the anchor line/indentation was taken by printing the dedented source with
    line numbers and reading it directly -- not from memory of source seen earlier in the
    session. NOTE: textwrap.dedent does NOT work here -- the toolbox source has two decorative
    comment banners ("# --- find company ---") sitting at column 0 inside the method body, which
    makes dedent's common-leading-whitespace computation come out to 0 (no-op). Strip exactly
    the 4-space class-body indent from lines that have it instead; column-0 comment lines are
    already valid at column 0 and are left alone."""
    raw = inspect.getsource(Portfolio.daily_hybrid_long_short_return)
    src = "\n".join(line[4:] if line.startswith("    ") else line for line in raw.splitlines()) + "\n"
    anchor = "            shorted_stock_yesterday = shorted_stock_today\n"
    if src.count(anchor) != 1:
        sys.exit("ERROR: expected anchor line not found exactly once in "
                 "daily_hybrid_long_short_return -- toolbox source changed, re-derive the patch")
    # Inserted at 8-space indent (same level as `if do_trade:` itself, i.e. still inside the
    # `for time in range(...)` loop but OUTSIDE the if-block) so it fires exactly once per day
    # regardless of whether that day traded or held -- end-of-day mark-to-market either way.
    record_line = (
        "        self._equity_curve.append(self.current_cash_amount + "
        "sum(c.share * c.stock_price[time] for c in self.portfolio_list))\n"
    )
    patched = src.replace(anchor, anchor + record_line)
    if patched == src:
        sys.exit("ERROR: patch insertion did not change the source")

    ns = {}
    exec(compile(patched, "<patched daily_hybrid_long_short_return>", "exec"),
         Portfolio.__dict__.copy() | vars(sys.modules[Portfolio.__module__]), ns)
    return ns["daily_hybrid_long_short_return"]


def build_portfolio(a):
    tickers = tp.discover_tickers(a.pred_dir, None, None)
    if a.expect_n and len(tickers) != a.expect_n:
        sys.exit(f"ERROR: expected {a.expect_n} tickers, found {len(tickers)}")
    records = [tp.load_stock(t, a.pred_dir, a.data_dir, False) for t in tickers]
    tp.align_suffix(records)
    tp.verify_market_agreement(records)
    if a.window_start or a.window_end:
        tp.apply_window(records, a.window_start, a.window_end)
    logger = Logger("NONE")
    portfolio = Portfolio(tickers)
    portfolio.set_initial_spend_per_stock(a.capital / len(tickers))
    portfolio.set_initial_capital(a.capital)
    portfolio.set_logger(logger)
    for r in records:
        r["stock"].set_logger(logger)
        portfolio.add_company_to_protfolio(r["stock"])
    portfolio.set_use_TC(a.use_tc)
    portfolio.set_trade_with_bid_ask(False)
    # trade_portfolio.py's run_strategy() sets this before calling the hybrid method directly
    # (Portfolio.trade() never dispatches to it) -- daily_hybrid_long_short_return's own
    # self.reset() reads it. Replicate that here or reset() throws.
    portfolio.strategy = "daily_hybrid_long_short_return"
    return portfolio, records[0]["dates"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--long", type=int, default=10, dest="long_n")
    ap.add_argument("--short", type=int, default=10, dest="short_n")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--use-tc", action="store_true")
    ap.add_argument("--window-start", default=None)
    ap.add_argument("--window-end", default=None)
    ap.add_argument("--expect-n", type=int, default=None)
    a = ap.parse_args()
    from pathlib import Path
    a.pred_dir, a.data_dir = Path(a.pred_dir), Path(a.data_dir)

    patched_method = build_instrumented_method()

    # ---- instrumented run
    portfolio, dates = build_portfolio(a)
    portfolio._equity_curve = [a.capital]
    bound = types.MethodType(patched_method, portfolio)
    bound(a.long_n, a.short_n)
    curve = np.array(portfolio._equity_curve)
    instrumented_final_return = (curve[-1] - a.capital) / a.capital * 100

    # ---- validation: fresh portfolio, REAL unpatched method, must match
    portfolio2, _ = build_portfolio(a)
    real_return = portfolio2.daily_hybrid_long_short_return(a.long_n, a.short_n)
    diff = abs(instrumented_final_return - real_return)
    print(f"validation: instrumented={instrumented_final_return:+.4f}%  "
          f"real={real_return:+.4f}%  diff={diff:.6f}pp")
    if diff > 1e-6:
        sys.exit("ERROR: instrumented curve's final value does NOT match the real, unpatched "
                 "strategy's reported return -- the patch diverged, do not trust this curve")

    # ---- Sharpe (annualized, on daily simple returns of the equity curve) and max drawdown
    daily_ret = curve[1:] / curve[:-1] - 1.0
    sharpe = daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252) if daily_ret.std(ddof=1) > 0 else float("nan")
    running_max = np.maximum.accumulate(curve)
    drawdown = (curve - running_max) / running_max
    mdd = drawdown.min() * 100

    print(f"days: {len(curve) - 1}   final return: {instrumented_final_return:+.2f}%")
    print(f"Sharpe (annualized, rf=0): {sharpe:+.3f}")
    print(f"Max drawdown: {mdd:.2f}%")


if __name__ == "__main__":
    main()
