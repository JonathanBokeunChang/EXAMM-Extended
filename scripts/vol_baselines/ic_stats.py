#!/usr/bin/env python3
"""Shared cross-sectional IC statistics -- the ONLY place standard errors are computed.

Why this module exists: our label is MASTER's `Ref($close,-5)/Ref($close,-1)-1`, which spans 4
trading days. Consecutive daily observations therefore share 3 of their 4 constituent daily
returns, so the per-date IC series is strongly autocorrelated. Measured on this dataset:

    rho_1 = +0.740   rho_2 = +0.467   rho_3 = +0.198   rho_4 = -0.044   (mean over 40 stocks)

The naive SE = sd/sqrt(T) assumes independent daily observations and is therefore WRONG here --
it understates the true SE by a factor of ~1.95, i.e. it roughly DOUBLES every t-statistic. Every
number previously reported on this dataset (ridge t=13.50, LSTM t=9.20, ...) is inflated by about
that factor.

The fix is a Newey-West HAC standard error with a Bartlett kernel. For the mean of a series:

    Var(xbar) = (1/T) * [ gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k ]

with gamma_k the lag-k autocovariance. L defaults to 4 (= label horizon - 1); the Bartlett weights
downweight the near-zero lag-4 term automatically, so the result is not sensitive to L=3 vs L=4.

Every evaluation script must import from here rather than computing sd/sqrt(n) locally -- that is
the whole point. Self-test: `python3 scripts/vol_baselines/ic_stats.py`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

DEFAULT_HAC_LAG = 4  # label spans 4 trading days -> overlap induces MA(3); 4 is the safe choice


def daily_ic(df, pred_col="pred", label_col="exp", date_col="date", min_names=5):
    """Per-date cross-sectional Spearman (rank) IC.

    Dates with fewer than `min_names` stocks, or with a degenerate (zero-variance) prediction or
    label, are skipped -- a rank correlation is undefined there and including it as 0 would bias
    the mean toward zero.
    """
    ics, dates = [], []
    for d, g in df.groupby(date_col):
        if len(g) < min_names:
            continue
        a = g[pred_col].to_numpy(float)
        b = g[label_col].to_numpy(float)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        v = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
        if np.isfinite(v):
            ics.append(v)
            dates.append(d)
    order = np.argsort(np.asarray(dates, dtype=object).astype(str))
    return np.asarray(ics)[order]  # chronological -- required for autocovariance to be meaningful


def hac_se(x, lag=DEFAULT_HAC_LAG):
    """Newey-West (Bartlett) HAC standard error of the MEAN of series `x`.

    `x` must be in chronological order. Returns the naive SE when the HAC estimate would be
    non-positive (possible with strongly negative autocovariance), which keeps the caller safe
    rather than emitting a NaN t-stat.
    """
    x = np.asarray(x, float)
    T = len(x)
    if T < 3:
        return float("nan")
    lag = int(min(lag, T - 1))
    xd = x - x.mean()
    var = np.dot(xd, xd) / T  # gamma_0
    for k in range(1, lag + 1):
        gk = np.dot(xd[k:], xd[:-k]) / T
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return float(x.std(ddof=1) / np.sqrt(T))
    return float(np.sqrt(var / T))


def summarize(ics, lag=DEFAULT_HAC_LAG):
    """All headline statistics for one model's daily-IC series."""
    ics = np.asarray(ics, float)
    T = len(ics)
    m = float(ics.mean())
    sd = float(ics.std(ddof=1))
    naive = sd / np.sqrt(T)
    h = hac_se(ics, lag)
    return {
        "ic": m,
        "n_dates": T,
        "sd": sd,
        "se_naive": naive,
        "se_hac": h,
        "t_naive": m / naive if naive > 0 else float("nan"),
        "t_hac": m / h if h > 0 else float("nan"),
        "inflation": naive and h / naive,  # how much the naive SE understated things
        "icir_ann": m / sd * np.sqrt(252) if sd > 0 else float("nan"),
        "hit": float((ics > 0).mean()),
        "ci95_lo": m - 1.96 * h,
        "ci95_hi": m + 1.96 * h,
    }


def report(name, ics, lag=DEFAULT_HAC_LAG, width=34):
    """Print one line with the HAC t as the headline and the naive t shown for transparency."""
    s = summarize(ics, lag)
    print(f"{name:<{width}} IC {s['ic']:+.4f}  "
          f"HAC-SE {s['se_hac']:.5f}  t_HAC {s['t_hac']:+6.2f}  "
          f"[95% {s['ci95_lo']:+.4f},{s['ci95_hi']:+.4f}]  "
          f"hit {s['hit']:5.1%}  n={s['n_dates']}"
          f"   (naive t {s['t_naive']:+.2f}, x{s['se_hac']/s['se_naive']:.2f})")
    return s


def paired(ics_a, ics_b, lag=DEFAULT_HAC_LAG):
    """HAC-corrected paired test on the DIFFERENCE of two models' daily IC series.

    Model-vs-model differences are far less noisy than either level, because both models see the
    same cross-section each day -- but the difference series inherits the same overlap-induced
    autocorrelation, so it needs the same HAC treatment.
    """
    a, b = np.asarray(ics_a, float), np.asarray(ics_b, float)
    if len(a) != len(b):
        raise ValueError(f"paired test needs aligned series, got {len(a)} vs {len(b)}")
    d = a - b
    s = summarize(d, lag)
    return {"delta": s["ic"], "se_hac": s["se_hac"], "t_hac": s["t_hac"],
            "win_rate": float((d > 0).mean()), "n": len(d)}


def _self_test():
    rng = np.random.default_rng(0)
    T = 4000

    # (1) white noise -> HAC should agree with naive (ratio ~1)
    x = rng.normal(size=T)
    r = hac_se(x) / (x.std(ddof=1) / np.sqrt(T))
    print(f"  white noise      : HAC/naive = {r:.3f}   (expect ~1.0)")
    assert 0.85 < r < 1.15, r

    # (2) MA(3) with equal weights -- the exact structure a 4-day overlapping label induces.
    #     True long-run variance ratio vs iid is (sum of weights)^2 / (sum of squared weights) = 4.
    #     So the SE ratio should approach sqrt(4) = 2 -- matching the ~1.95 we measured empirically.
    e = rng.normal(size=T + 3)
    y = (e[3:] + e[2:-1] + e[1:-2] + e[:-3]) / 4.0
    r2 = hac_se(y) / (y.std(ddof=1) / np.sqrt(T))
    print(f"  MA(3) overlap    : HAC/naive = {r2:.3f}   (expect ~2.0 -- the real-data case)")
    assert 1.6 < r2 < 2.4, r2

    # (3) chronological ordering matters: shuffling destroys the autocorrelation, so a shuffled
    #     series must look iid again. This guards against a caller passing unsorted dates.
    z = y.copy()
    rng.shuffle(z)
    r3 = hac_se(z) / (z.std(ddof=1) / np.sqrt(T))
    print(f"  shuffled MA(3)   : HAC/naive = {r3:.3f}   (expect ~1.0 -- ordering is load-bearing)")
    assert 0.85 < r3 < 1.15, r3

    print("  ic_stats self-test PASSED")


if __name__ == "__main__":
    _self_test()
