#!/usr/bin/env python3
"""EnbPI over a neuroevolution SEED POPULATION: are population-derived prediction
intervals calibrated, and does the seed spread carry any uncertainty signal?

    python3 .scratch/conformal/enbpi_population.py            # all cells
    python3 .scratch/conformal/enbpi_population.py --cells dsA:cohort_2021_aligned
    python3 .scratch/conformal/enbpi_population.py --window 120

NO TRAINING, NO RETRAINING, NO EVALUATION. Everything below is read off
test_output/<arm>/run_*/eval_test/<TICKER>_test_predictions.csv, which the sweeps
already wrote. The point of the exercise is precisely that the ten seeds are already
on disk and nine of them are normally thrown away.

------------------------------------------------------------------------------
WHAT ENbPI IS (Xu & Xie, ICML 2021, arXiv:2010.09107), AND WHAT WE CHANGED
------------------------------------------------------------------------------
Algorithm 1, verbatim in structure:

  train:   for b=1..B: bootstrap index set S_b from (1..T); f_b = A({(x_i,y_i): i in S_b})
           for i=1..T: f^phi_{-i}(x_i) = phi({f_b(x_i) : i not in S_b})
                       eps_i = y_i - f^phi_{-i}(x_i)                # LOO residual
  predict: f^phi_{-t}(x_t) = phi({f^phi_{-i}(x_t)})
           beta* = argmin_{beta in [0,alpha]} ( Q_{1-alpha+beta}(eps) - Q_beta(eps) )
           C_t = [ f_{-t}(x_t) + Q_{beta*}(eps),  f_{-t}(x_t) + Q_{1-alpha+beta*}(eps) ]
           every s steps: drop the s oldest residuals, append the s newest

Two structural facts about this repo change what "faithful" means:

1. THE SEEDS ARE NOT BOOTSTRAP REPLICATES. Every one of the ten EXAMM/EXA-GP runs
   trains on the *whole* training split; they differ only by RNG seed (and by the
   stochastic search path that seed induces). There is therefore no out-of-bag set
   and no f^phi_{-i} to build on the training data. EnbPI's LOO machinery exists
   solely to keep the calibration residuals honest -- to stop a model scoring its
   own training point. Here that problem is solved differently and more strongly:
   we calibrate on the TEST STREAM ITSELF, which no seed has seen at all. Every
   residual we ever put in the window is genuinely out-of-sample for all B members,
   which is a stronger guarantee than out-of-bag gives. phi = mean over all ten
   seeds, which is exactly the ensemble_test/ average the paper already uses.

2. THERE IS NO SEPARATE CALIBRATION SPLIT ON DISK. So the residual window is seeded
   by a BURN-IN of the first `--window` test days, and coverage is only ever scored
   on days after it. At target day t the window holds residuals from target days
   [t-W, t-1] and nothing later. Those are all known at the time the day-t forecast
   is made: the day-t forecast is emitted from day t-1's features, at which point
   y_{t-1} has printed. No lookahead. (Burn-in days are excluded from every method's
   score so all six are graded on the identical (stock, date) set.)

  s = 1 day (K tickers' residuals enter together), i.e. the window slides daily.

------------------------------------------------------------------------------
WHERE THIS DATA VIOLATES ENbPI'S ASSUMPTIONS
------------------------------------------------------------------------------
Assumption 1 (errors i.i.d. with Lipschitz CDF; Corollary 2 relaxes to stationary +
strongly mixing with summable mixing coefficients).
  VIOLATED, two ways. (a) Daily equity returns are volatility-clustered: the error
  scale is itself a persistent process, so the marginal error CDF is not constant
  over the window and mixing is slow relative to W. (b) CROSS-SECTIONAL DEPENDENCE.
  EnbPI is stated for one series. We have 50 stocks whose same-day errors are
  strongly correlated through the market factor, so a pooled window of W*50
  residuals has an effective sample size far below W*50 and the daily batch of 50
  new residuals is close to one draw, not fifty. That inflates the variance of the
  empirical quantile without inflating its nominal precision. This is why the
  per-ticker variant (`EnbPI-tkr`) is reported alongside the pooled one: it is the
  assumption-faithful single-series algorithm, at the cost of a 50x smaller window.

Assumption 2 (estimation quality: (1/T) sum (f_{-t}(x_t) - f(x_t))^2 <= delta_T^2,
delta_T -> 0).
  VACUOUSLY SATISFIED AND USELESS HERE. These models have rank IC near 0.01-0.03;
  the ensemble mean is nearly a constant close to zero. delta_T is not small in any
  meaningful sense -- but it is *bounded*, and because the predictor barely moves,
  the residual is essentially the raw return. Conformal coverage still works (it is
  distribution-free); the intervals just inherit the width of the unconditional
  return distribution. Coverage being fine here is therefore NOT evidence that the
  population told us anything.

Nonstationarity across the test span. cohort_2020's test split is 2022 AND 2023 --
a bear year followed by a rally. Coverage-by-quarter below is the check on that.

------------------------------------------------------------------------------
THE SIX METHODS (all scored on the identical (stock, date) set)
------------------------------------------------------------------------------
  EnbPI-tkr    EnbPI per ticker: signed residuals, own W-day window, beta-optimised
               asymmetric interval. The assumption-faithful single-series version.
  EnbPI-pool   Same but one residual pool shared by all K tickers (W*K residuals).
  Seed-Gauss   yhat +/- z_{1-alpha/2} * sd(seeds).  THE CLAIM UNDER TEST, uncalibrated:
               "the population spread IS the uncertainty, for free".
  Seed-Scaled  yhat +/- q * sd(seeds), q = trailing conformal quantile of
               |y-yhat|/sd(seeds). Calibrated by construction, so its *width* and its
               *conditional* coverage are the interesting part: it can only beat a
               fixed width if sd(seeds) RANKS uncertainty correctly.
  Trail-Fixed  yhat +/- Q_{1-alpha}(|y-yhat|) over the pooled trailing window.
               One width for every stock on a day. The honest do-nothing baseline.
  Vol-Scaled   Same as Seed-Scaled but with sigma = trailing 20-day realised vol of
               the stock instead of the seed spread. A conditioning variable anyone
               has for free WITHOUT a population. If this beats Seed-Scaled, the
               population adds nothing.

Quantiles are conformal-corrected order statistics (ceil((n+1)p)-th smallest), not
np.quantile, so a small per-ticker window is conservative rather than anti-conservative.

------------------------------------------------------------------------------
THE LOAD-BEARING NUMBER
------------------------------------------------------------------------------
Not coverage. Coverage is trivially achievable by widening. The question is whether
sd(seeds) is *proportional* to realised error:
  * spread ratio  = rms(sd_seeds) / rms(y - yhat).  If this is ~1 the population is
    a calibrated uncertainty estimate out of the box; if it is 0.03 it is not.
  * rank corr(sd_seeds, |y-yhat|), pooled / within-day / within-ticker. If this is
    ~0 the spread does not even ORDER the hard cases, and no rescaling can save it.
  Both are printed next to the same statistics for trailing realised vol, so the
  population is graded against a free alternative rather than against nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts/stock_run"))
from score_forecasting import data_dir, from_per_ticker  # noqa: E402

ALPHAS = [0.20, 0.10]                     # nominal 80% and 90%
BETA_GRID = 21                            # points in [0, alpha] for EnbPI's beta search
VOL_LOOKBACK = 20                         # trading days for realised vol
EPS = 1e-12

# (cell key, arm label, test_output root, panel, cohort)
CELLS = [
    ("dsA/2023", "EXA-GP-MIN", "exagpmin_cohort_2021_aligned_g500", "dsA", "cohort_2021_aligned"),
    ("dsA/2023", "EXAMM",      "examm_cohort_2021_aligned_g500",    "dsA", "cohort_2021_aligned"),
    ("dsA/2023", "EXAMM-pub",  "mse_cohort_2021_aligned",           "dsA", "cohort_2021_aligned"),
    ("dsA/22-23", "EXA-GP-MIN", "exagpmin_cohort_2020_aligned_g500", "dsA", "cohort_2020_aligned"),
    ("dsA/22-23", "EXAMM",      "examm_cohort_2020_aligned_g500",    "dsA", "cohort_2020_aligned"),
    ("dsA/22-23", "EXAMM-pub",  "mse_cohort_2020_aligned",           "dsA", "cohort_2020_aligned"),
    ("set1/2022", "EXA-GP-MIN", "exagpmin_set1_cohort_2020_aligned_g500", "set1", "cohort_2020_aligned"),
    ("set1/2022", "EXAMM",      "examm_set1_cohort_2020_aligned_g500",    "set1", "cohort_2020_aligned"),
]


# ----------------------------------------------------------------------------- loading
def load_population(root, panel, cohort):
    """(dates, tickers, Y[T,K], P[B,T,K]) -- one prediction plane per seed.

    NOT ensemble_test/. The whole premise is that the ten members are the ensemble,
    so the averaged plane is reconstructed here from the members rather than read,
    and `check_mean` below asserts it matches what eval_ic_run.sh wrote.
    """
    dd = data_dir(panel, cohort)
    runs = sorted(glob.glob(os.path.join(REPO, "test_output", root, "run_*", "eval_test")),
                  key=lambda p: int(p.split("run_")[-1].split("/")[0]))
    if len(runs) < 2:
        raise FileNotFoundError(f"{root}: {len(runs)} run_*/eval_test dirs (need >= 2)")
    Y = P = None
    planes = []
    for r in runs:
        df = from_per_ticker(r, dd)
        y = df.pivot(index="date", columns="ticker", values="expected_RET").sort_index()
        p = df.pivot(index="date", columns="ticker", values="predicted_RET").reindex_like(y)
        if Y is None:
            Y, cols, idx = y, y.columns, y.index
        elif not (y.index.equals(idx) and y.columns.equals(cols)):
            raise ValueError(f"{r}: (date,ticker) grid differs from run 1")
        elif not np.allclose(y.to_numpy(float), Y.to_numpy(float), equal_nan=True):
            raise ValueError(f"{r}: expected_RET differs from run 1 -- different targets")
        planes.append(p.to_numpy(float))
    P = np.stack(planes)                                   # (B, T, K)
    return np.asarray(idx, str), np.asarray(cols, str), Y.to_numpy(float), P


def check_mean(root, panel, cohort, dates, tickers, P):
    """Assert our seed mean reproduces ensemble_test/, so 'the seeds ARE the ensemble'
    is a fact about these files and not a hope."""
    d = os.path.join(REPO, "test_output", root, "ensemble_test")
    if not os.path.isdir(d):
        return None
    df = from_per_ticker(d, data_dir(panel, cohort))
    E = df.pivot(index="date", columns="ticker",
                 values="predicted_RET").reindex(index=dates, columns=tickers).to_numpy(float)
    return float(np.nanmax(np.abs(E - P.mean(0))))


def trailing_vol(panel, cohort, dates, tickers, lookback=VOL_LOOKBACK):
    """sd of the `lookback` daily returns ENDING THE DAY BEFORE the target date.

    Target date t is forecast from date t-1's features, so returns through t-1 are
    known. Row j of the aligned frame is test-csv row j+1, hence the window is raw
    test rows [j+1-lookback, j]. Anything needing rows before the split start is NaN
    and is dropped by the burn-in, which is >= lookback by construction.
    """
    dd = data_dir(panel, cohort)
    V = np.full((len(dates), len(tickers)), np.nan)
    for k, tic in enumerate(tickers):
        t = pd.read_csv(os.path.join(dd, f"{tic}_test.csv"))
        ret = t["RET"].astype(float).to_numpy()
        d = t["date"].astype(str).to_numpy()[1:]
        if not np.array_equal(d, dates):
            raise ValueError(f"{tic}: date grid mismatch vs aligned frame")
        for j in range(len(dates)):
            lo = j + 1 - lookback
            if lo < 0:
                continue
            V[j, k] = ret[lo:j + 1].std(ddof=1)
    return V


# ------------------------------------------------------------------ conformal quantiles
def _oshi(sorted_e, p):
    """ceil((n+1)p)-th smallest -- the conservative (upper) conformal quantile."""
    n = len(sorted_e)
    k = int(np.ceil((n + 1) * p))
    return sorted_e[-1] if k >= n else sorted_e[max(k - 1, 0)]


def _oslo(sorted_e, p):
    """floor((n+1)p)-th smallest -- the conservative (lower) conformal quantile."""
    n = len(sorted_e)
    k = int(np.floor((n + 1) * p))
    return sorted_e[0] if k <= 0 else sorted_e[min(k, n) - 1]


def enbpi_endpoints(sorted_e, alpha):
    """EnbPI line 13-15: beta* = argmin width over beta in [0, alpha], on SIGNED residuals."""
    best = None
    for beta in np.linspace(0.0, alpha, BETA_GRID):
        lo = _oslo(sorted_e, beta)
        hi = _oshi(sorted_e, 1.0 - alpha + beta)
        if best is None or (hi - lo) < (best[1] - best[0]):
            best = (lo, hi)
    return best


# ----------------------------------------------------------------------------- methods
def build_intervals(Y, P, V, W, alpha):
    """All six methods at one alpha. Returns {name: (LO[T,K], HI[T,K])}, NaN pre-burn-in."""
    T, K = Y.shape
    M = P.mean(0)                                   # phi = seed mean
    SD = P.std(0, ddof=1)                           # population spread
    E = Y - M                                       # signed residual (out-of-sample throughout)
    A = np.abs(E)
    z = {0.20: 1.2815515655446004, 0.10: 1.6448536269514722}[alpha]

    out = {n: (np.full((T, K), np.nan), np.full((T, K), np.nan))
           for n in ("EnbPI-tkr", "EnbPI-pool", "Seed-Gauss", "Seed-Scaled",
                     "Trail-Fixed", "Vol-Scaled")}

    # Seed-Gauss needs no window at all -- that is the point of the claim under test.
    lo, hi = out["Seed-Gauss"]
    lo[W:], hi[W:] = (M - z * SD)[W:], (M + z * SD)[W:]

    sdw = np.where(np.isfinite(SD) & (SD > EPS), SD, np.nan)
    volw = np.where(np.isfinite(V) & (V > EPS), V, np.nan)
    S_seed = A / sdw                                 # normalised nonconformity scores
    S_vol = A / volw

    for t in range(W, T):
        win = slice(t - W, t)                        # residuals known when day t is forecast

        # ---- pooled windows (W days x K tickers)
        pool_e = E[win].ravel()
        pool_e = np.sort(pool_e[np.isfinite(pool_e)])
        pool_a = A[win].ravel()
        pool_a = np.sort(pool_a[np.isfinite(pool_a)])
        if len(pool_e) >= 10:
            l, h = enbpi_endpoints(pool_e, alpha)
            out["EnbPI-pool"][0][t] = M[t] + l
            out["EnbPI-pool"][1][t] = M[t] + h
        if len(pool_a) >= 10:
            q = _oshi(pool_a, 1.0 - alpha)
            out["Trail-Fixed"][0][t] = M[t] - q
            out["Trail-Fixed"][1][t] = M[t] + q

        for name, S, sig in (("Seed-Scaled", S_seed, SD), ("Vol-Scaled", S_vol, V)):
            s = S[win].ravel()
            s = np.sort(s[np.isfinite(s)])
            if len(s) < 10:
                continue
            q = _oshi(s, 1.0 - alpha)
            half = q * sig[t]
            out[name][0][t] = M[t] - half
            out[name][1][t] = M[t] + half

        # ---- per-ticker EnbPI: the assumption-faithful single-series algorithm
        for k in range(K):
            e = E[win, k]
            e = np.sort(e[np.isfinite(e)])
            if len(e) < 10:
                continue
            l, h = enbpi_endpoints(e, alpha)
            out["EnbPI-tkr"][0][t, k] = M[t, k] + l
            out["EnbPI-tkr"][1][t, k] = M[t, k] + h
    return out


# ----------------------------------------------------------------------------- scoring
def residual_only_intervals(Y, M, W, alpha):
    """EnbPI-tkr and Trail-Fixed off an ARBITRARY point forecast M -- the B=1 control.

    Separating the population's two possible contributions. Averaging ten genomes gives a
    better POINT forecast (ordinary ensembling, known to work), which narrows a conformal
    interval by shrinking the residuals. That is different from the claim under test, which
    is that the population's DISAGREEMENT is an uncertainty signal. Running the identical
    conformal wrapper on one genome at a time isolates the first from the second: if B=1
    scores like B=10, the ten seeds bought nothing that conformal could not get from one.
    """
    T, K = Y.shape
    E = Y - M
    A = np.abs(E)
    tkr = (np.full((T, K), np.nan), np.full((T, K), np.nan))
    fix = (np.full((T, K), np.nan), np.full((T, K), np.nan))
    for t in range(W, T):
        win = slice(t - W, t)
        pa = A[win].ravel()
        pa = np.sort(pa[np.isfinite(pa)])
        if len(pa) >= 10:
            q = _oshi(pa, 1.0 - alpha)
            fix[0][t], fix[1][t] = M[t] - q, M[t] + q
        for k in range(K):
            e = E[win, k]
            e = np.sort(e[np.isfinite(e)])
            if len(e) < 10:
                continue
            l, h = enbpi_endpoints(e, alpha)
            tkr[0][t, k], tkr[1][t, k] = M[t, k] + l, M[t, k] + h
    return {"EnbPI-tkr": tkr, "Trail-Fixed": fix}


def winkler(y, lo, hi, alpha):
    """Interval score: width + 2/alpha * miss distance. One number that punishes BOTH
    under-coverage and lazy width, so 'wider is better' cannot win."""
    w = hi - lo
    return w + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def day_block_se(cov2d, ok2d, rng, n_boot=500):
    """Bootstrap SE of coverage resampling WHOLE DAYS.

    50 stocks on one day are one market shock, not fifty independent trials, so the
    naive sqrt(p(1-p)/n) over ~9000 (stock,date) cells understates the error by
    roughly sqrt(50). Every coverage difference below has to clear THIS bar, not that
    one -- otherwise Seed-Scaled's apparent conditional-coverage edge is unfalsifiable.
    """
    rows = np.where(ok2d.any(axis=1))[0]
    if len(rows) < 5:
        return float("nan")
    num = np.where(ok2d, cov2d, 0.0).sum(axis=1)[rows]
    den = ok2d.sum(axis=1)[rows].astype(float)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    return float(np.std(num[idx].sum(1) / np.maximum(den[idx].sum(1), 1), ddof=1))


def score(Y, lo, hi, alpha, mask, groups, center, rng=None):
    ok = mask & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(Y)
    if ok.sum() == 0:
        return None
    y, l, h, c = Y[ok], lo[ok], hi[ok], center[ok]
    cov = ((y >= l) & (y <= h)).astype(float)
    C2 = np.zeros_like(Y)
    C2[ok] = cov
    rec = {"n": int(ok.sum()), "cov": float(cov.mean()), "width": float((h - l).mean()),
           "med_width": float(np.median(h - l)),
           # EnbPI's beta search only earns its keep if it actually tilts the interval
           # off the point forecast; skew = ((hi-yhat)-(yhat-lo))/width, 0 if symmetric.
           "skew": float(np.mean(((h - c) - (c - l)) / np.maximum(h - l, EPS))),
           "cov_se": day_block_se(C2, ok, rng if rng is not None
                                  else np.random.default_rng(0)),
           "winkler": float(winkler(y, l, h, alpha).mean())}
    for gname, G in groups.items():
        g = G[ok]
        sub = {}
        for v in np.unique(g[g >= 0]):
            m = g == v
            if m.sum() >= 30:
                sub[str(v)] = float(cov[m].mean())
        if sub:
            rec[f"cov_{gname}"] = sub
            rec[f"maxdev_{gname}"] = float(max(abs(c - (1 - alpha)) for c in sub.values()))
    return rec


def rank_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return float("nan")
    return float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))


def diagnostics(Y, P, V, mask):
    """The load-bearing block: is sd(seeds) a scale, and does it rank?"""
    M, SD = P.mean(0), P.std(0, ddof=1)
    E = Y - M
    A = np.abs(E)
    m = mask & np.isfinite(E) & np.isfinite(SD)
    sd, a, e, v = SD[m], A[m], E[m], V[m]
    d = {
        "rms_spread": float(np.sqrt((sd ** 2).mean())),
        "rms_err": float(np.sqrt((e ** 2).mean())),
        "ratio_rms": float(np.sqrt((sd ** 2).mean()) / np.sqrt((e ** 2).mean())),
        "ratio_med": float(np.median(sd) / np.median(a)),
        "var_frac": float((sd ** 2).mean() / (e ** 2).mean()),
        "rho_spread_abserr": rank_corr(sd, a),
        "rho_vol_abserr": rank_corr(v, a),
        "rho_spread_vol": rank_corr(sd, v),
        "rms_pred": float(np.sqrt((M[m] ** 2).mean())),
    }
    # within-day (cross-sectional) and within-ticker (time-series) rank correlations:
    # a pooled rho can be manufactured entirely by between-stock scale differences.
    for tag, axis in (("xs", 0), ("ts", 1)):
        rs, rv = [], []
        it = range(Y.shape[0]) if axis == 0 else range(Y.shape[1])
        for i in it:
            sl = (i, slice(None)) if axis == 0 else (slice(None), i)
            mm = mask[sl]
            if mm.sum() < 30:
                continue
            rs.append(rank_corr(SD[sl][mm], A[sl][mm]))
            rv.append(rank_corr(V[sl][mm], A[sl][mm]))
        d[f"rho_spread_abserr_{tag}"] = float(np.nanmean(rs)) if rs else float("nan")
        d[f"rho_vol_abserr_{tag}"] = float(np.nanmean(rv)) if rv else float("nan")
    # how much of the population is even distinct: seeds that agree exactly add nothing
    d["frac_zero_spread"] = float((SD[m] < 1e-9).mean())
    # rms_pred vs rms_spread is the sharpest single contrast. If the spread EXCEEDS the
    # prediction magnitude the seeds violently disagree with each other -- and yet if
    # ratio_rms is still 0.03 that disagreement is happening two orders of magnitude below
    # the scale of the target. Relative disorder, absolute irrelevance.
    d["spread_over_pred"] = float(d["rms_spread"] / max(d["rms_pred"], EPS))
    B = P.shape[0]
    cs = []
    for i in range(B):
        for j in range(i + 1, B):
            a1, a2 = P[i][mask], P[j][mask]
            k = np.isfinite(a1) & np.isfinite(a2)
            if k.sum() > 30 and a1[k].std() > 0 and a2[k].std() > 0:
                cs.append(float(np.corrcoef(a1[k], a2[k])[0, 1]))
    d["mean_pairwise_seed_corr"] = float(np.mean(cs)) if cs else float("nan")
    return d


# -------------------------------------------------------------------------------- main
def run_cell(cell, arm, root, panel, cohort, W):
    dates, tickers, Y, P = load_population(root, panel, cohort)
    dmax = check_mean(root, panel, cohort, dates, tickers, P)
    V = trailing_vol(panel, cohort, dates, tickers)
    T, K = Y.shape

    quarter = np.array([f"{d[:4]}Q{(int(d[5:7]) - 1)//3 + 1}" for d in dates])
    qcodes = pd.Categorical(quarter).codes
    QG = np.repeat(qcodes[:, None], K, axis=1)

    mask = np.zeros((T, K), bool)
    mask[W:] = True
    mask &= np.isfinite(Y) & np.isfinite(V)

    # volatility quintiles, cut on the scored sample only
    vq = np.full((T, K), -1)
    vv = V[mask]
    if vv.size:
        edges = np.quantile(vv, [0.2, 0.4, 0.6, 0.8])
        vq[mask] = np.searchsorted(edges, V[mask])
    groups = {"vol": vq, "q": np.where(mask, QG, -1)}

    rec = {"cell": cell, "arm": arm, "root": root, "B": int(P.shape[0]),
           "T": T, "K": K, "W": W, "n_scored": int(mask.sum()),
           "span": f"{dates[0]}..{dates[-1]}",
           "ens_mean_maxdiff": dmax,
           "diag": diagnostics(Y, P, V, mask), "alpha": {}}

    rng = np.random.default_rng(0)
    center = P.mean(0)
    for alpha in ALPHAS:
        iv = build_intervals(Y, P, V, W, alpha)
        rec["alpha"][f"{alpha:.2f}"] = {
            name: score(Y, lo, hi, alpha, mask, groups, center, rng)
            for name, (lo, hi) in iv.items()}

    # B=1 control at 90% only: same conformal wrapper, each single genome in turn.
    b1 = {n: {"cov": [], "width": [], "winkler": []} for n in ("EnbPI-tkr", "Trail-Fixed")}
    for b in range(P.shape[0]):
        Mb = P[b]
        for name, (lo, hi) in residual_only_intervals(Y, Mb, W, 0.10).items():
            s = score(Y, lo, hi, 0.10, mask, groups, Mb, rng)
            for k in ("cov", "width", "winkler"):
                b1[name][k].append(s[k])
    rec["b1"] = {n: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                         "min": float(np.min(v)), "max": float(np.max(v))}
                     for k, v in d.items()} for n, d in b1.items()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60,
                    help="EnbPI residual window in trading days; also the burn-in")
    ap.add_argument("--cells", default="",
                    help="comma-separated substrings of the arm root to restrict to")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    a = ap.parse_args()

    todo = [c for c in CELLS
            if not a.cells or any(s in c[2] for s in a.cells.split(","))]
    recs = []
    for cell, arm, root, panel, cohort in todo:
        try:
            recs.append(run_cell(cell, arm, root, panel, cohort, a.window))
            print(f"ok  {cell:12s} {arm:11s} {root}", file=sys.stderr)
        except Exception as exc:                                          # noqa: BLE001
            print(f"ERR {cell:12s} {arm:11s} {root}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    json.dump(recs, open(a.out, "w"), indent=1)
    report(recs)
    print(f"\nwrote {a.out}", file=sys.stderr)


def report(recs):
    p = lambda *x: print(*x)                                              # noqa: E731
    p("\n" + "=" * 118)
    p("SEED-SPREAD DIAGNOSTIC  (does the population know anything?)")
    p("=" * 118)
    p(f"{'cell':11s} {'arm':11s} {'B':>2s} {'n':>6s} {'rms sd':>9s} {'rms err':>9s} "
      f"{'sd/err':>7s} {'varfrac':>8s} {'rho(sd,|e|)':>11s} {'  xs':>7s} {'  ts':>7s} "
      f"{'rho(vol,|e|)':>12s} {'  xs':>7s} {'  ts':>7s} {'rho(sd,vol)':>11s} {'infl':>6s}")
    for r in recs:
        d = r["diag"]
        a = r["alpha"]["0.10"]
        infl = (a["Seed-Scaled"]["width"] / a["Seed-Gauss"]["width"]
                if a.get("Seed-Scaled") and a.get("Seed-Gauss") else float("nan"))
        p(f"{r['cell']:11s} {r['arm']:11s} {r['B']:2d} {r['n_scored']:6d} "
          f"{d['rms_spread']:9.6f} {d['rms_err']:9.6f} {d['ratio_rms']:7.3f} "
          f"{d['var_frac']:8.5f} {d['rho_spread_abserr']:11.3f} "
          f"{d['rho_spread_abserr_xs']:7.3f} {d['rho_spread_abserr_ts']:7.3f} "
          f"{d['rho_vol_abserr']:12.3f} {d['rho_vol_abserr_xs']:7.3f} "
          f"{d['rho_vol_abserr_ts']:7.3f} {d['rho_spread_vol']:11.3f} {infl:5.0f}x")
    p("rms sd = rms across-seed sd of the 10 predictions; rms err = rms(y - seed mean).")
    p("varfrac = mean(sd^2)/mean(err^2): the share of squared error the population 'explains'.")
    p("rho = Spearman; xs = mean over days (cross-section), ts = mean over tickers (time series).")
    p("infl  = factor by which the raw Gaussian seed-spread interval must be widened to reach")
    p("        nominal 90% (= Seed-Scaled width / Seed-Gauss width).")
    p("")
    p(f"{'cell':11s} {'arm':11s} {'rms pred':>10s} {'pred/err':>9s} {'spread/pred':>12s} "
      f"{'mean pairwise seed corr':>24s}")
    for r in recs:
        d = r["diag"]
        p(f"{r['cell']:11s} {r['arm']:11s} {d['rms_pred']:10.6f} "
          f"{d['rms_pred']/d['rms_err']:9.4f} {d['spread_over_pred']:12.2f} "
          f"{d['mean_pairwise_seed_corr']:24.3f}")
    p("pred/err: how big the point forecast is next to the thing it is forecasting.")
    p("spread/pred > 1 means the seeds disagree with each other by MORE than the mean")
    p("  forecast they emit -- maximal relative disorder, on a scale irrelevant to the target.")

    for alpha in ALPHAS:
        key = f"{alpha:.2f}"
        p("\n" + "=" * 118)
        p(f"NOMINAL {100*(1-alpha):.0f}%  COVERAGE / MEAN WIDTH / WINKLER / worst-quintile "
          f"and worst-quarter coverage deviation")
        p("=" * 118)
        p(f"{'cell':11s} {'arm':11s} {'method':12s} {'cov':>7s} {'+-se':>6s} {'width':>9s} "
          f"{'Winkler':>9s} {'skew':>6s} {'dev(vol)':>9s} {'dev(qtr)':>9s}")
        for r in recs:
            for name, s in r["alpha"][key].items():
                if s is None:
                    continue
                p(f"{r['cell']:11s} {r['arm']:11s} {name:12s} {s['cov']:7.3f} "
                  f"{s['cov_se']:6.3f} {s['width']:9.5f} {s['winkler']:9.5f} "
                  f"{s['skew']:+6.2f} "
                  f"{s.get('maxdev_vol', float('nan')):9.3f} "
                  f"{s.get('maxdev_q', float('nan')):9.3f}")
        p("se = day-block bootstrap SE of coverage (whole days resampled; 50 stocks/day are")
        p("not 50 independent trials). Winkler = width + (2/alpha)*miss; lower is better.")
    p("\n" + "=" * 118)
    p("B=1 CONTROL, nominal 90%: the same conformal wrapper on ONE genome at a time")
    p("(mean +- sd over the 10 single-seed choices) against the 10-seed ensemble mean.")
    p("=" * 118)
    p(f"{'cell':11s} {'arm':11s} {'method':12s} {'B=1 cov':>9s} {'B=1 width':>19s} "
      f"{'B=1 Winkler':>19s} {'B=10 width':>11s} {'B=10 Wink':>10s} {'width gain':>11s}")
    for r in recs:
        for name in ("EnbPI-tkr", "Trail-Fixed"):
            b = r.get("b1", {}).get(name)
            s = r["alpha"]["0.10"].get(name)
            if not b or not s:
                continue
            p(f"{r['cell']:11s} {r['arm']:11s} {name:12s} "
              f"{b['cov']['mean']:9.3f} "
              f"{b['width']['mean']:.5f}+-{b['width']['sd']:.5f} "
              f"{b['winkler']['mean']:.5f}+-{b['winkler']['sd']:.5f} "
              f"{s['width']:11.5f} {s['winkler']:10.5f} "
              f"{100*(1 - s['width']/b['width']['mean']):9.1f}%")

    p("\n" + "=" * 118)
    p("COVERAGE BY VOLATILITY QUINTILE  (0 = calmest)   nominal 90%")
    p("=" * 118)
    for r in recs:
        for name, s in r["alpha"]["0.10"].items():
            if s is None or "cov_vol" not in s:
                continue
            cells = " ".join(f"{s['cov_vol'].get(str(i), float('nan')):6.3f}" for i in range(5))
            p(f"{r['cell']:11s} {r['arm']:11s} {name:12s} {cells}")


if __name__ == "__main__":
    main()
