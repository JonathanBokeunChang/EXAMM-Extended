"""Canonical forecast-evaluation metrics for the volatility paper.

SINGLE SOURCE OF TRUTH. Every script (baselines, EXAMM eval, LSTM eval, analysis)
must import from here rather than redefining metrics inline. An audit of the legacy
inline copies in scratchpad/ (qlib_vol_eval.py, pooled_big_eval.py, qlib_vol_ens_eval.py,
qlib_vol_eval_har.py, vol_diagnose.py, e1a_lstm_baseline.py) confirmed they all used
the identical R^2 and QLIKE definitions reproduced below, so centralising here changes
no historical number -- it only removes the risk of future drift.

CONVENTION: y and p are LOG VOLATILITY (natural log of a volatility, not variance).
The paper's target is TARGET(t) = mean(LV[t+1..t+H]) where LV = log Parkinson vol.

QLIKE derivation (why the formula looks like this):
    The standard QLIKE loss for a VARIANCE forecast h against realised variance r is
        QLIKE = log(h) + r/h
    Our quantities are log-vol, so variance = exp(2*logvol):
        h = exp(2p),  r = exp(2y)
        => QLIKE = log(exp(2p)) + exp(2y)/exp(2p) = 2p + exp(2y - 2p)
    The exponent is clipped to +/-20 purely to prevent overflow on pathological
    predictions; at |2(y-p)| = 20 the term is already ~4.9e8 and such a row dominates
    the mean regardless, so clipping does not change any ranking.

QLIKE and MSE are the two loss functions that are robust to noise in the volatility
proxy (Patton 2011), which is why both are reported. MAE and R^2 are included because
reviewers expect them and because single-model verdicts proved metric-dependent.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mse", "mae", "r2", "qlike", "all_metrics",
    "METRICS", "LOWER_IS_BETTER",
    "wilcoxon_paired", "ttest_paired", "diebold_mariano",
    "qlike_loss_series", "se_loss_series",
]

METRICS = ("mse", "mae", "r2", "qlike")
#: True  -> smaller is better (losses).  False -> larger is better (R^2).
LOWER_IS_BETTER = {"mse": True, "mae": True, "qlike": True, "r2": False}

_CLIP = 20.0


def _as_arrays(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y{y.shape} vs p{p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isfinite(y).all():
        raise ValueError("non-finite values in y_true")
    if not np.isfinite(p).all():
        raise ValueError("non-finite values in y_pred")
    return y, p


def mse(y, p) -> float:
    y, p = _as_arrays(y, p)
    return float(np.mean((y - p) ** 2))


def mae(y, p) -> float:
    y, p = _as_arrays(y, p)
    return float(np.mean(np.abs(y - p)))


def r2(y, p) -> float:
    """Out-of-sample R^2 against the *evaluation-set* mean (not the training mean).

    0 == no better than predicting the mean of the evaluation window; negative is
    possible and meaningful.
    """
    y, p = _as_arrays(y, p)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")  # constant target -> R^2 undefined
    return float(1.0 - np.sum((y - p) ** 2) / denom)


def qlike(y, p) -> float:
    """QLIKE loss on the variance scale, computed from log-vol inputs. Lower is better."""
    y, p = _as_arrays(y, p)
    return float(np.mean(2.0 * p + np.exp(np.clip(2.0 * y - 2.0 * p, -_CLIP, _CLIP))))


def all_metrics(y, p) -> dict[str, float]:
    """All four metrics as a dict -- the standard return shape for the harness."""
    return {"mse": mse(y, p), "mae": mae(y, p), "r2": r2(y, p), "qlike": qlike(y, p)}


# --------------------------------------------------------------------------------------
# Per-observation loss series (needed for Diebold-Mariano and the Model Confidence Set,
# which operate on loss *differentials*, not on aggregated metrics).
# --------------------------------------------------------------------------------------
def se_loss_series(y, p) -> np.ndarray:
    """Squared-error loss per observation."""
    y, p = _as_arrays(y, p)
    return (y - p) ** 2


def qlike_loss_series(y, p) -> np.ndarray:
    """QLIKE loss per observation."""
    y, p = _as_arrays(y, p)
    return 2.0 * p + np.exp(np.clip(2.0 * y - 2.0 * p, -_CLIP, _CLIP))


# --------------------------------------------------------------------------------------
# Paired significance tests
# --------------------------------------------------------------------------------------
def wilcoxon_paired(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired per-stock metrics.

    Primary test in the pre-registration: per-stock metric distributions are skewed,
    so a rank test is preferred over the paired t.
    """
    from scipy import stats
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.allclose(a[ok], b[ok]):
        return float("nan")
    return float(stats.wilcoxon(a[ok], b[ok]).pvalue)


def ttest_paired(a, b) -> float:
    """Two-sided paired t-test p-value (secondary test)."""
    from scipy import stats
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.ttest_rel(a[ok], b[ok]).pvalue)


def diebold_mariano(loss_a, loss_b, h: int = 5) -> tuple[float, float]:
    """Diebold-Mariano test on two per-observation loss series.

    Uses a Newey-West HAC variance with bandwidth h-1 (the forecast horizon overlaps,
    so the loss differential is autocorrelated), plus the Harvey-Leybourne-Newbold
    small-sample correction. Returns (DM statistic, two-sided p-value).

    Negative statistic => model A has lower loss (A is better).
    """
    from scipy import stats
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return float("nan"), float("nan")
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    var = gamma0
    for lag in range(1, max(1, h)):
        if lag >= n:
            break
        cov = float(np.mean(dc[lag:] * dc[:-lag]))
        var += 2.0 * (1.0 - lag / float(h)) * cov  # Bartlett kernel
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    corr = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    dm_hln = dm * corr
    pval = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(pval)
