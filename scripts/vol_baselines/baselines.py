"""Baseline volatility forecasters for the ICAIF paper.

All models predict the SAME target:  TARGET(t) = mean(LV[t+1..t+H]),  LV = log Parkinson vol.
All return LONG-FORMAT frames keyed by (stock, date) -- never positional arrays. This is
deliberate: the E1a incident (an LSTM harness silently dropping the first 22 test rows per
stock) produced two different "HAR" numbers that looked comparable but were not. Key-based
joins make that class of bug impossible.

ESTIMATION PROTOCOL (identical for every fitted model, for fairness):
  fit once on TRAIN, then forecast val/test with parameters held fixed.
  No model gets rolling re-estimation that another is denied.

CALIBRATION: models whose native output is a daily VARIANCE (EWMA, GARCH, EGARCH) do not
predict our target directly. Each produces a raw predictor, then an affine map
    TARGET ~ a + b * raw
is fit BY OLS ON TRAIN ROWS ONLY and applied unchanged to val/test. This is a
Mincer-Zarnowitz-style mapping applied uniformly to every non-native model, so the
comparison measures forecast quality rather than a units mismatch. Both `y_pred_raw`
(uncalibrated) and `y_pred` (calibrated) are persisted.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FEATS_HAR = ["LV", "MA5", "MA22"]
HORIZON = 5
EWMA_LAMBDA = 0.94          # RiskMetrics standard
ARCH_SCALE = 100.0          # `arch` wants percentage returns for numerical conditioning
SIMS = 250                  # simulation paths for EGARCH multi-step forecasts
EGARCH_SEED = 20260721      # EGARCH's multi-step forecast is Monte Carlo; unseeded it is
                            # NOT reproducible (caught by the determinism gate: egarch was
                            # the only model whose prediction hash changed between runs).

#: models whose native output needs the affine calibration described above
NEEDS_CALIBRATION = ("ewma", "garch", "egarch")
ALL_MODELS = ("naive", "har", "har_pooled", "har_x", "ewma", "garch", "egarch")


# ======================================================================================
# helpers
# ======================================================================================
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.hstack([X, np.ones((len(X), 1))])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _ols_pred(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ beta


def _calibrate(raw_tr: np.ndarray, y_tr: np.ndarray, raw_te: np.ndarray):
    """Fit TARGET ~ a + b*raw on TRAIN ONLY; return (calibrated_test, a, b)."""
    ok = np.isfinite(raw_tr) & np.isfinite(y_tr)
    if ok.sum() < 30:
        return raw_te.copy(), 0.0, 1.0          # too little data -> identity, flagged upstream
    b, a = np.polyfit(raw_tr[ok], y_tr[ok], 1)  # np.polyfit returns [slope, intercept]
    return a + b * raw_te, float(a), float(b)


def _long(stock: str, dates, y_true, y_pred, y_pred_raw=None) -> pd.DataFrame:
    return pd.DataFrame({
        "stock": stock,
        "date": pd.to_datetime(dates),
        "y_true": np.asarray(y_true, float),
        "y_pred": np.asarray(y_pred, float),
        "y_pred_raw": np.asarray(y_pred_raw if y_pred_raw is not None else y_pred, float),
    })


# ======================================================================================
# native-target models (predict TARGET directly; no calibration)
# ======================================================================================
def fit_naive(tr: pd.DataFrame, te: pd.DataFrame, stock: str):
    """Random-walk-in-vol: forecast future mean log-vol with the CURRENT trailing mean.

    Uses MA5(t) -- the trailing 5-day mean of LV, matching the target's 5-day averaging
    and using only information available at t.

    NOTE (important): the obvious alternative, TARGET(t-1), is INVALID here. TARGET is a
    5-day FORWARD rolling mean, so TARGET(t-1) and TARGET(t) share 4 of their 5 days; it
    scores R^2 ~ 0.87 and "beats" every real model purely by exploiting that overlap. It
    is not a forecast. This is a live trap in any overlapping-horizon setup.
    """
    return _long(stock, te["date"], te["TARGET"].values, te["MA5"].values), {}


def fit_har(tr: pd.DataFrame, te: pd.DataFrame, stock: str, feats=FEATS_HAR):
    """HAR: OLS on (LV, MA5, MA22) -- the domain-standard multi-scale linear model."""
    beta = _ols_fit(tr[feats].values, tr["TARGET"].values)
    pred = _ols_pred(te[feats].values, beta)
    diag = {f"har_beta_{i}": float(b) for i, b in enumerate(beta)}
    return _long(stock, te["date"], te["TARGET"].values, pred), diag


def fit_har_x(tr: pd.DataFrame, te: pd.DataFrame, stock: str):
    """HAR-X: HAR plus leverage terms (signed return and negative-return magnitude).

    The stronger linear bar -- volatility responds asymmetrically to negative returns,
    so a reviewer will expect HAR to be given this advantage before it is beaten.
    """
    def design(d):
        neg = np.maximum(-d["RET"].values, 0.0)
        return np.column_stack([d[FEATS_HAR].values, d["RET"].values, neg])
    beta = _ols_fit(design(tr), tr["TARGET"].values)
    return _long(stock, te["date"], te["TARGET"].values, _ols_pred(design(te), beta)), {}


# ======================================================================================
# variance models (native output = daily variance -> raw predictor -> calibrated)
# ======================================================================================
def _ewma_var(ret: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """RiskMetrics EWMA variance, recursive. var[t] uses returns up to and including t."""
    v = np.empty(len(ret), dtype=float)
    seed = float(np.var(ret[: min(22, len(ret))])) or 1e-8
    prev = seed
    for i, r in enumerate(ret):
        prev = lam * prev + (1.0 - lam) * r * r
        v[i] = prev
    return v


def fit_ewma(tr: pd.DataFrame, te: pd.DataFrame, stock: str):
    """EWMA/RiskMetrics. Flat h-step projection (EWMA is a martingale in variance),
    so the raw predictor is 0.5*log(var_t) -- the log-vol implied at the forecast origin."""
    full_ret = np.concatenate([tr["RET"].values, te["RET"].values])
    var = _ewma_var(full_ret)
    raw_all = 0.5 * np.log(np.maximum(var, 1e-12))
    n_tr = len(tr)
    raw_tr, raw_te = raw_all[:n_tr], raw_all[n_tr:]
    cal, a, b = _calibrate(raw_tr, tr["TARGET"].values, raw_te)
    return (_long(stock, te["date"], te["TARGET"].values, cal, raw_te),
            {"calib_a": a, "calib_b": b, "converged": True})


def _garch_raw(tr: pd.DataFrame, te: pd.DataFrame, kind: str):
    """Fit GARCH/EGARCH ONCE on train, then produce h=1..H ahead forecasts across the
    full series with parameters fixed. Returns (raw_train, raw_test, diagnostics).

    raw = mean over h=1..H of log(sigma_{t+h}) -- the model-implied analogue of our
    target (mean log vol over the next H days), before calibration.
    """
    from arch import arch_model

    full_ret = np.concatenate([tr["RET"].values, te["RET"].values]) * ARCH_SCALE
    n_tr = len(tr)
    kw = dict(vol="EGARCH", p=1, o=1, q=1) if kind == "egarch" else dict(vol="GARCH", p=1, q=1)
    am = arch_model(full_ret[:n_tr], mean="Constant", dist="normal", rescale=False, **kw)
    res = am.fit(disp="off", show_warning=False)

    conv = bool(getattr(res, "convergence_flag", 0) == 0)
    par = res.params.to_dict()
    persistence = float(par.get("alpha[1]", np.nan) + par.get("beta[1]", np.nan)) \
        if kind == "garch" else float(par.get("beta[1]", np.nan))

    # Forecast over the WHOLE series holding the train-fitted parameters fixed.
    full_am = arch_model(full_ret, mean="Constant", dist="normal", rescale=False, **kw)
    # start=0 is essential: arch's forecast() returns values only at the SAMPLE END by
    # default, which silently yields an all-NaN in-sample forecast matrix.
    # EGARCH has no closed-form multi-step forecast -> simulation.
    fkw = dict(horizon=HORIZON, start=0, reindex=True)
    if kind == "egarch":
        # seeded RNG -> byte-identical forecasts across runs (see EGARCH_SEED)
        fkw.update(method="simulation", simulations=SIMS,
                   rng=np.random.RandomState(EGARCH_SEED).standard_normal)
    fc = full_am.fix(res.params).forecast(**fkw)
    var_h = fc.variance.values                      # (n_obs, HORIZON), variance in %^2
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.sqrt(np.maximum(var_h, 1e-16)) / ARCH_SCALE   # back to return units
        raw_all = np.nanmean(np.log(np.maximum(sig, 1e-12)), axis=1)
    med_sigma = float(np.nanmedian(np.sqrt(np.maximum(var_h[:, 0], 1e-16)) / ARCH_SCALE))
    diag = {"converged": conv, "persistence": persistence, "median_sigma": med_sigma}
    return raw_all[:n_tr], raw_all[n_tr:], diag


def fit_garch(tr, te, stock, kind="garch"):
    try:
        raw_tr, raw_te, diag = _garch_raw(tr, te, kind)
    except Exception as e:                            # non-convergence / singular fit
        return None, {"converged": False, "error": str(e)[:120]}
    if not np.isfinite(raw_te).all():
        return None, {**diag, "converged": False, "error": "non-finite forecast"}
    cal, a, b = _calibrate(raw_tr, tr["TARGET"].values, raw_te)
    diag.update({"calib_a": a, "calib_b": b})
    return _long(stock, te["date"], te["TARGET"].values, cal, raw_te), diag


def fit_egarch(tr, te, stock):
    return fit_garch(tr, te, stock, kind="egarch")


# ======================================================================================
# pooled HAR (one model across all stocks) -- fairness protocol requires reporting
# HAR at its BEST configuration, so both per-stock and pooled are computed.
# ======================================================================================
def fit_har_pooled(train_all: pd.DataFrame, test_by_stock: dict[str, pd.DataFrame]):
    beta = _ols_fit(train_all[FEATS_HAR].values, train_all["TARGET"].values)
    out = [_long(s, d["date"], d["TARGET"].values, _ols_pred(d[FEATS_HAR].values, beta))
           for s, d in test_by_stock.items()]
    return pd.concat(out, ignore_index=True), {f"har_pooled_beta_{i}": float(b)
                                               for i, b in enumerate(beta)}


PER_STOCK_MODELS = {
    "naive": fit_naive,
    "har": fit_har,
    "har_x": fit_har_x,
    "ewma": fit_ewma,
    "garch": fit_garch,
    "egarch": fit_egarch,
}
