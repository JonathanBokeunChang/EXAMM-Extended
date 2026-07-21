"""Validation suite for the volatility baseline harness.

These numbers go in a paper, so the harness validates itself and FAILS LOUD rather than
writing results a human might later trust.

HARD GATES (abort the run, non-zero exit):
  1. regression anchor  -- pooled HAR must reproduce banked medians (.2473 all rows /
                           .2371 on rows offset by 22). If this fails the harness is
                           wrong and nothing else it produced can be trusted.
  2. finiteness         -- no NaN/inf in any prediction or metric
  3. coverage           -- every model spans an identical (stock,date) key set
  4. leakage            -- no fitted object saw val/test rows
  5. garch scale        -- median fitted sigma within [0.5x, 2x] empirical return sd
                           (catches the arch percentage-scaling error)
  6. determinism        -- re-running reproduces identical prediction hashes

SOFT CHECKS (recorded, never fatal):
  7. GARCH persistence in (0.5, 1.0)
  8. convergence rate >= 95%
  9. sanity ordering (naive worst; HAR >~ EWMA)
 10. legacy metric agreement (< 1e-9 vs the inline copies in scratchpad/)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

ANCHOR_TOL = 2e-3

#: Per-DATASET regression anchors: dataset dir basename -> (all-rows R^2, offset-22 R^2).
#: Matched EXACTLY, never by substring -- "qlib_vol_big_v2" must not inherit
#: "qlib_vol_big"'s anchor, since the two differ by the boundary-leak fix.
#: A value of None means "no anchor established yet": the measured number is reported as a
#: soft check so it can be promoted here after review, rather than hard-failing a new dataset.
ANCHORS: dict[str, tuple[float | None, float | None]] = {
    # FROZEN pre-leak-fix fixture. Do NOT rebuild this dataset -- it exists so the harness
    # keeps a known-good regression target that provably reproduced the banked E1a numbers.
    "qlib_vol_big": (0.2473, 0.2371),
    # Leak-fixed rebuilds (build_vol_dataset.py). qlib_vol_big_v2 is qlib_vol_big's exact
    # 120 tickers with the boundary trim, so .2475 vs .2473 measures the leak's whole
    # effect on the headline baseline: +0.0002. The leak was real but immaterial.
    "qlib_vol_big_v2": (0.2475, 0.2370),
    "qlib_vol_full": (0.2397, 0.2360),   # 238-stock universe (234 scored), E1b probe
}

ANCHOR_FULL = ANCHORS["qlib_vol_big"][0]        # back-compat for any external reader
ANCHOR_OFFSET22 = ANCHORS["qlib_vol_big"][1]


class Check:
    __slots__ = ("name", "hard", "passed", "detail")

    def __init__(self, name, hard, passed, detail):
        self.name, self.hard, self.passed, self.detail = name, hard, passed, detail

    def __str__(self):
        status = "PASS" if self.passed else ("FAIL" if self.hard else "WARN")
        kind = "HARD" if self.hard else "soft"
        return f"[{status}] ({kind}) {self.name}: {self.detail}"


# ======================================================================================
# gate 1 -- regression anchor
# ======================================================================================
def check_regression_anchor(per_stock: pd.DataFrame, anchor_csv: str | None,
                            dataset: str | None = None) -> list[Check]:
    """pooled HAR median R^2 must match this DATASET's banked values.

    `dataset` is the data dir (path or basename); it selects the anchor by exact name. An
    unknown dataset, or one registered with None, reports its measured value as a soft check
    instead of failing -- a new dataset has no banked truth to regress against yet.
    """
    out = []
    hp = per_stock[per_stock.model == "har_pooled"]
    if hp.empty:
        return [Check("regression_anchor", True, False, "har_pooled missing from results")]

    key = os.path.basename(str(dataset).rstrip("/")) if dataset else None
    exp_full, exp_off = ANCHORS.get(key, (None, None))
    known = key in ANCHORS

    got_full = float(hp["r2"].median())
    if exp_full is None:
        out.append(Check("regression_anchor/all_rows", False, True,
                         f"no anchor for dataset '{key}'"
                         f"{'' if known else ' (unregistered)'}; MEASURED pooled HAR median "
                         f"R2 = {got_full:.4f} -- promote into validate.ANCHORS once reviewed"))
    else:
        ok_full = abs(got_full - exp_full) <= ANCHOR_TOL
        out.append(Check("regression_anchor/all_rows", True, ok_full,
                         f"[{key}] pooled HAR median R2 = {got_full:.4f}, expected "
                         f"{exp_full:.4f} (tol {ANCHOR_TOL}); diff {got_full-exp_full:+.4f}"))

    if "r2_offset22" in hp.columns:
        got_off = float(hp["r2_offset22"].median())
        if exp_off is None:
            out.append(Check("regression_anchor/offset22", False, True,
                             f"no anchor for '{key}'; MEASURED offset-22 R2 = {got_off:.4f}"))
        else:
            ok_off = abs(got_off - exp_off) <= ANCHOR_TOL
            out.append(Check("regression_anchor/offset22", True, ok_off,
                             f"[{key}] pooled HAR median R2 (offset 22) = {got_off:.4f}, "
                             f"expected {exp_off:.4f}; diff {got_off-exp_off:+.4f}"))

    # cross-check against the E1a per-stock CSV -- only meaningful for the frozen fixture,
    # whose rows that CSV was actually produced from
    if anchor_csv and exp_off is not None:
        try:
            e1 = pd.read_csv(anchor_csv)
            if "HAR_r2" in e1.columns:
                ref = float(e1["HAR_r2"].median())
                out.append(Check("regression_anchor/e1a_csv", False,
                                 abs(ref - ANCHOR_OFFSET22) <= 1e-2,
                                 f"E1a HAR_r2 median = {ref:.4f} (reference {ANCHOR_OFFSET22:.4f})"))
        except Exception as e:
            out.append(Check("regression_anchor/e1a_csv", False, False, f"unreadable: {e}"))
    return out


# ======================================================================================
# gates 2-3 -- finiteness, coverage
# ======================================================================================
def check_finiteness(preds: dict[str, pd.DataFrame], per_stock: pd.DataFrame) -> list[Check]:
    bad = []
    for m, df in preds.items():
        for col in ("y_true", "y_pred"):
            n = int((~np.isfinite(df[col].values)).sum())
            if n:
                bad.append(f"{m}.{col}={n}")
    mcols = [c for c in ("mse", "mae", "r2", "qlike") if c in per_stock.columns]
    nm = int(per_stock[mcols].isna().sum().sum())
    ok = not bad and nm == 0
    return [Check("finiteness", True, ok,
                  "all predictions and metrics finite" if ok
                  else f"non-finite: {', '.join(bad) or 'none'}; NaN metrics={nm}")]


def check_coverage(preds: dict[str, pd.DataFrame],
                   per_stock: pd.DataFrame | None = None) -> list[Check]:
    """Two distinct things, deliberately separated.

    HARD -- the invariant the paper actually claims: every model is SCORED on an identical
    (stock, date) set. run_baselines intersects before computing metrics, so this holds by
    construction; the gate proves it rather than assuming it.

    SOFT -- raw prediction coverage. A model that fails to fit some stocks (EGARCH
    non-convergence is the usual cause) shrinks the common set for everyone. That is a
    disclosure item, not a correctness failure -- and it biases CONSERVATIVELY for us, since
    dropping a baseline's hardest stocks flatters that baseline, not our model.
    """
    out = []
    keysets = {m: set(zip(df["stock"], df["date"])) for m, df in preds.items()}
    inter = set.intersection(*keysets.values()) if keysets else set()

    short = {m: len(k) - len(inter) for m, k in keysets.items() if len(k) > len(inter)}
    limiting = sorted(m for m, k in keysets.items() if len(k) == len(inter))
    dropped_stocks = sorted({s for m, k in keysets.items() for s, _ in k} -
                            {s for s, _ in inter})
    sizes = ", ".join(f"{m}:{len(k):,}" for m, k in sorted(keysets.items()))
    frac = len(inter) / max(max((len(k) for k in keysets.values()), default=1), 1)
    out.append(Check(
        "coverage/raw_predictions", False, frac >= 0.99,
        f"common set {len(inter):,} keys = {frac:.4f} of the widest model; "
        f"{len(dropped_stocks)} stock(s) excluded from the comparison"
        f"{' (limiting: ' + ','.join(limiting) + ')' if short else ''}"
        f"{'; dropped: ' + ','.join(dropped_stocks[:8]) if dropped_stocks else ''}"
        f"{' ...' if len(dropped_stocks) > 8 else ''} | sizes: {sizes}"))

    if per_stock is not None and not per_stock.empty:
        sets = {m: frozenset(g["stock"]) for m, g in per_stock.groupby("model")}
        rows = {m: int(g["n"].sum()) for m, g in per_stock.groupby("model")}
        same_stocks = len(set(sets.values())) == 1
        same_rows = len(set(rows.values())) == 1
        out.append(Check(
            "coverage/scored_identical", True, same_stocks and same_rows,
            f"{len(next(iter(sets.values())))} stocks / "
            f"{next(iter(rows.values())):,} rows per model, identical across all "
            f"{len(sets)} models: stocks={same_stocks}, rows={same_rows}"))
    return out


# ======================================================================================
# gate 4 -- leakage
# ======================================================================================
def check_leakage(train_max_date, test_min_date, calib_fit_on_train: bool) -> list[Check]:
    ok_dates = pd.Timestamp(train_max_date) < pd.Timestamp(test_min_date)
    return [
        Check("leakage/train_before_test", True, ok_dates,
              f"train_max={pd.Timestamp(train_max_date).date()} < "
              f"test_min={pd.Timestamp(test_min_date).date()}"),
        Check("leakage/calibration_train_only", True, bool(calib_fit_on_train),
              "affine calibration fit on TRAIN rows only"),
    ]


# ======================================================================================
# gate 5 -- GARCH scale (catches the arch x100 error)
# ======================================================================================
def check_garch_scale(diag: pd.DataFrame, ret_sd_by_stock: dict[str, float]) -> list[Check]:
    rows = diag[(diag.model.isin(["garch", "egarch"])) & diag["median_sigma"].notna()]
    if rows.empty:
        return [Check("garch_scale", True, False, "no GARCH sigma diagnostics recorded")]
    ratios = []
    for _, r in rows.iterrows():
        sd = ret_sd_by_stock.get(r["stock"])
        if sd and sd > 0:
            ratios.append(r["median_sigma"] / sd)
    if not ratios:
        return [Check("garch_scale", True, False, "no comparable return sd")]
    med = float(np.median(ratios))
    ok = 0.5 <= med <= 2.0
    return [Check("garch_scale", True, ok,
                  f"median(fitted sigma / empirical return sd) = {med:.3f}, expected in "
                  f"[0.5, 2.0] -- outside implies the arch x100 percentage-scaling error")]


# ======================================================================================
# gate 6 -- determinism
# ======================================================================================
def hash_predictions(df: pd.DataFrame) -> str:
    d = df.sort_values(["stock", "date"]).reset_index(drop=True)
    payload = d[["stock", "y_true", "y_pred"]].round(10).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def check_determinism(hashes: dict[str, str], prev: dict[str, str] | None) -> list[Check]:
    if not prev:
        return [Check("determinism", False, True,
                      "first run -- hashes recorded for future comparison: "
                      + ", ".join(f"{m}={h}" for m, h in sorted(hashes.items())))]
    diffs = [m for m in hashes if prev.get(m) and prev[m] != hashes[m]]
    return [Check("determinism", True, not diffs,
                  "prediction hashes reproduce exactly" if not diffs
                  else f"hash mismatch in: {', '.join(diffs)}")]


# ======================================================================================
# soft checks
# ======================================================================================
def soft_checks(per_stock: pd.DataFrame, diag: pd.DataFrame) -> list[Check]:
    out = []
    g = diag[(diag.model == "garch") & diag["persistence"].notna()]
    if not g.empty:
        p = g["persistence"].values
        frac = float(np.mean((p > 0.5) & (p < 1.0)))
        out.append(Check("garch_persistence", False, frac >= 0.9,
                         f"alpha+beta in (0.5,1.0) for {frac:.1%} of stocks "
                         f"(median {np.median(p):.3f})"))
    for m in ("garch", "egarch"):
        sub = diag[diag.model == m]
        if not sub.empty:
            rate = float(sub["converged"].mean())
            out.append(Check(f"convergence/{m}", False, rate >= 0.95,
                             f"{rate:.1%} converged ({int(sub['converged'].sum())}/{len(sub)})"))
    med = per_stock.groupby("model")["r2"].median()
    if "naive" in med.index:
        worst = med.idxmin()
        note = "(as expected)" if worst == "naive" else "(expected 'naive')"
        out.append(Check("sanity/naive_is_worst", False, worst == "naive",
                         f"lowest median R2 is '{worst}' {note}"))
    if {"har", "ewma"} <= set(med.index):
        out.append(Check("sanity/har_ge_ewma", False, med["har"] >= med["ewma"],
                         f"HAR R2 {med['har']:.4f} vs EWMA {med['ewma']:.4f}"))
    return out


def check_legacy_metric_agreement() -> list[Check]:
    """Recompute metrics with the legacy inline formulas and compare to metrics.py."""
    from metrics import mse as c_mse, mae as c_mae, r2 as c_r2, qlike as c_ql
    rng = np.random.default_rng(0)
    y = rng.normal(-4.3, 0.55, 500); p = y + rng.normal(0, 0.3, 500)
    legacy_r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    legacy_ql = np.mean(2 * p + np.exp(np.clip(2 * y - 2 * p, -20, 20)))
    d = max(abs(legacy_r2 - c_r2(y, p)), abs(legacy_ql - c_ql(y, p)),
            abs(np.mean((y - p) ** 2) - c_mse(y, p)), abs(np.mean(np.abs(y - p)) - c_mae(y, p)))
    return [Check("legacy_metric_agreement", False, d < 1e-9,
                  f"max |canonical - legacy| = {d:.2e} (scratchpad inline formulas)")]


def write_report(checks: list[Check], path: str) -> bool:
    hard_fail = [c for c in checks if c.hard and not c.passed]
    lines = ["VOLATILITY BASELINE HARNESS -- VALIDATION REPORT", "=" * 70, ""]
    for c in checks:
        lines.append(str(c))
    lines += ["", "=" * 70,
              f"hard gates: {sum(1 for c in checks if c.hard and c.passed)}/"
              f"{sum(1 for c in checks if c.hard)} passed",
              f"soft checks: {sum(1 for c in checks if not c.hard and c.passed)}/"
              f"{sum(1 for c in checks if not c.hard)} passed",
              "", "VERDICT: " + ("PASS -- results safe to use"
                                 if not hard_fail else
                                 "FAIL -- DO NOT USE THESE RESULTS: "
                                 + "; ".join(c.name for c in hard_fail))]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return not hard_fail
