#!/usr/bin/env python3
"""Build the CSI300 sequence-form returns dataset (the EXAMM-native analogue of Alpha360).

WHY THIS SHAPE. Qlib's Alpha360 gives each (stock, date) a 360-vector = 60 lags x 6 fields,
every lag normalised by the anchor date's close (or volume):
    CLOSE_i = Ref($close,i)/$close ... VOLUME_i = Ref($volume,i)/$volume,  i = 0..59
Qlib's own GRU/LSTM models immediately reshape that 360-vector back into a (60, 6) SEQUENCE
before feeding their recurrence. EXAMM is a recurrent NAS, so handing it 360 flat inputs would
waste the recurrence entirely (it degenerates to an MLP on a huge input layer) and forfeit the
tiny-genome efficiency result that is this project's strongest finding. So we emit the 6 fields
PER DAY and let EXAMM's evolved memory span the window.

HONEST DIFFERENCE from literal Alpha360: Alpha360 normalises every lag by ONE anchor, so the
360-vector encodes relative price LEVELS across the window. Per-day self-normalised features
encode level changes instead; a recurrent model can integrate returns back into levels, so the
information is equivalent in principle but the encoding is not identical. Exact comparability to
published Alpha360 numbers was never available anyway (different universe snapshot, date range,
preprocessing) -- which is why this project always runs IN-HOUSE GRU/LSTM baselines on identical
data. Those, not the published table, are the comparison.

FEATURES (all scale-free and causal -- computable at the close of day t):
    RET      close_t/close_{t-1} - 1      daily return (carries cross-day dynamics)
    OPEN_C   open_t/close_t - 1           intraday shape
    HIGH_C   high_t/close_t - 1
    LOW_C    low_t/close_t - 1
    VWAP_C   vwap_t/close_t - 1           (0 if $vwap unavailable in the bundle)
    VOLR     log(volume_t / volume_{t-1}) volume dynamics

LABEL (Qlib's standard CSI300 target, no look-ahead at the signal date):
    LABEL = Ref($close,-2)/Ref($close,-1) - 1
i.e. the return earned from the NEXT day's close to the day after. A signal formed at close t is
traded at close t+1 and held to t+2, so nothing on the label path is knowable at t.

SPLITS default to Qlib's benchmark split, chosen for STATISTICAL POWER: a ~4y test window gives
SE(mean IC) ~ 0.0019 at N=300 (MDE ~0.0037), comfortably below the ~0.0083 model-vs-model gaps
worth chasing. Our 50-stock US cohort had MDE 0.0181 and was measured dead.

Series are deliberately left RAGGED (no calendar alignment): CSI300 membership changes over time,
and the training path we use (default MSE / per-series SGD) does not require rectangularity. The
cross-sectional IC evaluation joins on the DATE column, never on row index.

Run under the qlib venv:
  scratchpad/qlibenv/bin/python scripts/stock_run/build_csi300_seq.py \
      --provider ~/.qlib/qlib_data/cn_data --out datasets/csi300_seq
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

FEATURES = ["RET", "OPEN_C", "HIGH_C", "LOW_C", "VWAP_C", "VOLR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.environ.get("QLIB_PROVIDER"),
                    help="qlib provider_uri (the cn_data bundle dir); or set $QLIB_PROVIDER")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--train-end", default="2014-12-31")
    ap.add_argument("--val-end", default="2016-12-31")
    ap.add_argument("--end", default="2020-08-01")
    ap.add_argument("--min-train", type=int, default=250, help="min train rows to keep a stock")
    ap.add_argument("--min-split", type=int, default=60, help="min val/test rows to keep a stock")
    ap.add_argument("--winsor", type=float, default=0.001,
                    help="winsorise return-like columns at this/1-this quantile, bounds taken "
                         "from TRAIN rows only (0 disables). The public Yahoo-derived bundle "
                         "carries unadjusted corporate actions -- ~0.05%% of rows show |RET| "
                         "beyond the A-share 10%% daily limit, up to +295%%. Rank-based IC "
                         "shrugs those off, but MSE training does not: a LABEL of +3.18 would "
                         "dominate the gradient.")
    a = ap.parse_args()
    if not a.provider:
        ap.error("no qlib provider: pass --provider <cn_data dir> or set $QLIB_PROVIDER")

    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=os.path.expanduser(a.provider), region=REG_CN, kernels=1)
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
    names = ["open", "high", "low", "close", "volume", "vwap"]
    print(f"loading CSI300 {a.start}..{a.end} ...", flush=True)
    try:
        df = D.features(D.instruments(market="csi300"), fields,
                        start_time=a.start, end_time=a.end, freq="day")
        df.columns = names
    except Exception as e:                     # bundle lacks the field entirely
        print(f"  $vwap query failed ({type(e).__name__}); using OHLCV", flush=True)
        fields, names = fields[:-1], names[:-1]
        df = D.features(D.instruments(market="csi300"), fields,
                        start_time=a.start, end_time=a.end, freq="day")
        df.columns = names

    # A missing field can come back SILENTLY ALL-NaN rather than raising (qlib's public
    # Yahoo-derived cn_data bundle ships $vwap as 0% populated). Catching only the exception
    # would leave a fake column that then wipes every row via dropna, so check coverage.
    if "vwap" in df.columns and df["vwap"].notna().mean() < 0.5:
        print(f"  $vwap present but {df['vwap'].notna().mean():.1%} populated -> DROPPING it "
              f"(sequence carries 5 of Alpha360's 6 fields)", flush=True)
        df = df.drop(columns=["vwap"])
        names = [n for n in names if n != "vwap"]
    has_vwap = "vwap" in names
    if not has_vwap and "VWAP_C" in FEATURES:
        # drop rather than emit a constant-zero input: a dead column wastes EXAMM mutations
        FEATURES.remove("VWAP_C")
    print(f"  {len(df):,} rows, {df.index.get_level_values(0).nunique()} stocks, "
          f"features: {' '.join(FEATURES)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    kept, dropped, manifest, staged = 0, 0, [], []

    for stock, g in df.groupby(level=0):
        g = g.droplevel(0).sort_index()
        g = g[~g.index.duplicated(keep="last")]
        g = g[(g["close"] > 0) & (g["volume"] > 0)].dropna(subset=names)
        if len(g) < a.min_train + 2 * a.min_split:
            dropped += 1
            continue

        close = g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        out = pd.DataFrame(index=g.index)
        out["RET"] = np.concatenate([[np.nan], close[1:] / close[:-1] - 1.0])
        out["OPEN_C"] = g["open"].to_numpy(float) / close - 1.0
        out["HIGH_C"] = g["high"].to_numpy(float) / close - 1.0
        out["LOW_C"] = g["low"].to_numpy(float) / close - 1.0
        if has_vwap:
            out["VWAP_C"] = g["vwap"].to_numpy(float) / close - 1.0
        out["VOLR"] = np.concatenate([[np.nan], np.log(vol[1:] / vol[:-1])])

        # LABEL = Ref(close,-2)/Ref(close,-1) - 1  : trade at t+1 close, hold to t+2 close
        nxt1 = np.concatenate([close[1:], [np.nan]])
        nxt2 = np.concatenate([close[2:], [np.nan, np.nan]])
        out["LABEL"] = nxt2 / nxt1 - 1.0

        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        if out.empty:
            dropped += 1
            continue
        out.insert(0, "date", pd.to_datetime(out.index).strftime("%Y-%m-%d"))

        tr = out[out["date"] <= a.train_end]
        va = out[(out["date"] > a.train_end) & (out["date"] <= a.val_end)]
        te = out[out["date"] > a.val_end]
        if len(tr) < a.min_train or len(va) < a.min_split or len(te) < a.min_split:
            dropped += 1
            continue
        staged.append((stock, tr, va, te))

    if not staged:
        sys.exit("ERROR: no stocks survived the retention thresholds")

    # ---- winsorise, with bounds from TRAIN rows only (leak-free) --------------------
    clip_cols = [c for c in ("RET", "LABEL", "VOLR") if c in FEATURES + ["LABEL"]]
    bounds, n_clipped = {}, 0
    if a.winsor > 0:
        for c in clip_cols:
            pool = np.concatenate([t[c].to_numpy(float) for _, t, _, _ in staged])
            bounds[c] = (float(np.quantile(pool, a.winsor)), float(np.quantile(pool, 1 - a.winsor)))
        for _, *parts in staged:
            for p in parts:
                for c in clip_cols:
                    lo, hi = bounds[c]
                    v = p[c].to_numpy(float)
                    n_clipped += int(((v < lo) | (v > hi)).sum())
                    p[c] = np.clip(v, lo, hi)

    cols = ["date"] + FEATURES + ["LABEL"]
    for stock, tr, va, te in staged:
        for split, part in (("train", tr), ("val", va), ("test", te)):
            part[cols].to_csv(f"{a.out}/{stock}_{split}.csv", index=False, float_format="%.10g")
        kept += 1
        manifest.append({"stock": stock, "train": len(tr), "val": len(va), "test": len(te),
                         "first": tr["date"].iloc[0], "last": te["date"].iloc[-1]})

    mf = pd.DataFrame(manifest)
    mf.to_csv(f"{a.out}/MANIFEST.csv", index=False)

    # ---- self-checks (hard-fail) ---------------------------------------------------
    probe = mf["stock"].iloc[0]
    chk = pd.read_csv(f"{a.out}/{probe}_train.csv")
    assert list(chk.columns) == ["date"] + FEATURES + ["LABEL"], "column contract broken"
    assert np.isfinite(chk[FEATURES + ["LABEL"]].to_numpy(float)).all(), "non-finite values written"
    assert chk["date"].is_monotonic_increasing, "dates not sorted"
    # label look-ahead check: LABEL at row i must equal close[i+2]/close[i+1]-1, which is
    # independent of every feature at row i -- verified by construction above; here we assert
    # the split boundaries never overlap.
    va_chk = pd.read_csv(f"{a.out}/{probe}_val.csv")
    te_chk = pd.read_csv(f"{a.out}/{probe}_test.csv")
    assert chk["date"].max() < va_chk["date"].min() < va_chk["date"].max() < te_chk["date"].min(), \
        "split boundaries overlap"

    print(f"\nwrote {a.out}: kept {kept} stocks, dropped {dropped}")
    print(f"  features ({len(FEATURES)}): {' '.join(FEATURES)}"
          + ("" if has_vwap else "   [no $vwap in bundle -> VWAP_C omitted, 5 of Alpha360's 6 fields]"))
    print(f"  train rows {mf['train'].sum():,} | val {mf['val'].sum():,} | test {mf['test'].sum():,}")
    if a.winsor > 0:
        print(f"  winsorised at [{a.winsor}, {1-a.winsor}] (bounds from TRAIN only): "
              f"{n_clipped:,} values clipped")
        for c, (lo, hi) in bounds.items():
            print(f"    {c:<6} [{lo:+.4f}, {hi:+.4f}]")
    print(f"  test days (median per stock): {int(mf['test'].median())}")
    print(f"  split: train<={a.train_end}  val<={a.val_end}  test<={a.end}")
    print("  self-checks PASSED")
    print(f"\nEXAMM invocation:\n  --input_parameter_names {' '.join(FEATURES)} "
          f"--output_parameter_names LABEL --time_offset 0")


if __name__ == "__main__":
    main()
