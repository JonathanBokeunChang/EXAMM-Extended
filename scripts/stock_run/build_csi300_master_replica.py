#!/usr/bin/env python3
"""Research-grade replica of MASTER's (AAAI'24) CSI300 dataset -- v3.

Reproduces every part of MASTER's data pipeline that is specified in their code
(github.com/SJTU-DMTai/qlib, examples/benchmarks/MASTER + qlib/contrib/data/dataset.py), and fixes
four defects in v2 that each independently invalidated the benchmark. Each fix is listed with the
measurement that motivated it, so the rationale survives without needing the conversation.

WHAT IS REPRODUCED FROM MASTER (verbatim, not reconstructed)
  features   qlib's built-in Alpha158 handler, instruments=csi300
  market     63 features = 3 indices x 21, literal Mask(...) expressions copied from
             qlib/contrib/data/dataset.py::marketDataHandler.get_feature_config()
  label      Ref($close,-5)/Ref($close,-1)-1
  split      train 2008-01-01..2014-12-31 / val 2015-01-01..2016-12-31 / test 2017-01-01..2020-08-01
  processors RobustZScoreNorm(clip_outlier=True)+Fillna on features, fit on TRAIN only;
             CSRankNorm on the training label

DEFECTS FIXED SINCE v2
  1. Universe mismatch. v2 let each eval script pick its own population -- the ridge gate scored
     430 stocks (290/date) while the LSTM/GRU script scored 240 (189/date), so their numbers were
     never comparable. v3 emits eval_index.csv: the frozen (date, instrument) row set that EVERY
     model must score. This is the single most important change.
  2. Gap splicing. 93/430 test stocks (22%) had a >15-day hole where they left and rejoined the
     index -- largest 915 days. An 8-step recurrent window across such a hole silently spans ~900
     calendar days of "recent history". v3 assigns segment_id (break when the trading-day gap
     exceeds --max-gap) and requires SEQ real consecutive days WITHIN one segment for eligibility.
  3. No purge/embargo. v2's train ended 2014-12-31 while a label at that date reads prices ~5
     trading days into the validation window. v3 drops the last H trading days of train and of val
     (standard purging), so no training label can see the next split.
  4. Post-freeze label ranks. v2's CSRankNorm ranked over the pre-freeze qlib universe, not the
     population actually trained/evaluated on. v3 computes it per split over the retained rows.

STILL NOT REPRODUCED (unfixable, disclosed rather than glossed)
  - MASTER's raw feed is confidential and was never released. We use pinned open bundles
    (see datasets/_bundles/*/BUNDLE.json). The fork's own workflow yaml defaults provider_uri to
    qlib's public Yahoo bundle, which qlib documents as NOT the feed behind their published
    numbers either -- so no available bundle is verifiably closer to theirs.
  - Their MASTERTSDatasetH/marketDataHandler wrapper classes are fork-only; this script
    reimplements their logic on vanilla DataHandlerLP+QlibDataLoader. Formulas are verbatim; the
    join is ours.

TWO FEATURE VENUES (--handler), because they favour opposite model families
  alpha158  158 engineered tabular factors -- MASTER's choice. On qlib's own published CSI300
            leaderboard a plain LINEAR model scores Rank IC 0.0472 here, BEATING LightGBM
            (0.0469), MLP (0.0429), LSTM (0.0435), GRU (0.0428), Transformer (0.0407) and TCN
            (0.0421). Human feature engineering has already extracted the structure, leaving
            little for a sequence model to add.
  alpha360  60 lags x 6 raw price/volume fields, each divided by the current close/volume. Same
            leaderboard, same universe: GRU 0.0584, LSTM 0.0549, ALSTM 0.0599 now BEAT LightGBM
            (0.0499) and MLP (0.0396) -- the ordering inverts. This is the venue where recurrent
            architecture actually pays, and therefore the fair venue for a recurrent method.

Use --label qlib (next-day return, H=2) with alpha360 to make results directly comparable to
those published numbers; --label master (H=5) reproduces MASTER's setup instead.

Usage:
    python3 scripts/stock_run/build_csi300_master_replica.py --bundle invdata
    python3 scripts/stock_run/build_csi300_master_replica.py --bundle cndata
    python3 scripts/stock_run/build_csi300_master_replica.py --bundle invdata \
        --handler alpha360 --label qlib --no-market
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN = ("2008-01-01", "2014-12-31")
VAL = ("2015-01-01", "2016-12-31")
TEST = ("2017-01-01", "2020-08-01")
WARMUP_START = "2007-08-01"          # buffer for the 60-day rolling windows both handlers use

# Two label conventions, both standard, chosen with --label:
#   master  Ref($close,-5)/Ref($close,-1)-1  -- MASTER's 4-trading-day forward return (H=5)
#   qlib    Ref($close,-2)/Ref($close,-1)-1  -- the next-day return used by qlib's own public
#           leaderboard, so results are directly comparable to published benchmark numbers
LABEL_EXPRS = {
    "master": ("Ref($close,-5)/Ref($close,-1)-1", 5),
    "qlib": ("Ref($close,-2)/Ref($close,-1)-1", 2),
}

# How much REAL contiguous in-segment history each row needs before it can be evaluated.
#   alpha158  8  -- MASTER's step_len; the model itself windows 8 rows
#   alpha360 60  -- Alpha360 already embeds 60 lags IN the row (Ref($close,i)/$close), so the
#                   history requirement lives here rather than in the model's windowing. Without
#                   this, Ref() would silently reach across a membership gap.
HANDLER_SEQ = {"alpha158": 8, "alpha360": 60, "seq": 60}
SEQ_SENTINEL = -1  # "--seq not given"; resolved from HANDLER_SEQ

# Sequence form: the SAME six fields Alpha360 carries, but one row per day instead of 60 lags
# flattened into the row. Alpha360 anchors every lag to the CURRENT close (Ref($close,i)/$close),
# which encodes the price PATH as levels; per-day features encode CHANGES, and a recurrent model
# integrates changes back into the path. Equivalent information, different encoding -- an honest
# difference, not a claim of identity.
#
# Why this exists: Alpha360's 360 flat columns would give a model 360 INPUTS. For a method whose
# entire thesis is tiny evolved genomes, spending 360 nodes on the input layer concedes the point
# before training starts, and leaves the recurrence with nothing to traverse. qlib's own GRU/LSTM
# sidestep this by reshaping (N,360)->(N,60,6); this handler is the dataset-level equivalent, so a
# recurrent architecture search operates on 6 inputs across 60 timesteps.
# Feature definitions match scripts/stock_run/build_csi300_seq.py, this project's existing
# sequence-form precedent, so the two are directly comparable.
SEQ_FEATURES = [
    ("RET", "$close/Ref($close,1)-1"),        # cross-day return: carries the path
    ("OPEN_C", "$open/$close-1"),             # intraday shape
    ("HIGH_C", "$high/$close-1"),
    ("LOW_C", "$low/$close-1"),
    ("VWAP_C", "$vwap/$close-1"),
    ("VOLR", "Log($volume/Ref($volume,1))"),  # volume dynamics
]

# MASTER's exact index codes, from marketDataHandler.get_feature_config(). These are EXACT and
# never substituted: an earlier version of this script probed a candidate list and fell back to
# the first code carrying data, which silently resolved CSI100 to sz000903 on the invdata bundle
# -- a Shenzhen-listed COMPANY, not the index (return correlation with the real CSI100 was 0.47,
# level ratio 0.10). That produced 21 market features built from an unrelated stock. A missing
# index must therefore fail or be explicitly dropped, never quietly replaced.
MASTER_INDEX_CODES = {
    "csi100": "sh000903",
    "csi300": "sh000300",
    "csi500": "sh000905",
}
WINDOWS = [5, 10, 20, 30, 60]

STOCK_INFER_PROCESSORS = [
    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
]


def market_feature_exprs(idx):
    """MASTER's market features: per universe, the index's own return plus, for each of five
    windows, the mean/std of index return and the mean/std of a volume ratio -- 21 per universe,
    so 3 x (1 + 5x4) = 63 when all three indices are available."""
    names, exprs = [], []
    for uni, code in idx.items():
        names.append(f"MKT_{uni}_RET1")
        exprs.append(f'Mask($close/Ref($close,1)-1, "{code}")')
        for d in WINDOWS:
            for tag, e in [
                (f"RETMEAN{d}", f"Mean($close/Ref($close,1)-1,{d})"),
                (f"RETSTD{d}", f"Std($close/Ref($close,1)-1,{d})"),
                (f"VOLRMEAN{d}", f"Mean($volume,{d})/$volume"),
                (f"VOLRSTD{d}", f"Std($volume,{d})/$volume"),
            ]:
                names.append(f"MKT_{uni}_{tag}")
                exprs.append(f'Mask({e}, "{code}")')
    assert len(names) == 21 * len(idx), (len(names), len(idx))
    return names, exprs


def flatten_cols(df):
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    return df


def resolve_index_codes(D, wanted, on_missing):
    """Verify MASTER's EXACT index codes are present and populated. Never substitutes.

    Returns (resolved, dropped). With on_missing='fail' a missing index aborts the build; with
    'drop' its 21 features are omitted and the omission is recorded in MANIFEST.json, so a
    downstream reader can never mistake a 200-feature dataset for the full 221.
    """
    resolved, dropped = {}, {}
    for uni in wanted:
        code = MASTER_INDEX_CODES[uni]
        try:
            df = D.features([code.upper()], ["$close"], start_time=TRAIN[0],
                             end_time=TEST[1], freq="day")
        except Exception as e:
            df, err = None, str(e)[:80]
        else:
            err = None
        ok = df is not None and len(df) > 1000 and df["$close"].notna().mean() > 0.9
        if ok:
            resolved[uni] = code
            print(f"      {uni:6} -> {code}  ({len(df)} rows)  OK")
            continue
        why = err or (f"only {0 if df is None else len(df)} usable rows")
        if on_missing == "fail":
            sys.exit(f"ERROR: {uni} ({code}) missing/unusable in this bundle ({why}).\n"
                     f"  This bundle cannot reproduce MASTER's market block exactly. Re-run with\n"
                     f"  --on-missing-index drop to omit it (recorded in MANIFEST), or use a\n"
                     f"  bundle that carries {code}. It is NOT substituted with another code.")
        dropped[uni] = f"{code}: {why}"
        print(f"      {uni:6} -> {code}  MISSING ({why}) -- DROPPED, 21 features omitted")
    if not resolved:
        sys.exit("ERROR: no market indices resolved at all")
    return resolved, dropped


def csrank_norm(s, dates):
    """qlib's CSRankNorm, applied post-freeze (qlib/data/dataset/processor.py:326):
    per-date percentile rank, centred at 0.5 and scaled by 3.46 toward unit std.
    Monotonic within each date, so it cannot change what Rank IC measures.

    Returns a plain ndarray, NOT a Series: the caller assigns into df.loc[mask, col], and a
    Series carrying its own RangeIndex would be re-aligned against the frame's (offset) index,
    silently writing NaN for every split whose rows don't start at 0.
    """
    import pandas as pd
    t = pd.Series(s).groupby(dates).rank(pct=True)
    return ((t - 0.5) * 3.46).to_numpy()


def content_hash(out_dir, files):
    """Deterministic digest over the emitted CSV set -- lets a rebuild prove it reproduced."""
    h = hashlib.sha256()
    for fn in sorted(files):
        h.update(fn.encode())
        with open(f"{out_dir}/{fn}", "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                               text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=["invdata", "cndata"], default="invdata")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-gap", type=int, default=5,
                     help="a hole of more than this many TRADING days starts a new segment "
                          "(covers both index exit/re-entry and long trading suspensions)")
    ap.add_argument("--seq", type=int, default=SEQ_SENTINEL,
                     help="consecutive in-segment days required before a row is eval-eligible "
                          "(default: per-handler, see HANDLER_SEQ)")
    ap.add_argument("--indices", default="csi100,csi300,csi500",
                     help="which of MASTER's market universes to include (21 features each)")
    ap.add_argument("--on-missing-index", choices=["fail", "drop"], default="fail",
                     help="what to do when a bundle lacks one of MASTER's exact index codes. "
                          "Never substitutes another code -- see MASTER_INDEX_CODES.")
    ap.add_argument("--handler", choices=["alpha158", "alpha360", "seq"], default="alpha158",
                     help="alpha158 = 158 engineered tabular factors (MASTER's choice; linear/GBDT "
                          "territory). alpha360 = 60 lags x 6 raw price/volume fields, each "
                          "normalised by the current close/volume (recurrent-model territory).")
    ap.add_argument("--label", choices=["master", "qlib"], default="master")
    ap.add_argument("--no-market", action="store_true",
                     help="omit the market block entirely. Default for alpha360, whose published "
                          "leaderboard has no market features -- including them would break "
                          "comparability with those numbers.")
    a = ap.parse_args()
    label_expr, label_h = LABEL_EXPRS[a.label]
    seq = a.seq if a.seq != SEQ_SENTINEL else HANDLER_SEQ[a.handler]
    use_market = not a.no_market
    tag = f"{a.bundle}" if a.handler == "alpha158" else f"{a.bundle}_{a.handler}"
    out = a.out or f"{REPO}/datasets/csi300_master_replica_{tag}"

    import numpy as np
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN

    bj = f"{REPO}/datasets/_bundles/{a.bundle}/BUNDLE.json"
    if not os.path.exists(bj):
        sys.exit(f"ERROR: {bj} missing. Run:\n"
                 f"  python3 scripts/stock_run/acquire_bundle.py --bundle {a.bundle}")
    bundle = json.load(open(bj))
    provider_uri = bundle["provider_uri"]
    print(f"[bundle] {a.bundle}  {provider_uri}")
    print(f"         pinned id: {bundle.get('archive_sha256') or bundle.get('content_sha256')}")

    qlib.init(provider_uri=provider_uri, region=REG_CN, kernels=1)
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandlerLP

    print(f"[config] handler={a.handler}  label={a.label} ({label_expr}, H={label_h})  "
          f"seq={seq}  market_block={'on' if use_market else 'off'}")

    print("[1/9] verifying MASTER's exact market index codes in this bundle...")
    if use_market:
        idx, dropped_idx = resolve_index_codes(D, a.indices.split(","), a.on_missing_index)
    else:
        idx, dropped_idx = {}, {}
        print("      skipped (--no-market)")

    print(f"[2/9] {a.handler} features (fit on TRAIN) + raw label...")
    if a.handler == "alpha158":
        # RobustZScoreNorm+Fillna, matching workflow_config_master_Alpha158.yaml
        handler = Alpha158(
            instruments="csi300", start_time=WARMUP_START, end_time=TEST[1],
            fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
            infer_processors=STOCK_INFER_PROCESSORS,
            learn_processors=[],       # label kept RAW here; ranks are computed post-freeze
            label=[label_expr],
        )
    elif a.handler == "seq":
        # Same ProcessInf -> ZScoreNorm -> Fillna stack Alpha360 uses, so the sequence-form and
        # flat-form datasets differ ONLY in encoding, never in preprocessing.
        names = [n for n, _ in SEQ_FEATURES]
        exprs = [e for _, e in SEQ_FEATURES]
        handler = DataHandlerLP(
            instruments="csi300", start_time=WARMUP_START, end_time=TEST[1],
            data_loader={"class": "QlibDataLoader",
                         "kwargs": {"config": {"feature": (exprs, names),
                                                "label": ([label_expr], ["LABEL0"])}}},
            infer_processors=[
                {"class": "ProcessInf"},
                {"class": "ZScoreNorm", "kwargs": {"fit_start_time": TRAIN[0],
                                                    "fit_end_time": TRAIN[1],
                                                    "fields_group": "feature"}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[],
        )
    else:
        from qlib.contrib.data.handler import Alpha360
        # Alpha360's OWN default infer_processors (ProcessInf -> ZScoreNorm -> Fillna), which is
        # what qlib's published Alpha360 leaderboard runs. Deliberately NOT the Alpha158 stack:
        # matching the official preprocessing is what makes our numbers comparable to theirs.
        handler = Alpha360(
            instruments="csi300", start_time=WARMUP_START, end_time=TEST[1],
            fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
            learn_processors=[],
            label=[label_expr],
        )
    df = flatten_cols(handler.fetch(data_key=DataHandlerLP.DK_I))
    df = df.rename(columns={df.columns[-1]: "LABEL"})
    feat_cols = [c for c in df.columns if c != "LABEL"]
    print(f"      {df.shape}  ({len(feat_cols)} {a.handler} features)")

    print(f"[3/9] market block ({21 * len(idx)} features, RobustZScoreNorm fit on TRAIN)...")
    mkt = None
    mkt_names = []
    if use_market:
        mkt_names, mkt_exprs = market_feature_exprs(idx)
        mkt_handler = DataHandlerLP(
            instruments=[idx["csi300"].upper()],   # Mask() ignores the queried instrument
            start_time=WARMUP_START, end_time=TEST[1],
            data_loader={"class": "QlibDataLoader",
                         "kwargs": {"config": {"feature": (mkt_exprs, mkt_names)}}},
            infer_processors=[
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fields_group": "feature", "clip_outlier": True,
                            "fit_start_time": TRAIN[0], "fit_end_time": TRAIN[1]}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
        )
        mkt = flatten_cols(mkt_handler.fetch(data_key=DataHandlerLP.DK_I))
        mkt = mkt.reset_index(level="instrument", drop=True)
        print(f"      {mkt.shape}  ({len(mkt_names)} market features)")
    else:
        print("      skipped (--no-market)")

    print("[4/9] merging on date...")
    assert df.index.names == ["datetime", "instrument"], df.index.names
    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])
    if mkt is not None:
        assert mkt.index.name == "datetime", mkt.index.name
        mkt = mkt.reset_index()
        mkt["datetime"] = pd.to_datetime(mkt["datetime"])
        df = df.merge(mkt, on="datetime", how="left")
    all_feat = feat_cols + mkt_names
    df = df[(df["datetime"] >= TRAIN[0]) & (df["datetime"] <= TEST[1])]
    n0 = len(df)
    df = df.dropna(subset=all_feat + ["LABEL"]).reset_index(drop=True)
    print(f"      {n0} -> {len(df)} rows after dropna")

    print("[5/9] purge/embargo at split boundaries...")
    # ORDER MATTERS: purge BEFORE assigning segments. Purging removes the last H trading days of
    # train and of val, which punches a hole in the middle of the panel. If segments were assigned
    # first, that hole would sit INSIDE an already-assigned segment and silently violate the
    # contiguity invariant -- a model could then window across removed days. The alpha158/alpha360
    # builds happened not to trip this (their segments didn't straddle a purged boundary); the
    # sequence-form build did, and the intra-segment-gap assertion caught it.
    cal = [pd.Timestamp(d) for d in D.calendar(start_time=TRAIN[0], end_time=TEST[1], freq="day")]
    cal_idx = {d: i for i, d in enumerate(cal)}
    df["_ci"] = df["datetime"].map(cal_idx)
    if df["_ci"].isna().any():
        sys.exit("ERROR: rows carry dates absent from the trading calendar")
    val_first = min(i for d, i in cal_idx.items() if d >= pd.Timestamp(VAL[0]))
    test_first = min(i for d, i in cal_idx.items() if d >= pd.Timestamp(TEST[0]))
    ci = df["_ci"]
    is_train = ci < val_first
    is_val = (ci >= val_first) & (ci < test_first)
    is_test = ci >= test_first
    keep_train = is_train & (ci + label_h < val_first)
    keep_val = is_val & (ci + label_h < test_first)
    purged = int((is_train & ~keep_train).sum() + (is_val & ~keep_val).sum())
    df["split"] = np.where(keep_train, "train", np.where(keep_val, "val",
                            np.where(is_test, "test", "PURGED")))
    print(f"      purged {purged} rows whose label horizon (H={label_h}) reached into the "
          f"next split")
    df = df[df["split"] != "PURGED"].reset_index(drop=True)

    print("[6/9] assigning segments (gap-safe, AFTER purge) ...")
    df = df.sort_values(["instrument", "_ci"]).reset_index(drop=True)
    step = df.groupby("instrument")["_ci"].diff()
    new_seg = (step.isna()) | (step > a.max_gap + 1)   # >max_gap MISSING trading days
    df["segment_id"] = new_seg.groupby(df["instrument"]).cumsum().astype(int)
    nseg = df.groupby("instrument")["segment_id"].max()
    print(f"      {int((nseg > 1).sum())}/{len(nseg)} stocks are split into >1 segment "
          f"(max {int(nseg.max())} segments); gap threshold {a.max_gap} trading days")

    print("[7/9] CSRankNorm on the label, computed per split over RETAINED rows...")
    df["LABEL_CSRANK"] = np.nan
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        df.loc[m, "LABEL_CSRANK"] = csrank_norm(df.loc[m, "LABEL"].values,
                                                 df.loc[m, "datetime"].values)
    assert df["LABEL_CSRANK"].notna().all()

    print("[8/9] freezing the evaluation index (strict contiguity)...")
    # position of each row within its (instrument, segment); a test row is eligible only when SEQ
    # real consecutive in-segment trading days end at it -- so any window a model builds is genuine
    # observed history, never spliced across a gap and never synthetically padded.
    df["_pos"] = df.groupby(["instrument", "segment_id"]).cumcount()
    elig = (df["split"] == "test") & (df["_pos"] >= seq - 1)
    ev = (df.loc[elig, ["datetime", "instrument", "segment_id"]]
            .rename(columns={"datetime": "date"})
            .sort_values(["date", "instrument"]).reset_index(drop=True))
    n_test_rows = int((df["split"] == "test").sum())
    print(f"      eval rows {len(ev)} of {n_test_rows} test rows "
          f"({n_test_rows - len(ev)} dropped for <{seq} contiguous in-segment days)")
    print(f"      {ev['instrument'].nunique()} stocks, {ev['date'].nunique()} dates, "
          f"median {int(ev.groupby('date').size().median())} stocks/date")

    print("[9/9] self-checks (hard-fail)...")
    chk = all_feat + ["LABEL", "LABEL_CSRANK"]
    assert df[chk].isna().sum().sum() == 0, "NaNs present"
    assert np.isfinite(df[chk].to_numpy()).all(), "non-finite values present"
    absf = np.abs(df[all_feat].to_numpy())
    fmax = float(absf.max())
    outlier_stats = {f"pct_abs_gt_{t}": float((absf > t).mean()) for t in (3, 10, 50, 200)}
    outlier_stats["max_abs"] = fmax
    if a.handler == "alpha158":
        # RobustZScoreNorm(clip_outlier=True) hard-bounds to [-3,3]; anything outside means the
        # processor did not run as configured.
        assert fmax <= 3.0 + 1e-6, f"RobustZScoreNorm clip violated (max |z| = {fmax})"
    else:
        # Alpha360 uses qlib's plain ZScoreNorm, which does NOT clip -- kept deliberately, because
        # the published Alpha360 leaderboard we calibrate against ran this exact preprocessing.
        # Large values are structural, not corruption: the worst offenders are VOLUME ratios
        # (Ref($volume,k)/$volume) on near-zero-volume days, e.g. a resumption from suspension.
        # Measured on 2011-2014: |z|>50 in 0.00004% of values, |z|>200 in a single value.
        # Fail only on magnitudes that would indicate genuinely broken prices.
        assert fmax < 5000, f"implausible magnitude (max |z| = {fmax}) -- suspect broken prices"
        print(f"      no clipping (ZScoreNorm, faithful to the published Alpha360 pipeline); "
              f"max |z| {fmax:.0f}, "
              f"|z|>50 in {100*outlier_stats['pct_abs_gt_50']:.6f}% of values")
    assert df["LABEL_CSRANK"].abs().max() <= 1.73 + 1e-6, "CSRankNorm range violated"
    # CSRankNorm must preserve within-date rank order, or evaluation and training would be
    # measuring different things. Verified empirically, not assumed from the formula.
    corr = []
    for _, g in df[df["split"] == "train"].groupby("datetime"):
        if len(g) >= 5 and g["LABEL"].std() > 1e-12:
            corr.append(g[["LABEL", "LABEL_CSRANK"]].corr(method="spearman").iloc[0, 1])
    assert np.nanmin(corr) > 0.9999, f"CSRankNorm broke rank order (min {np.nanmin(corr)})"
    print(f"      rank-order: min per-date Spearman(LABEL, LABEL_CSRANK) = {np.nanmin(corr):.6f}")
    # no intra-segment gap may exceed the threshold
    d2 = df.sort_values(["instrument", "segment_id", "_ci"])
    gap = d2.groupby(["instrument", "segment_id"])["_ci"].diff()
    assert (gap.dropna() <= a.max_gap + 1).all(), f"intra-segment gap exceeds {a.max_gap}"
    # purge actually holds
    assert (df.loc[df["split"] == "train", "_ci"] + label_h < val_first).all()
    assert (df.loc[df["split"] == "val", "_ci"] + label_h < test_first).all()
    print(f"      purge verified; no intra-segment gap > {a.max_gap} trading days")

    print("[export] per-stock CSVs + eval_index.csv + MANIFEST.json...")
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(f"{out}/{f}")       # stale files would corrupt the content hash
    os.makedirs(out, exist_ok=True)
    cols = ["date", "segment_id"] + all_feat + ["LABEL", "LABEL_CSRANK"]
    written = []
    counts = {"train": 0, "val": 0, "test": 0}
    for (s, sp), g in df.groupby(["instrument", "split"]):
        g = (g.rename(columns={"datetime": "date"})[cols]
               .sort_values("date"))
        fn = f"{s}_{sp}.csv"
        # %.9g round-trips float32 exactly (qlib stores float32), so this is lossless, not a
        # precision shortcut -- a previous 6-sig-fig CSV defect in this project makes that worth
        # stating explicitly.
        g.to_csv(f"{out}/{fn}", index=False, float_format="%.9g")
        written.append(fn)
        counts[sp] += 1
    ev.to_csv(f"{out}/eval_index.csv", index=False)
    written.append("eval_index.csv")
    print(f"      {len(written)-1} per-stock files  (train {counts['train']}, "
          f"val {counts['val']}, test {counts['test']} stocks)")

    ch = content_hash(out, written)
    manifest = {
        "dataset": "MASTER AAAI24 replica v3 (research-grade)",
        "builder_git_sha": git_sha(),
        "bundle": {k: bundle.get(k) for k in
                   ("bundle", "release_tag", "archive_sha256", "content_sha256",
                    "target_trade_date", "provider_uri")},
        "qlib_version": getattr(qlib, "__version__", "unknown"),
        "index_codes_resolved": idx,
        "index_codes_dropped": dropped_idx,
        "market_block_complete": use_market and len(dropped_idx) == 0,
        "market_block_included": use_market,
        "n_features": len(all_feat),
        "n_alpha158": len(feat_cols),
        "n_market": len(mkt_names),
        "handler": a.handler,
        "label_kind": a.label,
        "label": label_expr,
        "label_horizon_trading_days": label_h,
        "seq_for_eligibility": seq,
        "max_gap_trading_days": a.max_gap,
        "splits": {"train": list(TRAIN), "val": list(VAL), "test": list(TEST)},
        "rows": {sp: int((df["split"] == sp).sum()) for sp in ("train", "val", "test")},
        "stocks": counts,
        "purged_rows": purged,
        "eval_index": {
            "rows": len(ev),
            "stocks": int(ev["instrument"].nunique()),
            "dates": int(ev["date"].nunique()),
            "median_stocks_per_date": int(ev.groupby("date").size().median()),
        },
        "feature_outlier_stats": outlier_stats,
        "content_sha256": ch,
        "float_format": "%.9g (lossless round-trip for the float32 qlib source)",
        "known_deviations": [
            "Raw feed is NOT MASTER's confidential source (unfixable).",
            "MASTERTSDatasetH/marketDataHandler reimplemented on vanilla qlib; formulas verbatim.",
        ],
    }
    with open(f"{out}/MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"      content_sha256 = {ch}")
    print(f"DONE -> {out}")


if __name__ == "__main__":
    sys.exit(main())
