#!/usr/bin/env python3
"""Zero-shot foundation-model forecasts on the portfolio cells, in the repo's predictions.csv shape.

    external/zs_env/bin/python scripts/transformer_bench/zeroshot_probe.py \
        --set set1 --cohort cohort_2020_aligned --model Salesforce/moirai-1.1-R-base

WHAT THIS IS FOR. Every trained model here is fit on this universe; a foundation model is not fit at
all. The question is whether a general-purpose forecaster, with no exposure to these stocks, ranks
them well enough to trade. The output is written in exactly the layout score_cells.py already reads,
so the scoring path -- ensembling, rank IC, Algorithm 2 via the verified trader -- is shared with
every other model rather than reimplemented here.

READ DISPERSION AND GATE DAYS BEFORE IC. The expected failure is not a low IC, it is a DEGENERATE
forecast: daily returns have a conditional mean near zero, and a well-calibrated forecaster asked
for tomorrow's return will emit ~0 for every stock every day. Cross-sectional dispersion then
collapses, ranking becomes arbitrary, and Algorithm 2's gate never opens -- which is exactly how
Crossformer failed on this universe (sd 2e-6, gate 0/249). A near-zero dispersion invalidates the
IC number regardless of its sign, so it is printed first and prominently.

CONTEXT LENGTH IS 96 TO MATCH THE PROTOCOL, not because it is best for this model. Every transformer
in this study gets L=96, so the comparison is like-for-like at 96. MOIRAI accepts far longer context
and would likely prefer it; --context is exposed so that can be tested separately, but a longer
context is a DIFFERENT experiment and must be reported as one.

PROVENANCE. Model code is the PyPI wheel `uni2ts==2.0.0` (the current release, 2025-11-04), with
torch 2.4.1 / gluonts 0.14.4; weights are the HF checkpoint at the revision recorded per run in
timing.json. The authors' `cli/eval.py` harness is deliberately NOT used: it is built around their
benchmark registries (Monash/LSF/PF) and their metric suite, whereas this study scores cross-
sectional rank IC and Algorithm 2 trading. Their tested INFERENCE path is used unchanged
(`MoiraiForecast.create_predictor`), so the forecasting behaviour is theirs and only the data
plumbing is ours -- and MOIRAI reaches the scorer by the identical route as every trained model.

NO AUTHOR DEFAULT EXISTS FOR context_length OR patch_size. In the authors' own eval configs both are
`???`, i.e. required-without-default, because they tune them per benchmark; only `num_samples: 100`
is fixed across their model configs. So --context is a choice this study must make and justify, not
a setting inherited from upstream. num_samples is deliberately NOT held at their 100: it is a Monte
Carlo estimation parameter rather than a model hyperparameter, and 100 has been measured here to be
short of convergence for cross-sectional IC (+0.0182 at 100 -> +0.0216 at 500 on set1/2022).

ALIGNMENT. The trained runs emit 250 predictions against a 251-row test file: one for every test day
except the first, each forecasting that day's RET from history ending the previous trading day, with
the 96-day context borrowing from the tail of the val file at the start of the window. That is
reproduced exactly and then asserted, because an off-by-one here would produce a plausible-looking
IC that is silently measuring the wrong day.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_REV = "unresolved"
DATA = os.path.join(REPO, "datasets/walkforward/mid_highmid_price")


def _ver(mod):
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception:
        return "?"


def build_windows(data_dir, ctx):
    """(windows, index) for one cell. windows[i] is the ctx-length RET history ending the day
    before index[i]'s date; index[i] carries that date's realised RET as the truth."""
    wins, rows = [], []
    for tf in sorted(glob.glob(os.path.join(data_dir, "*_test.csv"))):
        tic = os.path.basename(tf)[:-len("_test.csv")]
        te = pd.read_csv(tf)
        va = pd.read_csv(os.path.join(data_dir, f"{tic}_val.csv"))
        hist = np.concatenate([va["RET"].to_numpy(float), te["RET"].to_numpy(float)])
        off = len(va)                      # index in `hist` of test row 0
        for i in range(1, len(te)):        # every test day except the first
            end = off + i                  # exclusive: history through test day i-1
            if end - ctx < 0:
                raise SystemExit(f"{tic}: only {end} observations before test row {i}, need {ctx}")
            wins.append(hist[end - ctx:end])
            rows.append((tic, te["date"].iloc[i], float(te["RET"].iloc[i])))
    return np.asarray(wins, dtype=np.float32), pd.DataFrame(
        rows, columns=["ticker", "date", "expected_RET"])


def forecast(wins, model_id, ctx, batch, samples, device):
    """(array, kind) for the whole cell. kind is "samples" or "quantiles" depending on family.

    THE gluonts PREDICTOR PATH IS USED DELIBERATELY. MoiraiForecast.forward() is public and takes
    plain tensors, but driving it directly desynchronises the patch/mask bookkeeping (observed_mask
    ends up one patch shorter than the target) -- create_predictor is the path uni2ts actually
    tests, and it owns that bookkeeping.

    TWO FAMILIES, TWO OUTPUT SHAPES. Moirai 1.x is trained with a distributional loss and returns a
    SampleForecast: n_samples draws to be collapsed into a point estimate. Moirai 2.x replaced that
    with a QUANTILE loss and returns a QuantileForecast carrying nine levels, 0.1 to 0.9 -- it does
    not sample at all, and Moirai2Forecast takes neither num_samples nor patch_size. So `--samples`
    is silently inert for 2.x, and that is a property of the model, not an oversight.

    NEVER READ .mean OFF A QuantileForecast. gluonts does not store a mean for that type and
    returns the MEDIAN instead, with only a warning -- which would silently collapse the very
    mean-vs-median distinction this study is measuring.

    Each window is its own one-item series. Observations are trading days treated as a regular
    sequence, which is the same assumption every other model in this study makes about its input.
    """
    from gluonts.dataset.common import ListDataset
    global CKPT_REV
    is_v2 = "moirai-2" in model_id.lower()
    if is_v2:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        module = Moirai2Module.from_pretrained(model_id)
        model = Moirai2Forecast(module=module, prediction_length=1, context_length=ctx,
                                target_dim=1, feat_dynamic_real_dim=0,
                                past_feat_dynamic_real_dim=0)
    else:
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        module = MoiraiModule.from_pretrained(model_id)
        model = MoiraiForecast(module=module, prediction_length=1, context_length=ctx, target_dim=1,
                               feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                               patch_size="auto", num_samples=samples)
    try:                       # resolved commit of the downloaded snapshot, not just the tag
        import huggingface_hub as hh
        CKPT_REV = hh.HfApi().model_info(model_id).sha
    except Exception:
        CKPT_REV = "unresolved"

    ds = ListDataset([{"start": pd.Period("2000-01-03", "D"), "target": w} for w in wins], freq="D")
    predictor = model.create_predictor(batch_size=batch, device=device)
    out, kind, levels = None, None, None
    t0 = time.time()
    for i, fc in enumerate(predictor.predict(ds)):
        if out is None:
            if hasattr(fc, "forecast_array") and getattr(fc, "forecast_keys", None):
                kind, levels = "quantiles", [float(k) for k in fc.forecast_keys]
                out = np.empty((len(wins), len(levels)), dtype=np.float64)
            else:
                kind = "samples"
                out = np.empty((len(wins), samples), dtype=np.float64)
        if kind == "quantiles":
            out[i] = np.asarray(fc.forecast_array).reshape(len(levels), -1)[:, 0]
        else:
            out[i] = np.asarray(fc.samples).reshape(-1)[:samples]
        if (i + 1) % (batch * 10) == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(wins)}  {el:.0f}s  eta {el/(i+1)*(len(wins)-i-1):.0f}s", flush=True)
    return out, kind, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="st", default="set1")
    ap.add_argument("--cohort", default="cohort_2020_aligned")
    ap.add_argument("--model", default="Salesforce/moirai-1.1-R-base")
    ap.add_argument("--label", default=None, help="directory name; defaults from --model")
    ap.add_argument("--context", type=int, default=96)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows (timing probe)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch seed. MOIRAI's forecast is SAMPLED, so this changes the numbers -- "
                         "it is a real replicate axis here, not a formality.")
    ap.add_argument("--save-raw", default=None,
                    help="write the raw sample/quantile matrix as .npy (for the convergence curve)")
    ap.add_argument("--root", default=os.path.join(REPO, "results/zeroshot/mid_highmid"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    label = a.label or a.model.split("/")[-1].replace(".", "").replace("-", "")
    data_dir = os.path.join(DATA, a.st, a.cohort)
    wins, idx = build_windows(data_dir, a.context)
    if a.limit:
        wins, idx = wins[:a.limit], idx.iloc[:a.limit].copy()
    ntic = idx.ticker.nunique()
    print(f"{label} · {a.st}/{a.cohort} · {ntic} tickers · {len(wins)} windows · ctx {a.context}")
    if ntic != 50 and not a.limit:
        raise SystemExit(f"expected 50 tickers, got {ntic}")

    t0 = time.time()
    arr, kind, levels = forecast(wins, a.model, a.context, a.batch, a.samples, a.device)
    wall = time.time() - t0
    if a.save_raw:
        np.save(a.save_raw, arr.astype(np.float32))
        print(f"  raw {kind} matrix {arr.shape} -> {a.save_raw}")
    # BOTH point estimates. On 1.x the predictive distribution over a noisy return series is
    # fat-tailed, so the sample MEAN and MEDIAN rank the cross-section differently and the choice
    # is load-bearing; both are written and scored rather than picked after the fact. The authors
    # report both too (their metric list carries MSE and MedianMSE).
    #
    # ON 2.x THE "MEAN" IS TAIL-TRIMMED AND CANNOT BE OTHERWISE. Only the 0.1-0.9 quantiles are
    # emitted, so averaging them approximates the mean of the central 80% only -- the tail mass is
    # absent from the model output by construction. That is exactly the region carrying 1.x's
    # signal, so the two families' "mean" columns are NOT the same estimator and must not be read
    # as one. The suffix records which is which.
    if kind == "quantiles":
        q = np.asarray(levels)
        pts = {"": arr.mean(axis=1), "_med": arr[:, int(np.argmin(np.abs(q - 0.5)))]}
    else:
        pts = {"": arr.mean(axis=1), "_med": np.median(arr, axis=1)}

    if a.limit:
        print(f"  LIMIT RUN -- {len(wins)} windows in {wall:.0f}s "
              f"({wall/len(wins)*12500/60:.1f} min projected for a full cell); nothing written")
        return
    # Per-ticker prediction count is len(test)-1 and DIFFERS BY COHORT (250/249/251 for trade
    # 2022/2023/2024), so it is derived from the data rather than hardcoded -- a fixed 250 silently
    # killed every non-2022 cell. What must hold universally is n_price == n_pred + 1, which is the
    # relation trade_portfolio.py's Portfolio class requires.
    n_exp = len(pd.read_csv(sorted(glob.glob(os.path.join(data_dir, "*_test.csv")))[0])) - 1
    per = idx.groupby("ticker").size()
    assert per.nunique() == 1 and per.iloc[0] == n_exp, \
        f"expected {n_exp} preds/ticker (len(test)-1), got {sorted(per.unique())}"

    ref = 0.002617   # PatchTST set1/2022 ensemble: a healthy, non-collapsed model on this cell
    for suf, vals in pts.items():
        d = idx.copy()
        d["predicted_RET"] = vals
        assert d.predicted_RET.notna().all(), f"NaN in {suf or 'mean'} forecasts"
        lab = label + suf
        out = os.path.join(a.root, a.st, "author", a.cohort, lab, f"L{a.context}", f"seed_{a.seed}")
        os.makedirs(out, exist_ok=True)
        d[["ticker", "date", "predicted_RET", "expected_RET"]].to_csv(
            os.path.join(out, "predictions.csv"), index=False)
        json.dump({"model": lab, "hf_model": a.model, "set": a.st, "cohort": a.cohort, "seed": a.seed,
                   "seq_len": a.context, "profile": "zeroshot", "epochs": 0,
                   "samples": (a.samples if kind == "samples" else 0), "kind": kind,
                   "quantile_levels": levels, "point": ("median" if suf else
                       ("trimmed_mean_q10_q90" if kind == "quantiles" else "mean")),
                   "device": a.device, "batch_size": a.batch,
                   "ckpt_rev": CKPT_REV, "uni2ts": _ver("uni2ts"), "torch": _ver("torch"),
                   "gluonts": _ver("gluonts")},
                  open(os.path.join(out, "timing.json"), "w"))
        open(os.path.join(out, ".done"), "w").close()

        sd = float(d.predicted_RET.std())
        gate = sum(1 for _, g in d.groupby("date").predicted_RET
                   if len(g) >= 20 and g.sort_values().iloc[-10:].min() > 0
                   and g.sort_values().iloc[:10].max() < 0)
        print(f"\n  [{'median' if suf else 'mean'}]")
        print(f"    dispersion  {sd:.6f}   ({sd/ref:.2f}x a healthy trained model here)")
        print(f"    cs sd/day   {d.groupby('date').predicted_RET.std().mean():.6f}")
        print(f"    gate days   {gate}/{d.date.nunique()}")
        if sd < 1e-4:
            print("    COLLAPSED -- near-constant; an IC from this is not a skill measurement.")
    print(f"\n  wall {wall:.0f}s  ->  {a.root}")


if __name__ == "__main__":
    main()
