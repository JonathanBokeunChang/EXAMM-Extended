"""Run the full baseline suite on a volatility cohort directory.

  python3 scripts/vol_baselines/run_baselines.py --data datasets/qlib_vol_big \
      --tag qlib_vol_big_fixed

Cohort-general: the same code runs on the fixed split and on every walk-forward cohort.
Writes everything the paper needs to results/baselines/<tag>/ and EXITS NON-ZERO if any
hard validation gate fails.
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import baselines as B
import validate as V

REPO = "/Users/jonathanchang/EXAMM-Extended"


def load_cohort(data_dir: str):
    stocks, tr, va, te = [], {}, {}, {}
    for f in sorted(glob.glob(f"{data_dir}/*_train.csv")):
        s = os.path.basename(f)[: -len("_train.csv")]
        p_va, p_te = f.replace("_train", "_val"), f.replace("_train", "_test")
        if not (os.path.exists(p_va) and os.path.exists(p_te)):
            continue
        a, b, c = pd.read_csv(f), pd.read_csv(p_va), pd.read_csv(p_te)
        for d in (a, b, c):
            d["date"] = pd.to_datetime(d["date"])
        if len(a) < 100 or len(c) < 40:
            continue
        stocks.append(s); tr[s], va[s], te[s] = a, b, c
    return stocks, tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=f"{REPO}/results/baselines")
    ap.add_argument("--anchor-csv", default=f"{REPO}/results/e1a_lstm_per_stock.csv")
    ap.add_argument("--offset", type=int, default=22,
                    help="secondary scoring offset (matches the LSTM window) for the anchor")
    a = ap.parse_args()

    out = f"{a.out_root}/{a.tag}"
    os.makedirs(f"{out}/predictions", exist_ok=True)
    t0 = time.time()

    stocks, TR, VA, TE = load_cohort(a.data)
    if not stocks:
        sys.exit(f"ERROR: no usable stocks in {a.data}")
    print(f"[{a.tag}] {len(stocks)} stocks | train {sum(len(TR[s]) for s in stocks):,} rows "
          f"| test {sum(len(TE[s]) for s in stocks):,} rows", flush=True)

    preds: dict[str, list] = {m: [] for m in B.PER_STOCK_MODELS}
    diag_rows, failures = [], []

    for i, s in enumerate(stocks, 1):
        for mname, fn in B.PER_STOCK_MODELS.items():
            try:
                df, d = fn(TR[s], TE[s], s)
            except Exception as e:
                df, d = None, {"converged": False, "error": str(e)[:120]}
            if df is None or df.empty:
                failures.append((s, mname, d.get("error", "no output")))
            else:
                preds[mname].append(df)
            diag_rows.append({"stock": s, "model": mname, **d})
        if i % 25 == 0:
            print(f"  ...{i}/{len(stocks)} stocks ({time.time()-t0:.0f}s)", flush=True)

    # pooled HAR (fairness protocol: report HAR at its best configuration)
    train_all = pd.concat([TR[s] for s in stocks], ignore_index=True)
    hp, hp_diag = B.fit_har_pooled(train_all, {s: TE[s] for s in stocks})
    preds["har_pooled"] = [hp]
    diag_rows.append({"stock": "__pooled__", "model": "har_pooled", **hp_diag})

    missing = [m for m, v in preds.items() if not v]
    if missing:
        print(f"[{a.tag}] WARNING: no output from {missing}", flush=True)
        for m in missing:
            ex = [f for f in failures if f[1] == m][:2]
            for s_, m_, err in ex:
                print(f"    {m_} / {s_}: {err}", flush=True)
    P = {m: pd.concat(v, ignore_index=True) for m, v in preds.items() if v}
    diag = pd.DataFrame(diag_rows)

    # ---- per-stock metrics on IDENTICAL keys across models ---------------------------
    common = set.intersection(*[set(zip(d["stock"], d["date"])) for d in P.values()])
    print(f"[{a.tag}] scoring on {len(common):,} (stock,date) keys common to all "
          f"{len(P)} models", flush=True)
    rows = []
    for m, d in P.items():
        d = d[[k in common for k in zip(d["stock"], d["date"])]]
        for s, g in d.groupby("stock"):
            g = g.sort_values("date")
            r = {"stock": s, "model": m, "n": len(g), **M.all_metrics(g.y_true, g.y_pred)}
            if len(g) > a.offset:                      # secondary anchor scoring
                r["r2_offset22"] = M.r2(g.y_true.values[a.offset:], g.y_pred.values[a.offset:])
            rows.append(r)
    per_stock = pd.DataFrame(rows)

    # ---- summary vs HAR (best config) -------------------------------------------------
    med = per_stock.groupby("model")["r2"].median()
    har_ref = "har_pooled" if med.get("har_pooled", -9) >= med.get("har", -9) else "har"
    piv = {m: g.set_index("stock") for m, g in per_stock.groupby("model")}
    srows = []
    for m, g in piv.items():
        row = {"model": m, "n_stocks": len(g)}
        for k in M.METRICS:
            row[f"{k}_median"] = float(g[k].median())
            row[f"{k}_mean"] = float(g[k].mean())
        if m != har_ref and har_ref in piv:
            j = g.join(piv[har_ref], rsuffix="_har", how="inner")
            for k in M.METRICS:
                better = (j[k] < j[f"{k}_har"]) if M.LOWER_IS_BETTER[k] else (j[k] > j[f"{k}_har"])
                row[f"{k}_winrate_vs_{har_ref}"] = float(better.mean())
                row[f"{k}_wilcoxon_p"] = M.wilcoxon_paired(j[k], j[f"{k}_har"])
        srows.append(row)
    summary = pd.DataFrame(srows).sort_values("r2_median", ascending=False)

    # ---- persist --------------------------------------------------------------------
    for m, d in P.items():
        d.sort_values(["stock", "date"]).to_csv(f"{out}/predictions/{m}.csv", index=False)
    per_stock.to_csv(f"{out}/metrics_per_stock.csv", index=False)
    summary.to_csv(f"{out}/summary.csv", index=False)
    diag.to_csv(f"{out}/fit_diagnostics.csv", index=False)

    # ---- validation ------------------------------------------------------------------
    hashes = {m: V.hash_predictions(d) for m, d in P.items()}
    prev = None
    mpath = f"{out}/manifest.json"
    if os.path.exists(mpath):
        try:
            prev = json.load(open(mpath)).get("prediction_hashes")
        except Exception:
            prev = None

    ret_sd = {s: float(np.std(TR[s]["RET"].values)) for s in stocks}
    checks = []
    # Always run the anchor check; it resolves the expected value by EXACT dataset name and
    # degrades to a soft "measured = X" report for datasets with no banked truth yet.
    # (The old substring gate would have made qlib_vol_big_v2 inherit qlib_vol_big's anchor.)
    checks += V.check_regression_anchor(per_stock, a.anchor_csv, a.data)
    checks += V.check_finiteness(P, per_stock)
    checks += V.check_coverage(P, per_stock)
    checks += V.check_leakage(max(TR[s]["date"].max() for s in stocks),
                              min(TE[s]["date"].min() for s in stocks), True)
    checks += V.check_garch_scale(diag, ret_sd)
    checks += V.check_determinism(hashes, prev)
    checks += V.soft_checks(per_stock, diag)
    checks += V.check_legacy_metric_agreement()
    ok = V.write_report(checks, f"{out}/validation_report.txt")

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "n/a"
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        git = "n/a"
    json.dump({
        "tag": a.tag, "data_dir": a.data, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_stocks": len(stocks), "n_scored_keys": len(common), "har_reference": har_ref,
        "models": sorted(P), "failures": failures[:50], "n_failures": len(failures),
        "horizon": B.HORIZON, "ewma_lambda": B.EWMA_LAMBDA, "arch_scale": B.ARCH_SCALE,
        "calibrated_models": list(B.NEEDS_CALIBRATION), "offset_secondary": a.offset,
        "prediction_hashes": hashes, "git_commit": git,
        "versions": {m: ver(m) for m in ("numpy", "pandas", "scipy", "arch")},
        "runtime_sec": round(time.time() - t0, 1), "validation_passed": ok,
    }, open(mpath, "w"), indent=2)

    print(f"\n{'='*70}\n{open(f'{out}/validation_report.txt').read()}")
    print(summary[["model"] + [f"{k}_median" for k in M.METRICS]].to_string(index=False))
    print(f"\nartifacts -> {out}  ({time.time()-t0:.0f}s)")
    if not ok:
        sys.exit("HARD VALIDATION GATE FAILED -- results must not be used")


if __name__ == "__main__":
    main()
