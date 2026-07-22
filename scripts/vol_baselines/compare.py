"""Build the paper's model-comparison table from a harness run.

Every model in results/baselines/<tag>/metrics_per_stock.csv is scored on identical
(stock, date) keys, so the comparison is valid by construction. Reports medians, the
best baseline per metric, and paired Wilcoxon tests of the focus model against each
baseline with Holm correction across the baseline family (as pre-registered).

  python3 scripts/vol_baselines/compare.py --tag qlib_vol_big_fixed --focus examm
  python3 scripts/vol_baselines/compare.py --tag qlib_vol_full --focus examm \
      --exclude-halts --data datasets/qlib_vol_full     # halt-robustness column
"""
from __future__ import annotations

import argparse, glob, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HALT_FLOOR = np.log(1e-4)   # LV of a halt/limit-locked day (high==low -> Parkinson vol 0)
HALT_H = 5                  # forward target horizon; must match the dataset build


def halt_affected_keys(data_dir: str, horizon: int = HALT_H) -> set:
    """(stock, date) rows a halt touches: a halt day (LV==log 1e-4) within the trailing MA5
    window, the row itself, or the forward H-day target window. These are the rows where the
    Parkinson estimator is degenerate and QLIKE's exp() term blows up -- excluded ONLY in the
    robustness column, never from the headline. Model-agnostic (computed from the data)."""
    bad = set()
    for f in sorted(glob.glob(f"{data_dir}/*_test.csv")):
        s = os.path.basename(f)[: -len("_test.csv")]
        d = pd.read_csv(f)
        dates = pd.to_datetime(d["date"]).values
        halt = np.isclose(d["LV"].values, HALT_FLOOR, atol=1e-3)
        for i in range(len(d)):
            if halt[max(0, i - 4): i + 1].any() or halt[i + 1: i + 1 + horizon].any():
                bad.add((s, pd.Timestamp(dates[i])))
    return bad


def recompute_from_predictions(tag_dir: str, exclude: set | None = None) -> pd.DataFrame:
    """Rebuild per-stock metrics from predictions/*.csv, optionally dropping `exclude` keys.
    Used for the halt-robustness column so every model is re-scored on the SAME retained rows
    from its own persisted predictions -- identical-row invariant preserved by construction."""
    rows = []
    for f in sorted(glob.glob(f"{tag_dir}/predictions/*.csv")):
        model = os.path.basename(f)[:-4]
        p = pd.read_csv(f)
        p["date"] = pd.to_datetime(p["date"])
        if exclude:
            keep = [(s, ts) not in exclude for s, ts in zip(p["stock"], p["date"])]
            p = p[keep]
        for s, g in p.groupby("stock"):
            if len(g) > 10:
                rows.append({"stock": s, "model": model, "n": len(g),
                             **M.all_metrics(g.y_true, g.y_pred)})
    return pd.DataFrame(rows)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment across a family of comparisons."""
    items = sorted([(k, v) for k, v in pvals.items() if np.isfinite(v)], key=lambda x: x[1])
    n, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (n - i) * p))
        out[k] = adj
        prev = adj
    for k, v in pvals.items():
        out.setdefault(k, float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--focus", default="examm")
    ap.add_argument("--out-root", default=f"{REPO}/results/baselines")
    ap.add_argument("--exclude-halts", action="store_true",
                    help="robustness column: re-score from predictions with halt-affected "
                         "rows removed (requires --data)")
    ap.add_argument("--data", default=None,
                    help="dataset dir (for --exclude-halts halt detection)")
    a = ap.parse_args()

    d = f"{a.out_root}/{a.tag}"
    if a.exclude_halts:
        if not a.data:
            sys.exit("--exclude-halts requires --data <dataset dir> to locate halt days")
        bad = halt_affected_keys(a.data)
        ps = recompute_from_predictions(d, exclude=bad)
        print(f"\n  #### HALT-ROBUSTNESS VIEW: removed {len(bad):,} halt-affected (stock,date) "
              f"rows, re-scored every model from its predictions ####")
    else:
        ps = pd.read_csv(f"{d}/metrics_per_stock.csv")

    # A duplicate (model, stock) would cartesian-expand at the .join() below and silently
    # inflate every paired test. Fail loud instead -- this catches a partial pipeline re-run
    # that appended a model's rows twice.
    dup = ps.groupby(["model", "stock"]).size()
    dup = dup[dup > 1]
    if len(dup):
        sys.exit(f"ERROR: duplicate (model, stock) rows in metrics_per_stock.csv: "
                 f"{dup.index.tolist()[:8]}")

    # Restrict every model to the GLOBAL common stock set before anything is reported.
    # Models enter this table from different scripts (run_baselines, eval_lstm, eval_examm)
    # and can disagree on coverage -- e.g. EGARCH non-convergence drops stocks that the
    # LSTM arm still has. Without this, the medians in the summary table would each sit on a
    # slightly different universe while looking directly comparable, which is precisely the
    # defect this harness exists to prevent.
    per_model = {m: set(g["stock"]) for m, g in ps.groupby("model")}
    common = set.intersection(*per_model.values()) if per_model else set()
    dropped = {m: sorted(s - common) for m, s in per_model.items() if s - common}
    if dropped:
        n_all = len(set.union(*per_model.values()))
        print(f"\n  [coverage] restricting all models to {len(common)} stocks common to all "
              f"{len(per_model)} models (union was {n_all})")
        for m, ss in sorted(dropped.items()):
            print(f"    {m:12} dropped {len(ss)}: {','.join(ss[:6])}"
                  f"{' ...' if len(ss) > 6 else ''}")
    ps = ps[ps["stock"].isin(common)]

    # The table header claims identical (stock, date) keys; verify per-stock ROW COUNTS agree
    # across models (the `n` column), not just the stock set. Different scripts write these
    # rows and nothing else enforces it. A mismatch means the models were scored on different
    # dates for some stock and the comparison is invalid.
    if "n" in ps.columns:
        nwide = ps.pivot_table(index="stock", columns="model", values="n")
        bad = nwide[nwide.nunique(axis=1) > 1]
        if len(bad):
            sys.exit(f"ERROR: models disagree on row count for {len(bad)} stock(s) "
                     f"(not identical keys): e.g.\n{bad.head(5).to_string()}")

    piv = {m: g.set_index("stock") for m, g in ps.groupby("model")}
    if a.focus not in piv:
        sys.exit(f"focus model '{a.focus}' not in {sorted(piv)}")

    order = sorted(piv, key=lambda m: -piv[m]["r2"].median())
    print(f"\n#### MODEL COMPARISON -- {a.tag} "
          f"({piv[a.focus].shape[0]} stocks, identical (stock,date) keys) ####\n")
    print(f"  {'model':14}{'MSE':>10}{'MAE':>10}{'QLIKE':>11}{'R^2':>10}")
    for m in order:
        g = piv[m]
        mark = " *" if m == a.focus else ""
        print(f"  {m:14}{g.mse.median():>10.4f}{g.mae.median():>10.4f}"
              f"{g.qlike.median():>11.4f}{g.r2.median():>10.4f}{mark}")

    baselines = [m for m in piv if m != a.focus and not m.startswith(a.focus)]
    print(f"\n  best baseline per metric (excludes {a.focus}*):")
    best = {}
    for k in M.METRICS:
        vals = {m: piv[m][k].median() for m in baselines}
        best[k] = min(vals, key=vals.get) if M.LOWER_IS_BETTER[k] else max(vals, key=vals.get)
        print(f"    {k:6} -> {best[k]:14} {vals[best[k]]:+.4f}   "
              f"({a.focus} {piv[a.focus][k].median():+.4f})")

    print(f"\n  {a.focus} vs each baseline (paired Wilcoxon, Holm-adjusted across the family):")
    rows = []
    for k in M.METRICS:
        raw = {}
        for m in baselines:
            j = piv[a.focus].join(piv[m], rsuffix="_b", how="inner")
            raw[m] = M.wilcoxon_paired(j[k], j[f"{k}_b"])
        adj = holm(raw)
        for m in baselines:
            j = piv[a.focus].join(piv[m], rsuffix="_b", how="inner")
            better = (j[k] < j[f"{k}_b"]) if M.LOWER_IS_BETTER[k] else (j[k] > j[f"{k}_b"])
            rows.append({"metric": k, "baseline": m, "n": len(j),
                         "focus_median": j[k].median(), "baseline_median": j[f"{k}_b"].median(),
                         "win_rate": float(better.mean()),
                         "p_raw": raw[m], "p_holm": adj[m]})
    R = pd.DataFrame(rows)
    for k in M.METRICS:
        sub = R[R.metric == k].sort_values("baseline")
        print(f"\n    [{k}]")
        for _, r in sub.iterrows():
            flag = "WIN " if (r.win_rate > 0.5 and r.p_holm < 0.05) else \
                   ("LOSS" if (r.win_rate < 0.5 and r.p_holm < 0.05) else "ns  ")
            print(f"      {flag} vs {r.baseline:12} {r.win_rate*100:5.1f}% of {int(r.n)} stocks"
                  f"   p_holm={r.p_holm:.2g}")
    out_csv = f"{d}/comparison{'_exhalt' if a.exclude_halts else ''}.csv"
    R.to_csv(out_csv, index=False)

    # R2 is NOT an independent test. With models scored on identical rows, SST_s is a
    # per-stock constant, so r2 = 1 - n*mse/SST is a strictly decreasing affine function of
    # mse WITHIN each stock -- the paired sign is identical to MSE by construction (every MSE
    # win-rate equals its R2 win-rate exactly). Report the sweep over the THREE independent
    # metrics (MSE, MAE, QLIKE) and keep R2 as a descriptive scale-free statistic.
    indep = [k for k in M.METRICS if k != "r2"]
    sweep = all((R[R.metric == k].win_rate > 0.5).all() and (R[R.metric == k].p_holm < 0.05).all()
                for k in indep)
    print(f"\n  VERDICT: {a.focus} beats EVERY baseline on all THREE independent metrics "
          f"{tuple(m.upper() for m in indep)} (Holm-adjusted): {'YES' if sweep else 'NO'}")
    print(f"    (R2 is omitted as an independent test: within identical rows it is a monotone"
          f" transform of MSE, so its paired signs match MSE by construction.)")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
