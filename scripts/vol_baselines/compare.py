"""Build the paper's model-comparison table from a harness run.

Every model in results/baselines/<tag>/metrics_per_stock.csv is scored on identical
(stock, date) keys, so the comparison is valid by construction. Reports medians, the
best baseline per metric, and paired Wilcoxon tests of the focus model against each
baseline with Holm correction across the baseline family (as pre-registered).

  python3 scripts/vol_baselines/compare.py --tag qlib_vol_big_fixed --focus examm
"""
from __future__ import annotations

import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

REPO = "/Users/jonathanchang/EXAMM-Extended"


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
    a = ap.parse_args()

    d = f"{a.out_root}/{a.tag}"
    ps = pd.read_csv(f"{d}/metrics_per_stock.csv")

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
    R.to_csv(f"{d}/comparison.csv", index=False)

    sweep = all((R[R.metric == k].win_rate > 0.5).all() and (R[R.metric == k].p_holm < 0.05).all()
                for k in M.METRICS)
    print(f"\n  VERDICT: {a.focus} beats EVERY baseline on ALL FOUR metrics "
          f"(Holm-adjusted): {'YES' if sweep else 'NO'}")
    print(f"  -> {d}/comparison.csv")


if __name__ == "__main__":
    main()
