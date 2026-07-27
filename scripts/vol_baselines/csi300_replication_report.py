#!/usr/bin/env python3
"""Cross-source replication report: does the headline hold on an INDEPENDENTLY-SOURCED bundle?

The placebo test (labels shuffled within date -> IC ~ 0) proves the pipeline has no structural
leakage. It cannot prove the input data is sound: a bundle whose price series is systematically
cleaner, more autocorrelated, or differently survivorship-filtered than a real trading feed would
produce genuine-looking predictability that the placebo would happily pass.

The only check that catches that is replication on a different source. Here:

  invdata  chenditc/investment_data (Tushare/Baostock/Wind blend)  -- primary
  cndata   qlib's public example bundle (Yahoo-sourced)            -- independent replication

Same builder, same parameters, same protocol; only the raw feed differs. Agreement means the
effect is a property of the market, not the vendor. Divergence means any claim must be scoped to
the source it was measured on. This project has used exactly this triangulation before to separate
a real cross-sectional signal from a data artifact.

Prerequisite: build and gate both bundles first --
    for b in invdata cndata; do
      python3 scripts/stock_run/build_csi300_master_replica.py --bundle $b
      python3 scripts/vol_baselines/csi300_master_replica_gate.py \
          --data datasets/csi300_master_replica_$b
    done

Usage:
    python3 scripts/vol_baselines/csi300_replication_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(bundle):
    d = f"{REPO}/datasets/csi300_master_replica_{bundle}"
    mf, gf = f"{d}/MANIFEST.json", f"{d}/_gate_result.json"
    if not os.path.exists(mf):
        return None, None, None, f"dataset not built ({d})"
    if not os.path.exists(gf):
        return None, None, None, f"gate not run ({gf})"
    lf = f"{d}/_lstm_gru_result.json"
    lstm = json.load(open(lf)) if os.path.exists(lf) else None
    return json.load(open(mf)), json.load(open(gf)), lstm, None


def overlaps(a, b):
    """Do the two 95% HAC confidence intervals overlap?"""
    return not (a["ci95_hi"] < b["ci95_lo"] or b["ci95_hi"] < a["ci95_lo"])


def row(label, s):
    return (f"  {label:<34} {s['ic']:+.4f}  HAC t {s['t_hac']:+6.2f}   "
            f"[95% {s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}]   n={s['n_dates']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="invdata,cndata")
    a = ap.parse_args()
    bundles = a.bundles.split(",")

    data = {}
    for b in bundles:
        m, g, l, err = load(b)
        if err:
            print(f"[{b}] SKIPPED -- {err}")
            continue
        data[b] = (m, g, l)

    if len(data) < 2:
        sys.exit("\nNeed at least two built+gated bundles to report replication. "
                 "See this script's docstring for the build commands.")

    print("=" * 100)
    print("CROSS-SOURCE REPLICATION -- same builder, same protocol, different raw feed")
    print("=" * 100)

    for b, (m, g, l) in data.items():
        bd = m["bundle"]
        print(f"\n[{b}]  tag={bd.get('release_tag') or 'n/a'}  "
              f"content={m['content_sha256'][:16]}...")
        print(f"       indices={m['index_codes_resolved']}")
        print(f"       eval: {m['eval_index']['rows']} rows, {m['eval_index']['stocks']} stocks, "
              f"{m['eval_index']['dates']} dates, median {m['eval_index']['median_stocks_per_date']}/date")

    print("\n" + "-" * 100)
    print("RIDGE CEILING (221 features, CSRankNorm target)")
    print("-" * 100)
    for b, (m, g, l) in data.items():
        print(row(b, g["ridge_csrank"]))

    keys = list(data)
    a0, b0 = data[keys[0]][1]["ridge_csrank"], data[keys[1]][1]["ridge_csrank"]
    ok = overlaps(a0, b0)
    print(f"\n  95% HAC intervals {'OVERLAP' if ok else 'DO NOT OVERLAP'}  ->  "
          f"{'consistent across sources' if ok else 'SOURCE-DEPENDENT: scope the claim'}")
    print(f"  point-estimate spread: {abs(a0['ic']-b0['ic']):.4f}")

    if all(d[2] for d in data.values()):
        for model in ("lstm", "gru"):
            if all(model in d[2] for d in data.values()):
                print("\n" + "-" * 100)
                print(f"{model.upper()} (2x64, 3-seed ensemble)")
                print("-" * 100)
                for b, (m, g, l) in data.items():
                    print(row(b, l[model]))
                x, y = [data[k][2][model] for k in keys]
                o = overlaps(x, y)
                print(f"\n  95% HAC intervals {'OVERLAP' if o else 'DO NOT OVERLAP'}  ->  "
                      f"{'consistent' if o else 'SOURCE-DEPENDENT'}")
    else:
        print("\n(LSTM/GRU results absent for at least one bundle -- ridge comparison only)")

    print("\n" + "=" * 100)
    print("READING THIS: overlapping intervals across two independently-sourced feeds is the")
    print("strongest evidence available that a result reflects the market rather than the vendor.")
    print("It is NOT evidence that our feed matches MASTER's confidential one -- that remains")
    print("unverifiable, and any comparison to their published numbers stays uncontrolled.")
    print("=" * 100)


if __name__ == "__main__":
    main()
