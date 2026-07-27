#!/usr/bin/env python3
"""Cross-sectional (relational) premise gate -- run BEFORE writing any C++ operator.

THE QUESTION
Would giving EXAMM an evolvable cross-sectional operator (a node that sees peer stocks, not just
its own history) buy anything? Building one is weeks of C++. This tests the premise in hours with
a linear proxy, the same discipline that killed the volatility cross-sectional idea for ~1 hour of
work instead of three weeks.

THE BAR, from qlib's own 20-seed leaderboard on Alpha360 (Rank IC):
    HIST  (cross-stock concepts) 0.0667   = +0.0083 over GRU   <- the best relational model
    IGMTF (cross-stock)          0.0606   = +0.0022 over GRU
    GATs  (graph attention)      0.0598   = +0.0014 over GRU
    GRU                          0.0584
Hand-designed, expert-built relational architectures buy +0.0014 to +0.0083. That is the entire
prize. Our paired-HAC resolution on this cross-section is ~0.0023 SE, so only a HIST-scale effect
is even detectable; a GATs-scale effect is statistically invisible here no matter what we build.

WHAT IS TESTED
    base      the dataset's own features (Alpha360: 60 lags x 6 fields, own-stock only)
    base+CS   the same, plus peer/relational features that a cross-sectional operator could learn

*** READ THIS BEFORE INTERPRETING A NULL ***
This gate is ASYMMETRIC. A positive result is strong evidence the premise holds. A NULL result is
NOT evidence against it, because the test is linear and the architectures in question are not:
HIST/GATs/IGMTF/MASTER all work by MODULATING how own-stock features are used (attention, learned
concept membership, market-conditioned gating), which is multiplicative and conditional. A linear
model can only add peer features on top, and cannot represent that mechanism at all.

An earlier run of this script printed "VERDICT: FAIL" on a null. That was wrong and has been
removed: it conflated "a linear additive combination finds nothing" with "cross-sectional structure
is useless", when qlib's own leaderboard shows relational models beating GRU on this very venue.
The deciding test is the NONLINEAR arm (GRU with vs without these features) -- see
csi300_cs_nonlinear_gate.py.

CRITICAL DESIGN CONSTRAINT (learned the hard way this session)
Cross-sectional rank IC compares stocks WITHIN a date. A feature that is constant across stocks on
a given date shifts every prediction identically and CANNOT change their ordering -- its
contribution is exactly zero BY CONSTRUCTION. MASTER's 63 "market" features are exactly this, and
an earlier ablation of ours appeared to show them helping (+0.0200, t=3.95) when the real cause was
two fits selecting different ridge penalties. So every CS feature here is asserted to have non-zero
within-date variance before it is allowed into the test. A gate that silently included date-constant
features would be rigged to return null.

LEAK SAFETY
Peer sets, peer weights and betas are estimated on TRAIN ROWS ONLY and then frozen. All features
are causal (built from data up to and including t). The label horizon is purged at split boundaries
by the dataset itself.

Usage:
    python3 scripts/vol_baselines/csi300_cs_operator_gate.py \
        --data datasets/csi300_master_replica_invdata_alpha360
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi300_master_replica_gate import fit_ridge, load, restrict_to_eval_index  # noqa: E402
from ic_stats import DEFAULT_HAC_LAG, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RET_WINDOWS = [1, 5, 20]


def raw_returns(bundle_uri, insts, start, end):
    """Raw close-to-close returns straight from the bundle.

    Not derived from the dataset's feature columns: those are z-scored by the handler, which
    destroys the ratio structure needed to recover returns. Peer/beta estimation must run on real
    returns, so we re-query the source.
    """
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=bundle_uri, region=REG_CN, kernels=1)
    from qlib.data import D
    df = D.features(list(insts), ["$close/Ref($close,1)-1"], start_time=start, end_time=end,
                     freq="day")
    df.columns = ["RET1"]
    df = df.reset_index()
    df.columns = ["instrument", "datetime", "RET1"] if df.columns[0] == "instrument" else \
                 ["datetime", "instrument", "RET1"]
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
    return df[["date", "instrument", "RET1"]]


def build_cs_features(rets, train_dates, k=20, min_obs=250):
    """Peer-relative features -- the linear proxy for what a relational operator would compute.

    Peer sets are the k most TRAIN-correlated stocks for each name; peer/beta statistics are frozen
    from train and applied unchanged to val/test. Every feature varies across stocks within a date
    (different names have different peers and betas), which is what makes them capable of moving a
    cross-sectional ranking at all.
    """
    wide = rets.pivot(index="date", columns="instrument", values="RET1").sort_index()
    tr = wide.loc[wide.index.isin(train_dates)]
    keep = tr.columns[tr.notna().sum() >= min_obs]
    corr = tr[keep].corr(min_periods=min_obs // 2)
    np.fill_diagonal(corr.values, np.nan)

    peers = {}
    for s in corr.columns:
        c = corr[s].dropna()
        if len(c) >= k:
            peers[s] = list(c.nlargest(k).index)

    mkt = wide.mean(axis=1)                      # equal-weight market return, all names present
    trm = mkt.loc[mkt.index.isin(train_dates)]
    betas = {}
    for s in keep:
        a = tr[s].dropna()
        b = trm.reindex(a.index)
        m = a.notna() & b.notna()
        if m.sum() >= min_obs and b[m].var() > 0:
            betas[s] = float(np.cov(a[m], b[m])[0, 1] / b[m].var())

    cum = {w: wide.rolling(w).sum() for w in RET_WINDOWS}   # log-free approx, fine at daily scale
    mkt_cum = {w: mkt.rolling(w).sum() for w in RET_WINDOWS}

    out = []
    for s, pl in peers.items():
        if s not in betas:
            continue
        d = pd.DataFrame(index=wide.index)
        for w in RET_WINDOWS:
            own = cum[w][s]
            peer = cum[w][pl].mean(axis=1)       # what this stock's peer group did
            d[f"CS_PEER{w}"] = peer
            d[f"CS_REL{w}"] = own - peer         # own move relative to its peers
            d[f"CS_BETAADJ{w}"] = own - betas[s] * mkt_cum[w]   # beta-adjusted idiosyncratic move
        d["instrument"] = s
        d["date"] = d.index
        out.append(d)
    cs = pd.concat(out, ignore_index=True)
    return cs, len(peers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--k", type=int, default=20, help="peers per stock")
    ap.add_argument("--hac-lag", type=int, default=None)
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = a.hac_lag if a.hac_lag is not None else max(0, man["label_horizon_trading_days"] - 1)
    print(f"[dataset] {man['dataset']}  handler={man.get('handler')} label={man.get('label_kind')}")
    print(f"          content={man['content_sha256'][:16]}...  HAC lag={hac}\n")

    tr, va, te = (load(a.data, s) for s in ("train", "val", "test"))
    te = restrict_to_eval_index(te, a.data)
    drop = {"date", "instrument", "segment_id", "LABEL", "LABEL_CSRANK"}
    base_cols = [c for c in tr.columns if c not in drop]

    insts = sorted(set(tr["instrument"]) | set(te["instrument"]))
    print(f"[cs] querying raw returns for {len(insts)} instruments from the pinned bundle...")
    rets = raw_returns(man["bundle"]["provider_uri"], insts,
                        man["splits"]["train"][0], man["splits"]["test"][1])
    cs, n_peered = build_cs_features(rets, set(tr["date"]), k=a.k)
    cs_cols = [c for c in cs.columns if c.startswith("CS_")]
    print(f"[cs] built {len(cs_cols)} CS features for {n_peered} stocks (k={a.k} train-selected peers)")

    def attach(d):
        n0 = len(d)
        m = d.merge(cs, on=["date", "instrument"], how="left")
        m[cs_cols] = m[cs_cols].fillna(0.0)   # names without a peer set fall back to no CS info
        assert len(m) == n0
        return m

    tr, va, te = attach(tr), attach(va), attach(te)

    # GUARD: a CS feature that is constant within a date cannot move a cross-sectional ranking.
    print("\n[guard] within-date variance of each CS feature (must be > 0 to be testable):")
    sd = te.groupby("date")[cs_cols].std().max()
    dead = [c for c in cs_cols if not (sd[c] > 1e-12)]
    for c in cs_cols:
        print(f"   {c:<16} max within-date sd across stocks = {sd[c]:.3e}"
              f"{'   <-- DATE-CONSTANT, cannot affect rank IC' if c in dead else ''}")
    if dead:
        sys.exit(f"\nERROR: {len(dead)} CS features are date-constant. Including them would make "
                 f"this gate structurally incapable of showing an effect. Fix the features.")
    print("   all CS features vary across stocks within a date -- the test is capable of a result")

    print(f"\n[gate] base {len(base_cols)} features vs base+CS {len(base_cols) + len(cs_cols)}")
    target = "LABEL_CSRANK" if "LABEL_CSRANK" in tr.columns else "LABEL"
    print(f"       training target = {target}\n")

    al_b, vic_b, _, ics_b = fit_ridge(tr, va, te, base_cols, target)
    print(f"   base   ridge alpha={al_b:.0f} (val IC {vic_b:+.4f})")
    s_b = report("base (own-stock only)", ics_b, hac)

    al_c, vic_c, _, ics_c = fit_ridge(tr, va, te, base_cols + cs_cols, target)
    print(f"   base+CS ridge alpha={al_c:.0f} (val IC {vic_c:+.4f})")
    s_c = report("base + cross-sectional", ics_c, hac)

    p = paired(ics_c, ics_b, hac)
    print(f"\n=== PAIRED TEST (the number that decides this) ===")
    print(f"   delta = {p['delta']:+.4f}   HAC t = {p['t_hac']:+.2f}   "
          f"win-rate {p['win_rate']:.1%}   SE {p['se_hac']:.5f}")

    print(f"\n=== calibration against hand-designed relational models (qlib leaderboard) ===")
    print(f"   HIST  vs GRU : +0.0083   {'DETECTABLE' if 0.0083 > 2*p['se_hac'] else 'below our resolution'}")
    print(f"   IGMTF vs GRU : +0.0022   {'DETECTABLE' if 0.0022 > 2*p['se_hac'] else 'below our resolution'}")
    print(f"   GATs  vs GRU : +0.0014   {'DETECTABLE' if 0.0014 > 2*p['se_hac'] else 'below our resolution'}")
    print(f"   our 2xSE resolution floor = {2*p['se_hac']:.4f}")

    print(f"\n=== WHAT THIS TEST CAN AND CANNOT CONCLUDE ===")
    print("   A POSITIVE result here would be strong: it would mean cross-sectional signal is")
    print("   available even to a linear, additive combination -- an easy thing to exploit.")
    print("")
    print("   A NULL result here is NOT evidence against relational architectures, and must not")
    print("   be reported as one. HIST (+0.0083), IGMTF (+0.0022) and GATs (+0.0014) all beat GRU")
    print("   on this exact venue in qlib's own 20-seed runs, so cross-sectional information")
    print("   demonstrably helps. They do it by MODULATING how own-stock features are used --")
    print("   attention weights, learned concept membership, market-conditioned gating. Those are")
    print("   multiplicative/conditional mechanisms. A linear model adds peer features:")
    print("       pred_i = beta*own_i + gamma*peer_i")
    print("   and literally cannot represent 'weight momentum more when peers are trending'.")
    print("   So this test is structurally blind to the mechanism the architectures actually use.")
    print("")
    print("   Caveat on the numbers above: the two fits selected their own ridge penalties, so")
    print("   they differ in SHRINKAGE as well as in features -- the delta is not a clean feature")
    print("   ablation. (The same confound made MASTER's date-constant market block appear to")
    print("   'help' by +0.0200 in an earlier ablation of ours.)")
    print("")
    print("   VERDICT ON THE PREMISE: this test does not settle it either way. Only a NONLINEAR")
    print("   arm can -- GRU with vs without these features, matched seeds, paired HAC. That is")
    print("   the test that decides whether an evolved cross-sectional operator has anything")
    print("   to search for.")
    if p["t_hac"] > 2:
        print("\n   (Observed: POSITIVE and significant even under the linear handicap -- notable,")
        print("    since this is the harder direction for a result to appear.)")

    with open(f"{a.data}/_cs_operator_gate.json", "w") as f:
        json.dump({"base": s_b, "base_plus_cs": s_c, "paired": p,
                   "n_cs_features": len(cs_cols), "k_peers": a.k, "hac_lag": hac},
                  f, indent=2, default=float)
    print(f"\nwrote {a.data}/_cs_operator_gate.json")


if __name__ == "__main__":
    main()
