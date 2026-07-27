#!/usr/bin/env python3
"""NONLINEAR cross-sectional gate -- the test that can actually decide the relational premise.

WHY THIS EXISTS
The linear version (csi300_cs_operator_gate.py) returned a null, and that null is uninformative.
Relational architectures do not add peer information; they MODULATE own-stock features with it:

    HIST   learned concept membership shared across stocks
    GATs   attention weights over a stock graph
    MASTER market-conditioned gating of stock features

All multiplicative/conditional. A linear model can only form  beta*own + gamma*peer  and cannot
represent "weight momentum more when this stock's peers are trending", so a linear gate is
structurally blind to the mechanism. Reading its null as "cross-sectional structure is useless"
contradicts qlib's own 20-seed leaderboard, where on this exact venue relational models beat GRU:

    HIST  0.0667   (+0.0083 over GRU)
    IGMTF 0.0606   (+0.0022)
    GATs  0.0598   (+0.0014)
    GRU   0.0584

WHAT THIS TESTS
The same GRU, with and without cross-sectional inputs appended to each timestep. A GRU CAN learn
multiplicative interactions between its inputs, so if cross-sectional information is usable at all,
this arm can find it. Held constant across arms: architecture, hidden size, layers, learning rate,
epochs, early stopping, seeds, and the frozen eval index. Only the input width differs.

    base    (T, F)          own-stock features only
    base+CS (T, F + n_cs)   same, with CS features broadcast to every timestep

The CS features are contemporaneous per date, so they are broadcast across the sequence rather
than lagged -- the model sees "what the cross-section looks like now" alongside its own history,
which is the information a relational operator would have.

READING THE RESULT
  positive & significant -> cross-sectional information is usable by a nonlinear model on this
                            data. An evolved operator has something real to search for, and the
                            delta sizes the prize against HIST's +0.0083.
  null                   -> a genuine negative, since the model class CAN represent the mechanism.
                            Worth reconciling against HIST's published gain before acting on it:
                            our CS features are fixed peer averages, whereas HIST LEARNS its
                            concept memberships, so a null here still would not rule out a learned
                            relational structure -- only a hand-specified one.

Paired HAC test on daily IC differences, seeds matched pairwise so the comparison is within-seed.

Usage:
    python3 scripts/vol_baselines/csi300_cs_nonlinear_gate.py \
        --data datasets/csi300_master_replica_invdata_alpha360
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi300_master_replica_lstm_gru_fit import LR, Net, load_panel, predict  # noqa: E402
from ic_stats import DEFAULT_HAC_LAG, daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_samples(panel, FX, cs_cols, target, eval_keys, n_field, use_cs):
    """Alpha360 reshape mode: each row is (T, F). CS features are appended to every timestep."""
    Xtr, ytr, Xva, yva, Xte, meta = [], [], [], [], [], []
    for s, g in panel.items():
        for _, seg in g.groupby("segment_id", sort=True):
            seg = seg.reset_index(drop=True)
            X = seg[FX].to_numpy(np.float32)
            W = X.reshape(len(X), n_field, -1).transpose(0, 2, 1)   # (n, T, F)
            if use_cs:
                C = seg[cs_cols].to_numpy(np.float32)               # (n, n_cs)
                C = np.repeat(C[:, None, :], W.shape[1], axis=1)    # broadcast over timesteps
                W = np.concatenate([W, C], axis=2)
            y = seg[target].to_numpy(np.float32)
            sp, dt, lb = seg["split"].to_numpy(), seg["date"].to_numpy(), seg["LABEL"].to_numpy()
            for i in range(len(seg)):
                if sp[i] == "train":
                    Xtr.append(W[i]); ytr.append(y[i])
                elif sp[i] == "val":
                    Xva.append(W[i]); yva.append(y[i])
                elif (dt[i], s) in eval_keys:
                    Xte.append(W[i]); meta.append((s, dt[i], lb[i]))
    return (np.asarray(Xtr), np.asarray(ytr), np.asarray(Xva), np.asarray(yva),
            np.asarray(Xte), pd.DataFrame(meta, columns=["instrument", "date", "LABEL"]))


def train_eval(kind, Xtr, ytr, Xva, yva, Xte, hidden, layers, epochs, batch, patience, seed,
               log_every=1, tag="", dev=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n_in = Xtr.shape[2]
    mu = Xtr.reshape(-1, n_in).mean(0); sd = Xtr.reshape(-1, n_in).std(0)
    sd[sd < 1e-12] = 1.0
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32))
    dev = dev or torch.device('cpu')
    Xt, Xv, Xe = f(Xtr).to(dev), f(Xva).to(dev), f(Xte).to(dev)
    yt = torch.from_numpy((ytr - ymu) / ysd).to(dev)
    yv = torch.from_numpy((yva - ymu) / ysd).to(dev)

    m = Net(kind, hidden, n_in, layers).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR[kind]); lf = nn.MSELoss()
    best, state, bad, n = np.inf, None, 0, len(Xt)
    t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(predict(m, Xv), yv).item()
        improved = vl < best - 1e-5
        if improved:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
        if log_every and (ep % log_every == 0 or improved or bad >= patience):
            print(f"      [{tag} s{seed}] epoch {ep+1:>3}  val {vl:.6f}  best {best:.6f}  "
                  f"patience {bad}/{patience}  {time.time()-t0:.0f}s "
                  f"({(time.time()-t0)/(ep+1):.0f}s/ep)", flush=True)
        if bad >= patience:
            break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return predict(m, Xe).cpu().numpy(), sum(p.numel() for p in m.parameters()), ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--model", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--build-cs-cache", action="store_true",
                     help="only build the qlib-dependent peer-feature cache and exit; run this with the qlib interpreter, then train with the torch one")
    ap.add_argument("--log-every", type=int, default=1,
                     help="print val loss every N epochs (0 = silent); each epoch on "
                          "Alpha360's 60-step sequences takes minutes, so per-epoch "
                          "output is the only way to see progress inside a seed")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[device] {dev}')

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    assert man.get("handler") == "alpha360", "this gate assumes Alpha360 reshape layout"
    hac = max(1, man["label_horizon_trading_days"] - 1)
    print(f"[dataset] {man['dataset']}  content={man['content_sha256'][:16]}...  HAC lag={hac}")

    ev = pd.read_csv(f"{a.data}/eval_index.csv")
    eval_keys = set(zip(ev["date"], ev["instrument"]))
    panel = load_panel(a.data)
    FX = [c for c in next(iter(panel.values())).columns
          if c not in ("date", "segment_id", "split", "LABEL", "LABEL_CSRANK")]

    # qlib and torch live in different interpreters here (qlib in scratchpad/qlibenv, torch in the
    # system python), so the qlib-dependent peer construction is cached to disk by a separate
    # process and this one only ever reads the CSV. Build it with:
    #   scratchpad/qlibenv/bin/python scripts/vol_baselines/csi300_cs_nonlinear_gate.py \
    #       --data <dir> --build-cs-cache
    cache = f"{a.data}/_cs_features.csv"
    insts = sorted(panel)
    if os.path.exists(cache) and not a.build_cs_cache:
        cs = pd.read_csv(cache)
        n_peered = cs["instrument"].nunique()
        print(f"[cs] loaded cached peer features from {os.path.basename(cache)}")
    else:
        print(f"[cs] building peer features for {len(insts)} instruments (k={a.k}, train-only)...")
        train_dates = set(pd.concat([g[g['split'] == 'train'][['date']]
                                     for g in panel.values()])['date'])
        from csi300_cs_operator_gate import build_cs_features, raw_returns
        rets = raw_returns(man["bundle"]["provider_uri"], insts,
                            man["splits"]["train"][0], man["splits"]["test"][1])
        cs, n_peered = build_cs_features(rets, train_dates, k=a.k)
        cs.to_csv(cache, index=False)
        print(f"[cs] wrote {cache}")
        if a.build_cs_cache:
            print("DONE (cache only; re-run without --build-cs-cache to train)")
            return
    cs_cols = [c for c in cs.columns if c.startswith("CS_")]
    for s in panel:
        panel[s] = panel[s].merge(cs, on=["date", "instrument"], how="left") \
            if "instrument" in panel[s].columns else \
            panel[s].assign(instrument=s).merge(cs, on=["date", "instrument"], how="left")
        panel[s][cs_cols] = panel[s][cs_cols].fillna(0.0)
    print(f"[cs] {len(cs_cols)} CS features on {n_peered} stocks")

    results = {}
    for use_cs in (False, True):
        tag = "base+CS" if use_cs else "base"
        Xtr, ytr, Xva, yva, Xte, md = build_samples(
            panel, FX, cs_cols, "LABEL_CSRANK", eval_keys, 6, use_cs)
        if len(md) != len(ev):
            sys.exit(f"ERROR: {tag} produced {len(md)} eval rows, expected {len(ev)}")
        print(f"\n[{tag}] train {Xtr.shape} val {Xva.shape} eval {Xte.shape}")
        P = []
        t0 = time.time()
        for sd_ in range(a.seeds):
            p, npar, eps = train_eval(a.model, Xtr, ytr, Xva, yva, Xte, a.hidden, a.layers,
                                       a.epochs, a.batch, a.patience, sd_,
                                       a.log_every, tag, dev)
            P.append(p)
            print(f"   seed {sd_}: {eps} epochs, {npar} params, {time.time()-t0:.0f}s")
        ics = daily_ic(md.assign(pred=np.mean(P, axis=0)),
                        pred_col="pred", label_col="LABEL", date_col="date")
        results[tag] = (ics, P, md)
        report(f"{a.model} {tag}", ics, hac)

        # Persist the arm the moment it finishes. The paired test needs per-DATE IC series and raw
        # predictions, not just the summary, so holding them in memory until the end meant an
        # interrupted run lost a completed arm entirely and had to redo it. Written per-arm, a
        # rerun can reuse whatever already landed.
        ap_path = f"{a.data}/_cs_arm_{tag.replace('+', '_')}_{a.model}.npz"
        np.savez_compressed(ap_path, preds=np.asarray(P), ics=ics,
                             instrument=md["instrument"].to_numpy(),
                             date=md["date"].to_numpy(),
                             label=md["LABEL"].to_numpy())
        print(f"   saved arm -> {os.path.basename(ap_path)}", flush=True)

    ics_b, Pb, md = results["base"]
    ics_c, Pc, _ = results["base+CS"]
    p_ens = paired(ics_c, ics_b, hac)
    print(f"\n=== PAIRED TEST, ensemble ===")
    print(f"   delta = {p_ens['delta']:+.4f}   HAC t = {p_ens['t_hac']:+.2f}   "
          f"win-rate {p_ens['win_rate']:.1%}   2xSE floor = {2*p_ens['se_hac']:.4f}")

    # Within-seed pairing removes seed-to-seed variance, the dominant noise source with 3 seeds.
    print(f"\n=== PAIRED TEST, within-seed (same init, only inputs differ) ===")
    per = []
    for i, (pb, pc) in enumerate(zip(Pb, Pc)):
        ib = daily_ic(md.assign(pred=pb), pred_col="pred", label_col="LABEL", date_col="date")
        ic = daily_ic(md.assign(pred=pc), pred_col="pred", label_col="LABEL", date_col="date")
        q = paired(ic, ib, hac)
        per.append(q["delta"])
        print(f"   seed {i}: base {ib.mean():+.4f} -> base+CS {ic.mean():+.4f}   "
              f"delta {q['delta']:+.4f}  HAC t {q['t_hac']:+.2f}")
    print(f"   mean within-seed delta = {np.mean(per):+.4f}  "
          f"(all same sign: {len(set(np.sign(per))) == 1})")

    print(f"\n=== calibration ===")
    print(f"   HIST vs GRU (published, this venue) : +0.0083")
    print(f"   our resolution floor (2xSE)          : {2*p_ens['se_hac']:.4f}")
    print(f"   published GRU / LSTM on Alpha360     : 0.0584 / 0.0549")

    with open(f"{a.data}/_cs_nonlinear_gate.json", "w") as f:
        json.dump({"model": a.model, "paired_ensemble": p_ens,
                   "within_seed_deltas": [float(x) for x in per],
                   "n_cs_features": len(cs_cols), "seeds": a.seeds, "hac_lag": hac},
                  f, indent=2, default=float)
    print(f"\nwrote {a.data}/_cs_nonlinear_gate.json")


if __name__ == "__main__":
    main()
