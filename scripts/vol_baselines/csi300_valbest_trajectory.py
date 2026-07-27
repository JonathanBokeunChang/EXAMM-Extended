#!/usr/bin/env python3
"""DIAGNOSTIC: what happens to test IC and validation IC at each new validation-best?

NOT a reportable experiment. Run it only after the honest numbers are banked.

Two questions it answers that a normal run cannot:

1. TRAJECTORY OF SELECTED MODELS. Early stopping keeps a single best-so-far checkpoint that gets
   silently replaced each time validation improves; only the last survives. This logs test IC at
   every one of those checkpoints, showing whether the model being *selected* is genuinely getting
   better or wandering.

2. IS MSE THE WRONG STOPPING CRITERION? We train and early-stop on validation **MSE** but evaluate
   on **rank IC**. In the Alpha360 GRU calibration, BOTH seeds kept improving on test IC after
   validation MSE had peaked (seed 0: 8 of 10 post-best epochs higher; seed 1: post-best mean
   +0.0592 vs selected +0.0583). If that is real, MSE-based stopping halts before the rank-IC
   optimum -- and since qlib's own configs use `metric: loss` too, it would understate the entire
   published Alpha360 leaderboard, not just our runs.

   The decisive column here is **VAL-IC**, not test IC. Validation IC is a legitimate selection
   criterion (it never touches test), so if val IC keeps climbing after val MSE stalls, the fix is
   to stop on val IC -- a change we could make honestly. Test IC is logged alongside only to
   confirm the story, and must never drive the choice.

STOPPING CRITERION IS UNCHANGED AND MUST STAY THAT WAY
This script stops on validation **MSE**, exactly like the calibration and both gate arms, so every
run in the comparison remains directly comparable. Validation IC is only OBSERVED here, never used
to stop. If val IC ever becomes the stopping criterion, EVERY arm has to switch together as a
separate, clearly-labelled experiment -- an MSE-stopped baseline against an IC-stopped treatment
would confound the stopping rule with whatever the treatment was, and the delta would mean nothing.
That mistake is easy to make precisely because switching looks like an improvement.

Usage (after the CS gate finishes):
    python3 scripts/vol_baselines/csi300_valbest_trajectory.py \
        --data datasets/csi300_master_replica_invdata_alpha360 --arm base+cs
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
from csi300_cs_nonlinear_gate import build_samples  # noqa: E402
from csi300_master_replica_lstm_gru_fit import LR, Net, load_panel, predict  # noqa: E402
from ic_stats import daily_ic  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--arm", default="base+cs", choices=["base", "base+cs"])
    ap.add_argument("--model", default="gru", choices=["gru", "lstm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--ckpt-dir", default=None,
                     help="save each val-best checkpoint here (default: <data>/_valbest_ckpts)")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 78)
    print("DIAGNOSTIC RUN -- numbers here are NOT reportable.")
    print("Test IC is observed during training; any tuning informed by it is contaminated.")
    print("The actionable column is VAL-IC, which never touches test.")
    print("=" * 78)
    print(f"[device] {dev}  arm={a.arm}  seed={a.seed}")

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    ckpt_dir = a.ckpt_dir or f"{a.data}/_valbest_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    ev = pd.read_csv(f"{a.data}/eval_index.csv")
    eval_keys = set(zip(ev["date"], ev["instrument"]))
    panel = load_panel(a.data)
    FX = [c for c in next(iter(panel.values())).columns
          if c not in ("date", "segment_id", "split", "LABEL", "LABEL_CSRANK")]

    use_cs = a.arm == "base+cs"
    cs_cols = []
    if use_cs:
        cs = pd.read_csv(f"{a.data}/_cs_features.csv")
        cs_cols = [c for c in cs.columns if c.startswith("CS_")]
        for s in panel:
            g = panel[s]
            if "instrument" not in g.columns:
                g = g.assign(instrument=s)
            g = g.merge(cs, on=["date", "instrument"], how="left")
            g[cs_cols] = g[cs_cols].fillna(0.0)
            panel[s] = g
        print(f"[cs] {len(cs_cols)} CS features attached")

    Xtr, ytr, Xva, yva, Xte, md = build_samples(
        panel, FX, cs_cols, "LABEL_CSRANK", eval_keys, 6, use_cs)
    del panel
    import gc; gc.collect()
    print(f"[fit] train {Xtr.shape} val {Xva.shape} eval {Xte.shape}")

    # validation IC needs (date, instrument, LABEL) for the val rows, which build_samples does not
    # return -- rebuild that index here so val IC can be computed without ever touching test
    vmeta = []
    for split in ("val",):
        for f in sorted(__import__("glob").glob(f"{a.data}/*_{split}.csv")):
            s = os.path.basename(f)[: -len(f"_{split}.csv")]
            d = pd.read_csv(f, usecols=["date", "LABEL"])
            vmeta.append(d.assign(instrument=s))
    vmd = pd.concat(vmeta, ignore_index=True).sort_values(["instrument", "date"])
    if len(vmd) != len(Xva):
        print(f"[warn] val meta {len(vmd)} != val samples {len(Xva)}; val IC disabled")
        vmd = None

    n_in = Xtr.shape[2]
    mu = Xtr.reshape(-1, n_in).mean(0); sd = Xtr.reshape(-1, n_in).std(0)
    sd[sd < 1e-12] = 1.0
    ymu, ysd = float(ytr.mean()), float(ytr.std())
    f = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32)).to(dev)
    Xt, Xv, Xe = f(Xtr), f(Xva), f(Xte)
    yt = torch.from_numpy((ytr - ymu) / ysd).to(dev)
    yv = torch.from_numpy((yva - ymu) / ysd).to(dev)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    m = Net(a.model, a.hidden, n_in, a.layers).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR[a.model]); lf = nn.MSELoss()
    best, bad, n, t0 = np.inf, 0, len(Xt), time.time()
    rows = []

    print(f"\n{'ep':>3} {'val_mse':>9} {'VAL-IC':>8} {'TEST-IC':>8}  note")
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = lf(predict(m, Xv), yv).item()
        improved = vl < best - 1e-5
        if improved:
            best, bad = vl, 0
        else:
            bad += 1

        # evaluate ONLY at val-bests: that is the set of models early stopping actually considers
        if improved:
            with torch.no_grad():
                vp = predict(m, Xv).cpu().numpy() * ysd + ymu
                tp = predict(m, Xe).cpu().numpy() * ysd + ymu
            vic = (daily_ic(vmd.assign(pred=vp), pred_col="pred", label_col="LABEL",
                             date_col="date").mean() if vmd is not None else float("nan"))
            tic = daily_ic(md.assign(pred=tp), pred_col="pred", label_col="LABEL",
                            date_col="date").mean()
            torch.save(m.state_dict(), f"{ckpt_dir}/{a.arm}_s{a.seed}_ep{ep+1}.pt")
            rows.append({"epoch": ep + 1, "val_mse": vl, "val_ic": float(vic),
                          "test_ic": float(tic)})
            print(f"{ep+1:>3} {vl:>9.6f} {vic:>+8.4f} {tic:>+8.4f}  <- NEW VAL-BEST (checkpointed)",
                  flush=True)
        else:
            print(f"{ep+1:>3} {vl:>9.6f} {'':>8} {'':>8}  patience {bad}/{a.patience}", flush=True)
        if bad >= a.patience:
            break

    df = pd.DataFrame(rows)
    out = f"{a.data}/_valbest_trajectory_{a.arm}_s{a.seed}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== val-best trajectory ({len(df)} checkpoints, {time.time()-t0:.0f}s) ===")
    print(df.to_string(index=False))
    if len(df) > 1 and df["val_ic"].notna().all():
        print(f"\n   val IC at first vs last val-best : {df['val_ic'].iloc[0]:+.4f} -> "
              f"{df['val_ic'].iloc[-1]:+.4f}")
        print(f"   test IC at first vs last val-best: {df['test_ic'].iloc[0]:+.4f} -> "
              f"{df['test_ic'].iloc[-1]:+.4f}")
        peak = df.loc[df["val_ic"].idxmax()]
        print(f"   val IC PEAKS at epoch {int(peak['epoch'])} (val_mse {peak['val_mse']:.6f})")
        if int(peak["epoch"]) != int(df["epoch"].iloc[-1]):
            print("   => val IC peaked BEFORE the final val-MSE-best: MSE and IC disagree about")
            print("      when to stop. Stopping on val IC would be a legitimate, test-free fix.")
        else:
            print("   => val IC peaks at the same checkpoint MSE selects: the criteria agree,")
            print("      and the post-best test-IC drift seen earlier was noise, not bias.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
