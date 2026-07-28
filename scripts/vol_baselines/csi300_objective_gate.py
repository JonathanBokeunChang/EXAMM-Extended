#!/usr/bin/env python3
"""Does a cross-sectional IC objective beat MSE on THIS venue? (cheap PyTorch gate)

WHY THIS EXISTS, AND WHY IT IS NOT ALREADY ANSWERED
---------------------------------------------------
Six objectives have been tested on this project and all lost to raw MSE (raw-MSE +0.0195 >
z-score-MSE +0.0044 > mean-IC +0.008 > ICIR +0.0002, plus Huber and dispersion-penalized MSE). That
looks decisive, but it was measured on the US walk-forward cohort -- 50 stocks, 1 year -- where the
recorded mechanism was:

    "the more an objective amplifies cross-sectional dispersion, the worse it overfits;
     raw-MSE's tiny natural target scale is a REGULARIZER"

That mechanism DOES NOT EXIST on this venue. Here the target is LABEL_CSRANK, already
cross-sectionally rank-normalized to +-1.73 with unit variance -- so "raw MSE" here is itself a
dispersion-amplified objective, and the regularizer that won on the cohort is absent. Separately,
the cohort's per-date IC was ~pure noise (sd 0.143 = 1/sqrt(49)); here it is 1/sqrt(433) = 0.048.
Both legs of the prior null fail to transfer, so the question is genuinely open.

WHY PYTORCH AND NOT EXAMM
-------------------------
EXAMM's cross-sectional loss path REQUIRES calendar-aligned equal-length training series (asserted
in both anvil_ic.sb and the C++). The CSI300 seq dataset is ragged -- train lengths run 55..1690 --
so aligning it would drop ~half the universe and ~45% of training rows, which would weaken the very
cross-section the objective needs. In torch, date-grouped batching handles ragged series natively
and no alignment is needed.

So this gate answers the OBJECTIVE question at full universe strength for the cost of an afternoon.
If the IC objective does not beat MSE for a GRU here, it will not beat it for EXAMM on a halved
universe, and the whole EXAMM IC-loss track can be dropped without building an aligner.

THE SURROGATE IS EXACT ENOUGH TO BE HONEST ABOUT
------------------------------------------------
Within a date, the target is already a rank transform, so Pearson(pred, LABEL_CSRANK) equals
Spearman(pred, LABEL) up to the tie structure of the ranking -- i.e. the differentiable surrogate is
essentially the eval metric, not a loose proxy. Predictions are NOT ranked (that would kill the
gradient); only the target is, and it arrives pre-ranked from the builder.

COLLAPSE GUARD. Correlation is scale- and shift-invariant, so it is degenerate at constant
predictions (0/0) and can be gamed by shrinking dispersion. This project already established that
the fix is a VICReg-style VARIANCE FLOOR, and that the obvious alternative (-IC + lambda*MSE) FAILS.
So the loss is -corr + lambda * relu(target_std - pred_std)^2.

SELECTION IS SYMMETRIC AND IS NOT val MSE. Both arms select on VALIDATION IC. Selecting on val MSE
would handicap the arm that is not optimizing MSE, and this venue has two recorded inversions where
val MSE moved opposite to test IC (grow3 vs the seed; the CS-features arm in section 7). Val IC is
the metric both arms are ultimately judged on, so it is the only symmetric choice -- while noting
val IC is itself noisy here (486 val dates, SE ~0.008), which is a limitation of the gate, not a bug.

Usage:
    python3 scripts/vol_baselines/csi300_objective_gate.py --seeds 3
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi300_encoder_sensitivity import RelModel, date_batches  # noqa: E402
from csi300_master_replica_lstm_gru_fit import build_windows, load_panel  # noqa: E402
from ic_stats import daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = 0.005   # pre-registered: IC-loss minus MSE-loss must clear the ~0.0053 paired floor


def cs_ic_loss(pred, y, var_lambda, eps=1e-8):
    """-Pearson(pred, y) within one date's cross-section, plus a variance floor.

    Correlation is invariant to scale and shift, so it is undefined at constant predictions and
    rewards nothing for matching the target's spread. The floor penalises pred_std falling below
    the target's, which is the anti-collapse form this project found to work (a -IC + lambda*MSE
    hybrid was tried and failed).
    """
    p = pred - pred.mean()
    t = y - y.mean()
    corr = (p * t).sum() / (p.norm() * t.norm() + eps)
    loss = -corr
    if var_lambda > 0:
        loss = loss + var_lambda * torch.relu(y.std() - pred.std()).pow(2)
    return loss


def val_ic(m, X, y_df, va_b, n):
    """Mean daily cross-sectional Spearman IC on validation -- the SYMMETRIC selection metric."""
    m.eval()
    out = np.full(n, np.nan)
    with torch.no_grad():
        for idx in va_b:
            out[idx.cpu().numpy()] = m(X[idx]).cpu().numpy().reshape(-1)
    d = y_df.assign(pred=out)
    ics = daily_ic(d, pred_col="pred", label_col="LABEL", date_col="date")
    return float(np.mean(ics))


def fit(Xtr, ytr, tr_b, Xva, yva_df, va_b, n_va, hidden, objective, var_lambda,
        seed, epochs, patience, lr, log_every):
    torch.manual_seed(seed)
    m = RelModel(hidden_size=hidden, relational=False).to(Xtr.device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    mse = nn.MSELoss()
    best, best_state, bad, used = -np.inf, None, 0, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for bi in rng.permutation(len(tr_b)):
            idx = tr_b[bi]
            opt.zero_grad()
            pred = m(Xtr[idx])
            loss = mse(pred, ytr[idx]) if objective == "mse" \
                else cs_ic_loss(pred, ytr[idx], var_lambda)
            loss.backward()
            opt.step()
        vi = val_ic(m, Xva, yva_df, va_b, n_va)
        used = ep + 1
        # HIGHER val IC is better -- both arms selected on the same metric (see module docstring)
        if vi > best + 1e-7:
            best, bad = vi, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if log_every and ep % log_every == 0:
            print(f"      ep {ep:3d} val_IC {vi:+.5f} best {best:+.5f} pat {bad}/{patience}",
                  flush=True)
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best, used


def predict(m, X, te_b, n):
    m.eval()
    out = np.full(n, np.nan)
    with torch.no_grad():
        for idx in te_b:
            out[idx.cpu().numpy()] = m(X[idx]).cpu().numpy().reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--var-lambda", type=float, default=1.0,
                    help="VICReg-style variance floor on the IC arm; 0 disables (collapse risk)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--target", default="LABEL_CSRANK")
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = max(1, man["label_horizon_trading_days"] - 1)
    mode = "reshape" if man.get("handler") == "alpha360" else "window"
    print(f"[dataset] {os.path.basename(a.data)} mode={mode} HAC lag={hac} device={a.device}")

    ev = pd.read_csv(f"{a.data}/eval_index.csv")
    eval_keys = set(zip(ev["date"], ev["instrument"]))
    panel = load_panel(a.data)
    head = pd.read_csv(sorted(glob.glob(f"{a.data}/*_train.csv"))[0], nrows=0).columns
    FX = [c for c in head if c not in ("date", "segment_id", "LABEL", "LABEL_CSRANK", "split")]
    if mode == "reshape" and len(FX) != 360:
        sys.exit(f"ERROR: reshape mode expects 360 features, got {len(FX)}")
    print(f"[data] {len(panel)} stocks, {len(FX)} features, {len(ev)} eval rows")

    Xtr, ytr, Xva, yva, Xte, meta, dtr, dva = build_windows(
        panel, FX, a.target, a.seq, eval_keys, mode)
    del panel
    gc.collect()
    print(f"[windows] train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    if len(meta) != len(ev):
        sys.exit(f"ERROR: built {len(meta)} test rows but eval_index has {len(ev)}")

    dev = torch.device(a.device)
    tr_b, va_b = date_batches(dtr, dev), date_batches(dva, dev)
    te_b = date_batches(meta["date"].to_numpy(), dev)
    if sum(len(b) for b in te_b) != len(meta):
        sys.exit("ERROR: some eval_index rows sit on dates with <2 stocks and would go unscored")
    # val selection frame: the CSRANK target is rank-equivalent to LABEL within a date (the builder
    # asserts per-date Spearman(LABEL, LABEL_CSRANK) = 1.000000), so it is a valid ranking target.
    yva_df = pd.DataFrame({"date": dva, "LABEL": yva})
    print(f"[batches] train {len(tr_b)} dates, val {len(va_b)}, test {len(te_b)}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    del Xtr, Xva, Xte
    gc.collect()

    results = {}
    for objective in ("mse", "ic"):
        ics_seeds, preds = [], []
        for s in range(a.seeds):
            t0 = time.time()
            m, vi, eps_used = fit(Xtr_t, ytr_t, tr_b, Xva_t, yva_df, va_b, len(yva),
                                  a.hidden, objective, a.var_lambda, s,
                                  a.epochs, a.patience, a.lr, a.log_every)
            p = predict(m, Xte_t, te_b, len(meta))
            if np.isnan(p).any():
                sys.exit(f"ERROR: {objective} seed {s} left rows unscored")
            sd = float(np.std(p))
            if sd < 1e-6:
                print(f"   WARNING: {objective} seed {s} collapsed (pred sd {sd:.2e}) -- raise "
                      f"--var-lambda")
            ics_seeds.append(daily_ic(meta.assign(pred=p), pred_col="pred",
                                      label_col="LABEL", date_col="date"))
            preds.append(p)
            print(f"   {objective:<3} seed {s}: {eps_used} ep, val_IC {vi:+.5f}, "
                  f"pred sd {sd:.4f}, {time.time() - t0:.0f}s", flush=True)
        ens = meta.assign(pred=np.mean(preds, axis=0))
        results[objective] = {"ics": daily_ic(ens, pred_col="pred", label_col="LABEL",
                                              date_col="date"),
                              "per_seed": ics_seeds}

    if len({len(v["ics"]) for v in results.values()}) != 1:
        sys.exit("ERROR: arms scored different numbers of dates -- paired test would be misaligned")

    print(f"\n=== per-objective ({a.seeds}-seed prediction ensemble) ===")
    for k, v in results.items():
        report(f"loss={k}", v["ics"], hac)

    p = paired(results["ic"]["ics"], results["mse"]["ics"], hac)
    ws = [float(np.mean(i - m_)) for i, m_ in zip(results["ic"]["per_seed"],
                                                  results["mse"]["per_seed"])]
    print(f"\n=== IC-loss minus MSE-loss (paired, same dates) ===")
    print(f"   delta {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}   "
          f"95% [{p['delta'] - 2 * p['se_hac']:+.4f}, {p['delta'] + 2 * p['se_hac']:+.4f}]")
    print(f"   within-seed: {['%+.4f' % w for w in ws]}")

    print(f"\n=== PRE-REGISTERED GATE ===")
    print(f"   delta {p['delta']:+.4f} vs gate +{GATE:.4f} -> "
          f"{'PASS' if p['delta'] >= GATE else 'FAIL'}")
    if p["delta"] < GATE:
        print(f"   The cross-sectional IC objective does not beat MSE for a GRU at FULL universe")
        print(f"   strength. It will not beat it for EXAMM on the ~181-stock aligned subset that")
        print(f"   EXAMM's cross-sectional path would require. DROP the EXAMM IC-loss track --")
        print(f"   do not build the aligner.")
    else:
        print(f"   Objective helps here. Building the aligner and running EXAMM's --loss ic arm")
        print(f"   is now justified by evidence. Expect attenuation: EXAMM's cross-section would")
        print(f"   be ~181 stocks vs {len(ev) // len(te_b)} here.")

    out = a.out or f"{a.data}/_objective_gate.json"
    with open(out, "w") as f:
        json.dump({"config": {"hidden": a.hidden, "seeds": a.seeds, "lr": a.lr,
                              "var_lambda": a.var_lambda, "hac_lag": hac, "gate": GATE},
                   "ic": float(np.mean(results["ic"]["ics"])),
                   "mse": float(np.mean(results["mse"]["ics"])),
                   "paired": {k: float(v) for k, v in p.items()},
                   "within_seed": ws}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
