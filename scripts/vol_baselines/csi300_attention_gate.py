#!/usr/bin/env python3
"""Does TEMPORAL attention pay on this venue -- and if so, is it the attention or just pooling?

WHY THIS GATE
-------------
EXAMM sits ON the GRU capacity curve at every size tested, so the only levers that could move it
OFF the curve change the SEARCH SPACE rather than the parameter count. A temporal-attention node
type is the leading candidate: it is EXAMM-native (a new node is EXAMM's normal extension path), and
unlike a cross-sectional operator it needs NO change to the execution model -- ALSTM's attention is
over TIMESTEPS within a single series (qlib pytorch_alstm.py:331, Softmax(dim=1) over seq_len), which
is exactly what EXAMM's per-series forward pass already provides.

But adding a node type is days of C++ plus Anvil time. This gate answers, for an afternoon and with
no C++ at all, whether the mechanism pays for a HAND-DESIGNED model here. If it does not, evolving
attention nodes will not either.

WHY THREE ARMS, NOT TWO
-----------------------
"ALSTM vs GRU" conflates two different things. Decomposed:

    last : fc_out(cat(h_T, h_T))              -- no temporal pooling (plain recurrent readout)
    mean : fc_out(cat(h_T, mean_t h_t))       -- UNIFORM attention over time
    attn : fc_out(cat(h_T, sum_t a_t h_t))    -- LEARNED attention (ALSTM proper)

  attn - mean  isolates LEARNED attention.
  mean - last  isolates temporal POOLING, which is far cheaper to implement as an EXAMM node.

That distinction is the point of the gate. If most of the gain is mean-vs-last, the node type to
build is a trivial pooling node, not an attention node -- days of C++ saved. If the gain is
attn-vs-mean, the attention machinery is doing real work and is worth building properly. If neither
clears the floor, drop the whole node-type track.

FAIRNESS. One class, one branch switched -- identical RNN, identical fc_in/fc_out, identical init
order (att_net is constructed in every arm so RNG consumption matches; unused parameters get
grad=None and Adam skips them). n_params() counts only parameters actually used, so the arms are
comparable on size. A separately written control would confound architecture with implementation.

Effects here are small relative to resolution: ALSTM's published gain over LSTM is ~+0.005 against a
paired HAC floor of ~0.0053. So a single arm's point estimate is marginal by construction -- report
per-seed sign consistency alongside the ensembled delta, and treat a near-floor result as
"insufficient", not as a verdict.

Usage:
    python3 scripts/vol_baselines/csi300_attention_gate.py --seeds 3
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
from csi300_master_replica_lstm_gru_fit import build_windows, load_panel  # noqa: E402
from ic_stats import daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = 0.005
ARMS = ("last", "mean", "attn")


class TemporalModel(nn.Module):
    """qlib's ALSTM (pytorch_alstm.py:299-343) with the temporal-readout branch switchable."""

    def __init__(self, d_feat=6, hid_size=64, num_layers=2, dropout=0.0, mode="attn",
                 rnn_type="GRU"):
        super().__init__()
        assert mode in ARMS
        self.mode, self.d_feat, self.hid_size = mode, d_feat, hid_size
        self.net = nn.Sequential(nn.Linear(d_feat, hid_size), nn.Tanh())
        klass = getattr(nn, rnn_type.upper())
        self.rnn = klass(input_size=hid_size, hidden_size=hid_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_size * 2, 1)
        # built in EVERY arm so RNG order (and thus rnn/fc_out init) is identical across arms
        self.att_net = nn.Sequential(
            nn.Linear(hid_size, hid_size // 2), nn.Dropout(dropout), nn.Tanh(),
            nn.Linear(hid_size // 2, 1, bias=False), nn.Softmax(dim=1))

    def forward(self, x):                      # x: [N, T, F]
        out, _ = self.rnn(self.net(x))         # [N, T, H]
        h_last = out[:, -1, :]
        if self.mode == "last":
            pooled = h_last
        elif self.mode == "mean":
            pooled = out.mean(dim=1)
        else:
            pooled = torch.sum(out * self.att_net(out), dim=1)
        return self.fc_out(torch.cat((h_last, pooled), dim=1))[..., 0]

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters()
                   if self.mode == "attn" or not n.startswith("att_net"))


def fit(Xtr, ytr, Xva, yva, mode, seed, epochs, patience, lr, batch, hid, log_every):
    torch.manual_seed(seed)
    m = TemporalModel(hid_size=hid, mode=mode).to(Xtr.device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best, best_state, bad, used = float("inf"), None, 0, 0
    n = len(Xtr)
    for ep in range(epochs):
        m.train()
        for i in range(0, n, batch):
            idx = torch.tensor(rng.choice(n, size=min(batch, n - i), replace=False),
                               dtype=torch.long, device=Xtr.device)
            opt.zero_grad()
            lossf(m(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([float(lossf(m(Xva[j:j + 4096]), yva[j:j + 4096]))
                                for j in range(0, len(Xva), 4096)]))
        used = ep + 1
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if log_every and ep % log_every == 0:
            print(f"      ep {ep:3d} val {vl:.6f} best {best:.6f} pat {bad}/{patience}", flush=True)
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best, used


def predict(m, X, chunk=4096):
    m.eval()
    with torch.no_grad():
        return np.concatenate([m(X[i:i + chunk]).cpu().numpy().reshape(-1)
                               for i in range(0, len(X), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3, help="qlib ALSTM default; same for all arms")
    ap.add_argument("--batch", type=int, default=2000, help="qlib ALSTM default")
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

    Xtr, ytr, Xva, yva, Xte, meta, _dtr, _dva = build_windows(
        panel, FX, a.target, a.seq, eval_keys, mode)
    del panel
    gc.collect()
    print(f"[windows] train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    if len(meta) != len(ev):
        sys.exit(f"ERROR: built {len(meta)} test rows but eval_index has {len(ev)}")

    dev = torch.device(a.device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    del Xtr, Xva, Xte
    gc.collect()

    results = {}
    for arm in ARMS:
        ics_seeds, preds, npar = [], [], None
        for s in range(a.seeds):
            t0 = time.time()
            m, vl, eps = fit(Xtr_t, ytr_t, Xva_t, yva_t, arm, s, a.epochs, a.patience,
                             a.lr, a.batch, a.hidden, a.log_every)
            p = predict(m, Xte_t)
            npar = m.n_params()
            ics_seeds.append(daily_ic(meta.assign(pred=p), pred_col="pred",
                                      label_col="LABEL", date_col="date"))
            preds.append(p)
            print(f"   {arm:<4} seed {s}: {eps} ep, val {vl:.6f}, {npar:,}p, "
                  f"{time.time() - t0:.0f}s", flush=True)
        ens = meta.assign(pred=np.mean(preds, axis=0))
        results[arm] = {"ics": daily_ic(ens, pred_col="pred", label_col="LABEL", date_col="date"),
                        "per_seed": ics_seeds, "n_params": npar}

    if len({len(v["ics"]) for v in results.values()}) != 1:
        sys.exit("ERROR: arms scored different numbers of dates -- paired test would be misaligned")

    print(f"\n=== per-arm ({a.seeds}-seed prediction ensemble) ===")
    for k, v in results.items():
        report(f"{k} ({v['n_params']:,}p)", v["ics"], hac)

    print(f"\n=== DECOMPOSITION (paired, same dates) ===")
    contrasts = {}
    for name, hi, lo in (("learned attention (attn - mean)", "attn", "mean"),
                         ("temporal pooling  (mean - last)", "mean", "last"),
                         ("total             (attn - last)", "attn", "last")):
        p = paired(results[hi]["ics"], results[lo]["ics"], hac)
        contrasts[f"{hi}_minus_{lo}"] = p
        ws = [float(np.mean(x - y)) for x, y in zip(results[hi]["per_seed"],
                                                    results[lo]["per_seed"])]
        signs = "same" if len({w > 0 for w in ws}) == 1 else "MIXED"
        print(f"   {name}: {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}   "
              f"per-seed {['%+.4f' % w for w in ws]} ({signs})")

    att = contrasts["attn_minus_mean"]["delta"]
    pool = contrasts["mean_minus_last"]["delta"]
    print(f"\n=== WHAT TO BUILD ===")
    if max(att, pool) < GATE:
        print(f"   Neither contrast clears +{GATE:.4f}. Temporal attention does not pay for a")
        print(f"   hand-designed model on this venue, so an evolved attention node will not")
        print(f"   either. DROP the node-type track.")
    elif pool >= att:
        print(f"   POOLING dominates ({pool:+.4f} vs attention's {att:+.4f}). Build a simple")
        print(f"   temporal-POOLING node, not an attention node -- far less C++ for the same gain.")
    else:
        print(f"   LEARNED ATTENTION dominates ({att:+.4f} vs pooling's {pool:+.4f}). The")
        print(f"   attention machinery is doing real work; building the attention node is")
        print(f"   justified. EXAMM's edge would be placing SEVERAL at heterogeneous depths,")
        print(f"   which a fixed ALSTM cannot express.")
    print(f"\n   Caveat: ALSTM's published gain over LSTM is ~+0.005 against a ~{GATE:.4f} floor,")
    print(f"   so a near-floor result is 'insufficient resolution', not 'no effect'. Weigh the")
    print(f"   per-seed sign consistency above at least as heavily as the point estimate.")

    out = a.out or f"{a.data}/_attention_gate.json"
    with open(out, "w") as f:
        json.dump({"config": {"hidden": a.hidden, "seeds": a.seeds, "lr": a.lr,
                              "batch": a.batch, "hac_lag": hac, "gate": GATE},
                   "per_arm": {k: {"ic": float(np.mean(v["ics"])), "n_params": v["n_params"]}
                               for k, v in results.items()},
                   "contrasts": {k: {kk: float(vv) for kk, vv in v.items()}
                                 for k, v in contrasts.items()}}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
