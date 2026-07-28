#!/usr/bin/env python3
"""Does a LEARNED relational module produce any gain on this venue at all?

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is NOT an "efficiency at the frontier" test. That idea -- "SOTA-class accuracy with an N-fold
smaller temporal encoder" -- is already falsified by section 4 of results/csi300_benchmark_results.md
and needed no new experiment:

    GRU 2x64 (38,849p) +0.0564      GRU 2x2 (99p) +0.0295      encoder deficit -0.0269

Published relational gains on Alpha360 are GATs +0.0014, IGMTF +0.0022, HIST +0.0083 -- 5%, 8% and
31% of that deficit. So even at PERFECT additivity, a tiny encoder plus the best published relational
head reaches +0.0378 and still loses to a large encoder with NO relational head (+0.0564) by -0.0186.
The encoder supplies 69-95% of the signal; the relational module cannot substitute for it. Any
"compact evolved encoder + relational head" claim is arithmetically closed before it starts.

What IS open, and what this script measures: does GATs' LEARNED peer attention beat its own no-
attention control on this venue? Section 7 makes that the sharp question -- HAND-SPECIFIED peer
features measured -0.0078 (HAC t -2.96) here, while HIST's LEARNED relational structure is worth
+0.0083 elsewhere. Same magnitude, opposite sign. If learning the structure is what matters, a
learned attention module should land on the positive side of that contrast. If it also comes out
negative, the relational route on this venue is empirically dead and no amount of architecture work
rescues it.

PRE-REGISTERED GATE (fix before looking): the h64 relational gain must be >= +0.005.
Below that, stop. A module that adds nothing at a STRONG encoder cannot rescue a weak one, and the
arithmetic above says even a large additive gain would not close the encoder deficit anyway.
Note +0.005 is ~3.5x the published GATs gain, so clearing it would itself be the finding.

DESIGN
------
Paired: {relational on, relational off} at one encoder size, several seeds, everything else
identical. The control is THE SAME CLASS with the attention-mixing branch skipped -- identical RNN,
head, init order, batching and training loop; a separately written GRU would confound architecture
with implementation. (Verified: all parameters are bit-identical across the two arms at a given seed,
because nn.GRU is constructed first and self.a's randn is consumed in both.)

Base architecture is qlib's GATs (qlib/contrib/model/pytorch_gats.py:326), inlined because qlibenv
has qlib but no torch while system python has torch but no qlib. GATs rather than HIST because HIST
needs an external stock2concept matrix we do not have; GATs learns its graph from the data.

DATE-GROUPED BATCHING IS MANDATORY, IN BOTH ARMS. GAT attention is computed ACROSS THE BATCH
(cal_attention builds an NxN matrix over samples), so a batch must be exactly one date's
cross-section or the "peers" are unrelated dates -- a different model, not a degraded one. The
control batches identically although it does not need to, because batch composition changes gradient
noise and would otherwise be a second difference between arms.

WHY --hidden TAKES ONE SIZE BY DEFAULT
A ratio of gains across encoder sizes is NOT estimable here. With a true gain near GATs' +0.0014 and
a paired HAC SE of ~0.0026, the denominator is insignificant (|t| ~ 0.5) and by Fieller's theorem the
ratio's confidence set is unbounded -- additive and multiplicative truths produce nearly identical
ratio distributions. If several sizes are passed anyway, this reports DIFFERENCE-IN-DIFFERENCES with
its own paired HAC SE, which has finite variance and a well-defined null. (For reference, hidden=16
gives 3,073 control params, a 6-parameter match to EXAMM's 3,079-weight operating point.)

h64_base here will NOT reproduce the +0.0581 GRU baseline and must not be quoted against it: no
input/target standardisation, an extra fc+LeakyReLU head, date-batching, and a different validation
criterion. It is a matched control for THIS contrast only.

Usage:
    python3 scripts/vol_baselines/csi300_encoder_sensitivity.py --seeds 3
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
GATE = 0.005  # pre-registered; see module docstring


class RelModel(nn.Module):
    """qlib GATs (pytorch_gats.py:326) with the attention step made optional."""

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, relational=True):
        super().__init__()
        self.rnn = nn.GRU(input_size=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.hidden_size, self.d_feat, self.relational = hidden_size, d_feat, relational
        # Constructed in BOTH arms so RNG consumption order -- and therefore every other tensor's
        # init for a given seed -- is identical. Unused ones get grad=None and Adam skips them.
        self.transformation = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.randn(hidden_size * 2, 1))
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)
        n, dim = x.shape[0], x.shape[1]
        e_x = x.expand(n, n, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        out = torch.t(self.a).mm(torch.t(attention_in)).view(n, n)
        return self.softmax(self.leaky_relu(out))

    def forward(self, x):
        out, _ = self.rnn(x)                 # x: [N, T, F]
        hidden = out[:, -1, :]
        if self.relational:
            hidden = self.cal_attention(hidden, hidden).mm(hidden) + hidden
        return self.fc_out(self.leaky_relu(self.fc(hidden))).squeeze(-1)

    def n_params(self):
        """Params ACTUALLY used. The control never invokes transformation/a, so counting them would
        overstate its size."""
        return sum(p.numel() for n, p in self.named_parameters()
                   if self.relational or not n.startswith(("transformation", "a")))


def date_batches(dates, device, min_size=2):
    """Indices grouped by date, as device tensors. One batch == one date's cross-section.

    Returned on-device (not numpy) because indexing a device tensor with a numpy array forces a
    host<->device sync on every step, and there are ~1,700 steps per epoch.
    Dates with < min_size members are dropped: attention over a 1-stock cross-section is a no-op, so
    such a batch would train the two arms on different effective objects.
    """
    order = np.argsort(dates, kind="stable")
    d = dates[order]
    cuts = np.flatnonzero(np.r_[True, d[1:] != d[:-1], True])
    out = [order[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    return [torch.tensor(b, dtype=torch.long, device=device) for b in out if len(b) >= min_size]


def fit(Xtr, ytr, tr_b, Xva, yva, va_b, hidden, relational, seed, epochs, patience, lr, log_every):
    torch.manual_seed(seed)
    m = RelModel(hidden_size=hidden, relational=relational).to(Xtr.device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best, best_state, bad, used = float("inf"), None, 0, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for bi in rng.permutation(len(tr_b)):
            idx = tr_b[bi]
            opt.zero_grad()
            lossf(m(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([float(lossf(m(Xva[i]), yva[i])) for i in va_b]))
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
    ap.add_argument("--hidden", type=int, nargs="+", default=[64],
                    help="encoder size(s). ONE by default -- the gate is whether learned attention "
                         "helps at a strong encoder. Several sizes -> difference-in-differences.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4, help="matches the GRU baseline's LR['gru']")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--target", default="LABEL_CSRANK")
    ap.add_argument("--seq", type=int, default=60)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = max(1, man["label_horizon_trading_days"] - 1)
    mode = "reshape" if man.get("handler") == "alpha360" else "window"
    print(f"[dataset] {os.path.basename(a.data)} handler={man.get('handler')} mode={mode} "
          f"HAC lag={hac} device={a.device}")

    ev = pd.read_csv(f"{a.data}/eval_index.csv")
    eval_keys = set(zip(ev["date"], ev["instrument"]))
    panel = load_panel(a.data)
    head = pd.read_csv(sorted(glob.glob(f"{a.data}/*_train.csv"))[0], nrows=0).columns
    FX = [c for c in head if c not in ("date", "segment_id", "LABEL", "LABEL_CSRANK", "split")]
    # mode='reshape' hardcodes n_field=6 and reshapes each row to (60, 6). A dataset with a
    # different width would reshape into garbage silently, so check rather than trust the handler.
    if mode == "reshape" and len(FX) != 360:
        sys.exit(f"ERROR: reshape mode expects 360 features (60 lags x 6 fields), got {len(FX)}")
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
    covered = sum(len(b) for b in te_b)
    print(f"[batches] train {len(tr_b)} dates, val {len(va_b)}, test {len(te_b)} "
          f"(rows covered {covered}/{len(meta)})")
    if covered != len(meta):
        sys.exit(f"ERROR: {len(meta) - covered} test rows sit on dates with <2 stocks and would go "
                 f"unscored; every eval_index row must get a prediction")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)
    del Xtr, Xva, Xte
    gc.collect()

    results = {}
    for hidden in a.hidden:
        for relational in (True, False):
            tag = f"h{hidden}_{'rel' if relational else 'base'}"
            ics_seeds, preds, npar = [], [], None
            for s in range(a.seeds):
                t0 = time.time()
                m, vl, eps = fit(Xtr_t, ytr_t, tr_b, Xva_t, yva_t, va_b, hidden, relational,
                                 s, a.epochs, a.patience, a.lr, a.log_every)
                p = predict(m, Xte_t, te_b, len(meta))
                if np.isnan(p).any():
                    sys.exit(f"ERROR: {tag} seed {s} left {int(np.isnan(p).sum())} rows unscored")
                npar = m.n_params()
                ics_seeds.append(daily_ic(meta.assign(pred=p), pred_col="pred",
                                          label_col="LABEL", date_col="date"))
                preds.append(p)
                print(f"   {tag} seed {s}: {eps} ep, val {vl:.6f}, {npar:,}p, "
                      f"{time.time() - t0:.0f}s", flush=True)
            ens = meta.assign(pred=np.mean(preds, axis=0))
            results[tag] = {"ics": daily_ic(ens, pred_col="pred", label_col="LABEL",
                                            date_col="date"),
                            "per_seed": ics_seeds, "n_params": npar}

    # paired() checks only length, and daily_ic silently drops zero-variance dates. If two arms
    # dropped DIFFERENT dates and the counts happened to match, the paired test would be silently
    # misaligned. Guard explicitly.
    lens = {k: len(v["ics"]) for k, v in results.items()}
    if len(set(lens.values())) != 1:
        sys.exit(f"ERROR: configs scored different numbers of dates {lens} -- paired tests would be "
                 f"misaligned")

    print(f"\n=== per-configuration ({a.seeds}-seed prediction ensemble) ===")
    for tag, r in results.items():
        report(f"{tag} ({r['n_params']:,}p)", r["ics"], hac)

    print(f"\n=== RELATIONAL GAIN (paired: same dates, same seeds, only attention differs) ===")
    gains = {}
    for hidden in a.hidden:
        rel, base = results[f"h{hidden}_rel"], results[f"h{hidden}_base"]
        p = paired(rel["ics"], base["ics"], hac)
        gains[hidden] = p
        lo, hi = p["delta"] - 2 * p["se_hac"], p["delta"] + 2 * p["se_hac"]
        print(f"   hidden {hidden:>3}: rel - base = {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}"
              f"   95% [{lo:+.4f}, {hi:+.4f}]")
        ws = [float(np.mean(r - b)) for r, b in zip(rel["per_seed"], base["per_seed"])]
        print(f"                within-seed: {['%+.4f' % w for w in ws]}")

    big = max(a.hidden)
    g = gains[big]["delta"]
    print(f"\n=== PRE-REGISTERED GATE ===")
    print(f"   h{big} relational gain {g:+.4f} vs gate +{GATE:.4f} -> "
          f"{'PASS' if g >= GATE else 'FAIL'}")
    if g < GATE:
        print(f"   Learned attention does not clear the gate at a strong encoder. It cannot rescue")
        print(f"   a weak one, and section 4's arithmetic (encoder deficit h64->h2 = -0.0269 vs a")
        print(f"   best published relational gain of +0.0083) closes the compact-encoder route")
        print(f"   regardless. STOP -- do not invest further in relational architecture here.")
    else:
        print(f"   Clears the gate at {g / 0.0014:.1f}x the published GATs gain -- that is itself")
        print(f"   the finding. Re-run with --hidden 64 16 2 for the size curve (h16 = 3,073p")
        print(f"   matches EXAMM's 3,079-weight operating point).")

    if len(a.hidden) > 1:
        # difference-in-differences, NOT a ratio: with an insignificant denominator a ratio's
        # confidence set is unbounded (Fieller), and additive vs multiplicative truths produce
        # near-identical ratio distributions at this effect size.
        small = min(a.hidden)
        d = paired(results[f"h{small}_rel"]["ics"] - results[f"h{small}_base"]["ics"],
                   results[f"h{big}_rel"]["ics"] - results[f"h{big}_base"]["ics"], hac)
        print(f"\n=== difference-in-differences (gain@h{small} - gain@h{big}) ===")
        print(f"   {d['delta']:+.4f}  HAC t {d['t_hac']:+.2f}   "
              f"(0 = gain is size-independent/additive; negative = gain shrinks at the small "
              f"encoder)")

    out = a.out or f"{a.data}/_encoder_sensitivity.json"
    with open(out, "w") as f:
        json.dump({"config": {"hidden": a.hidden, "seeds": a.seeds, "epochs": a.epochs,
                              "lr": a.lr, "hac_lag": hac, "n_eval_rows": int(len(meta)),
                              "gate": GATE},
                   "per_config": {k: {"ic": float(np.mean(v["ics"])), "n_params": v["n_params"]}
                                  for k, v in results.items()},
                   "gains": {str(k): {kk: float(vv) for kk, vv in v.items()}
                             for k, v in gains.items()}}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
