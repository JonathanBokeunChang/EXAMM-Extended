"""LSTM/GRU baselines on the MASTER-replica CSI300 dataset.

v2 protocol changes (see the dataset builder's header for the measurements behind each):

  * Windows are SEGMENT-AWARE. A window may never span a gap where the stock left and rejoined
    the index -- 430/712 stocks have such gaps (up to 9 segments, largest hole 915 days), so v1's
    naive per-stock windowing was feeding models 8-step "recent history" spanning ~900 calendar
    days. Windows are built strictly within one (instrument, segment_id).
  * Scores ONLY eval_index.csv, the dataset's frozen row set. v1 scored 240 stocks / 189 per date
    here while the ridge gate scored 430 / 290 -- the two numbers were never comparable.
  * Standard errors from ic_stats (Newey-West HAC). v1's naive t = 9.20 for LSTM was ~2x inflated
    by the 4-day overlapping label.
  * Trains on LABEL_CSRANK (MASTER's actual target); always evaluates against raw LABEL.
  * Per-model learning rates matching the official qlib benchmark configs (LSTM 1e-3, GRU 2e-4).
    v1 used 1e-3 for both, which plausibly explains why GRU (+0.0313) landed below even the ridge
    ceiling there -- that anomaly is retested here rather than left as folklore.

Otherwise mirrors eval_lstm.py's documented fairness protocol: pooled training over all stocks,
train+val standardisation, window ending AT row i inclusive, val-based early stopping, multi-seed
ensemble. Architecture (hidden 64 x 2 layers) matches the official Alpha158 GRU/LSTM benchmarks.

Usage:
    python3 scripts/vol_baselines/csi300_master_replica_lstm_gru_fit.py \
        --data datasets/csi300_master_replica_invdata
"""
from __future__ import annotations

import argparse
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
from ic_stats import DEFAULT_HAC_LAG, daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LR = {"lstm": 1e-3, "gru": 2e-4}  # official qlib workflow_config_{lstm,gru}_Alpha158.yaml


class Net(nn.Module):
    def __init__(self, kind, hid, n_in, layers):
        super().__init__()
        R = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = R(n_in, hid, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def predict(model, X, chunk=8192):
    """Forward in chunks. A single pass over the 250k x 60 x 6 eval tensor exhausts MPS memory,
    and chunking costs nothing since there is no cross-sample dependence."""
    outs = []
    for i in range(0, len(X), chunk):
        outs.append(model(X[i:i + chunk]))
    return torch.cat(outs)


def load_panel(data_dir):
    """One long frame per stock across all splits, in date order, carrying its split label.

    Feature columns are read as float32, not pandas' default float64. On Alpha360 the panel is
    ~918k rows x 362 columns, which is ~2.7 GB in float64 and gets the process OOM-killed on a
    machine that also has to hold the numpy sample arrays; float32 halves it and is lossless here
    anyway, since qlib stores float32 and the CSVs were written with %.9g (float32 round-trip).
    """
    files = sorted(glob.glob(f"{data_dir}/*_train.csv"))
    if not files:
        sys.exit(f"ERROR: no *_train.csv under {data_dir}")
    head = pd.read_csv(files[0], nrows=0).columns
    dtypes = {c: np.float32 for c in head if c not in ("date", "segment_id")}
    dtypes["segment_id"] = np.int32

    per = {}
    for split in ("train", "val", "test"):
        for f in sorted(glob.glob(f"{data_dir}/*_{split}.csv")):
            s = os.path.basename(f)[: -len(f"_{split}.csv")]
            d = pd.read_csv(f, dtype=dtypes)
            d["split"] = split
            per.setdefault(s, []).append(d)
    out = {}
    for s in list(per):
        g = pd.concat(per.pop(s), ignore_index=True).sort_values(["date"]).reset_index(drop=True)
        out[s] = g
    return out


def build_windows(panel, FX, target, seq, eval_keys, mode, n_field=6):
    """Emit (X, y) sequence samples. Two modes, because the two feature venues encode history
    in fundamentally different places:

    mode='window' (Alpha158): each row holds ONE day of features, so a sample is `seq` consecutive
        rows. Windows never cross a segment boundary, so every timestep is real observed history.

    mode='reshape' (Alpha360): each row ALREADY holds 60 lags x 6 fields flattened field-major
        (CLOSE59..CLOSE0, OPEN59..OPEN0, ...). A sample is therefore a SINGLE row reshaped to
        (60, 6) -- exactly what qlib's own GRU/LSTM do internally
        (x.reshape(N, d_feat, -1).permute(0, 2, 1) with d_feat=6). Windowing rows here would be
        wrong twice over: it would stack 60-day histories 60 deep, and it would break comparability
        with the published Alpha360 numbers. The 60-contiguous-day requirement is already enforced
        by the dataset's eval index, so each row's embedded history is genuine.
    """
    Xtr, ytr, Xva, yva, Xte, meta = [], [], [], [], [], []
    for s, g in panel.items():
        for _, seg in g.groupby("segment_id", sort=True):
            seg = seg.reset_index(drop=True)
            X = seg[FX].to_numpy(np.float32)
            y = seg[target].to_numpy(np.float32)
            sp = seg["split"].to_numpy()
            dt = seg["date"].to_numpy()
            lb = seg["LABEL"].to_numpy()
            if mode == "reshape":
                # (n_rows, 360) -> (n_rows, 6, 60) -> (n_rows, 60, 6)
                W = X.reshape(len(X), n_field, -1).transpose(0, 2, 1)
                idx = range(len(seg))
            else:
                if len(seg) < seq:
                    continue
                W = None
                idx = range(seq - 1, len(seg))
            for i in idx:
                w = W[i] if mode == "reshape" else X[i - seq + 1:i + 1]
                if sp[i] == "train":
                    Xtr.append(w); ytr.append(y[i])
                elif sp[i] == "val":
                    Xva.append(w); yva.append(y[i])
                elif (dt[i], s) in eval_keys:
                    Xte.append(w); meta.append((s, dt[i], lb[i]))
    return (np.asarray(Xtr), np.asarray(ytr), np.asarray(Xva), np.asarray(yva),
            np.asarray(Xte), pd.DataFrame(meta, columns=["instrument", "date", "LABEL"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--hac-lag", type=int, default=None,
                     help="default: label_horizon-1 from MANIFEST")
    ap.add_argument("--target", choices=["LABEL", "LABEL_CSRANK"], default="LABEL_CSRANK")
    ap.add_argument("--models", default="lstm,gru")
    ap.add_argument("--train-universe", default=None,
                     help="file listing instruments to TRAIN on (one per line, # comments). "
                          "Exists for fairness with EXAMM, which pairs training/validation "
                          "filenames index-wise and so can only use stocks having BOTH splits. "
                          "Evaluation is unaffected -- all models still score the full "
                          "frozen eval_index.")
    ap.add_argument("--log-every", type=int, default=1,
                     help="print val loss every N epochs (0 = silent); each epoch on "
                          "Alpha360's 60-step sequences takes minutes, so per-epoch "
                          "output is the only way to see progress inside a seed")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"],
                     help="auto picks Apple-GPU MPS when available. A 60-timestep RNN cannot "
                          "parallelise over time, so CPU runs are dominated by sequential depth.")
    ap.add_argument("--log-test-ic", action="store_true",
                     help="ALSO print TEST rank IC every epoch. DIAGNOSTIC ONLY -- see the warning "
                          "printed at startup. Never use a run with this flag as a reported number.")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("mps" if (a.device in ("auto", "mps")
                                  and torch.backends.mps.is_available()) else "cpu")
    print(f"[device] {dev} (torch threads {torch.get_num_threads()})")
    if a.log_test_ic:
        print("=" * 78)
        print("DIAGNOSTIC MODE: test IC is printed every epoch.")
        print("This run's numbers are NOT reportable. Watching test during training makes any")
        print("subsequent choice (epochs, patience, seeds, architecture) contaminated by test")
        print("knowledge -- the winner's-curse pattern this project already hit once with a live")
        print("tracker (+0.0188 live vs +0.0113 honest). Use it to SEE the training dynamics;")
        print("bank reportable numbers from a run WITHOUT this flag.")
        print("=" * 78)

    mf = f"{a.data}/MANIFEST.json"
    seq, handler, hac_default = 8, "alpha158", DEFAULT_HAC_LAG
    if os.path.exists(mf):
        m = json.load(open(mf))
        seq = m["seq_for_eligibility"]
        handler = m.get("handler", "alpha158")
        hac_default = max(1, m.get("label_horizon_trading_days", DEFAULT_HAC_LAG + 1) - 1)
        print(f"[dataset] {m['dataset']}  bundle={m['bundle'].get('bundle')} "
              f"handler={handler} content={m['content_sha256'][:16]}...")
    if a.hac_lag is None:
        a.hac_lag = hac_default
    # Alpha360 rows already embed the 60-lag history; Alpha158 rows are single days.
    mode = "reshape" if handler == "alpha360" else "window"

    ev = pd.read_csv(f"{a.data}/eval_index.csv")
    eval_keys = set(zip(ev["date"], ev["instrument"]))
    panel = load_panel(a.data)
    if a.train_universe:
        keep = {ln.strip() for ln in open(a.train_universe)
                if ln.strip() and not ln.startswith('#')}
        # Mark non-universe TRAIN/VAL rows as history-only rather than DROPPING them. Dropping
        # removes the 60-day history that those stocks' early TEST rows need to form a window,
        # which silently shrinks the eval set (measured: 235,277 -> 232,846) and breaks the
        # frozen-index guarantee. Marked rows still participate as history; they just never
        # become training samples.
        for st in list(panel):
            if st not in keep:
                g = panel[st].copy()
                m = g['split'].isin(['train', 'val'])
                g.loc[m, 'split'] = 'histonly'
                panel[st] = g
        n_tr = sum((g['split'] == 'train').any() for g in panel.values())
        print(f'[universe] training restricted to {len(keep)} stocks ({n_tr} contribute train rows); eval index unchanged')
    FX = [c for c in next(iter(panel.values())).columns
          if c not in ("date", "segment_id", "split", "LABEL", "LABEL_CSRANK")]
    n_field = 6
    if mode == "reshape":
        assert len(FX) % n_field == 0, f"{len(FX)} features not divisible by {n_field} fields"
        t_steps, n_in = len(FX) // n_field, n_field
    else:
        t_steps, n_in = seq, len(FX)
    print(f"[fit] {len(panel)} stocks, {len(FX)} cols -> mode={mode}, "
          f"input=({t_steps} timesteps x {n_in} feat), target={a.target}, "
          f"hidden={a.hidden}x{a.layers}, seeds={a.seeds}, HAC lag={a.hac_lag}")

    Xtr, ytr, Xva, yva, Xte, md = build_windows(panel, FX, a.target, seq, eval_keys, mode, n_field)
    del panel                      # ~1.3 GB of DataFrames no longer needed once sampled
    import gc; gc.collect()
    print(f"[fit] windows train {Xtr.shape} val {Xva.shape} eval {Xte.shape}")
    if len(md) != len(ev):
        sys.exit(f"ERROR: produced {len(md)} eval windows but eval_index.csv has {len(ev)} rows. "
                 f"Every eval row must be scorable -- the dataset guarantees >= seq contiguous "
                 f"in-segment history for each. Dataset and index are out of sync.")
    print(f"[fit] eval rows match eval_index exactly ({len(md)})")

    # train+val statistics only -- never test
    stack = np.concatenate([Xtr.reshape(-1, n_in), Xva.reshape(-1, n_in)])
    mu, sd = stack.mean(0), stack.std(0)
    sd[sd < 1e-12] = 1.0
    ymu, ysd = float(np.concatenate([ytr, yva]).mean()), float(np.concatenate([ytr, yva]).std())
    norm = lambda A: ((A - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(norm(Xtr)).to(dev); yt = torch.from_numpy((ytr - ymu) / ysd).to(dev)
    Xv = torch.from_numpy(norm(Xva)).to(dev); yv = torch.from_numpy((yva - ymu) / ysd).to(dev)
    Xe = torch.from_numpy(norm(Xte)).to(dev)

    def train_one(kind, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Net(kind, a.hidden, n_in, a.layers).to(dev)
        npar = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=LR[kind]); lf = nn.MSELoss()
        best, state, bad, n = np.inf, None, 0, len(Xt)
        t0 = time.time()
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
                best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
            tstr = ""
            if a.log_test_ic:
                with torch.no_grad():
                    tp = predict(m, Xe).cpu().numpy()
                tic = daily_ic(md.assign(pred=tp), pred_col="pred", label_col="LABEL",
                                date_col="date").mean()
                tstr = f"  TEST-IC {tic:+.4f}"
            if a.log_every and (ep % a.log_every == 0 or improved or bad >= a.patience):
                print(f"      [{kind} s{seed}] epoch {ep+1:>3}  val {vl:.6f}  "
                      f"best {best:.6f}  patience {bad}/{a.patience}{tstr}  "
                      f"{time.time()-t0:.0f}s ({(time.time()-t0)/(ep+1):.0f}s/ep)", flush=True)
            if bad >= a.patience:
                break
        m.load_state_dict(state); m.eval()
        with torch.no_grad():
            p = predict(m, Xe).cpu().numpy()
        return p, npar, ep + 1

    results, out = {}, {}
    for kind in a.models.split(","):
        t0 = time.time()
        P, npar = [], None
        for s_ in range(a.seeds):
            p, npar, e = train_one(kind, s_)
            P.append(p)
            print(f"  {kind} seed {s_}: {e} epochs, lr={LR[kind]:g}, {time.time()-t0:.0f}s")
        ics = daily_ic(md.assign(pred=np.mean(P, axis=0)),
                        pred_col="pred", label_col="LABEL", date_col="date")
        print(f"  {kind}: {npar} params, {time.time()-t0:.0f}s")
        results[kind] = ics
        out[kind] = report(f"{kind} (2x{a.hidden}, {a.seeds}-seed ens)", ics, a.hac_lag)
        # persist the DAILY IC SERIES, not just summary stats: a paired test against another
        # model needs day-by-day differences, and paired SEs here are ~2x tighter than
        # marginal ones. Storing only means makes the sensitive comparison impossible later.
        out[kind]["_daily_ic"] = [float(x) for x in ics]
        out[kind]["_n_params"] = int(npar)

    if len(results) == 2:
        k1, k2 = list(results)
        p = paired(results[k1], results[k2], a.hac_lag)
        print(f"\n   {k1} - {k2}: {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}  "
              f"win-rate {p['win_rate']:.1%}")

    gp = f"{a.data}/_gate_result.json"
    if os.path.exists(gp):
        g = json.load(open(gp))
        print(f"\n=== same eval rows, from the linear gate ===")
        print(f"   ridge (221, CSRankNorm target) : {g['ridge_csrank']['ic']:+.4f}  "
              f"HAC t {g['ridge_csrank']['t_hac']:+.2f}")
        print(f"   ridge (Alpha158 only)           : {g['ridge_alpha158']['ic']:+.4f}  "
              f"HAC t {g['ridge_alpha158']['t_hac']:+.2f}")

    print("\n=== context (different data/protocol -- NOT targets) ===")
    print("   MASTER's own GRU column (their protocol) : +0.0520 RankIC")
    print("   MASTER (published, confidential data)     : +0.0760 RankIC")

    with open(f"{a.data}/_lstm_gru_result.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {a.data}/_lstm_gru_result.json")


if __name__ == "__main__":
    main()
