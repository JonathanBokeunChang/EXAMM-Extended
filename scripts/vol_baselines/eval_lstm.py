"""Fixed-architecture LSTM/GRU baselines -- the "did you need NAS?" comparison.

Supersedes scratchpad/e1a_lstm_baseline.py, which had TWO fairness defects that both
favoured EXAMM. This script fixes them and writes predictions in the harness long format
so every model is scored on identical (stock, date) keys.

FIX 1 -- information asymmetry (the serious one).
    Old: window X[i-seq : i]  -> rows i-seq .. i-1, EXCLUDING row i.
    EXAMM at --time_offset 0 consumes row i when predicting TARGET(i). LV/MA5/MA22 at
    row i are the most predictive inputs for TARGET(i) = mean(LV[i+1..i+5]), so the old
    window denied the LSTM exactly the information EXAMM was given.
    New: window X[i-seq+1 : i+1] -> rows i-seq+1 .. i, INCLUDING row i. Same length,
    both models now see information up to and including t.

FIX 2 -- row coverage.
    Old: windows built within each split, so the first seq test rows had no history and
    were dropped (47,291 of 49,931 rows scored). EXAMM scored all rows, so the two were
    compared on different denominators -- which is why "HAR" read .2371 in the LSTM
    script and .2473 everywhere else.
    New: test windows draw history from the concatenated val+test series. Val precedes
    test, so this is past information (no leakage) and is what a deployment would do.
    Every test row is now scored, identical to EXAMM.

FAIRNESS PROTOCOL (matches the EXAMM arm exactly):
    pooled training over all stocks * same 5 inputs * train-only standardisation *
    val-based early stopping * same ensemble size (mean over seeds).
Variants: LSTM-32, GRU-32 (standard) and LSTM-4 (~parameter parity with EXAMM's evolved
genomes, which carry only tens of weights). Parameter counts are reported.
"""
from __future__ import annotations

import argparse, glob, os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = ["LV", "MA5", "MA22", "RET", "LOGVOL"]


def load(data_dir):
    S, TR, VA, TE = [], {}, {}, {}
    for f in sorted(glob.glob(f"{data_dir}/*_train.csv")):
        s = os.path.basename(f)[: -len("_train.csv")]
        pv, pt = f.replace("_train", "_val"), f.replace("_train", "_test")
        if not (os.path.exists(pv) and os.path.exists(pt)):
            continue
        a, b, c = pd.read_csv(f), pd.read_csv(pv), pd.read_csv(pt)
        for d in (a, b, c):
            d["date"] = pd.to_datetime(d["date"])
        if len(a) < 100 or len(c) < 40:
            continue
        S.append(s); TR[s], VA[s], TE[s] = a, b, c
    return S, TR, VA, TE


class Net(nn.Module):
    def __init__(self, kind, hid, n_in):
        super().__init__()
        R = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = R(n_in, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/qlib_vol_big")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seq", type=int, default=22)
    ap.add_argument("--seeds", type=int, default=3, help="match the EXAMM ensemble size")
    ap.add_argument("--features", nargs="+", default=None,
                    help="input features; default LV MA5 MA22 RET LOGVOL. The raw-input "
                         "ablation passes 'LV RET LOGVOL' so the network must DISCOVER the "
                         "multi-scale memory that MA5/MA22 otherwise hand it.")
    ap.add_argument("--suffix", default="",
                    help="appended to model names (e.g. _raw) so ablation rows do not "
                         "overwrite the main comparison in metrics_per_stock.csv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out-root", default=f"{REPO}/results/baselines")
    a = ap.parse_args()
    global FX
    if a.features:
        FX = list(a.features)          # raw-input ablation
    print(f"[lstm] features: {FX}")
    torch.set_num_threads(4)
    SEQ = a.seq

    S, TR, VA, TE = load(a.data)
    print(f"[lstm] {len(S)} stocks, seq={SEQ} (window INCLUDES row i), seeds={a.seeds}")

    # Pooled standardisation over TRAIN + VALIDATION, to MATCH EXAMM. EXAMM computes its
    # avg_std_dev normalisation bounds over training_filenames UNION validation_filenames
    # (time_series.cxx:760-774,1033-1064) and stores them in the genome. A train-only scaler
    # here would hand EXAMM an information advantage in the very fairness comparison this
    # harness exists to make. The pool contains no test rows, so there is no test leakage;
    # this is disclosed in the methods as "standardised over train+validation".
    allc = pd.concat([TR[s] for s in S] + [VA[s] for s in S], ignore_index=True)
    mu, sd = allc[FX].mean().values, allc[FX].std().replace(0, 1).values
    tmu, tsd = allc["TARGET"].mean(), allc["TARGET"].std()

    def windows(hist: pd.DataFrame, emit_from: int):
        """Windows ending AT row i (inclusive), emitted for rows >= emit_from."""
        X = ((hist[FX].values - mu) / sd).astype(np.float32)
        y = ((hist["TARGET"].values - tmu) / tsd).astype(np.float32)
        xs, ys, idx = [], [], []
        for i in range(max(emit_from, SEQ - 1), len(hist)):
            xs.append(X[i - SEQ + 1: i + 1])   # <-- FIX 1: includes row i
            ys.append(y[i]); idx.append(i)
        if not xs:
            return None, None, []
        return np.asarray(xs), np.asarray(ys), idx

    Xtr, ytr = [], []
    Xva, yva = [], []
    Xte, yte, meta = [], [], []
    for s in S:
        xt, yt, _ = windows(TR[s], 0)
        if xt is not None:
            Xtr.append(xt); ytr.append(yt)
        # val windows draw history from train+val
        tv = pd.concat([TR[s], VA[s]], ignore_index=True)
        xv, yv, _ = windows(tv, len(TR[s]))
        if xv is not None:
            Xva.append(xv); yva.append(yv)
        # FIX 2: test windows draw history from val+test -> every test row is scored
        vt = pd.concat([VA[s], TE[s]], ignore_index=True)
        xe, ye, idx = windows(vt, len(VA[s]))
        if xe is not None:
            Xte.append(xe); yte.append(ye)
            meta += [(s, vt["date"].values[i], vt["TARGET"].values[i]) for i in idx]
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xva = np.concatenate(Xva); yva = np.concatenate(yva)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)
    n_test_rows = sum(len(TE[s]) for s in S)
    print(f"[lstm] windows train {Xtr.shape} val {Xva.shape} test {Xte.shape} "
          f"(test rows available {n_test_rows:,} -> coverage {len(Xte)/n_test_rows:.4f})")

    Xt, yt_ = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv_ = torch.from_numpy(Xva), torch.from_numpy(yva)
    Xe = torch.from_numpy(Xte)

    def train_one(kind, hid, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = Net(kind, hid, len(FX))
        npar = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.MSELoss()
        best, state, bad = np.inf, None, 0
        n = len(Xt)
        for ep in range(a.epochs):
            m.train(); perm = torch.randperm(n)
            for i in range(0, n, a.batch):
                b = perm[i:i + a.batch]
                opt.zero_grad(); lf(m(Xt[b]), yt_[b]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = lf(m(Xv), yv_).item()
            if vl < best - 1e-5:
                best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        m.load_state_dict(state); m.eval()
        with torch.no_grad():
            p = m(Xe).numpy()
        return p * tsd + tmu, npar, ep + 1

    md = pd.DataFrame(meta, columns=["stock", "date", "y_true"])
    out = f"{a.out_root}/{a.tag}"
    os.makedirs(f"{out}/predictions", exist_ok=True)
    mp = f"{out}/metrics_per_stock.csv"
    per_all = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame()
    params = {}

    for kind, hid in [("lstm", 32), ("gru", 32), ("lstm", 4)]:
        name = f"{kind}{hid}{a.suffix}"
        t0 = time.time(); P, npar, eps = [], None, []
        for sd_ in range(a.seeds):
            p, npar, e = train_one(kind, hid, sd_); P.append(p); eps.append(e)
        params[name] = npar
        ens = np.mean(P, axis=0)
        df = md.copy(); df["y_pred"] = ens; df["y_pred_raw"] = ens
        df.sort_values(["stock", "date"]).to_csv(f"{out}/predictions/{name}.csv", index=False)
        rows = [{"stock": s, "model": name, "n": len(g), **M.all_metrics(g.y_true, g.y_pred)}
                for s, g in df.groupby("stock")]
        D = pd.DataFrame(rows)
        per_all = pd.concat([per_all[per_all.model != name] if len(per_all) else per_all, D],
                            ignore_index=True)
        print(f"  {name:7} {npar:>6} params, epochs {eps}, {time.time()-t0:.0f}s  "
              f"| R2 {D.r2.median():+.4f}  QLIKE {D.qlike.median():+.4f}")
    per_all.to_csv(mp, index=False)
    print(f"\nparams: {params}")
    print(f"-> {out}/predictions/{{lstm32,gru32,lstm4}}.csv   metrics merged into {mp}")


if __name__ == "__main__":
    main()
