"""SYMMETRIC best-of-N: every stochastic model gets the same N draws and the same
selection rules, so no method is handed a larger search budget than another.

THE QUESTION THIS ANSWERS
-------------------------
A run of EXAMM (or a seed of LSTM/GRU) is a DRAW from a distribution, not a model. How you
collapse N draws into one reported number is a methodological choice, and the choice changes
the ranking:

  ensemble    mean of all N predictions           -- no selection, uses every draw
  best|val    the draw with the lowest VALIDATION loss, scored on test  -- honest selection
  best|TEST   the draw with the best TEST score                         -- leaks test

best|TEST is the convention that inflates high-variance methods: with N draws you keep the
luckiest, so the reported number rises with variance rather than with skill. best|val is the
defensible version of the same protocol -- selection happens on data the model may legally see.

Validation losses are compared ONLY WITHIN a model family (argmin over that family's own N
draws). They are NOT comparable across families: EXAMM normalises targets as
(x - pooled_mean)/(global_max - pooled_mean) while the torch baselines standardise as
(y - mu)/sd, so the two MSEs differ by a scale factor. Ranking draws inside one family is
scale-invariant and therefore valid; ranking families by raw val loss would be meaningless.

Supersedes scratchpad/bestof_all.py, which printed to stdout and persisted nothing (so the
compute had to be repeated) and picked genomes with a lexicographic sort.
"""
from __future__ import annotations

import argparse, glob, os, re, subprocess, sys, tempfile
import numpy as np, pandas as pd
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = f"{REPO}/build/rnn_examples/evaluate_rnn"
FX = ["LV", "MA5", "MA22", "RET", "LOGVOL"]
SEQ = 22


def newest_genome(run_dir: str) -> str | None:
    """Highest generation id. `sorted(glob(...))[0]` is LEXICOGRAPHIC and puts _1000
    before _998 -- the same defect fixed in the .sb launchers. Sort numerically."""
    cands = glob.glob(f"{run_dir}/global_best_genome_*.bin")
    if not cands:
        return None
    return max(cands, key=lambda p: int(re.search(r"_(\d+)\.bin$", p).group(1)))


def load(dd):
    S, TR, VA, TE = [], {}, {}, {}
    for f in sorted(glob.glob(f"{dd}/*_train.csv")):
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
    def __init__(self, kind, hid, nin):
        super().__init__()
        R = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = R(nin, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def neural_draws(dd, kind, hid, seeds):
    """N independently seeded trainings. Each draw keeps its early-stopping validation
    loss -- that is the selection key for best|val."""
    S, TR, VA, TE = load(dd)
    allc = pd.concat([TR[s] for s in S] + [VA[s] for s in S], ignore_index=True)
    mu, sd = allc[FX].mean().values, allc[FX].std().replace(0, 1).values
    tmu, tsd = allc["TARGET"].mean(), allc["TARGET"].std()

    def win(h, emit):
        X = ((h[FX].values - mu) / sd).astype(np.float32)
        y = ((h["TARGET"].values - tmu) / tsd).astype(np.float32)
        xs, ys, idx = [], [], []
        for i in range(max(emit, SEQ - 1), len(h)):
            xs.append(X[i - SEQ + 1: i + 1]); ys.append(y[i]); idx.append(i)
        return (np.asarray(xs), np.asarray(ys), idx) if xs else (None, None, [])

    Xtr, ytr, Xva, yva, Xte, meta = [], [], [], [], [], []
    for s in S:
        a, b, _ = win(TR[s], 0)
        if a is not None:
            Xtr.append(a); ytr.append(b)
        tv = pd.concat([TR[s], VA[s]], ignore_index=True)
        a, b, _ = win(tv, len(TR[s]))
        if a is not None:
            Xva.append(a); yva.append(b)
        vt = pd.concat([VA[s], TE[s]], ignore_index=True)
        a, b, idx = win(vt, len(VA[s]))
        if a is not None:
            Xte.append(a); meta += [(s, vt["TARGET"].values[i]) for i in idx]

    Xt = torch.from_numpy(np.concatenate(Xtr)); yt = torch.from_numpy(np.concatenate(ytr))
    Xv = torch.from_numpy(np.concatenate(Xva)); yv = torch.from_numpy(np.concatenate(yva))
    Xe = torch.from_numpy(np.concatenate(Xte))
    md = pd.DataFrame(meta, columns=["stock", "y_true"])

    draws = []
    for sd_ in range(seeds):
        torch.manual_seed(sd_); np.random.seed(sd_)
        m = Net(kind, hid, len(FX))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.MSELoss()
        best, state, bad = np.inf, None, 0
        for ep in range(60):
            m.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                opt.zero_grad(); lf(m(Xt[b]), yt[b]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = lf(m(Xv), yv).item()
            if vl < best - 1e-5:
                best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        m.load_state_dict(state); m.eval()
        with torch.no_grad():
            p = m(Xe).numpy() * tsd + tmu
        draws.append({"val": float(best), "pred": p})
        print(f"    seed {sd_}: val {best:.6f}", flush=True)
    return md, draws


def examm_draws(dd, runs_dir):
    """One draw per EXAMM run. Selection key is the run's final Best Val. MSE, straight
    from its fitness log -- the same number the search itself optimised."""
    tmp = tempfile.mkdtemp(); draws = []; md = None
    runs = sorted(glob.glob(f"{runs_dir}/run_*"), key=lambda p: int(p.split("_")[-1]))
    for run in runs:
        g = newest_genome(run)
        if g is None:
            print(f"    {os.path.basename(run)}: NO GENOME, skipped", flush=True); continue
        fl = pd.read_csv(f"{run}/fitness_log.csv"); fl.columns = [c.strip() for c in fl.columns]
        rows = []
        for f in sorted(glob.glob(f"{dd}/*_test.csv")):
            s = os.path.basename(f)[: -len("_test.csv")]
            for x in glob.glob(f"{tmp}/*.csv"):
                os.remove(x)
            subprocess.run([EV, "--genome_file", g, "--testing_filenames", f, "--time_offset", "0",
                            "--output_directory", tmp, "--std_message_level", "ERROR",
                            "--file_message_level", "NONE"], capture_output=True)
            h = glob.glob(f"{tmp}/*_predictions.csv")
            if h:
                df = pd.read_csv(h[0])
                pc = [c for c in df.columns if c.lower().startswith("predicted")][0]
                te = pd.read_csv(f); n = min(len(df), len(te))
                rows.append(pd.DataFrame({"stock": s, "y_true": te["TARGET"].values[:n],
                                          "p": df[pc].values[:n]}))
        P = pd.concat(rows, ignore_index=True)
        md = P[["stock", "y_true"]]
        v = float(fl["Best Val. MSE"].values[-1])
        draws.append({"val": v, "pred": P["p"].values})
        print(f"    {os.path.basename(run)}: val {v:.8f}", flush=True)
    return md, draws


def summarize(md, draws, label):
    def r2_of(p):
        d = md.copy(); d["p"] = p
        return float(np.median([M.r2(g.y_true, g.p) for _, g in d.groupby("stock")]))

    r2s = np.array([r2_of(d["pred"]) for d in draws])
    vals = np.array([d["val"] for d in draws])
    ens = r2_of(np.mean([d["pred"] for d in draws], axis=0))
    pick = int(np.argmin(vals))                      # honest selection: lowest val loss
    # Where did the val-winner actually land on test? rank 1 = best test R2 of the N.
    rank = int((r2s > r2s[pick]).sum()) + 1
    sp = pd.Series(vals).corr(pd.Series(r2s), method="spearman")
    per = pd.DataFrame({"model": label, "draw": np.arange(len(draws)),
                        "val": vals, "test_r2": r2s,
                        "picked_by_val": np.arange(len(draws)) == pick})
    return {"model": label, "n": len(draws), "ensemble": ens,
            "best_by_val": float(r2s[pick]), "best_by_test": float(r2s.max()),
            "median_single": float(np.median(r2s)), "worst": float(r2s.min()),
            "val_pick_test_rank": rank, "spearman_val_vs_test": float(sp)}, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="dir of EXAMM run_* subdirs")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2,
                    help="keep low when sharing the box with another training job")
    ap.add_argument("--out-root", default=f"{REPO}/results/best_of_n")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    out = f"{a.out_root}/{a.tag}"
    os.makedirs(out, exist_ok=True)
    rows, pers = [], []

    print(f"[{a.tag}] EXAMM draws ...", flush=True)
    md, dr = examm_draws(a.data, a.runs)
    r, p = summarize(md, dr, "EXAMM"); rows.append(r); pers.append(p)

    for kind, hid, lab in [("gru", 32, "GRU-32"), ("lstm", 32, "LSTM-32"), ("lstm", 4, "LSTM-4")]:
        print(f"[{a.tag}] {lab} draws ...", flush=True)
        md2, dr2 = neural_draws(a.data, kind, hid, a.seeds)
        r, p = summarize(md2, dr2, lab); rows.append(r); pers.append(p)

    D = pd.DataFrame(rows); P = pd.concat(pers, ignore_index=True)
    D.to_csv(f"{out}/summary.csv", index=False)
    P.to_csv(f"{out}/per_draw.csv", index=False)

    print(f"\n{'model':10}{'ensemble':>10}{'best|val':>10}{'best|TEST':>11}"
          f"{'median':>9}{'worst':>9}{'valpick_rk':>12}{'rho(v,t)':>10}")
    for _, r in D.iterrows():
        print(f"{r.model:10}{r.ensemble:>10.4f}{r.best_by_val:>10.4f}{r.best_by_test:>11.4f}"
              f"{r.median_single:>9.4f}{r.worst:>9.4f}{r.val_pick_test_rank:>9d}/{r.n:<2d}"
              f"{r.spearman_val_vs_test:>10.3f}")
    print(f"\n-> {out}/summary.csv  {out}/per_draw.csv")


if __name__ == "__main__":
    main()
