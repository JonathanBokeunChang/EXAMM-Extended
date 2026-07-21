"""Evaluate EXAMM genome ensembles into the harness's long format.

Closes a real reproducibility hole: the banked EXAMM headline numbers were produced by
an inline heredoc that was never saved, so they could not be regenerated. This script
persists predictions keyed by (stock, date) exactly like the baselines, so every model in
the paper is scored on provably identical rows.

  python3 scripts/vol_baselines/eval_examm.py \
      --data datasets/qlib_vol_big --runs <dir-with-run_*/global_best_genome_*.bin> \
      --tag qlib_vol_big_fixed --name examm

Cross-checks `expected_TARGET` in evaluate_rnn's output against the dataset's TARGET
column -- nothing else in the codebase ever verified that alignment.
"""
from __future__ import annotations

import argparse, glob, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

REPO = "/Users/jonathanchang/EXAMM-Extended"
EV = f"{REPO}/build/rnn_examples/evaluate_rnn"


def predict(genome: str, csv: str, tmp: str) -> np.ndarray | None:
    for f in glob.glob(f"{tmp}/*.csv"):
        os.remove(f)
    subprocess.run([EV, "--genome_file", genome, "--testing_filenames", csv,
                    "--time_offset", "0", "--output_directory", tmp,
                    "--std_message_level", "ERROR", "--file_message_level", "NONE"],
                   capture_output=True)
    hits = glob.glob(f"{tmp}/*_predictions.csv")
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    pcol = [c for c in df.columns if c.lower().startswith("predicted")][0]
    ecol = [c for c in df.columns if c.lower().startswith("expected")]
    return df[pcol].values, (df[ecol[0]].values if ecol else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="dir containing run_*/global_best_genome_*.bin")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default="examm")
    ap.add_argument("--out-root", default=f"{REPO}/results/baselines")
    a = ap.parse_args()

    genomes = [g for r in sorted(glob.glob(f"{a.runs}/run_*"))
               for g in sorted(glob.glob(f"{r}/global_best_genome_*.bin"))[:1]]
    if not genomes:
        sys.exit(f"ERROR: no genomes under {a.runs}/run_*/")
    print(f"[{a.name}] ensemble of {len(genomes)} genomes")

    out = f"{a.out_root}/{a.tag}"
    os.makedirs(f"{out}/predictions", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="examm_eval_")
    rows, align_err, skipped = [], 0.0, []

    for f in sorted(glob.glob(f"{a.data}/*_train.csv")):
        s = os.path.basename(f)[: -len("_train.csv")]
        csv = f"{a.data}/{s}_test.csv"
        if not os.path.exists(csv):
            continue
        te = pd.read_csv(csv)
        te["date"] = pd.to_datetime(te["date"])
        preds, exp = [], None
        for g in genomes:
            r = predict(g, csv, tmp)
            if r is None:
                continue
            p, e = r
            preds.append(p)
            if e is not None and exp is None:
                exp = e
        if not preds:
            skipped.append(s); continue
        n = min(min(len(p) for p in preds), len(te))
        # alignment cross-check: evaluate_rnn's expected_TARGET vs the dataset TARGET
        if exp is not None:
            align_err = max(align_err,
                            float(np.max(np.abs(exp[:n] - te["TARGET"].values[:n]))))
        rows.append(pd.DataFrame({
            "stock": s, "date": te["date"].values[:n],
            "y_true": te["TARGET"].values[:n],
            "y_pred": np.mean([p[:n] for p in preds], axis=0),
            "y_pred_raw": np.mean([p[:n] for p in preds], axis=0),
        }))

    P = pd.concat(rows, ignore_index=True)
    P.sort_values(["stock", "date"]).to_csv(f"{out}/predictions/{a.name}.csv", index=False)

    per = [{"stock": s, "model": a.name, "n": len(g),
            **M.all_metrics(g.y_true, g.y_pred)} for s, g in P.groupby("stock")]
    D = pd.DataFrame(per)
    mp = f"{out}/metrics_per_stock.csv"
    if os.path.exists(mp):
        old = pd.read_csv(mp)
        old = old[old.model != a.name]
        D = pd.concat([old, D], ignore_index=True)
    D.to_csv(mp, index=False)

    tol_ok = align_err < 1e-3          # evaluate_rnn writes 6 significant figures
    print(f"  stocks={P.stock.nunique()}  rows={len(P):,}  skipped={len(skipped)}")
    print(f"  ALIGNMENT CHECK expected_TARGET vs dataset TARGET: max|diff| = {align_err:.2e} "
          f"-> {'PASS' if tol_ok else 'FAIL'} (tol 1e-3, 6-sig-fig output)")
    d = D[D.model == a.name]
    print(f"  {a.name}: MSE {d.mse.median():.4f}  MAE {d.mae.median():.4f}  "
          f"QLIKE {d.qlike.median():.4f}  R2 {d.r2.median():.4f}")
    if not tol_ok:
        sys.exit("ALIGNMENT FAILED -- predictions do not correspond to the dataset rows")


if __name__ == "__main__":
    main()
