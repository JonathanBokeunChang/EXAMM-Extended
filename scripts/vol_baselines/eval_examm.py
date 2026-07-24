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

import argparse, glob, json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

# REPO is this file's grandparent (scripts/vol_baselines/eval_examm.py -> repo root), so the
# script is portable rather than pinned to one machine.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = f"{REPO}/build/rnn_examples/evaluate_rnn"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def predict(genome: str, csv: str, tmp: str):
    for f in glob.glob(f"{tmp}/*.csv"):
        os.remove(f)
    r = subprocess.run([EV, "--genome_file", genome, "--testing_filenames", csv,
                        "--time_offset", "0", "--output_directory", tmp,
                        "--std_message_level", "ERROR", "--file_message_level", "NONE"],
                       capture_output=True, text=True)
    # Surface the one channel that reports the C++ loader's NON-FATAL failures: a bad CSV
    # cell logs `invalid argument:` / `doesn't equal number of rows` and then reads out of
    # bounds. Silently discarding rc+stderr would hide exactly that.
    err = (r.stderr or "") + (r.stdout or "")
    if r.returncode != 0 or "invalid argument:" in err or "doesn't equal number of rows" in err:
        sys.stderr.write(f"  [evaluate_rnn] rc={r.returncode} on {os.path.basename(csv)} "
                         f"/ {os.path.basename(genome)}: {err.strip()[:300]}\n")
        return None
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
    ap.add_argument("--members", choices=["global_best", "island_best", "population"],
                    default="global_best",
                    help="which genomes to ensemble per run. global_best (default, historical: "
                         "1/run) | island_best (the top genome of each island: n_islands/run) | "
                         "population (every genome in every island: n_islands*island_size/run). "
                         "island_best/population need --save_genome_option entire_population, "
                         "which writes island_<i>_genome_<j>.bin (j=0 is that island's best -- "
                         "islands keep their genomes sorted by fitness).")
    a = ap.parse_args()

    def latest_genome(run_dir: str) -> str | None:
        # Numeric sort on the trailing generation id; sorted()/[:1] is lexicographic and
        # would pick _1000 before _998 if a run ever saved more than one global best.
        gs = glob.glob(f"{run_dir}/global_best_genome_*.bin")
        if not gs:
            return None
        return max(gs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))

    def population_genomes(run_dir: str, best_only: bool) -> list:
        """island_<i>_genome_<j>.bin written by Island::save_population. Genomes are stored
        fitness-sorted within an island, so j=0 is the island champion."""
        gs = glob.glob(f"{run_dir}/island_*_genome_*.bin")
        if best_only:
            gs = [g for g in gs if os.path.basename(g).rsplit("_", 1)[1] == "0.bin"]
        return sorted(gs)

    run_dirs = sorted(glob.glob(f"{a.runs}/run_*"))
    genomes = []
    for r in run_dirs:
        if a.members == "global_best":
            if (g := latest_genome(r)):
                genomes.append(g)
        else:
            got = population_genomes(r, best_only=(a.members == "island_best"))
            if not got:
                sys.exit(f"ERROR: --members {a.members} but no island_*_genome_*.bin in {r}\n"
                         f"       (re-run training with --save_genome_option entire_population)")
            genomes += got
    if not genomes:
        sys.exit(f"ERROR: no genomes under {a.runs}/run_*/")
    print(f"[{a.name}] members={a.members}: ensemble of {len(genomes)} genomes "
          f"from {len(run_dirs)} run(s)")
    for g in genomes[:12]:
        print(f"    {os.path.relpath(g, a.runs)}")
    if len(genomes) > 12:
        print(f"    ... and {len(genomes) - 12} more")

    out = f"{a.out_root}/{a.tag}"
    os.makedirs(f"{out}/predictions", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="examm_eval_")
    rows, align_err, skipped, checked = [], 0.0, [], 0

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
        if len(preds) < len(genomes):
            sys.stderr.write(f"  [WARN] {s}: {len(preds)}/{len(genomes)} genomes produced "
                             f"predictions; ensembled over {len(preds)}\n")
        # alignment cross-check: evaluate_rnn's expected_TARGET vs the dataset TARGET
        if exp is not None:
            align_err = max(align_err,
                            float(np.max(np.abs(exp[:n] - te["TARGET"].values[:n]))))
            checked += 1
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

    # Provenance: the EXACT genomes ensembled, the size, dataset, and git SHA. Without this
    # the headline EXAMM numbers cannot be traced back to the runs that produced them.
    prov = {"model": a.name, "tag": a.tag, "data_dir": a.data, "git_sha": git_sha(),
            "members": a.members, "n_runs": len(run_dirs),
            "ensemble_size": len(genomes),
            "genomes": [os.path.relpath(g, a.runs) for g in genomes],
            "runs_dir": a.runs, "stocks_scored": int(P.stock.nunique()),
            "stocks_skipped": skipped, "alignment_checked_stocks": checked,
            "alignment_max_abs_diff": align_err}
    with open(f"{out}/{a.name}_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    # An alignment gate that never ran is not a pass. `checked` counts stocks where an
    # expected_* column was actually compared; if it is zero the "PASS" below would be
    # vacuous (align_err never moved off its 0.0 initial value).
    tol_ok = checked > 0 and align_err < 1e-3     # evaluate_rnn writes 6 significant figures
    print(f"  stocks={P.stock.nunique()}  rows={len(P):,}  skipped={len(skipped)}  "
          f"ensemble={len(genomes)}")
    print(f"  ALIGNMENT CHECK expected_TARGET vs dataset TARGET on {checked} stock(s): "
          f"max|diff| = {align_err:.2e} -> {'PASS' if tol_ok else 'FAIL'} "
          f"(tol 1e-3, 6-sig-fig output)")
    d = D[D.model == a.name]
    print(f"  {a.name}: MSE {d.mse.median():.4f}  MAE {d.mae.median():.4f}  "
          f"QLIKE {d.qlike.median():.4f}  R2 {d.r2.median():.4f}")
    if not tol_ok:
        sys.exit("ALIGNMENT FAILED -- gate did not run (checked=0) or predictions do not "
                 "correspond to the dataset rows")


if __name__ == "__main__":
    main()
