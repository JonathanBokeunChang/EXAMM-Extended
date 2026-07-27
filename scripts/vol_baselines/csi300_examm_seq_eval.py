#!/usr/bin/env python3
"""Evaluate pooled EXAMM on the sequence-form CSI300 dataset, on the SAME terms as the GRU.

Differs from csi300_examm_eval.py in three ways that all exist to make the comparison honest:

1. FROZEN EVAL INDEX. Scores exactly the (date, instrument) rows in eval_index.csv and hard-fails
   if any are missing. The older evaluator scored whatever test CSVs happened to exist, which is
   how the Alpha158 comparison ended up with the ridge on 430 stocks and LSTM/GRU on 240 -- numbers
   that were never comparable. Every model now scores an identical row set.

2. CSRANK TARGET. EXAMM trains on LABEL_CSRANK because the GRU does, and that target was worth
   +0.0277 (HAC t 3.97) on Alpha158 -- training EXAMM on raw returns against a rank-trained
   baseline would hand the baseline a large known advantage. Predictions therefore come out in rank
   space, and are scored against the RAW LABEL by cross-sectional rank IC. That is valid because
   rank IC depends only on ordering, and the builder verified per-date
   Spearman(LABEL, LABEL_CSRANK) = 1.000000 -- the two label forms induce identical orderings.

3. HAC STANDARD ERRORS via ic_stats, matching every other result in this benchmark.

Genome selection: the single validation-selected global_best_genome per run, evaluated once.
Never a max over island-bests -- that is the winner's-curse pattern this project already hit with a
live tracker (+0.0188 shown vs +0.0113 honest).

Usage:
    python3 scripts/vol_baselines/csi300_examm_seq_eval.py \
        --runs-root <scratchpad>/csi300_seq_examm \
        --data datasets/csi300_master_replica_invdata_seq
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ic_stats import DEFAULT_HAC_LAG, daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = f"{REPO}/build/rnn_examples/evaluate_rnn"
TARGET = "LABEL_CSRANK"


def global_best(run_dir):
    """The single validation-selected genome. global_best_genome_*.bin is written once at the end
    by EXAMM::update_log via speciation_strategy->get_global_best_genome(); rnn_genome_*.bin files
    are ISLAND-local bests (insert_position==0) and must never be max-ed over on test."""
    gs = glob.glob(f"{run_dir}/global_best_genome_*.bin")
    if not gs:
        return None
    return max(gs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))


def eval_genome(genome, data_dir, cache_root, run_tag):
    gdir = f"{cache_root}/{run_tag}"
    os.makedirs(gdir, exist_ok=True)
    frames = []
    for f in sorted(glob.glob(f"{data_dir}/*_test.csv")):
        stock = os.path.basename(f)[: -len("_test.csv")]
        pred_csv = f"{gdir}/{stock}_test_predictions.csv"
        if not os.path.exists(pred_csv):
            subprocess.run([BIN, "--genome_file", genome, "--testing_filenames", f,
                            "--time_offset", "0", "--output_directory", gdir,
                            "--std_message_level", "ERROR", "--file_message_level", "ERROR"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(pred_csv):
            continue
        d = pd.read_csv(pred_csv)
        pcol = f"predicted_{TARGET}"
        if pcol not in d.columns:
            continue
        # EXAMM emits predictions in source row order; attach dates and the RAW label by position
        src = pd.read_csv(f, usecols=["date", "LABEL"])
        n = min(len(d), len(src))
        frames.append(pd.DataFrame({"date": src["date"].values[:n],
                                     "instrument": stock,
                                     "pred": d[pcol].values[:n],
                                     "LABEL": src["LABEL"].values[:n]}))
    return pd.concat(frames, ignore_index=True) if frames else None


def restrict(df, data_dir, tag):
    ev = pd.read_csv(f"{data_dir}/eval_index.csv")
    m = ev.merge(df, on=["date", "instrument"], how="left")
    miss = m["pred"].isna().sum()
    if miss:
        sys.exit(f"ERROR: {tag} is missing {miss}/{len(ev)} eval_index rows. EXAMM must score the "
                 f"same rows as every other model; a short join would silently reintroduce the "
                 f"population mismatch this benchmark exists to prevent.")
    return m


STATS_BIN = f"{REPO}/build/rnn_examples/rnn_statistics"


def genome_stats(genome):
    """Exact weight count and node-type composition, via rnn_statistics.

    The efficiency claim is "matches the GRU at N-fold fewer parameters", so this number has to be
    the real trainable-weight count -- not a file size, which varies with serialization detail and
    would silently misstate the headline by an arbitrary factor. rnn_statistics prints:
        RNN INFO FOR '<file>', nodes: 8, edges: 10, rec: 0, weights: 26
    plus the per-cell-type breakdown, which doubles as the memory-cell inductive-bias analysis
    (which cells evolution actually selects for financial sequences).
    """
    out = {"weights": None, "nodes": None, "edges": None, "rec": None}
    try:
        r = subprocess.run([STATS_BIN, "--rnn_filenames", genome,
                            "--output_directory", "/tmp/_gstats",
                            "--std_message_level", "INFO", "--file_message_level", "NONE"],
                           capture_output=True, text=True, timeout=120)
        for line in (r.stdout + r.stderr).splitlines():
            if "RNN INFO FOR" in line:
                for key in out:
                    m = re.search(rf"\b{key}:\s*(\d+)", line)
                    if m:
                        out[key] = int(m.group(1))
                break
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_seq")
    ap.add_argument("--hac-lag", type=int, default=None)
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = a.hac_lag if a.hac_lag is not None else max(1, man["label_horizon_trading_days"] - 1)
    print(f"[dataset] handler={man.get('handler')} content={man['content_sha256'][:16]}...  "
          f"HAC lag={hac}")

    runs = sorted(d for d in glob.glob(f"{a.runs_root}/run_*") if os.path.isdir(d))
    pairs = [(d, global_best(d)) for d in runs]
    missing = [d for d, g in pairs if g is None]
    pairs = [(d, g) for d, g in pairs if g]
    print(f"[runs] {len(pairs)} with a global_best_genome"
          + (f"; MISSING in {[os.path.basename(m) for m in missing]}" if missing else ""))
    if not pairs:
        sys.exit("ERROR: no global_best_genome_*.bin found -- did the runs finish?")

    cache = f"{a.runs_root}/_eval_cache"
    per_run, ics_per_run, sizes = {}, {}, {}
    for d, g in pairs:
        tag = os.path.basename(d)
        df = eval_genome(g, a.data, cache, tag)
        if df is None:
            print(f"   {tag}: no predictions produced -- skipped")
            continue
        df = restrict(df, a.data, tag)
        per_run[tag] = df
        ics_per_run[tag] = daily_ic(df, pred_col="pred", label_col="LABEL", date_col="date")
        st = genome_stats(g)
        sizes[tag] = st
        print(f"   {tag}: {os.path.basename(g)}  weights={st['weights']} "
              f"nodes={st['nodes']} edges={st['edges']} rec={st['rec']}")

    print(f"\n=== per-run (validation-selected global best, scored once) ===")
    for tag, ics in ics_per_run.items():
        report(tag, ics, hac)

    # ensemble: average predictions across runs on the frozen index
    base = next(iter(per_run.values()))[["date", "instrument", "LABEL"]].copy()
    P = np.column_stack([per_run[t]["pred"].to_numpy(float) for t in per_run])
    base["pred"] = P.mean(axis=1)
    ics_ens = daily_ic(base, pred_col="pred", label_col="LABEL", date_col="date")
    print(f"\n=== ENSEMBLE ({len(per_run)} runs) ===")
    s = report(f"EXAMM {len(per_run)}-run ensemble", ics_ens, hac)

    gp = f"{a.data}/_lstm_gru_result.json"
    if os.path.exists(gp):
        g = json.load(open(gp))
        print(f"\n=== same rows, same target, same HAC lag ===")
        for k, v in g.items():
            print(f"   {k:<6} {v['ic']:+.4f}  HAC t {v['t_hac']:+.2f}   (38,849 params)")
        if "gru" in g:
            gi = np.asarray(g["gru"].get("_daily_ic", []), float)
            if gi.size == len(ics_ens):
                p = paired(ics_ens, gi, hac)
                print(f"   EXAMM - GRU: {p['delta']:+.4f}  HAC t {p['t_hac']:+.2f}")
            else:
                print("   (paired test needs the GRU daily-IC series; not stored in that JSON)")

    ws = [v["weights"] for v in sizes.values() if v.get("weights")]
    if ws:
        mean_w = sum(ws) / len(ws)
        print(f"\n=== EFFICIENCY (the claim this experiment exists to test) ===")
        print(f"   EXAMM weights per genome : {ws}  (mean {mean_w:.0f})")
        print(f"   GRU 2x64 parameters      : 38849")
        print(f"   ratio                    : {38849/mean_w:.0f}x fewer")
        print(f"   NOTE: a parameter-count advantage only counts if accuracy is comparable --")
        print(f"   read it against the paired IC delta above, never on its own.")

    print(f"\n=== context (different encoding -- NOT a target) ===")
    print(f"   published qlib GRU, Alpha360 flat : 0.0584")
    print(f"   our GRU, Alpha360 flat            : +0.0581")
    print(f"   published HIST (SOTA, Alpha360)   : 0.0667")

    with open(f"{a.runs_root}/_examm_result.json", "w") as f:
        json.dump({"ensemble": s, "per_run": {k: float(v.mean()) for k, v in ics_per_run.items()},
                   "n_runs": len(per_run), "hac_lag": hac}, f, indent=2, default=float)
    print(f"\nwrote {a.runs_root}/_examm_result.json")


if __name__ == "__main__":
    main()
