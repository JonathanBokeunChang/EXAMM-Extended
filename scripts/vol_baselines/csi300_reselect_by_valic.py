#!/usr/bin/env python3
"""Was the POPULATION bad, or just the SELECTOR? Re-select saved genomes on validation rank IC.

THE QUESTION
------------
Seeded EXAMM produced 6 runs whose validation-MSE-selected champions all scored below their own
pretrained seed (+0.0429 / +0.0431 ensembles vs the seed's +0.0452). Two very different explanations
fit that:

  (A) the population is uniformly worse than the seed  -> evolution genuinely failed
  (B) the population contains better genomes and EXAMM picked the wrong ones -> the SELECTOR failed

EXAMM selects on validation MSE. On this venue that metric has now ranked things backwards against
test rank IC four separate times (grow3 vs the seed; section 7's CS features; the attention gate;
and today's arms), and the measured val->test correlation across EXAMM configurations is r = +0.065.
So (B) is not a stretch -- it is the default expectation.

This distinguishes them for FREE. Every genome is already on disk (global bests + island champions),
and their TEST predictions are already cached by csi300_examm_seq_eval.py. The only new work is
scoring each genome on the VALIDATION split to get a val rank IC, which is the selection criterion
EXAMM should arguably have used.

WHAT IT REPORTS
---------------
  1. corr(val IC, test IC) across all genomes -- directly comparable to the val-MSE-vs-test r=+0.065.
     If val IC is a materially better selector, this is where it shows.
  2. The genome that val IC would have chosen, vs the one val MSE actually chose, vs the seed.
  3. Whether ANY genome in the population beats the seed -- which settles (A) vs (B) regardless of
     which selector is better.

Point 3 is the load-bearing one and it needs no selector at all: if nothing beats the seed, the
population really is worse and no amount of better selection rescues it.

HONEST LIMIT: choosing the max over N genomes by a noisy validation metric is itself a
winner's-curse operation. A val-IC-selected genome that beats the seed on test is suggestive, NOT
proof that val-IC selection generalises -- that would need a held-out confirmation. This tells you
whether the raw material was there, not that you can reliably find it.

Usage:
    python3 scripts/vol_baselines/csi300_reselect_by_valic.py \
        --runs-root test_output/seeded_examm_seed32_plain
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
from ic_stats import daily_ic, paired, report  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = f"{REPO}/build/rnn_examples/evaluate_rnn"
TARGET = "LABEL_CSRANK"


def members(run_dir):
    """(tag, path) for this run's global best and every island champion."""
    out = []
    gs = glob.glob(f"{run_dir}/global_best_genome_*.bin")
    if gs:
        g = max(gs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        out.append((f"{os.path.basename(run_dir)}_gb", g))
    for g in sorted(glob.glob(f"{run_dir}/island_*_genome_0.bin"),
                    key=lambda p: int(re.search(r"island_(\d+)_", p).group(1))):
        i = int(re.search(r"island_(\d+)_", g).group(1))
        out.append((f"{os.path.basename(run_dir)}_i{i}", g))
    return out


def score_split(genome, data_dir, cache_root, tag, split):
    """Daily-IC series for one genome on one split. Reuses cached predictions when present, so
    re-running is cheap and the TEST pass costs nothing after csi300_examm_seq_eval.py."""
    gdir = f"{cache_root}/{tag}"
    os.makedirs(gdir, exist_ok=True)
    frames = []
    for f in sorted(glob.glob(f"{data_dir}/*_{split}.csv")):
        stock = os.path.basename(f)[: -len(f"_{split}.csv")]
        pred_csv = f"{gdir}/{stock}_{split}_predictions.csv"
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
        src = pd.read_csv(f, usecols=["date", "LABEL"])
        n = min(len(d), len(src))
        frames.append(pd.DataFrame({"date": src["date"].values[:n],
                                    "pred": d[pcol].values[:n],
                                    "LABEL": src["LABEL"].values[:n]}))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return daily_ic(df, pred_col="pred", label_col="LABEL", date_col="date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_seq")
    ap.add_argument("--seed-genome", default=f"{REPO}/seeds/gru_2x32_pretrained.bin")
    ap.add_argument("--hac-lag", type=int, default=None)
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = a.hac_lag if a.hac_lag is not None else max(1, man["label_horizon_trading_days"] - 1)
    cache = f"{a.runs_root}/_eval_cache"

    runs = sorted(d for d in glob.glob(f"{a.runs_root}/run_*") if os.path.isdir(d))
    pool = [(t, g) for d in runs for t, g in members(d)]
    print(f"[pool] {len(pool)} genomes from {len(runs)} runs  (HAC lag {hac})")
    if not pool:
        sys.exit("ERROR: no genomes found")

    # the seed is the reference and must be scored the SAME way, on the same splits
    print("[seed] scoring reference genome...", flush=True)
    seed_val = score_split(a.seed_genome, a.data, cache, "_seed", "val")
    seed_test = score_split(a.seed_genome, a.data, cache, "_seed", "test")
    if seed_val is None or seed_test is None:
        sys.exit("ERROR: could not score the seed genome")
    print(f"[seed] val IC {np.mean(seed_val):+.5f}   test IC {np.mean(seed_test):+.5f}")

    rows = []
    for i, (tag, g) in enumerate(pool, 1):
        v = score_split(g, a.data, cache, tag, "val")
        t = score_split(g, a.data, cache, tag, "test")
        if v is None or t is None:
            print(f"   {tag}: unscorable -- skipped")
            continue
        rows.append({"tag": tag, "genome": g, "val_ic": float(np.mean(v)),
                     "test_ic": float(np.mean(t)), "_t": t})
        print(f"   [{i}/{len(pool)}] {tag}: val IC {np.mean(v):+.5f}  test IC {np.mean(t):+.5f}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("val_ic", ascending=False).reset_index(drop=True)
    st, sv = float(np.mean(seed_test)), float(np.mean(seed_val))

    print(f"\n=== does validation IC predict test IC? ===")
    r = float(np.corrcoef(df.val_ic, df.test_ic)[0, 1])
    print(f"   corr(val IC, test IC) over {len(df)} genomes = {r:+.3f}")
    print(f"   for comparison, val-MSE -> test-IC across EXAMM configs was r = +0.065")

    print(f"\n=== (A) is the population worse than the seed, or (B) was the selector wrong? ===")
    beat = df[df.test_ic > st]
    print(f"   genomes beating the seed on TEST: {len(beat)}/{len(df)}  "
          f"(seed test IC {st:+.5f})")
    print(f"   population test IC: mean {df.test_ic.mean():+.5f}  sd {df.test_ic.std():.5f}  "
          f"max {df.test_ic.max():+.5f}")
    pct = 100.0 * (df.test_ic < st).mean()
    print(f"   the seed sits at the {pct:.0f}th percentile of the evolved population")
    if len(beat) == 0:
        print(f"   -> (A): NO genome beats the seed. The population really is worse; better")
        print(f"      selection cannot rescue it.")
    else:
        print(f"   -> (B) is possible: better genomes EXIST. Whether a selector can find them")
        print(f"      reliably is a separate question this cannot answer (see docstring).")

    print(f"\n=== what each selector would have chosen ===")
    best_valic = df.iloc[0]
    gb = df[df.tag.str.endswith("_gb")]
    print(f"   val-IC pick   : {best_valic.tag:<14} val {best_valic.val_ic:+.5f} -> "
          f"TEST {best_valic.test_ic:+.5f}  ({best_valic.test_ic - st:+.5f} vs seed)")
    if len(gb):
        print(f"   val-MSE picks : (EXAMM's actual global bests)")
        for _, r_ in gb.iterrows():
            print(f"                   {r_.tag:<14} val {r_.val_ic:+.5f} -> "
                  f"TEST {r_.test_ic:+.5f}  ({r_.test_ic - st:+.5f} vs seed)")
        print(f"   val-MSE mean test IC {gb.test_ic.mean():+.5f}  vs val-IC pick "
              f"{best_valic.test_ic:+.5f}")

    # paired HAC on the val-IC pick vs the seed -- the only inferential number here
    p = paired(np.asarray(best_valic._t, float), np.asarray(seed_test, float), hac)
    print(f"\n   val-IC pick - seed (paired, HAC): {p['delta']:+.5f}  t {p['t_hac']:+.2f}")
    print(f"   NOTE: this genome was chosen as the max over {len(df)} noisy validation scores, so")
    print(f"   it carries winner's curse. Treat a positive delta as 'the raw material exists',")
    print(f"   not as 'val-IC selection works'.")

    out = f"{a.runs_root}/_reselect_valic.json"
    with open(out, "w") as f:
        json.dump({"seed": {"val_ic": sv, "test_ic": st},
                   "corr_val_ic_test_ic": r,
                   "n_beating_seed": int(len(beat)),
                   "population": {"mean": float(df.test_ic.mean()),
                                  "sd": float(df.test_ic.std()),
                                  "max": float(df.test_ic.max())},
                   "genomes": df.drop(columns=["_t"]).to_dict("records")}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
