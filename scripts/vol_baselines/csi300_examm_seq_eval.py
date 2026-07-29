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

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts/stock_run"))
from salvage_best_genome import salvage  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = f"{REPO}/build/rnn_examples/evaluate_rnn"
TARGET = "LABEL_CSRANK"


def global_best(run_dir, allow_salvage=False):
    """The single validation-selected genome. global_best_genome_*.bin is written once at the end
    by EXAMM::update_log via speciation_strategy->get_global_best_genome(); rnn_genome_*.bin files
    are ISLAND-local bests (insert_position==0) and must never be max-ed over on test.

    With allow_salvage, a run KILLED before its genome budget (which therefore has no global best,
    examm.cxx:416-418 only fires on completion) falls back to the argmin-best_validation_mse saved
    island best. That reproduces EXAMM's own global-best rule on VALIDATION fitness -- it is not a
    test-set max. Off by default so a truncated run can never be silently scored as a complete one.
    """
    gs = glob.glob(f"{run_dir}/global_best_genome_*.bin")
    if gs:
        return max(gs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])), "global_best"
    if not allow_salvage:
        return None, None
    r = salvage(run_dir)
    if "error" in r:
        print(f"   {os.path.basename(run_dir)}: salvage failed -- {r['error']}")
        return None, None
    print(f"   {os.path.basename(run_dir)}: SALVAGED gen {r['generation_id']} "
          f"val_mse {r['val_mse']:.6f} (best of {r['n_saved']} saved island bests)")
    return r["genome"], "salvaged"


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
    # --genome scores ONE .bin directly. Needed for the seeded experiment's control arm: the
    # pretrained seed and the compute-matched-trained seed are single genomes, not campaigns, and
    # they must be scored by this evaluator -- same frozen eval_index, same CSRANK target, same HAC
    # lag -- or the B - A delta compares numbers produced by different harnesses.
    ap.add_argument("--genome", help="score a single genome .bin (e.g. a pretrained seed)")
    ap.add_argument("--runs-root")
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_seq")
    ap.add_argument("--hac-lag", type=int, default=None)
    # island_best ensembles each island's champion (island_<i>_genome_0.bin, written by
    # --save_genome_option entire_population; islands keep genomes fitness-sorted so j=0 is the
    # champion). That is n_islands genomes per run instead of one, and it is the ONLY readout that
    # tests architectural diversity -- islands are separate lineages, whereas runs differ only by
    # seed. Prior evidence says expect little: on the volatility venue the 10 -> 100 -> 1000 member
    # dose-response moved median R2 by +-0.002, and the WIDEST ensemble was slightly worse than the
    # narrowest. Running it here tests whether that flat response replicates on returns/rank IC.
    # NOT 'population' (all 100/run): 600 genomes x 452 test files is ~271k evaluate_rnn calls, and
    # the vol dose-response says island mates within a lineage add nothing over the champion.
    ap.add_argument("--members", choices=["global_best", "island_best"], default="global_best",
                    help="global_best: 1 genome/run (default). island_best: each island's champion")
    ap.add_argument("--allow-salvage", action="store_true",
                    help="for runs killed before their genome budget: fall back to the "
                         "argmin-best_validation_mse saved island best (EXAMM's own global-best "
                         "rule). Results are PARTIAL-BUDGET and must be reported as such.")
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    hac = a.hac_lag if a.hac_lag is not None else max(1, man["label_horizon_trading_days"] - 1)
    print(f"[dataset] handler={man.get('handler')} content={man['content_sha256'][:16]}...  "
          f"HAC lag={hac}")

    if not a.genome and not a.runs_root:
        sys.exit("ERROR: pass either --genome <file.bin> or --runs-root <dir>")
    if a.genome:
        if not os.path.exists(a.genome):
            sys.exit(f"ERROR: genome '{a.genome}' not found")
        tag = os.path.splitext(os.path.basename(a.genome))[0]
        cache = a.runs_root or os.path.dirname(os.path.abspath(a.genome))
        df = eval_genome(a.genome, a.data, f"{cache}/_eval_cache", tag)
        if df is None:
            sys.exit(f"ERROR: {a.genome} produced no predictions")
        df = restrict(df, a.data, tag)
        ics = daily_ic(df, pred_col="pred", label_col="LABEL", date_col="date")
        st = genome_stats(a.genome)
        print(f"\n=== single genome ===")
        print(f"   {tag}: weights={st['weights']} nodes={st['nodes']} edges={st['edges']} "
              f"rec={st['rec']}")
        report(tag, ics, hac)
        return

    runs = sorted(d for d in glob.glob(f"{a.runs_root}/run_*") if os.path.isdir(d))
    if a.members == "island_best":
        pairs, missing, salvaged = [], [], []
        for d in runs:
            champs = sorted(glob.glob(f"{d}/island_*_genome_0.bin"),
                            key=lambda p: int(re.search(r"island_(\d+)_", p).group(1)))
            if not champs:
                missing.append(d)
                continue
            # tag must be unique per genome: the prediction cache is keyed on it, so a collision
            # would silently score one genome and reuse its predictions for another.
            pairs += [(f"{os.path.basename(d)}_i{i}", g) for i, g in enumerate(champs)]
        print(f"[runs] {len(runs)} runs -> {len(pairs)} island champions"
              + (f"; NONE found in {[os.path.basename(m) for m in missing]}" if missing else ""))
    else:
        resolved = [(d,) + global_best(d, a.allow_salvage) for d in runs]
        missing = [d for d, g, _ in resolved if g is None]
        pairs = [(os.path.basename(d), g) for d, g, _ in resolved if g]
        salvaged = [os.path.basename(d) for d, g, src in resolved if src == "salvaged"]
        print(f"[runs] {len(pairs)} usable genome(s)"
              + (f"; MISSING in {[os.path.basename(m) for m in missing]}" if missing else ""))
    if not pairs:
        sys.exit("ERROR: no global_best_genome_*.bin found -- did the runs finish? "
                 "If a run was killed by walltime, re-run with --allow-salvage.")
    if salvaged:
        print(f"[WARNING] PARTIAL-BUDGET runs (walltime-killed, validation-selected from saved "
              f"island bests): {salvaged}. Report the genome count reached, not '10,000 genomes'.")

    cache = f"{a.runs_root}/_eval_cache"
    per_run, ics_per_run, sizes = {}, {}, {}
    for tag, g in pairs:
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

    # The baseline's parameter count MUST be read from the JSON, never hardcoded: this file is
    # rewritten by every baseline run, including capacity-sweep configs. A hardcoded 38,849 silently
    # mislabelled a 10,209-param sweep GRU as the 2x64 baseline and inflated the efficiency ratio
    # by 3.8x. The whole efficiency claim rests on this number being the one actually compared.
    gp = f"{a.data}/_lstm_gru_result.json"
    base_params = None
    if os.path.exists(gp):
        g = json.load(open(gp))
        print(f"\n=== same rows, same target, same HAC lag ===")
        for k, v in g.items():
            np_ = v.get("_n_params")
            print(f"   {k:<6} {v['ic']:+.4f}  HAC t {v['t_hac']:+.2f}   "
                  + (f"({np_:,} params)" if np_ else "(param count NOT in JSON)"))
            if k == "gru":
                base_params = np_
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
        if base_params:
            print(f"   GRU parameters (from JSON): {base_params:,}")
            print(f"   ratio                    : {base_params/mean_w:.0f}x fewer")
        else:
            print(f"   GRU parameters           : UNKNOWN -- _n_params absent from "
                  f"_lstm_gru_result.json, so no ratio is reported. Re-run the baseline rather "
                  f"than assuming a param count.")
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
