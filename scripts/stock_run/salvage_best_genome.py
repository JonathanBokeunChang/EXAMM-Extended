#!/usr/bin/env python3
"""Recover the validation-selected best genome from a run that was KILLED before completion.

Why this exists
---------------
EXAMM writes ``global_best_genome_*.bin`` exactly once, at examm.cxx:416-418, and only when
``evaluated_genomes > max_genomes``. A SLURM walltime kill therefore leaves NO global best, even
though ~80% of the search may have completed. If the run was submitted with
``--save_genome_option all_best_genomes``, every genome that was best-in-its-island at insert time
was written to ``rnn_genome_<generation_id>.bin`` (examm.cxx:377-379), and the global best is
necessarily among them -- a genome cannot be the global best without also being best in its island
at the moment it was inserted.

Why this is NOT winner's-curse selection
----------------------------------------
Selection here is on ``best_validation_mse``, which is precisely what EXAMM itself selects on:
``RNN_Genome::get_fitness()`` returns ``best_validation_mse`` (rnn_genome.cxx:928), and
``IslandSpeciationStrategy::insert_genome`` updates the global best via
``global_best_genome->get_fitness() > genome->get_fitness()`` (island_speciation_strategy.cxx:170),
i.e. lower is better. Taking the argmin over saved genomes reproduces the same choice the run would
have made had it finished the genome budget.

This is the distinction that matters: selecting on VALIDATION fitness is legitimate; selecting on
TEST IC is the winner's-curse pattern that already cost this project a bogus +0.0188 (vs +0.0113
honest) when a live tracker max-ed over island bests. Never sort these files by test metric.

The fitness is read from the ``.txt`` sibling that ``EXAMM::save_genome`` writes alongside each
``.bin`` via ``RNN_Genome::write_equations``, whose last line is ``best_validation_mse: <value>``
(rnn_genome.cxx:4835). That avoids parsing the binary, whose layout has variable-length fields
before the fitness and no stable byte offset.

Caveat: ``to_string()`` emits 6 decimal places, so genomes whose validation MSE agrees to <1e-6 are
indistinguishable here and the tie is broken by the LATER generation_id (more search behind it).
Ties are reported so they are never silent.

Usage:
    python3 scripts/stock_run/salvage_best_genome.py --run-dir test_output/seq_examm_grow3/run_1
    python3 scripts/stock_run/salvage_best_genome.py --runs-root test_output/seq_examm_grow3 --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

FITNESS_RE = re.compile(r"^best_validation_mse:\s*([-+0-9.eEnaN]+)\s*$", re.MULTILINE)
GEN_RE = re.compile(r"rnn_genome_(\d+)\.bin$")


def read_fitness(txt_path):
    """Last `best_validation_mse:` line in a genome's .txt sibling, or None."""
    try:
        with open(txt_path, "r", errors="replace") as f:
            hits = FITNESS_RE.findall(f.read())
    except OSError:
        return None
    if not hits:
        return None
    try:
        v = float(hits[-1])
    except ValueError:
        return None
    # a NaN/inf fitness means the genome never trained; it must never win an argmin
    return v if v == v and abs(v) != float("inf") else None


def salvage(run_dir):
    """Pick the min-validation-MSE saved genome in run_dir.

    Returns a dict describing the pick, or one with 'error' set.
    """
    done = os.path.exists(f"{run_dir}/.done")
    gb = sorted(glob.glob(f"{run_dir}/global_best_genome_*.bin"))
    if gb:
        return {"run": os.path.basename(run_dir), "completed": True, "genome": gb[-1],
                "source": "global_best", "note": "run finished normally; no salvage needed"}

    bins = glob.glob(f"{run_dir}/rnn_genome_*.bin")
    if not bins:
        return {"run": os.path.basename(run_dir), "completed": done, "error":
                "no global_best_genome_*.bin AND no rnn_genome_*.bin -- the run was killed and was "
                "NOT submitted with --save_genome_option all_best_genomes, so nothing is "
                "recoverable. Resubmit with a longer walltime."}

    cands = []
    for b in bins:
        m = GEN_RE.search(b)
        if not m:
            continue
        fit = read_fitness(b[:-4] + ".txt")
        if fit is None:
            continue
        cands.append((fit, int(m.group(1)), b))
    if not cands:
        return {"run": os.path.basename(run_dir), "completed": done, "error":
                f"{len(bins)} rnn_genome_*.bin found but none had a readable "
                f"'best_validation_mse:' in its .txt sibling -- cannot select on validation "
                f"fitness, and selecting any other way would bias the result."}

    # argmin on validation MSE; ties broken by LATER generation_id
    cands.sort(key=lambda t: (t[0], -t[1]))
    best_fit, best_gen, best_bin = cands[0]
    tied = [c for c in cands if c[0] == best_fit]

    return {"run": os.path.basename(run_dir), "completed": done, "genome": best_bin,
            "source": "salvaged", "val_mse": best_fit, "generation_id": best_gen,
            "n_saved": len(cands), "n_tied": len(tied),
            "worst_saved": cands[-1][0]}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="a single run_R directory")
    g.add_argument("--runs-root", help="a directory containing run_*/ subdirectories")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    a = ap.parse_args()

    if a.run_dir:
        runs = [a.run_dir]
    else:
        runs = sorted(d for d in glob.glob(f"{a.runs_root}/run_*") if os.path.isdir(d))
    if not runs:
        sys.exit("ERROR: no run directories matched")

    out = [salvage(r) for r in runs]

    if a.json:
        print(json.dumps(out, indent=2))
        return

    for r in out:
        if "error" in r:
            print(f"{r['run']}: ERROR {r['error']}")
            continue
        if r["source"] == "global_best":
            print(f"{r['run']}: COMPLETE -- {os.path.basename(r['genome'])} (no salvage needed)")
            continue
        print(f"{r['run']}: SALVAGED  {os.path.basename(r['genome'])}")
        print(f"    validation MSE : {r['val_mse']:.6f}   (best of {r['n_saved']} saved island "
              f"bests; worst {r['worst_saved']:.6f})")
        print(f"    generation_id  : {r['generation_id']}")
        if r["n_tied"] > 1:
            print(f"    NOTE: {r['n_tied']} genomes tie at this validation MSE to 6dp; broke the "
                  f"tie by latest generation_id.")
        print(f"    selection      : argmin best_validation_mse == EXAMM's own global-best rule.")

    if any(r.get("source") == "salvaged" for r in out):
        print("\nThis genome is validation-selected, exactly as global_best would have been, but the"
              "\nsearch was TRUNCATED by the walltime kill -- report it as a partial-budget result"
              "\n(state the genome count reached), never as a completed run.")


if __name__ == "__main__":
    main()
