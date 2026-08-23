#!/usr/bin/env python3
"""Compare EXA-GP / EXA-GP-MIN against EXAMM on a Dataset A cohort.

    # after eval_ic_run.sh has produced ensemble_test/ for each arm:
    python3 scripts/stock_run/compare_exagp.py --cohort cohort_2021_aligned \
        --arm "EXA-GP-MIN=test_output/exagpmin_cohort_2021_aligned" \
        --arm "EXA-GP=test_output/exagp_cohort_2021_aligned" \
        --arm "EXAMM (local)=test_output/examm_cohort_2021_aligned" \
        --equations

METRICS COME FROM score_forecasting.py, IMPORTED RATHER THAN REIMPLEMENTED. The whole
point of this comparison is that an EXA-GP number is commensurable with the EXAMM number
already in the paper, and the surest way to lose that is to write a second IC function
that differs in some detail -- rank-vs-Pearson, the forward-compounding convention, which
days are kept. Importing means a change to the paper's scorer moves both sides together.

THE YEAR SLICE IS NOT OPTIONAL. cohort_2020's test split spans 2022 AND 2023 (501 rows),
so scoring it whole returns a two-year blend: results/hp_grid_orig records +0.0157 that
way against a true +0.0030. The cohort -> trade-year map below is the same one
score_forecasting.cells() uses.

EXAMM (ANVIL) IS SHOWN AS A REFERENCE ROW, NOT AS A CONTROL. It is the paper's own
published tree, produced by examm_mpi on Anvil. The local EXAMM arm is the control -- the
one that shares a binary and a machine with the EXA-GP arms. Reading the Anvil row against
a local arm measures method PLUS binary PLUS parallelism; reading local-EXAMM against
Anvil-EXAMM isolates the last two, which is what tells you whether the twelve
mid_highmid cells can be compared against the paper's existing EXAMM numbers without
re-running EXAMM on all of them.

SEED DISPERSION IS REPORTED BECAUSE THIS REPO HAS BEEN BITTEN BY IT. score_forecasting's
reps_for() documents per-cell swings of 26-36 points between trees of an identical config.
An ensemble IC quoted without the spread of its members invites exactly the over-reading
that produced those surprises, so mean/sd/range across the ten single runs are printed
next to every ensemble figure.
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_forecasting import PRIMARY, data_dir, from_per_ticker, load, metrics  # noqa: E402

# Train through N-1, validate on N, trade N+1. Dataset A supplies two such cells; the
# mid_highmid_price panels supply three apiece, which is the same rule with cohort_2022 added.
TRADE_YEAR = {"cohort_2020_aligned": 2022, "cohort_2021_aligned": 2023,
              "cohort_2022_aligned": 2024}


def arm_metrics(root, cohort, year, panel="dsA"):
    """Ensemble metrics plus the per-run IC spread behind them."""
    dd = data_dir(panel, cohort)
    ens = os.path.join(root, "ensemble_test")
    if not os.path.isdir(ens):
        raise FileNotFoundError(
            f"{ens} missing -- run: bash scripts/stock_run/eval_ic_run.sh {root} test")
    out = metrics(from_per_ticker(ens, dd), year)

    # Per-run ICs from the same eval_test dirs eval_ic_run.sh already wrote. A run whose
    # genome failed to evaluate leaves no dir; skip it loudly rather than averaging over a
    # silently smaller set, which is the failure eval_ic_run.sh's own hard-fail guards exist for.
    singles = []
    for d in sorted(glob.glob(os.path.join(root, "run_*", "eval_test"))):
        try:
            singles.append(metrics(from_per_ticker(d, dd), year)["ic1"])
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"  ! skipping {d}: {e}", file=sys.stderr)
    out["n_runs"] = len(singles)
    if singles:
        out["single_mean"] = statistics.fmean(singles)
        out["single_sd"] = statistics.stdev(singles) if len(singles) > 1 else 0.0
        out["single_min"], out["single_max"] = min(singles), max(singles)
    return out


def equations(root):
    """The evolved closed form per run -- the thing EXAMM cannot produce at all.

    ONE GENOME PER RUN, THE HIGHEST-INDEXED ONE. A run directory can hold more than one
    global_best_genome_* file (run_7 of the g500 2021 tree holds both 325 and 363), so a
    plain glob reports more "runs" than exist and double-counts whichever seed happened to
    flush twice. eval_ic_run.sh scores the highest index -- the best fitness found -- so
    this must select identically or the equations shown would not be the equations scored.

    THE WHOLE BLOCK, NOT THE FIRST LINE. Equations with hidden nodes are written across
    several lines (H0 = ..., H1 = ..., then RET(t + 1) in terms of them). Taking only the
    first line renders such a genome as "H0 = sin(...)", which silently misreports a
    two-term model as a one-term one -- and hides the output line entirely.
    """
    seen = {}
    for d in sorted(glob.glob(os.path.join(root, "run_*"))):
        fs = glob.glob(os.path.join(d, "global_best_genome_*.txt"))
        if not fs:
            continue
        best = max(fs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        body = []
        for line in open(best).read().splitlines():
            if line.startswith("best_validation_mse"):
                break
            if line.strip():
                body.append(line.strip())
        seen.setdefault(" ; ".join(body), []).append(os.path.basename(d))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=sorted(TRADE_YEAR))
    ap.add_argument("--panel", default="dsA",
                    help="dsA (default) or set1..set4 for the mid_highmid_price cells")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=ROOT",
                    help="repeatable; e.g. 'EXA-GP-MIN=test_output/exagpmin_cohort_2021_aligned'")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the paper's Anvil EXAMM row")
    ap.add_argument("--equations", action="store_true",
                    help="print the evolved closed forms per arm")
    a = ap.parse_args()

    year = TRADE_YEAR[a.cohort]
    rows = []

    if not a.no_reference:
        try:
            rows.append(("EXAMM (Anvil, paper)",
                         metrics(load("EXAMM", a.panel, a.cohort, rep=""), year)))
        except (FileNotFoundError, ValueError) as e:
            print(f"! reference EXAMM unavailable: {e}", file=sys.stderr)

    for spec in a.arm:
        if "=" not in spec:
            ap.error(f"--arm wants LABEL=ROOT, got {spec!r}")
        label, root = spec.split("=", 1)
        rows.append((label, arm_metrics(root, a.cohort, year, a.panel)))

    hdr = ["model", "days", "runs"] + [f"IC h={h}" for h in PRIMARY] + \
          ["MSE", "MAE", "single IC mean", "sd", "range"]
    print(f"\n=== {a.panel} {a.cohort} -> trade {year} (rank IC, cross-sectional) ===\n")
    print(" | ".join(hdr))
    print("-|-".join("-" * len(h) for h in hdr))
    for label, m in rows:
        cells = [label, str(m["days"]), str(m.get("n_runs", "--"))]
        cells += [f"{m.get(f'ic{h}', float('nan')):+.4f}" for h in PRIMARY]
        cells += [f"{m['mse']:.3e}", f"{m['mae']:.5f}"]
        if "single_mean" in m:
            cells += [f"{m['single_mean']:+.4f}", f"{m['single_sd']:.4f}",
                      f"[{m['single_min']:+.4f}, {m['single_max']:+.4f}]"]
        else:
            cells += ["--", "--", "--"]
        print(" | ".join(cells))

    if a.equations:
        for spec in a.arm:
            label, root = spec.split("=", 1)
            eqs = equations(root)
            if not eqs:
                continue
            print(f"\n--- {label}: {len(eqs)} distinct closed form(s) over "
                  f"{sum(len(v) for v in eqs.values())} run(s) ---")
            for eq, runs in sorted(eqs.items(), key=lambda kv: -len(kv[1])):
                print(f"  [{len(runs)}x] {eq}")


if __name__ == "__main__":
    main()
