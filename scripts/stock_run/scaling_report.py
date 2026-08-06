#!/usr/bin/env python3
"""Turn results/scaling/timings.csv into a speedup / parallel-efficiency table.

    python3 scripts/stock_run/scaling_report.py            # table + LaTeX
    python3 scripts/stock_run/scaling_report.py --csv path/to/timings.csv

THREE THINGS THIS CHECKS BEFORE REPORTING A SPEEDUP, because each can produce a
scaling curve that looks clean and means nothing:

  1. Did every arm do the SAME WORK? --max_genomes is a ceiling. If one arm stopped
     early its wall-clock is not comparable to the others, and dividing them produces
     a speedup that is really a work-ratio.

  2. Are the evolved genomes the SAME SIZE? EXAMM's insertion order is wall-clock
     dependent (examm_mpi.cxx: MPI_Probe(MPI_ANY_SOURCE)), so rank count changes which
     topologies get explored. An arm that happens to evolve bigger networks pays more
     per genome for reasons unrelated to parallelism.

  3. Is the spread across repeats smaller than the differences between arms? If not,
     the table is reporting node noise.

Speedup is reported against WORKERS (ranks - 1), not ranks. EXAMM is master-worker:
rank 0 coordinates and evaluates nothing. Charging the master to the parallel algorithm
understates efficiency by 12.5% at 8 ranks and 1.6% at 64 -- i.e. by a different amount
in each arm, which would bend the curve on its own.
"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/scaling/timings.csv")
    ap.add_argument("--baseline", type=int, default=None,
                    help="rank count to use as the speedup reference (default: smallest)")
    a = ap.parse_args()

    if not os.path.exists(a.csv):
        sys.exit(f"no timings at {a.csv} -- run submit_examm_scaling.sh first")

    rows = []
    with open(a.csv) as f:
        hdr = f.readline().strip().split(",")
        for ln in f:
            if not ln.strip():
                continue
            d = dict(zip(hdr, ln.strip().split(",")))
            rows.append(d)
    if not rows:
        sys.exit("timings.csv has a header but no data rows")

    def num(x, default=float("nan")):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    by = defaultdict(list)
    for r in rows:
        by[int(r["cores"])].append(r)

    counts = sorted(by)
    base = a.baseline if a.baseline is not None else counts[0]
    if base not in by:
        sys.exit(f"baseline {base} ranks has no rows (have {counts})")

    def stats(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return float("nan"), float("nan")
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0.0
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return m, var ** 0.5

    # ---- integrity checks, before any speedup is printed ----------------------
    print("=== integrity ===")
    ok = True

    def spread_check(col, label, tol, unit=""):
        """True if every arm agrees on `col` to within `tol` (relative)."""
        nonlocal ok
        vals = {c: stats([num(r.get(col)) for r in by[c]])[0] for c in counts}
        valid = {c: v for c, v in vals.items() if v == v}
        if len(valid) < 2:
            print(f"  [--]   {label}: not recorded, cannot check")
            return
        lo, hi = min(valid.values()), max(valid.values())
        if hi > 0 and (hi - lo) / hi > tol:
            print(f"  [FAIL] {label} differs across arms: {lo:,.0f}..{hi:,.0f}{unit}")
            ok = False
        else:
            print(f"  [ok]   {label} equal across arms: {lo:,.0f}..{hi:,.0f}{unit}")

    # Genome count AND backprop epochs. Genome count alone is not enough: two arms can insert
    # the same number of genomes while training them for different totals if any terminated
    # mid-evaluation. BP epochs is the actual compute unit.
    spread_check("genomes_done", "genomes inserted", 0.02)
    spread_check("bp_epochs", "backprop epochs (true work unit)", 0.02)

    # The genome-size confound: EXAMM's insertion order is wall-clock dependent
    # (examm_mpi.cxx, MPI_Probe(MPI_ANY_SOURCE)), so rank count changes which topologies get
    # explored. An arm that evolved larger networks pays more per genome for reasons that have
    # nothing to do with parallelism.
    spread_check("edges_mean", "mean best-genome edges", 0.15, " edges")

    # Wall-clock vs EXAMM's own clock. The gap is srun launch + MPI init + data load, which is
    # ~constant in rank count and therefore a LARGER fraction of the fast arms -- it depresses
    # measured speedup at 64 ranks specifically. Report it so the reader can see its size.
    for c in counts:
        w, _ = stats([num(r.get("seconds")) for r in by[c]])
        e, _ = stats([num(r.get("examm_internal_s")) for r in by[c]])
        if e == e and w == w and w > 0:
            oh = (w - e) / w * 100
            flag = "  <- startup is a large share of this arm" if oh > 15 else ""
            print(f"  [--]   {c:>3} ranks: wall {w:.0f}s, EXAMM-internal {e:.0f}s, "
                  f"startup {oh:.0f}%{flag}")

    for c in counts:
        n = len(by[c])
        if n < 3:
            print(f"  [WARN] {c} ranks has only {n} repeat(s) -- no usable variance")
            ok = False

    # ---- the table -----------------------------------------------------------
    bt, _ = stats([num(r["seconds"]) for r in by[base]])
    bw = int(by[base][0]["workers"])

    print(f"\n=== EXAMM strong scaling (baseline {base} ranks / {bw} workers) ===")
    print(f"{'ranks':>6} {'workers':>8} {'reps':>5} {'seconds':>18} {'genomes/s':>10} "
          f"{'speedup':>8} {'ideal':>7} {'efficiency':>11}")
    print("-" * 80)
    tex = []
    for c in counts:
        rs = by[c]
        t, sd = stats([num(r["seconds"]) for r in rs])
        gps, _ = stats([num(r["genomes_per_sec"]) for r in rs])
        w = int(rs[0]["workers"])
        speed = bt / t if t else float("nan")
        ideal = w / bw
        eff = speed / ideal if ideal else float("nan")
        cv = (sd / t * 100) if t else float("nan")
        print(f"{c:>6} {w:>8} {len(rs):>5} {t:>10.1f} +/-{sd:>5.1f} {gps:>10.2f} "
              f"{speed:>8.2f} {ideal:>7.2f} {eff:>10.0%}")
        tex.append((c, w, t, sd, gps, speed, ideal, eff))
        if cv == cv and cv > 10:
            print(f"       ^ repeat spread is {cv:.0f}% of the mean -- treat this row as indicative only")

    print("\n  speedup   = baseline_time / this_time")
    print("  ideal     = this_workers / baseline_workers  (master excluded)")
    print("  efficiency= speedup / ideal;  100% = perfect strong scaling")
    if not ok:
        print("\n  NOTE: an integrity check above failed -- read the table with that caveat.")

    # ---- LaTeX ---------------------------------------------------------------
    print("\n=== LaTeX ===")
    print(r"\begin{table}[t]")
    print(r"\centering")
    def tex_num(vals):
        """Largest finite value, with LaTeX-safe thousands separators ({,} not ,)."""
        v = max([x for x in vals if x == x] or [0])
        return f"{v:,.0f}".replace(",", "{,}")
    g_rep  = tex_num([num(r.get("genomes_done")) for rs in by.values() for r in rs])
    bp_rep = tex_num([num(r.get("bp_epochs"))    for rs in by.values() for r in rs])
    caption = (
        r"EXAMM strong scaling on one Anvil node: fixed search budget "
        rf"(${g_rep}$ genomes, ${bp_rep}$ backpropagation epochs), varying MPI rank "
        rf"count. Speedup is measured against the {base}-rank run; \emph{{ideal}} accounts "
        r"for EXAMM's master--worker structure, in which rank~0 coordinates and evaluates "
        r"no genomes. Times are means over repeated runs on an exclusively allocated node."
    )
    print(rf"\caption{{{caption}}}")
    print(r"\label{tab:examm-scaling}")
    print(r"\begin{tabular}{rrrrrr}")
    print(r"\toprule")
    print(r"Ranks & Workers & Time (s) & Genomes/s & Speedup & Efficiency \\")
    print(r"\midrule")
    for c, w, t, sd, gps, speed, ideal, eff in tex:
        print(rf"${c}$ & ${w}$ & ${t:.0f} \pm {sd:.0f}$ & ${gps:.2f}$ & "
              rf"${speed:.2f}\times$ & ${eff:.0%}$ \\".replace("%", r"\%"))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
