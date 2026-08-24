#!/usr/bin/env python3
"""Build an EXAMM panel from Barkjohn's own 49-sensor collocation set.

    python3 scripts/purpleair/build_barkjohn_dataset.py

WHY, RATHER THAN THE 7-SITE LTPP PANEL WE ALREADY HAVE. The deployed incumbent was fitted
on this data: 12,430 paired 24-hour records, 49 sensors, 5 regions. Our LTPP panel has six
training sites, and that shortfall is measurable rather than theoretical -- the held-out wi
site fails by 85-98% under every hyperparameter configuration tried, with roughly +3.7
ug/m3 bias, because a model trained on six sites averaging 8-10.6 ug/m3 learns a baseline
LEVEL instead of a site-invariant mapping. Training on the same 49 sensors the incumbent
saw removes that as an excuse and makes the comparison fair in the direction that currently
favours us being wrong.

IT ALSO CARRIES INPUTS THE INCUMBENT'S FUNCTIONAL FORM CANNOT USE: particle size-bin counts
(b0.3um through b10um) and dewpoint. Humidity biases a low-cost sensor by swelling
particles, which is a change in the SIZE DISTRIBUTION -- exactly what those bins measure.
A three-term linear equation in (cf_1, RH) cannot represent that; an evolved graph can. So
the panel is emitted twice: `full` with the bins, and `minimal` with only the two inputs
the incumbent uses. Comparing the two is the ablation that separates "our search is better"
from "we used information the equation form cannot hold", and only the second is a real
contribution.

NUMERIC-CLEAN OR NOT AT ALL. time_series.cxx:400-409 drops a non-numeric cell rather than
NaN-filling it, which SHORTENS that column and silently desynchronises every other column
in the file, with only a non-fatal log line. Rows with any missing used field are therefore
dropped here, up front. TempC and Dewpoint account for nearly all of it (184 of 12,430).

SHORT SERIES ARE DROPPED. Sensors range from 3 to 830 rows; a 3-row series cannot support
backprop through time and would contribute noise to the ensemble while counting as a
training file. MIN_ROWS is deliberately explicit rather than a silent filter.
"""
from __future__ import annotations

import os
import shutil

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, ".scratch/purpleair/barkjohn_24h",
                   "Data_DevelopmentUSPAcorrection_210408/Full24hrdataset.csv")
OUT = os.path.join(REPO, "datasets/purpleair_bk")
MIN_ROWS = 30

# (source column, emitted name). ref_pm25 last: it is the target.
FULL = [("PM25cf1", "pa_cf1"), ("RH", "pa_rh"), ("TempC", "pa_temp"), ("Dewpoint", "pa_dew"),
        ("b0.3um", "b03"), ("b0.5um", "b05"), ("b1um", "b1"), ("b2.5um", "b25"),
        ("b5um", "b5"), ("b10um", "b10"), ("PM25FM", "ref_pm25")]
MINIMAL = [("PM25cf1", "pa_cf1"), ("RH", "pa_rh"), ("PM25FM", "ref_pm25")]


def emit(df, variant, cols):
    root = os.path.join(OUT, variant)
    if os.path.isdir(root):
        shutil.rmtree(root)
    src_cols = [c for c, _ in cols]
    kept = dropped_rows = dropped_sensors = 0
    for (region, sid), g in df.groupby(["region", "ID"], sort=True):
        g = g.sort_values("Date")
        before = len(g)
        g = g.dropna(subset=src_cols)
        dropped_rows += before - len(g)
        if len(g) < MIN_ROWS:
            dropped_sensors += 1
            continue
        out = pd.DataFrame({name: g[src].astype(float).to_numpy() for src, name in cols})
        # Guard the exact failure mode the header describes, rather than trusting dropna.
        assert out.notna().all().all(), f"non-finite value survived in {region}/{sid}"
        d = os.path.join(root, str(region).replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        out.to_csv(os.path.join(d, f"{sid}.csv"), index=False)
        kept += 1
    print(f"{variant:<8} sensors kept {kept:>3}  dropped(<{MIN_ROWS} rows) {dropped_sensors:>2}  "
          f"rows dropped for NaN {dropped_rows:>4}  cols {len(cols)}")
    return root


def main():
    df = pd.read_csv(SRC, low_memory=False)
    print(f"source: {len(df)} rows, {df['ID'].nunique()} sensors, {df['region'].nunique()} regions")

    # BOTH VARIANTS MUST SEE IDENTICAL ROWS. Dropping NaNs per-variant gave `full` 45 sensors
    # and `minimal` 46, because TempC/Dewpoint are missing where cf_1 and RH are not. The
    # full-vs-minimal comparison is the ablation that decides whether the size bins carry
    # real information, so letting the two differ in their row sets would confound "extra
    # features help" with "different data", which is precisely the question being asked.
    df = df.dropna(subset=[c for c, _ in FULL])
    print(f"restricted to rows complete in every FULL column: {len(df)} rows, "
          f"{df['ID'].nunique()} sensors\n")

    for variant, cols in (("full", FULL), ("minimal", MINIMAL)):
        root = emit(df, variant, cols)
        for region in sorted(os.listdir(root)):
            files = os.listdir(os.path.join(root, region))
            n = sum(sum(1 for _ in open(os.path.join(root, region, f))) - 1 for f in files)
            print(f"    {region:<20} {len(files):>2} sensors  {n:>5} rows")
        print()


if __name__ == "__main__":
    main()
