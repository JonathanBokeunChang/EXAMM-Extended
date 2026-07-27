#!/usr/bin/env python3
"""Build the cross-sectional peer-feature cache. qlib + pandas only -- NO torch.

Exists because this repo's two interpreters are disjoint:
    scratchpad/qlibenv/bin/python   has qlib, no torch
    system python3                  has torch, no qlib
The nonlinear CS gate needs peer features (qlib) AND a GRU (torch), so the qlib half runs here,
writes a CSV, and the torch half just reads it. Importing the gate module directly would pull in
torch and fail, which is exactly what happened the first time.

Usage:
    scratchpad/qlibenv/bin/python scripts/vol_baselines/build_cs_features.py \
        --data datasets/csi300_master_replica_invdata_alpha360
then train with the torch interpreter:
    python3 scripts/vol_baselines/csi300_cs_nonlinear_gate.py --data <same dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi300_cs_operator_gate import build_cs_features, raw_returns  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{REPO}/datasets/csi300_master_replica_invdata_alpha360")
    ap.add_argument("--k", type=int, default=20, help="peers per stock (train-selected)")
    a = ap.parse_args()

    man = json.load(open(f"{a.data}/MANIFEST.json"))
    out = f"{a.data}/_cs_features.csv"

    # instruments and train dates come from the dataset itself, so the cache is always aligned
    # with the exact panel the gate will train on
    insts, train_dates = set(), set()
    for f in sorted(glob.glob(f"{a.data}/*_train.csv")):
        s = os.path.basename(f)[: -len("_train.csv")]
        insts.add(s)
        train_dates |= set(pd.read_csv(f, usecols=["date"])["date"])
    for split in ("val", "test"):
        for f in sorted(glob.glob(f"{a.data}/*_{split}.csv")):
            insts.add(os.path.basename(f)[: -len(f"_{split}.csv")])
    insts = sorted(insts)
    print(f"[cs] {len(insts)} instruments, {len(train_dates)} train dates, k={a.k}")

    print(f"[cs] querying raw returns from {man['bundle']['provider_uri']} ...")
    rets = raw_returns(man["bundle"]["provider_uri"], insts,
                        man["splits"]["train"][0], man["splits"]["test"][1])
    print(f"[cs] {len(rets):,} return rows")

    cs, n_peered = build_cs_features(rets, train_dates, k=a.k)
    cs_cols = [c for c in cs.columns if c.startswith("CS_")]
    cs = cs.dropna(subset=cs_cols, how="all")
    cs.to_csv(out, index=False, float_format="%.9g")
    print(f"[cs] {len(cs_cols)} features for {n_peered} stocks -> {out}")
    print(f"[cs] {len(cs):,} rows, {os.path.getsize(out)/1e6:.0f} MB")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
