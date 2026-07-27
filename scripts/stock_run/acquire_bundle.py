#!/usr/bin/env python3
"""Acquire and pin a qlib data bundle to a PERSISTENT, checksummed location.

Why this exists: the `investment_data` bundle used for the first MASTER-replica build lived in a
/tmp scratchpad and was deleted by temp cleanup, leaving a 2 GB dataset that could not be rebuilt
or verified. That is a reproducibility failure -- a benchmark dataset whose source has vanished
cannot back a paper. This script makes bundle acquisition explicit, pinned, and verifiable.

Two bundles are supported:

  invdata  chenditc/investment_data -- Tushare/Baostock/Wind blend. Higher quality than the public
           bundle: populated $vwap, and fine-grained point-in-time CSI300 membership (16k+ spans,
           every rebalance) vs the public bundle's ~820 coarse spans. Releases are ROLLING DAILY,
           so the tag MUST be pinned -- "latest" is not reproducible. Their release ships a
           manifest.json carrying archive_sha256 plus the dolt/qlib/repo commits that produced it,
           all recorded here.

  cndata   qlib's public example bundle (Yahoo-sourced). qlib itself documents its quality as
           imperfect, and $vwap came back 0% populated in our earlier build. Used ONLY as an
           independent second source for cross-source replication -- if a result appears on
           invdata but not here, it is a data artifact, not a finding. Already present under
           ~/.qlib; recorded in place (with a checksum) rather than copied.

Usage:
    python3 scripts/stock_run/acquire_bundle.py --bundle invdata --tag 2026-07-26
    python3 scripts/stock_run/acquire_bundle.py --bundle cndata
    python3 scripts/stock_run/acquire_bundle.py --bundle invdata --verify-only

Idempotent: if the target already exists and its checksum matches BUNDLE.json, this is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE_ROOT = f"{REPO}/datasets/_bundles"

INVDATA_REPO = "chenditc/investment_data"
CNDATA_DEFAULT = os.path.expanduser("~/.qlib/qlib_data/cn_data")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    """Stream to disk, hashing as we go, with coarse progress."""
    h = hashlib.sha256()
    t0 = time.time()
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        nxt = 10
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            got += len(b)
            if total:
                pct = 100 * got / total
                if pct >= nxt:
                    print(f"      {pct:5.1f}%  ({got/1e6:.0f}/{total/1e6:.0f} MB, "
                          f"{time.time()-t0:.0f}s)", flush=True)
                    nxt += 10
    return h.hexdigest()


def qlib_version():
    try:
        import qlib
        return qlib.__version__
    except Exception:
        return "unknown"


def validate_layout(provider_uri):
    """A usable qlib bundle needs these three directories; fail loudly rather than let a
    half-extracted archive surface later as an empty-DataFrame mystery."""
    missing = [d for d in ("features", "instruments", "calendars")
               if not os.path.isdir(f"{provider_uri}/{d}")]
    if missing:
        sys.exit(f"ERROR: {provider_uri} is not a valid qlib bundle (missing: {missing})")
    n_feat = len(os.listdir(f"{provider_uri}/features"))
    insts = sorted(os.listdir(f"{provider_uri}/instruments"))
    return n_feat, insts


def write_bundle_json(out_dir, payload):
    os.makedirs(out_dir, exist_ok=True)
    p = f"{out_dir}/BUNDLE.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  wrote {p}")


def acquire_invdata(tag, verify_only):
    out_dir = f"{BUNDLE_ROOT}/invdata"
    provider_uri = f"{out_dir}/qlib_bin"
    bj = f"{out_dir}/BUNDLE.json"

    if verify_only or os.path.isdir(provider_uri):
        if not os.path.exists(bj):
            sys.exit(f"ERROR: {provider_uri} exists but {bj} does not -- unverifiable bundle. "
                     f"Delete the directory and re-run to acquire cleanly.")
        rec = json.load(open(bj))
        n_feat, insts = validate_layout(provider_uri)
        print(f"  bundle present: tag={rec['release_tag']} sha256={rec['archive_sha256'][:16]}... "
              f"({n_feat} feature dirs, instruments={insts})")
        if verify_only:
            return provider_uri
        print("  already acquired -- nothing to do (idempotent).")
        return provider_uri

    print(f"[invdata] resolving release {tag} from {INVDATA_REPO}...")
    man_url = (f"https://github.com/{INVDATA_REPO}/releases/download/{tag}/"
               f"qlib_bin.manifest.json")
    tar_url = f"https://github.com/{INVDATA_REPO}/releases/download/{tag}/qlib_bin.tar.gz"
    with urllib.request.urlopen(man_url) as r:
        man = json.load(r)
    expect = man["archive_sha256"].replace("sha256:", "")
    print(f"  release_tag       : {man['release_tag']}")
    print(f"  target_trade_date : {man['target_trade_date']}")
    print(f"  archive_sha256    : {expect}")
    print(f"  size              : {man['archive_size_bytes']/1e6:.0f} MB")

    os.makedirs(out_dir, exist_ok=True)
    tgz = f"{out_dir}/qlib_bin.tar.gz"
    print(f"  downloading -> {tgz}")
    got = download(tar_url, tgz)
    if got != expect:
        os.remove(tgz)
        sys.exit(f"ERROR: checksum mismatch!\n  expected {expect}\n  got      {got}\n"
                 f"Download corrupted or the release was mutated; refusing to use it.")
    print(f"  checksum OK ({got[:16]}...)")

    print("  extracting...")
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(out_dir)
    if not os.path.isdir(provider_uri):
        # archive root name differs from expectation -- find the real one
        cands = [d for d in os.listdir(out_dir)
                 if os.path.isdir(f"{out_dir}/{d}") and
                 os.path.isdir(f"{out_dir}/{d}/features")]
        if len(cands) != 1:
            sys.exit(f"ERROR: cannot locate qlib bundle root in {out_dir} (found {cands})")
        os.rename(f"{out_dir}/{cands[0]}", provider_uri)
    n_feat, insts = validate_layout(provider_uri)
    print(f"  extracted: {n_feat} feature dirs, instruments={insts}")
    os.remove(tgz)  # 558 MB archive; the extracted bundle + checksum record is what we keep

    write_bundle_json(out_dir, {
        "bundle": "invdata",
        "source": f"https://github.com/{INVDATA_REPO}",
        "release_tag": man["release_tag"],
        "release_url": tar_url,
        "archive_sha256": expect,
        "archive_size_bytes": man["archive_size_bytes"],
        "target_trade_date": man["target_trade_date"],
        "dolt_commit": man.get("dolt_commit"),
        "investment_data_commit": man.get("investment_data_commit"),
        "qlib_commit": man.get("qlib_commit"),
        "image_digest": man.get("image_digest"),
        "provider_uri": provider_uri,
        "n_feature_dirs": n_feat,
        "instruments": insts,
        "qlib_version_at_acquire": qlib_version(),
        "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("Releases are rolling daily -- this tag is pinned. Re-acquiring a DIFFERENT tag "
                 "will produce a different dataset; record the tag alongside any published result."),
    })
    return provider_uri


def acquire_cndata(cn_path, verify_only):
    """Recorded IN PLACE (not copied): ~/.qlib is user-home persistent, and copying 500 MB to
    duplicate it buys nothing. What matters is that its identity is pinned by checksum."""
    out_dir = f"{BUNDLE_ROOT}/cndata"
    if not os.path.isdir(cn_path):
        sys.exit(f"ERROR: {cn_path} not found. Acquire it with qlib's own downloader:\n"
                 f"  python -m qlib.run.get_data qlib_data --target_dir {cn_path} --region cn")
    n_feat, insts = validate_layout(cn_path)

    # Prefer hashing the original distribution zip if it is still alongside the extracted data:
    # one stable file identifies the release exactly. Otherwise fall back to a directory digest.
    zips = sorted(f for f in os.listdir(cn_path) if f.endswith(".zip"))
    if zips:
        zp = f"{cn_path}/{zips[0]}"
        print(f"  hashing distribution archive {zips[0]} ({os.path.getsize(zp)/1e6:.0f} MB)...")
        digest, kind = sha256_file(zp), f"archive:{zips[0]}"
    else:
        print("  no distribution zip present; computing directory digest (slower)...")
        h = hashlib.sha256()
        for root, dirs, files in os.walk(cn_path):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, cn_path).encode())
                h.update(str(os.path.getsize(p)).encode())
        digest, kind = h.hexdigest(), "directory-listing-digest"

    cal = f"{cn_path}/calendars/day.txt"
    span = None
    if os.path.exists(cal):
        with open(cal) as f:
            days = f.read().split()
        span = {"first": days[0], "last": days[-1], "n_trading_days": len(days)}

    print(f"  {kind} sha256 = {digest[:16]}...  ({n_feat} feature dirs)")
    if verify_only:
        bj = f"{out_dir}/BUNDLE.json"
        if os.path.exists(bj):
            rec = json.load(open(bj))
            ok = rec.get("content_sha256") == digest
            print(f"  verify: {'MATCH' if ok else 'MISMATCH -- bundle changed since acquisition!'}")
            if not ok:
                sys.exit(1)
        return cn_path

    write_bundle_json(out_dir, {
        "bundle": "cndata",
        "source": "qlib public example bundle (Yahoo-sourced), via qlib.run.get_data",
        "provider_uri": cn_path,
        "recorded_in_place": True,
        "content_sha256": digest,
        "content_sha256_kind": kind,
        "n_feature_dirs": n_feat,
        "instruments": insts,
        "calendar": span,
        "qlib_version_at_acquire": qlib_version(),
        "acquired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "known_limitations": [
            "Yahoo-sourced; qlib documents quality as 'might not be perfect'.",
            "$vwap was 0% populated in an earlier build from this bundle.",
            "Coarse point-in-time membership (~820 spans vs invdata's 16k+).",
            "Use ONLY as an independent replication source, never as the primary.",
        ],
    })
    return cn_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=["invdata", "cndata"], required=True)
    ap.add_argument("--tag", default="2026-07-26",
                     help="invdata release tag to PIN (rolling daily; do not use 'latest')")
    ap.add_argument("--cn-path", default=CNDATA_DEFAULT)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(BUNDLE_ROOT, exist_ok=True)
    if a.bundle == "invdata":
        uri = acquire_invdata(a.tag, a.verify_only)
    else:
        uri = acquire_cndata(a.cn_path, a.verify_only)
    print(f"\nprovider_uri = {uri}")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
