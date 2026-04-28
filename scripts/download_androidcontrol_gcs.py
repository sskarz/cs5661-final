#!/usr/bin/env python3
"""Download the AndroidControl GCS shards (with a11y trees).

The HuggingFace mirror (smolagents/android-control) drops the accessibility
trees. Original Google release at gs://gresearch/android_control/ keeps them.
Format: 20 GZIP-compressed TFRecord shards, ~2.5 GB each, ~50 GB total.

Resumable + idempotent. Validates by HTTP HEAD content-length. Run from repo
root. No auth required (public bucket).

Usage:
    uv run python scripts/download_androidcontrol_gcs.py \\
        --output-dir data/androidcontrol_gcs \\
        --shards 0-19          # all shards
        --shards 0              # smoke (one shard, ~2.5 GB)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BUCKET_BASE = "https://storage.googleapis.com/gresearch/android_control"
SHARD_FMT = "android_control-{:05d}-of-00020"
SPLITS_OBJECT = ""  # the bucket directory marker ALSO contains the splits json
TOTAL_SHARDS = 20


def parse_shard_spec(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(s for s in out if 0 <= s < TOTAL_SHARDS)


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers["Content-Length"])
    except Exception:
        return None


def download_shard(shard_idx: int, output_dir: Path, force: bool = False) -> bool:
    name = SHARD_FMT.format(shard_idx)
    url = f"{BUCKET_BASE}/{name}"
    dest = output_dir / name

    expected = head_size(url)
    if expected is None:
        print(f"[shard {shard_idx:02d}] HEAD failed; skipping", file=sys.stderr)
        return False

    if dest.exists() and not force:
        actual = dest.stat().st_size
        if actual == expected:
            print(f"[shard {shard_idx:02d}] OK ({actual / 1e9:.2f} GB, complete)")
            return True
        print(f"[shard {shard_idx:02d}] partial {actual}/{expected} — resuming")
        # Resume via Range
        start = actual
    else:
        start = 0

    headers = {"Range": f"bytes={start}-"} if start > 0 else {}
    req = urllib.request.Request(url, headers=headers)

    t0 = time.time()
    written = 0
    mode = "ab" if start > 0 else "wb"
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
            while True:
                buf = r.read(1 << 22)  # 4 MB
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                # progress every ~100MB
                if written // (100 * 1 << 20) != (written - len(buf)) // (100 * 1 << 20):
                    elapsed = time.time() - t0
                    rate = (written / 1e6) / max(elapsed, 1e-3)
                    pct = 100 * (start + written) / expected
                    print(f"[shard {shard_idx:02d}] {pct:5.1f}% {rate:6.1f} MB/s")
    except Exception as e:
        print(f"[shard {shard_idx:02d}] download error: {e}", file=sys.stderr)
        return False

    final = dest.stat().st_size
    if final != expected:
        print(
            f"[shard {shard_idx:02d}] size mismatch {final}/{expected}",
            file=sys.stderr,
        )
        return False
    print(f"[shard {shard_idx:02d}] DONE ({final / 1e9:.2f} GB)")
    return True


def download_splits(output_dir: Path) -> bool:
    """Download the small splits-index object (96 KB)."""
    url = f"{BUCKET_BASE}/"
    dest = output_dir / "splits.json"
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        # Validate: should be JSON with train/validation/test keys
        d = json.loads(data)
        assert {"train", "validation", "test"} <= set(d.keys())
        dest.write_bytes(data)
        print(f"[splits] {len(d['train'])}/{len(d['validation'])}/{len(d['test'])} ep ids")
        return True
    except Exception as e:
        print(f"[splits] failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("data/androidcontrol_gcs"))
    ap.add_argument("--shards", default="0-19", help='e.g. "0", "0-3", "0,5,7"')
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--splits-only", action="store_true",
                    help="Only download the small splits index (96 KB)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not download_splits(args.output_dir):
        sys.exit(1)
    if args.splits_only:
        return

    shards = parse_shard_spec(args.shards)
    print(f"[plan] downloading {len(shards)} shards: {shards}")
    failed: list[int] = []
    for s in shards:
        ok = download_shard(s, args.output_dir, force=args.force)
        if not ok:
            failed.append(s)
    if failed:
        print(f"[fail] {len(failed)} shards failed: {failed}", file=sys.stderr)
        sys.exit(2)
    print(f"[done] {len(shards)} shards in {args.output_dir}")


if __name__ == "__main__":
    main()
