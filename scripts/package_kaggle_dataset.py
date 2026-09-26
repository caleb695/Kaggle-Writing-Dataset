#!/usr/bin/env python3
"""Copy the training-ready artifacts into ``kaggle_dataset/`` for GitHub → Kaggle.

Raw books stay gitignored. Only chunked/tokenized files that a trainer needs
are copied, all under GitHub's 100 MB file limit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

KEEP = (
    "train.jsonl",
    "val.jsonl",
    "input_ids.npy",
    "offsets.npy",
    "train_offsets.npy",
    "val_offsets.npy",
    "tokenized_index.jsonl",
    "stats.json",
    "tokenizer_info.json",
    "REPORT.md",
    "oversize_blocks.jsonl",
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("data/out/ministral3-14b-base-8192"),
        help="build output directory",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("kaggle_dataset"),
        help="folder that gets committed and imported by Kaggle",
    )
    args = p.parse_args()
    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"build output not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for name in KEEP:
        s = src / name
        if not s.exists():
            missing.append(name)
            continue
        shutil.copy2(s, dst / name)
        copied.append((name, s.stat().st_size))

    # Drop a tiny machine-readable manifest for the notebook's path finder.
    index = {
        "run_name": "ministral3-14b-base-8192",
        "model": "mistralai/Ministral-3-14B-Base-2512",
        "max_seq_len": 8192,
        "bos_id": 1,
        "eos_id": 2,
        "pad_id": 11,
        "files": {name: size for name, size in copied},
    }
    stats_path = dst / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        ds = stats.get("dataset", {})
        index["train_chunks"] = ds.get("train_chunks")
        index["val_chunks"] = ds.get("val_chunks")
        index["training_tokens"] = ds.get("training_tokens_including_boundaries")
    (dst / "manifest.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    print(f"packaged {len(copied)} files into {dst}/")
    for name, size in copied:
        print(f"  {size:12,}  {name}")
    if missing:
        print("missing (skipped):", ", ".join(missing))
        if "input_ids.npy" in missing or "train.jsonl" in missing:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
