#!/usr/bin/env python3
"""Fetch the model's own ``tekken.json`` for exact token counting.

Why this exists
---------------
``mistral-common`` bundles Tekken files with their **special tokens stripped**:
ranks 0..999 are raw control bytes, so ``"<s>"`` encodes to three ordinary
tokens instead of id 1. The file shipped alongside Ministral-3-14B-Base-2512
(16.8 MB, next to ``special_tokens_map.json``) has the real specials.

For plain prose the two agree, because the BPE merge table for ranks >= 1000 is
shared, and the stripped file can only *over*-count -- so it is a safe upper
bound. Use this script when you want counts that are exact rather than merely
conservative.

Run it somewhere with Hugging Face access (your training box), then pass the
result to the builder with ``--tekken``.

    python scripts/fetch_tekken.py --out data/tokenizer/tekken.json
    python -m writing_dataset.cli --check-tokenizer --tekken data/tokenizer/tekken.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = "mistralai/Ministral-3-14B-Base-2512"
FILENAMES = ("tekken.json",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/tokenizer/tekken.json"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("error: pip install huggingface_hub", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    for name in FILENAMES:
        try:
            path = hf_hub_download(
                repo_id=args.repo, filename=name, revision=args.revision
            )
        except Exception as exc:
            print(f"error: could not download {name} from {args.repo}: {exc}", file=sys.stderr)
            return 2
        data = Path(path).read_bytes()
        args.out.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        print(f"{name}: {len(data):,} bytes -> {args.out}")
        print(f"  sha256 {digest}")

    print(
        "\nNext: check it.\n"
        f"  python -m writing_dataset.cli --check-tokenizer --tekken {args.out}\n"
        "Expect BOS 1, EOS 2 and 'real specials: True'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
