"""Command-line entry point.

Exit codes
----------
0   success
2   configuration or source problem (nothing usable found, bad manifest, ...)
3   oversize stop threshold reached (nothing is written unless --force)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .build import DEFAULT_EXCLUDE_PATTERNS, BuildConfig, run_build, write_outputs
from .chunking import OVERSIZE_SCENE_EXCLUDE, OVERSIZE_SCENE_SPLIT
from .tokenizer import MAX_SEQ_LEN, RESERVED_BOUNDARY_TOKENS, load_tokenizer


#: Maps a nested key in a --config YAML file to (argparse dest, transform).
#: ``transform`` lets the YAML express things positively ("chunks_jsonl: true")
#: while argparse uses inverted flags ("--no-chunks-jsonl").
def _config_map() -> dict:
    negate = lambda v: not v  # noqa: E731
    return {
        "run_name": ("run_name", None),
        "budget.max_seq_len": ("max_seq_len", None),
        "budget.reserved_boundary_tokens": ("reserved_boundary_tokens", None),
        "budget.block_separator": ("block_separator", None),
        "chunking.include_chapter_headings": ("no_chapter_headings", negate),
        "chunking.include_scene_headings": ("no_scene_headings", negate),
        "chunking.repeat_chapter_title": ("repeat_chapter_title", None),
        "chunking.min_chunk_tokens": ("min_chunk_tokens", None),
        "chunking.merge_across_chapters": ("merge_across_chapters", None),
        "chunking.oversize_scene_policy": ("oversize_scene_policy", None),
        "sources.exclude_patterns": ("exclude", lambda v: list(v)),
        "sources.drop_repeated_headings": ("keep_repeated_headings", negate),
        "sources.strip_page_numbers": ("strip_page_numbers", None),
        "sources.aggressive_breaks": ("aggressive_breaks", None),
        "sources.numbered_breaks": ("numbered_breaks", None),
        "splits.strategy": ("split_by", None),
        "splits.val_fraction": ("val_fraction", None),
        "splits.val_books": ("val_books", lambda v: list(v)),
        "review_gates.oversize_warn_fraction": ("oversize_warn_fraction", None),
        "review_gates.oversize_stop_fraction": ("oversize_stop_fraction", None),
        "outputs.chunks_jsonl": ("no_chunks_jsonl", negate),
        "outputs.emit_tokenized": ("emit_tokenized", None),
        "outputs.emit_hf_dataset": ("emit_hf_dataset", None),
    }


def _apply_yaml_config(parser: argparse.ArgumentParser, path: Path) -> None:
    """Load a recipe file and use its values as defaults.

    Anything passed on the command line still wins, because argparse only falls
    back to a default when the flag is absent.
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    defaults: dict = {}
    for dotted, (dest, transform) in _config_map().items():
        section, _, key = dotted.rpartition(".")
        node = data.get(section) if section else data
        if not isinstance(node, dict) or key not in node:
            continue
        value = node[key]
        defaults[dest] = transform(value) if transform else value

    if defaults:
        parser.set_defaults(**defaults)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="writing-dataset",
        description=(
            "Chunk source manuscripts into boundary-respecting training chunks for "
            "continued-pretraining LoRA on Ministral-3-14B-Base-2512 (8192-token budget)."
        ),
    )

    p.add_argument("--raw-root", type=Path, default=Path("data/raw"),
                   help="folder of source books (default: data/raw)")
    p.add_argument("--out-root", type=Path, default=Path("data/out"),
                   help="output folder (default: data/out)")
    p.add_argument("--run-name", default="ministral3-14b-base-8192",
                   help="subfolder name under --out-root")
    p.add_argument("--manifest", type=Path, default=None,
                   help="YAML manifest declaring which files make up each book")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML recipe file, e.g. configs/ministral3-14b-base-8192.yaml. "
                        "Explicit flags override values from the file.")

    g = p.add_argument_group("sequence budget")
    g.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                   help=f"total sequence budget (default {MAX_SEQ_LEN})")
    g.add_argument("--reserved-boundary-tokens", type=int, default=RESERVED_BOUNDARY_TOKENS,
                   help=f"positions held back for [BOS]/[EOS] (default {RESERVED_BOUNDARY_TOKENS})")
    g.add_argument("--tekken", default=None,
                   help="path to the model's tekken.json for exact token counts")
    g.add_argument("--allow-proxy-tokenizer", action="store_true",
                   help="permit an approximate token counter (dry runs only)")

    g = p.add_argument_group("chunking")
    g.add_argument("--block-separator", default="\n\n",
                   help="separator inserted between blocks inside a chunk")
    g.add_argument("--no-chapter-headings", action="store_true",
                   help="do not emit chapter headings")
    g.add_argument("--no-scene-headings", action="store_true",
                   help="do not emit scene sub-headings")
    g.add_argument("--repeat-chapter-title", action="store_true",
                   help="re-emit the chapter heading on continuation chunks (duplicates text)")
    g.add_argument("--min-chunk-tokens", type=int, default=0,
                   help="merge chunks smaller than this into the previous chunk (0 = off)")
    g.add_argument("--merge-across-chapters", action="store_true",
                   help="allow the small-chunk merge pass to cross chapter boundaries")
    g.add_argument("--oversize-scene-policy", choices=[OVERSIZE_SCENE_SPLIT, OVERSIZE_SCENE_EXCLUDE],
                   default=OVERSIZE_SCENE_SPLIT,
                   help="'split' cuts oversize scenes at paragraph boundaries (default); "
                        "'exclude' drops them")

    g = p.add_argument_group("source handling")
    g.add_argument("--exclude", action="append", default=None, metavar="GLOB",
                   help="exclude books matching GLOB (repeatable). "
                        f"Default: {', '.join(DEFAULT_EXCLUDE_PATTERNS)}")
    g.add_argument("--no-default-excludes", action="store_true",
                   help="drop the built-in Wings of Fire / Onyx exclusions")
    g.add_argument("--aggressive-breaks", action="store_true",
                   help="treat a lone dash/underscore line as a scene break")
    g.add_argument("--numbered-breaks", action="store_true",
                   help="treat a bare '1'/'2'/'3' line as a scene break")
    g.add_argument("--strip-page-numbers", action="store_true",
                   help="drop lines that are only a page number")
    g.add_argument("--keep-repeated-headings", action="store_true",
                   help="keep headings repeated 4+ times (running heads)")
    g.add_argument("--no-split-omnibuses", action="store_true",
                   help="treat a multi-novel .mobi as a single book instead of "
                        "splitting it at its table-of-contents offsets")

    g = p.add_argument_group("splits")
    g.add_argument("--val-fraction", type=float, default=0.10,
                   help="fraction held out for validation (default 0.10)")
    g.add_argument("--split-by", choices=["book", "chapter", "none"], default="book",
                   help="hold out whole books (default), chapters, or nothing")
    g.add_argument("--val-books", nargs="*", default=(),
                   help="explicit book ids to hold out for validation")

    g = p.add_argument_group("review gates")
    g.add_argument("--oversize-warn-fraction", type=float, default=0.005,
                   help="warn when excluded content exceeds this fraction (default 0.005)")
    g.add_argument("--oversize-stop-fraction", type=float, default=0.02,
                   help="stop for review above this fraction (default 0.02)")
    g.add_argument("--force", action="store_true",
                   help="write outputs even if the oversize stop threshold was hit")

    g = p.add_argument_group("outputs")
    g.add_argument("--no-chunks-jsonl", action="store_true",
                   help="skip the combined chunks.jsonl")
    g.add_argument("--emit-tokenized", action="store_true",
                   help="also write input_ids.npy / offsets.npy with boundaries applied")
    g.add_argument("--emit-hf-dataset", action="store_true",
                   help="also write a datasets.DatasetDict (requires the 'datasets' package)")
    g.add_argument("--dry-run", action="store_true",
                   help="analyse and report, write nothing")
    g.add_argument("--json", action="store_true",
                   help="print stats.json to stdout instead of the summary")

    g = p.add_argument_group("utilities")
    g.add_argument("--check-tokenizer", action="store_true",
                   help="print tokenizer provenance and exit")
    return p


def _config_from_args(args) -> BuildConfig:
    if args.exclude is None:
        excludes = () if args.no_default_excludes else DEFAULT_EXCLUDE_PATTERNS
    else:
        excludes = tuple(args.exclude)

    return BuildConfig(
        raw_root=args.raw_root,
        out_root=args.out_root,
        run_name=args.run_name,
        max_seq_len=args.max_seq_len,
        reserved_boundary_tokens=args.reserved_boundary_tokens,
        tekken_path=args.tekken,
        block_separator=args.block_separator,
        include_chapter_headings=not args.no_chapter_headings,
        include_scene_headings=not args.no_scene_headings,
        repeat_chapter_title=args.repeat_chapter_title,
        min_chunk_tokens=args.min_chunk_tokens,
        merge_across_chapters=args.merge_across_chapters,
        oversize_scene_policy=args.oversize_scene_policy,
        manifest=args.manifest,
        exclude_patterns=excludes,
        aggressive_breaks=args.aggressive_breaks,
        numbered_breaks=args.numbered_breaks,
        strip_page_numbers=args.strip_page_numbers,
        drop_repeated_headings=not args.keep_repeated_headings,
        split_omnibuses=not args.no_split_omnibuses,
        val_fraction=args.val_fraction,
        split_by=args.split_by,
        val_books=tuple(args.val_books),
        oversize_warn_fraction=args.oversize_warn_fraction,
        oversize_stop_fraction=args.oversize_stop_fraction,
        fail_on_oversize_stop=not args.force,
        emit_tokenized=args.emit_tokenized,
        emit_hf_dataset=args.emit_hf_dataset,
        write_chunks=not args.no_chunks_jsonl,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    # Two-pass parse: the first pass only looks for --config so the recipe can
    # be applied as defaults before the real parse.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    if pre_args.config is not None:
        if not pre_args.config.is_file():
            print(f"error: config file not found: {pre_args.config}", file=sys.stderr)
            return 2
        try:
            _apply_yaml_config(parser, pre_args.config)
        except Exception as exc:
            print(f"error: could not read {pre_args.config}: {exc}", file=sys.stderr)
            return 2

    args = parser.parse_args(argv)

    if args.check_tokenizer:
        counter = load_tokenizer(
            args.tekken,
            allow_proxy_fallback=args.allow_proxy_tokenizer,
            max_seq_len=args.max_seq_len,
        )
        print(counter.info.describe())
        probe = "The dragon spread its wings and rose above the mountain."
        print(f"\nsanity check   : {counter.count(probe)} tokens for a "
              f"{len(probe.split())}-word sentence")
        return 0

    cfg = _config_from_args(args)

    try:
        result = run_build(cfg, progress=not args.json)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from .report import render_console_summary

    if args.json:
        print(json.dumps(result.stats, indent=2, ensure_ascii=False))
    else:
        print(render_console_summary(result, cfg))

    if result.review_required and cfg.fail_on_oversize_stop and result.stats["oversize"]["stop"]:
        print(
            "\nSTOP FOR REVIEW: oversize content exceeded the stop threshold. "
            "Inspect oversize_blocks.jsonl, then re-run with --force to write outputs anyway.",
            file=sys.stderr,
        )
        if not args.dry_run:
            return 3

    if args.dry_run:
        # Keep stdout pure JSON when --json is set, so it can be piped to jq.
        print("\n(dry run: nothing written)", file=sys.stderr if args.json else sys.stdout)
        return 0

    out_dir = write_outputs(result, cfg)
    if not args.json:
        print(f"\nwrote {out_dir}/")
        for name in ("chunks.jsonl", "train.jsonl", "val.jsonl", "oversize_blocks.jsonl",
                     "stats.json", "REPORT.md"):
            if (out_dir / name).is_file():
                print(f"  {name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
