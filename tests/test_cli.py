"""CLI behaviour: argument mapping, recipe files, exit codes."""

from __future__ import annotations

import json

import pytest

from fixtures import make_corpus

from writing_dataset.cli import build_parser, main
from writing_dataset.tokenizer import CONTENT_BUDGET, MAX_SEQ_LEN, RESERVED_BOUNDARY_TOKENS


def test_defaults_match_the_spec():
    args = build_parser().parse_args([])
    assert args.max_seq_len == MAX_SEQ_LEN == 8192
    assert args.reserved_boundary_tokens == RESERVED_BOUNDARY_TOKENS == 2
    assert MAX_SEQ_LEN - RESERVED_BOUNDARY_TOKENS == CONTENT_BUDGET == 8190
    assert args.split_by == "book"


def test_default_exclusions_cover_wings_of_fire_and_onyx():
    args = build_parser().parse_args([])
    from writing_dataset.build import BuildConfig

    patterns = BuildConfig().exclude_patterns
    assert any("wings of fire" in p for p in patterns)
    assert any("onyx" in p for p in patterns)


def test_check_tokenizer_exits_zero(capsys):
    assert main(["--check-tokenizer"]) == 0
    out = capsys.readouterr().out
    assert "vocab size" in out
    assert "8190" in out


def test_missing_raw_root_is_configuration_error(capsys, tmp_path):
    code = main(["--raw-root", str(tmp_path / "nope"), "--out-root", str(tmp_path / "out")])
    assert code == 2
    assert "raw root does not exist" in capsys.readouterr().err


def test_recipe_file_applies_as_defaults(tmp_path):
    from writing_dataset.cli import _apply_yaml_config

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "run_name: from-yaml\n"
        "budget:\n  max_seq_len: 4096\n  reserved_boundary_tokens: 2\n"
        "chunking:\n  include_chapter_headings: false\n  min_chunk_tokens: 700\n"
        "sources:\n  exclude_patterns: [onyx*]\n"
        "outputs:\n  emit_tokenized: true\n  chunks_jsonl: false\n",
        encoding="utf-8",
    )

    parser = build_parser()
    _apply_yaml_config(parser, recipe)
    args = parser.parse_args([])

    assert args.run_name == "from-yaml"
    assert args.max_seq_len == 4096
    assert args.no_chapter_headings is True      # inverted flag
    assert args.min_chunk_tokens == 700
    assert args.exclude == ["onyx*"]
    assert args.emit_tokenized is True
    assert args.no_chunks_jsonl is True          # inverted flag


def test_cli_flags_override_the_recipe(tmp_path):
    from writing_dataset.cli import _apply_yaml_config

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("budget:\n  max_seq_len: 4096\n", encoding="utf-8")

    parser = build_parser()
    _apply_yaml_config(parser, recipe)
    args = parser.parse_args(["--max-seq-len", "2048"])
    assert args.max_seq_len == 2048


def test_missing_recipe_file_reports_cleanly(capsys, tmp_path):
    code = main(["--config", str(tmp_path / "absent.yaml")])
    assert code == 2
    assert "config file not found" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path, capsys):
    raw = tmp_path / "raw"
    make_corpus(raw)
    out = tmp_path / "out"
    code = main([
        "--raw-root", str(raw), "--out-root", str(out),
        "--run-name", "dry", "--dry-run", "--json",
    ])
    assert code == 0
    assert not out.exists() or not any(out.rglob("*"))

    stats = json.loads(capsys.readouterr().out)
    assert stats["budget"]["content_budget"] == 8190
    assert stats["dataset"]["chunks"] > 0


def test_end_to_end_cli_build(tmp_path):
    raw = tmp_path / "raw"
    make_corpus(raw)
    out = tmp_path / "out"
    code = main([
        "--raw-root", str(raw), "--out-root", str(out),
        "--run-name", "cli", "--val-fraction", "0.5",
    ])
    assert code == 0
    run = out / "cli"
    for name in ("chunks.jsonl", "train.jsonl", "val.jsonl",
                 "oversize_blocks.jsonl", "stats.json", "REPORT.md"):
        assert (run / name).is_file()

    stats = json.loads((run / "stats.json").read_text(encoding="utf-8"))
    assert stats["corpus"]["books_excluded"] == 2
    assert stats["dataset"]["chunk_tokens"]["max"] <= 8190
