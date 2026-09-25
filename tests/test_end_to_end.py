"""End-to-end build over a small on-disk corpus (EPUB + TXT, with exclusions)."""

from __future__ import annotations

import json

import pytest

from fixtures import make_corpus

from writing_dataset.build import BuildConfig, run_build, write_outputs


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("corpus")
    raw = tmp / "raw"
    make_corpus(raw)

    cfg = BuildConfig(
        raw_root=raw,
        out_root=tmp / "out",
        run_name="e2e",
        val_fraction=0.5,
        split_by="book",
    )
    result = run_build(cfg, progress=False)
    out_dir = write_outputs(result, cfg)
    return cfg, result, out_dir


def test_excluded_folders_absent_from_dataset(built):
    cfg, result, out_dir = built
    included = {b.book_id for b in result.books}
    assert not any("wings" in b for b in included), "Wings of Fire must be excluded"
    assert not any("onyx" in b for b in included), "Onyx must be excluded"

    assert len(result.excluded) == 2
    patterns = {rec["matched_pattern"] for rec in result.excluded}
    assert patterns

    blob = (out_dir / "chunks.jsonl").read_text(encoding="utf-8")
    assert "never reach the dataset" not in blob
    assert "also never reach the dataset" not in blob


def test_expected_files_written(built):
    _, _, out_dir = built
    for name in (
        "chunks.jsonl",
        "train.jsonl",
        "val.jsonl",
        "oversize_blocks.jsonl",
        "stats.json",
        "REPORT.md",
        "tokenizer_info.json",
    ):
        assert (out_dir / name).is_file(), f"missing {name}"


def test_chunks_respect_budget_and_schema(built):
    cfg, _, out_dir = built
    rows = [
        json.loads(line)
        for line in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    required = {
        "id", "text", "n_tokens", "book_id", "chapter_index", "chapter_title",
        "scene_index_start", "scene_index_end", "paragraph_index_start",
        "paragraph_index_end", "source_files", "flags",
    }
    for row in rows:
        assert required <= set(row)
        assert row["n_tokens"] <= cfg.content_budget
        assert row["text"].strip()
        assert len({f for f in row["source_files"]}) == len(row["source_files"])


def test_train_and_val_are_disjoint_by_book(built):
    _, result, _ = built
    train_books = {c.book_id for c in result.train}
    val_books = {c.book_id for c in result.val}
    assert not (train_books & val_books), "a book leaked across the split"


def test_stats_capture_the_budget_contract(built):
    _, _, out_dir = built
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["budget"]["max_seq_len"] == 8192
    assert stats["budget"]["reserved_boundary_tokens"] == 2
    assert stats["budget"]["content_budget"] == 8190
    assert stats["model"]["hf_repo"] == "mistralai/Ministral-3-14B-Base-2512"
    assert stats["dataset"]["chunks"] > 0
    assert stats["dataset"]["training_tokens_including_boundaries"] == (
        stats["dataset"]["content_tokens"] + stats["dataset"]["boundary_tokens"]
    )


def test_report_mentions_the_key_facts(built):
    _, _, out_dir = built
    report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "8192" in report and "8190" in report
    assert "Ministral-3-14B-Base-2512" in report
    assert "Oversize" in report


def test_epub_and_txt_both_extracted(built):
    _, result, _ = built
    sources = {f for b in result.books for f in b.source_files}
    assert any(s.endswith(".epub") for s in sources)
    assert any(s.endswith(".txt") for s in sources)


def test_txt_scene_breaks_detected(built):
    _, result, _ = built
    txt_book = next(b for b in result.books if b.source_files[0].endswith(".txt"))
    scene_counts = [c.n_scenes for c in txt_book.chapters]
    assert max(scene_counts) >= 2, "the '* * *' break should split Chapter 1 into two scenes"


def test_emit_tokenized_roundtrip(tmp_path):
    import numpy as np

    from fixtures import make_corpus

    raw = tmp_path / "raw"
    make_corpus(raw)
    cfg = BuildConfig(
        raw_root=raw, out_root=tmp_path / "out", run_name="tok", emit_tokenized=True
    )
    result = run_build(cfg, progress=False)
    out_dir = write_outputs(result, cfg)

    ids = np.load(out_dir / "input_ids.npy")
    offsets = np.load(out_dir / "offsets.npy")
    assert ids.ndim == 1
    assert offsets.shape == (len(result.chunks), 3)

    info = json.loads((out_dir / "tokenizer_info.json").read_text(encoding="utf-8"))
    bos, eos = info["bos_id"], info["eos_id"]
    for i, chunk in enumerate(result.chunks):
        start, end, length = offsets[i]
        assert ids[start] == bos
        assert ids[end - 1] == eos
        assert length == end - start
        assert length <= cfg.max_seq_len
