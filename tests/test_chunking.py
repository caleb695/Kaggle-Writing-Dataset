"""Invariant tests for the chunking rules.

Each test names the rule it protects, so a failure tells you which
non-negotiable guarantee regressed.
"""

from __future__ import annotations

import pytest

from fixtures import counter, make_book, paragraph_of_tokens

from writing_dataset.chunking import (
    ChunkConfig,
    assert_budget,
    assert_no_duplicates,
    assert_order,
    chunk_book,
)

BUDGET = 8190


@pytest.fixture(scope="module")
def tok():
    return counter()


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def test_budget_respected_for_fitting_book(tok):
    """8192 total, 2 reserved: no chunk may exceed 8190 content tokens."""
    book = make_book(
        "b1",
        [("Chapter One", [([], [paragraph_of_tokens(tok, 3000, i)[0] for i in range(5)])])],
    )
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert chunks
    assert_budget(chunks, BUDGET)
    for chunk in chunks:
        assert chunk.n_tokens <= BUDGET


def test_boundary_tokens_complete_the_budget(tok):
    """Content budget + 2 reserved positions is exactly the sequence budget."""
    cfg = ChunkConfig()
    assert cfg.max_seq_len == 8192
    assert cfg.reserved_boundary_tokens == 2
    assert cfg.content_budget == 8190

    book = make_book("b1", [("Ch1", [([], [paragraph_of_tokens(tok, 9000, 0)[0]])])])
    chunks, _ = chunk_book(book, tok, cfg)
    # The 9000-token paragraph is oversize, so it must not appear in any chunk.
    assert all(c.n_tokens <= 8190 for c in chunks)
    for chunk in chunks:
        assert chunk.n_tokens + cfg.reserved_boundary_tokens <= cfg.max_seq_len


# --------------------------------------------------------------------------- #
# Paragraph integrity
# --------------------------------------------------------------------------- #


def test_oversize_paragraph_excluded_and_logged(tok):
    """A paragraph over the whole budget is recorded, never silently split."""
    big, big_tokens = paragraph_of_tokens(tok, 12000, 7)
    assert big_tokens > BUDGET, "fixture must actually exceed the budget"

    book = make_book(
        "b1",
        [("Chapter One", [([], [big, "A short trailing paragraph."])])],
    )
    chunks, oversize = chunk_book(book, tok, ChunkConfig())

    kinds = [r["kind"] for r in oversize]
    assert "oversize_paragraph" in kinds

    rec = next(r for r in oversize if r["kind"] == "oversize_paragraph")
    assert rec["n_tokens"] == big_tokens
    assert rec["budget"] == BUDGET
    assert rec["action"] == "excluded"
    assert rec["chapter"] == 0
    assert rec["scene"] == 0
    assert rec["paragraph"] is not None
    assert "reason" in rec and rec["reason"]

    for chunk in chunks:
        assert big[:200] not in chunk.text


def test_paragraph_is_never_split(tok):
    """Every emitted chunk's paragraphs appear verbatim and whole."""
    paras = [paragraph_of_tokens(tok, 2500, i)[0] for i in range(6)]
    book = make_book("b1", [("Chapter One", [([], paras)])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())

    joined = "\n\n".join(c.text for c in chunks)
    for p in paras:
        assert p in joined, "a whole paragraph went missing"
    assert len(paras) == 6


# --------------------------------------------------------------------------- #
# Scene integrity
# --------------------------------------------------------------------------- #


def test_scene_not_split_to_fill_a_chunk(tok):
    """Two ~5000-token scenes must not be merged-and-cut to fill 8190."""
    s1, n1 = paragraph_of_tokens(tok, 5000, 1)
    s2, n2 = paragraph_of_tokens(tok, 5000, 2)
    assert n1 + n2 > BUDGET, "fixture must not fit in a single chunk"

    book = make_book("b1", [("Chapter One", [([], [s1]), ([], [s2])])])
    chunks, oversize = chunk_book(book, tok, ChunkConfig())

    assert len(chunks) == 2, "each scene should get its own chunk"
    assert all(c.n_tokens <= BUDGET for c in chunks)
    # Scene 1 was not cut to make room for scene 2.
    assert chunks[0].text.endswith(s1)
    assert chunks[1].text.endswith(s2)
    assert oversize == []


def test_prefers_ending_chunk_at_scene_boundary(tok):
    """When a scene does not fit, the chunk ends before the scene starts."""
    small = paragraph_of_tokens(tok, 4000, 3)[0]
    big = paragraph_of_tokens(tok, 6000, 4)[0]
    book = make_book("b1", [("Chapter One", [([], [small]), ([], [big])])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert chunks[0].ends_scene, "chunk should finish exactly on the scene break"


def test_oversize_scene_split_only_at_paragraph_boundaries(tok):
    """A scene bigger than the whole budget is cut only between paragraphs."""
    paras = [paragraph_of_tokens(tok, 4000, i)[0] for i in range(4)]
    book = make_book("b1", [("Chapter One", [([], paras)])])
    chunks, oversize = chunk_book(book, tok, ChunkConfig())

    scenes = [r for r in oversize if r["kind"] == "oversize_scene"]
    assert len(scenes) == 1
    rec = scenes[0]
    assert rec["action"] == "split_at_paragraph_boundaries"
    assert rec["n_tokens"] > BUDGET
    assert rec["chunk_ids"], "the record must name the chunks it became"

    assert_budget(chunks, BUDGET)
    assert_no_duplicates(chunks)

    joined = "\n\n".join(c.text for c in chunks)
    for p in paras:
        assert p in joined


def test_oversize_scene_exclude_policy(tok):
    """--oversize-scene-policy exclude drops the scene but still records it."""
    paras = [paragraph_of_tokens(tok, 4000, i)[0] for i in range(4)]
    book = make_book("b1", [("Chapter One", [([], paras)])])
    chunks, oversize = chunk_book(
        book, tok, ChunkConfig(oversize_scene_policy="exclude")
    )
    rec = next(r for r in oversize if r["kind"] == "oversize_scene")
    assert rec["action"] == "excluded"
    assert chunks == []


# --------------------------------------------------------------------------- #
# Chapter integrity
# --------------------------------------------------------------------------- #


def test_chapter_title_stays_with_its_content(tok):
    """The heading is attached to the first chunk carrying that chapter."""
    body = [paragraph_of_tokens(tok, 3000, i)[0] for i in range(6)]
    book = make_book("b1", [("The Ember Gate", [([], body)])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())

    assert chunks[0].text.startswith("The Ember Gate")
    assert chunks[0].starts_chapter
    assert sum(1 for c in chunks if "The Ember Gate" in c.text) == 1, (
        "the heading must not be duplicated across chunks"
    )


def test_chapter_title_not_repeated_by_default(tok):
    """No duplicated heading text unless explicitly requested."""
    body = [paragraph_of_tokens(tok, 3000, i)[0] for i in range(6)]
    book = make_book("b1", [("Chapter Twelve", [([], body)])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert len(chunks) > 1
    occurrences = sum(c.text.count("Chapter Twelve") for c in chunks)
    assert occurrences == 1


# --------------------------------------------------------------------------- #
# Book / file isolation
# --------------------------------------------------------------------------- #


def test_never_joins_text_from_different_books(tok):
    a = make_book("book_a", [("Ch1", [([], [paragraph_of_tokens(tok, 2000, 1)[0]])])])
    b = make_book("book_b", [("Ch1", [([], [paragraph_of_tokens(tok, 2000, 2)[0]])])])

    chunks_a, _ = chunk_book(a, tok, ChunkConfig())
    chunks_b, _ = chunk_book(b, tok, ChunkConfig())

    assert all(c.book_id == "book_a" for c in chunks_a)
    assert all(c.book_id == "book_b" for c in chunks_b)
    for chunk in chunks_a:
        assert chunk.text not in "\n".join(c.text for c in chunks_b)


def test_chunk_metadata_lists_only_its_own_source_files(tok):
    book = make_book("b1", [("Ch1", [([], [paragraph_of_tokens(tok, 2000, 1)[0]])])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    for chunk in chunks:
        assert chunk.source_files == ["b1.epub"]


# --------------------------------------------------------------------------- #
# Ordering and duplication
# --------------------------------------------------------------------------- #


def test_no_duplicated_prose_across_chunks(tok):
    paras = [paragraph_of_tokens(tok, 3500, i)[0] for i in range(8)]
    book = make_book("b1", [("Ch1", [([], paras[:4]), ([], paras[4:])])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert_no_duplicates(chunks)
    assert len(chunks) > 1


def test_source_order_preserved(tok):
    paras = [paragraph_of_tokens(tok, 3500, i)[0] for i in range(6)]
    book = make_book("b1", [("Ch1", [([], paras)])])
    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert_order(chunks)

    positions = [chunks[0].text.find(paras[0])]
    joined = "\n\n".join(c.text for c in chunks)
    positions = [joined.find(p) for p in paras]
    assert positions == sorted(positions), "paragraphs came out of order"


def test_small_chunk_merge_is_opt_in_and_respects_budget(tok):
    body = [paragraph_of_tokens(tok, 7900, 0)[0], "One short line."]
    book = make_book("b1", [("Ch1", [([], body)])])

    strict, _ = chunk_book(book, tok, ChunkConfig())
    merged, _ = chunk_book(book, tok, ChunkConfig(min_chunk_tokens=400))
    assert_budget(merged, BUDGET)
    assert len(merged) <= len(strict)


# --------------------------------------------------------------------------- #
# Heading behaviour
# --------------------------------------------------------------------------- #


def test_multi_part_headings_are_preserved_in_order(tok):
    from writing_dataset.types import Block, BLOCK_HEADING

    from writing_dataset.structure import blocks_to_book

    blocks = [
        Block(BLOCK_HEADING, "Part One", level=1),
        Block(BLOCK_HEADING, "Chapter One", level=1),
        Block("paragraph", paragraph_of_tokens(tok, 50, 0)[0]),
    ]
    book = blocks_to_book(blocks, "b1", "B1", ["b1.txt"])
    assert book.chapters[0].headings == ["Part One", "Chapter One"]

    chunks, _ = chunk_book(book, tok, ChunkConfig())
    assert chunks[0].text.startswith("Part One\n\nChapter One")


def test_headings_can_be_disabled(tok):
    book = make_book("b1", [("Chapter One", [([], [paragraph_of_tokens(tok, 100, 0)[0]])])])
    chunks, _ = chunk_book(
        book, tok, ChunkConfig(include_chapter_headings=False, include_scene_headings=False)
    )
    assert "Chapter One" not in chunks[0].text
