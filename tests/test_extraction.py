"""Regression tests for the extractors.

Two bugs these tests exist to prevent from ever coming back:

1. The scene-break character class was built by string interpolation, so the
   bare ``-`` inside it defined the range U+003D..U+2013 -- every letter and
   digit. Ordinary sentences were being classified as ornaments.
2. Leaf-block text was assembled from raw inner HTML, so inline markup such as
   ``<em>`` leaked straight into the corpus.
"""

from __future__ import annotations

import pytest

from fixtures import make_corpus, write_epub

from writing_dataset.extract import (
    extract_epub,
    extract_text,
    html_to_blocks,
    is_scene_break_line,
)


# --------------------------------------------------------------------------- #
# 1. Scene-break detection must not fire on prose
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence",
    [
        "The gate opened at dusk.",
        "Nobody was watching.",
        "She counted to ten.",
        "He said nothing.",
        "Yes.",
        "No!",
        "OK?",
        "A lantern in the hollow.",
        "Go on.",
    ],
)
def test_ordinary_sentence_is_not_a_scene_break(sentence):
    assert not is_scene_break_line(sentence)


@pytest.mark.parametrize("ornament", ["* * *", "***", "~~~", "###", "- - -", "\u2042", "****"])
def test_real_ornaments_are_scene_breaks(ornament):
    assert is_scene_break_line(ornament)


def test_lone_dash_is_not_aggressive_by_default_but_can_be():
    assert not is_scene_break_line("-")
    assert not is_scene_break_line("--")
    assert is_scene_break_line("---", aggressive=True)


# --------------------------------------------------------------------------- #
# 2. Inline markup must not leak into the prose
# --------------------------------------------------------------------------- #


def test_inline_tags_do_not_leak_into_text():
    html = (
        "<html><body>"
        "<p>The <em>dragon</em> <strong>roared</strong> once.</p>"
        "<p>Then <span class='x'>silence</span>.</p>"
        "</body></html>"
    )
    blocks = html_to_blocks(html, "t.xhtml")
    texts = [b.text for b in blocks]
    assert "The dragon roared once." in texts
    assert "Then silence." in texts
    for t in texts:
        assert "<" not in t and ">" not in t, f"markup leaked: {t!r}"


def test_br_splits_a_leaf_block_into_separate_paragraphs():
    html = "<html><body><div>First line.<br/>Second line.</div></body></html>"
    texts = [b.text for b in html_to_blocks(html, "t.xhtml")]
    assert "First line." in texts
    assert "Second line." in texts


def test_hr_and_dinkus_paragraph_become_scene_breaks():
    html = (
        "<html><body><p>Before.</p><hr/><p>* * *</p><p>After.</p></body></html>"
    )
    kinds = [b.kind for b in html_to_blocks(html, "t.xhtml")]
    assert kinds.count("scene_break") == 2
    assert kinds.count("paragraph") == 2


# --------------------------------------------------------------------------- #
# EPUB reading
# --------------------------------------------------------------------------- #


def test_epub_with_empty_nav_parses(tmp_path):
    """An empty nav document used to crash ebooklib; it must not crash us."""
    path = tmp_path / "empty_nav.epub"
    write_epub(
        path,
        "Empty Nav",
        [("Chapter One", ["The gate opened at dusk.", "Nobody was watching."])],
        empty_nav=True,
    )
    doc = extract_epub(path)
    assert doc.n_paragraphs == 2
    assert doc.metadata["title"] == "Empty Nav"
    assert any(b.text == "Chapter One" and b.kind == "heading" for b in doc.blocks)


def test_epub_spine_order_is_respected(tmp_path):
    path = tmp_path / "order.epub"
    write_epub(
        path,
        "Order",
        [("One", ["alpha."]), ("Two", ["bravo."]), ("Three", ["charlie."])],
    )
    texts = [b.text for b in extract_epub(path).blocks if b.kind == "paragraph"]
    assert texts == ["alpha.", "bravo.", "charlie."]


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #


def test_text_scene_break_and_headings(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "Chapter One\n\n"
        "A lantern in the hollow. It burned low.\n\n"
        "* * *\n\n"
        "Morning came late that year.\n\n"
        "Chapter Two\n\n"
        "The river ran backwards.\n",
        encoding="utf-8",
    )
    doc = extract_text(path)
    kinds = [b.kind for b in doc.blocks]
    assert kinds.count("heading") == 2
    assert kinds.count("scene_break") == 1
    assert kinds.count("paragraph") == 3

    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert headings == ["Chapter One", "Chapter Two"]


def test_text_dehyphenation_rejoins_wrapped_words(tmp_path):
    path = tmp_path / "wrap.txt"
    path.write_text("A won-\nderful thing happened.\n", encoding="utf-8")
    doc = extract_text(path)
    assert doc.blocks[0].text == "A wonderful thing happened."


def test_corpus_excludes_only_the_requested_folders(tmp_path):
    info = make_corpus(tmp_path / "raw")
    from writing_dataset.build import DEFAULT_EXCLUDE_PATTERNS, is_excluded
    from writing_dataset.structure import discover_books

    specs = discover_books(info["root"])
    excluded = {s.book_id for s in specs if is_excluded(s, DEFAULT_EXCLUDE_PATTERNS)[0]}
    kept = {s.book_id for s in specs if not is_excluded(s, DEFAULT_EXCLUDE_PATTERNS)[0]}

    assert not any("wings" in b for b in kept)
    assert not any("onyx" in b for b in kept)
    assert any("wings" in b for b in excluded)
    assert any("onyx" in b for b in excluded)
    assert len(kept) == 2
