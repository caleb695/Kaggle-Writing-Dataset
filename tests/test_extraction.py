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
    extract_epub_collection,
    extract_file,
    extract_mobi,
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


def test_soft_wrapped_paragraph_is_not_split_on_newlines():
    """Pretty-printed XHTML wraps lines at ~60 chars; that is not a paragraph."""
    html = (
        "<html><body><p>Did you ever notice the keyhole on the belly of the "
        "unicorn?” Seth asked. He was lying on the floor beside the fanciful "
        "rocking horse, hands\n laced behind his head.</p></body></html>"
    )
    paras = [b.text for b in html_to_blocks(html, "t.xhtml") if b.kind == "paragraph"]
    assert len(paras) == 1
    assert "hands laced" in paras[0]
    assert "\n" not in paras[0]


def test_chapter_class_paragraph_is_a_heading():
    html = (
        "<html><body>"
        '<p class="ChapNo">Chapter 5</p>'
        '<p class="Chapter"> Journal of Secrets </p>'
        "<p>Did you ever notice the keyhole?</p>"
        "</body></html>"
    )
    blocks = html_to_blocks(html, "t.xhtml")
    headings = [b.text for b in blocks if b.kind == "heading"]
    paras = [b.text for b in blocks if b.kind == "paragraph"]
    assert headings == ["Chapter 5", "Journal of Secrets"]
    assert paras == ["Did you ever notice the keyhole?"]


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


# --------------------------------------------------------------------------- #
# KF8 / AZW3 and EPUB omnibus
# --------------------------------------------------------------------------- #


def test_kf8_unpacked_epub_is_read_as_epub(tmp_path, monkeypatch):
    """KindleUnpack returns an EPUB zip for KF8; that zip must be parsed as EPUB."""
    epub_path = tmp_path / "unpacked.epub"
    write_epub(
        epub_path,
        "The False Prince",
        [("Chapter One", ["Sage stole the throne.", "The prince ran."])],
    )

    monkeypatch.setattr("mobi.extract", lambda _p: (str(tmp_path), str(epub_path)))

    from writing_dataset.mobi import MobiHeader
    import writing_dataset.mobi as mobi_mod

    def fake_header(_path):
        header = MobiHeader(
            num_records=2,
            record_offsets=[0, 100],
            compression=2,
            text_length=10,
            record_count=1,
            record_size=4096,
            encryption=0,
        )
        header.mobi_version = 8
        header.full_name = "The False Prince"
        header.exth = {100: "Jennifer A. Nielsen"}
        return header

    monkeypatch.setattr(mobi_mod, "parse_header", fake_header)

    docs = extract_mobi(tmp_path / "dummy.azw3")
    assert len(docs) == 1
    paras = [b.text for b in docs[0].blocks if b.kind == "paragraph"]
    assert "Sage stole the throne." in paras
    assert "The prince ran." in paras
    assert docs[0].source_path == tmp_path / "dummy.azw3"


def _write_isbn_omnibus(path):
    """Two-novel EPUB whose spine files are named with distinct ISBN-13 prefixes."""
    import zipfile

    def chapter(isbn: str, n: int, title: str, body: str) -> tuple[str, str]:
        # Zip path and OPF href: OPF lives at ops/content.opf, so href is relative
        # to that folder (xhtml/...), while the zip name is ops/xhtml/...
        href = f"xhtml/{isbn}_ch{n:02d}.html"
        zip_name = f"ops/{href}"
        html = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            f"<h1>{title}</h1><p>{body}</p></body></html>"
        )
        return zip_name, href, html

    book_a = [chapter("9781111111111", i, f"A{i}", f"Alpha paragraph {i}.") for i in range(1, 5)]
    book_b = [chapter("9782222222222", i, f"B{i}", f"Bravo paragraph {i}.") for i in range(1, 5)]
    files = book_a + book_b

    manifest = "\n".join(
        f'<item id="c{i}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, (_zip_name, href, _html) in enumerate(files)
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(files)))
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Some Trilogy</dc:title>"
        "<dc:language>en</dc:language>"
        "<dc:description>An eBook boxed set of the complete trilogy.</dc:description>"
        "</metadata>"
        f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    ncx_nav = []
    play = 1
    for isbn, title, kids in (
        ("9781111111111", "Alpha Book", 4),
        ("9782222222222", "Bravo Book", 4),
    ):
        inner = "".join(
            f'<navPoint playOrder="{play + j + 1}" id="k{isbn}{j}">'
            f"<navLabel><text>Ch {j}</text></navLabel>"
            f'<content src="xhtml/{isbn}_ch{j + 1:02d}.html"/>'
            "</navPoint>"
            for j in range(kids)
        )
        ncx_nav.append(
            f'<navPoint playOrder="{play}" id="b{isbn}">'
            f"<navLabel><text>{title}</text></navLabel>"
            f'<content src="xhtml/{isbn}_ch01.html"/>'
            f"{inner}</navPoint>"
        )
        play += kids + 1
    ncx = (
        '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">'
        "<docTitle><text>Some Trilogy</text></docTitle>"
        f"<navMap>{''.join(ncx_nav)}</navMap></ncx>"
    )
    container = (
        '<?xml version="1.0"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="ops/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("ops/content.opf", opf)
        zf.writestr("ops/toc.ncx", ncx)
        for zip_name, _href, html in files:
            zf.writestr(zip_name, html)


def test_epub_isbn_omnibus_splits_into_separate_novels(tmp_path):
    path = tmp_path / "boxed.epub"
    _write_isbn_omnibus(path)
    docs = extract_epub_collection(path)
    assert len(docs) == 2
    titles = [d.metadata["title"] for d in docs]
    assert titles == ["Alpha Book", "Bravo Book"]
    a_paras = [b.text for b in docs[0].blocks if b.kind == "paragraph"]
    b_paras = [b.text for b in docs[1].blocks if b.kind == "paragraph"]
    assert all(t.startswith("Alpha") for t in a_paras)
    assert all(t.startswith("Bravo") for t in b_paras)
    assert not any(t.startswith("Bravo") for t in a_paras)

    via_file = extract_file(path)
    assert len(via_file) == 2


def test_epub_omnibus_can_be_left_unsplit(tmp_path):
    path = tmp_path / "boxed.epub"
    _write_isbn_omnibus(path)
    docs = extract_epub_collection(path, split_omnibus=False)
    assert len(docs) == 1
    paras = [b.text for b in docs[0].blocks if b.kind == "paragraph"]
    assert any(t.startswith("Alpha") for t in paras)
    assert any(t.startswith("Bravo") for t in paras)
