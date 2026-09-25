"""Helpers for building synthetic books and a small on-disk corpus."""

from __future__ import annotations

import random
import zlib
from pathlib import Path
from typing import Iterable, Optional, Sequence

from writing_dataset.structure import _renumber
from writing_dataset.tokenizer import TokenCounter, load_tokenizer
from writing_dataset.types import Book, Chapter, Paragraph, Scene

_WORDS = (
    "dragon wing scale fire shadow mountain river storm silver ember ash "
    "tower gate river stone banner hollow crown raven frost lantern tide "
    "harbor meadow thorn cinder willow arrow compass thornlight cave sunglass "
    "keeper watcher wanderer the and of in a to it was with that his her"
).split()


def paragraph_text(n_words: int, seed: int = 0) -> str:
    rnd = random.Random(seed)
    words = [rnd.choice(_WORDS) for _ in range(n_words)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def paragraph_of_tokens(
    counter: TokenCounter, approx_tokens: int, seed: int = 0
) -> tuple[str, int]:
    """Return ``(text, actual_token_count)`` sized near ``approx_tokens``.

    Random dictionary words average ~1.25 tokens each, so we start there and
    report the measured count so tests can assert on facts rather than guesses.
    """
    n_words = max(1, int(approx_tokens / 1.25))
    text = paragraph_text(n_words, seed)
    actual = counter.count(text)

    # Nudge toward the target with a few adjustment rounds.
    for _ in range(12):
        if actual >= approx_tokens:
            break
        text = paragraph_text(n_words, seed) + " " + paragraph_text(10, seed + 1)
        n_words += 10
        text = paragraph_text(n_words, seed)
        actual = counter.count(text)
    return text, actual


def make_book(
    book_id: str,
    spec: Sequence[tuple[Optional[str], Sequence[tuple[Sequence[str], Sequence[str]]]]],
    series: Optional[str] = None,
) -> Book:
    """Build a :class:`Book` from a compact spec.

    ``spec`` is a list of chapters: ``(chapter_title, scenes)`` where each scene
    is ``(scene_headings, [paragraph_text, ...])``.
    """
    book = Book(
        book_id=book_id,
        title=book_id,
        source_files=[f"{book_id}.epub"],
        series=series,
    )
    for chapter_title, scenes in spec:
        chapter = Chapter(index=len(book.chapters))
        if chapter_title:
            chapter.headings.append(chapter_title)
        for headings, paragraphs in scenes:
            scene = Scene(index=len(chapter.scenes))
            scene.headings.extend(headings)
            for text in paragraphs:
                scene.paragraphs.append(Paragraph(index=0, text=text))
            chapter.scenes.append(scene)
        book.chapters.append(chapter)
    _renumber(book)
    return book


def counter() -> TokenCounter:
    return load_tokenizer()


# --------------------------------------------------------------------------- #
# On-disk corpus
# --------------------------------------------------------------------------- #


def write_epub(
    path: Path,
    title: str,
    chapters: Sequence[tuple[str, Sequence[str]]],
    empty_nav: bool = False,
) -> None:
    """Write a minimal, spec-shaped EPUB by hand.

    Built with :mod:`zipfile` rather than a library so the test exercises our own
    reader directly. ``empty_nav=True`` reproduces the real-world case that made
    ``ebooklib`` raise ``lxml.etree.ParserError`` during ``read_epub``.
    """
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)

    chap_files = []
    for i, (chapter_title, paragraphs) in enumerate(chapters):
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        name = f"chap_{i:02d}.xhtml"
        chap_files.append((name, chapter_title))
        xhtml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            f"<title>{chapter_title}</title></head><body>"
            f"<h2>{chapter_title}</h2>{body}</body></html>"
        )
        (path.parent / name).write_text(xhtml, encoding="utf-8")

    nav_body = "" if empty_nav else "".join(
        f'<li><a href="{n}">{t}</a></li>' for n, t in chap_files
    )
    nav = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head>'
        '<body><nav epub:type="toc" id="toc"><ol>'
        f"{nav_body}</ol></nav></body></html>"
    )

    manifest = "\n".join(
        f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>'
        for i, (n, _) in enumerate(chap_files)
    )
    spine = "\n".join(f'<itemref idref="c{i}"/>' for i in range(len(chap_files)))
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:title>{title}</dc:title><dc:language>en</dc:language>'
        f'<dc:identifier id="uid">urn:uuid:{zlib.crc32(title.encode()):08x}</dc:identifier>'
        "</metadata>"
        f'<manifest><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
        f'properties="nav"/>\n{manifest}</manifest>'
        f"<spine>{spine}</spine></package>"
    )

    container = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )

    with zipfile.ZipFile(path, "w") as zf:
        # 'mimetype' must be first and stored uncompressed.
        zf.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip", zipfile.ZIP_STORED
        )
        zf.writestr("META-INF/container.xml", container, zipfile.ZIP_DEFLATED)
        zf.writestr("content.opf", opf, zipfile.ZIP_DEFLATED)
        zf.writestr("nav.xhtml", nav, zipfile.ZIP_DEFLATED)
        for name, _ in chap_files:
            zf.write(path.parent / name, name)
            (path.parent / name).unlink()


def write_txt(path: Path, chapters: Sequence[tuple[str, Sequence[str]]]) -> None:
    """Write a plain-text manuscript with 'Chapter N' headings and scene breaks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for chapter_title, blocks in chapters:
        lines += [chapter_title, ""]
        for block in blocks:
            if block == "<scene-break>":
                lines += ["* * *", ""]
            else:
                lines += [block, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_corpus(root: Path) -> dict:
    """Create a small corpus that exercises exclusions, multi-book splits and txt/epub."""
    kept_dir = root / "Test Series"
    wof_dir = root / "Wings of Fire Arc 1"
    onyx_dir = root / "Onyx"

    write_epub(
        kept_dir / "Book One.epub",
        "Book One",
        [
            ("Chapter One", ["The gate opened at dusk.", "Nobody was watching."]),
            ("Chapter Two", ["Smoke rose beyond the ridge.", "She counted to ten."]),
        ],
    )

    write_txt(
        kept_dir / "Book Two.txt",
        [
            ("Chapter 1", ["A lantern in the hollow.", "<scene-break>", "Morning came late."]),
            ("Chapter 2", ["The river ran backwards.", "He said nothing."]),
        ],
    )

    write_epub(
        wof_dir / "Should Be Excluded.epub",
        "Should Be Excluded",
        [("Chapter One", ["This text must never reach the dataset."])],
    )

    write_epub(
        onyx_dir / "Also Excluded.epub",
        "Also Excluded",
        [("Chapter One", ["This text must also never reach the dataset."])],
    )

    return {
        "root": root,
        "kept": [kept_dir / "Book One.epub", kept_dir / "Book Two.txt"],
        "excluded": [wof_dir / "Should Be Excluded.epub", onyx_dir / "Also Excluded.epub"],
    }
