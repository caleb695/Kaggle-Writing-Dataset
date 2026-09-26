"""Block list -> Book / Chapter / Scene / Paragraph hierarchy.

The inference here is deliberately conservative. When the source is ambiguous
we prefer to *keep* text and to create fewer, larger structural units, because:

* a spurious scene split only costs a slightly smaller chunk;
* a missed scene split costs nothing at all;
* a wrongly dropped block loses prose permanently.

Heading policy
--------------
A chapter-level heading (``level <= 1``) is appended to the *current* chapter's
``headings`` list when that chapter has no prose yet, and otherwise opens a new
chapter. So ``Part One`` followed by ``Chapter 1`` becomes one chapter whose
headings are ``["Part One", "Chapter 1"]`` -- both strings are preserved
verbatim and in source order, with nothing synthesised or discarded.
Scene-level headings (``level >= 2``) follow the same rule within a scene.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from .extract import ExtractedDoc
from .types import (
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_SCENE_BREAK,
    Block,
    Book,
    Chapter,
    Paragraph,
    Scene,
)


# --------------------------------------------------------------------------- #
# Book-level configuration
# --------------------------------------------------------------------------- #


class BookSpec:
    """Declares which source files make up one book, in reading order.

    Grouping is explicit on purpose: the training rules forbid joining text
    from different books or unrelated source files, so the only way a ``Book``
    can span multiple files is if a manifest says so.
    """

    def __init__(
        self,
        book_id: str,
        title: str,
        sources: Sequence[Path],
        series: Optional[str] = None,
        order_hint: int = 0,
        group_key: Optional[str] = None,
    ) -> None:
        self.book_id = book_id
        self.title = title
        self.sources = list(sources)
        self.series = series
        self.order_hint = order_hint
        #: Folder/manifest group this book came from (used for exclusions + splits).
        self.group_key = group_key if group_key is not None else book_id.split("/")[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"BookSpec({self.book_id!r}, {len(self.sources)} source(s))"


# --------------------------------------------------------------------------- #
# Hierarchy construction
# --------------------------------------------------------------------------- #


def blocks_to_book(
    blocks: Sequence[Block],
    book_id: str,
    book_title: str,
    source_files: Sequence[str],
    series: Optional[str] = None,
    order_hint: int = 0,
    drop_repeated_headings: bool = True,
) -> Book:
    """Fold a flat, source-ordered block list into the chapter/scene hierarchy."""
    book = Book(
        book_id=book_id,
        title=book_title,
        source_files=list(source_files),
        series=series,
        order_hint=order_hint,
    )

    if drop_repeated_headings:
        blocks = _drop_running_headings(blocks, book_title)

    chapters: list[Chapter] = []
    chapter: Optional[Chapter] = None
    scene: Optional[Scene] = None
    para_counter = 0

    def ensure_chapter() -> Chapter:
        nonlocal chapter
        if chapter is None:
            chapter = Chapter(index=len(chapters))
            chapters.append(chapter)
        return chapter

    def ensure_scene() -> Scene:
        nonlocal scene
        ch = ensure_chapter()
        if scene is None:
            scene = Scene(index=len(ch.scenes), origin=blocks[0].origin if blocks else "")
            ch.scenes.append(scene)
        return scene

    for block in blocks:
        if block.kind == BLOCK_HEADING and block.level <= 1:
            if chapter is None or chapter.n_paragraphs > 0:
                chapter = Chapter(index=len(chapters), origin=block.origin)
                chapters.append(chapter)
                scene = None
            chapter.headings.append(block.text)
            continue

        if block.kind == BLOCK_HEADING:  # level >= 2 -> scene heading
            if scene is None or scene.n_paragraphs > 0:
                prev_origin = block.origin
                ch = ensure_chapter()
                scene = Scene(index=len(ch.scenes), origin=prev_origin)
                ch.scenes.append(scene)
            scene.headings.append(block.text)
            continue

        if block.kind == BLOCK_SCENE_BREAK:
            if scene is not None and scene.n_paragraphs > 0:
                scene.break_after = "ornament"
                scene = None
            continue

        if block.kind == BLOCK_PARAGRAPH:
            sc = ensure_scene()
            sc.paragraphs.append(Paragraph(index=para_counter, text=block.text))
            para_counter += 1
            continue

    # Drop content-free scenes and chapters, but never drop a paragraph.
    for ch in chapters:
        ch.scenes = [s for s in ch.scenes if not s.is_empty]
    book.chapters = [c for c in chapters if c.scenes or c.has_title]

    _renumber(book)
    return book


def _renumber(book: Book) -> None:
    """Re-index chapters, scenes and paragraphs to be dense and 0-based."""
    para = 0
    for ci, chapter in enumerate(book.chapters):
        chapter.index = ci
        for si, scene in enumerate(chapter.scenes):
            scene.index = si
            for p in scene.paragraphs:
                p.index = para
                para += 1


def _drop_running_headings(blocks: Sequence[Block], book_title: str) -> list[Block]:
    """Remove page furniture headings without touching real chapter titles.

    An earlier version dropped any heading seen four or more times that was also
    short. That is wrong for books whose chapters are named by a repeating
    label -- e.g. a multi-POV novel whose chapters are titled "JASON", "PIPER",
    "LEO" dozens of times -- so those titles were being deleted along with the
    genuine running heads.

    The rule is now explicit: drop a heading only when it matches the book title
    or appears in the known furniture list. Real titles survive, even repeated
    ones.
    """
    from .extract import FURNITURE_HEADINGS

    title_norm = _norm(book_title)
    drop: set[str] = set()

    for block in blocks:
        if block.kind != BLOCK_HEADING:
            continue
        text = block.text.strip()
        if not text:
            continue
        normalised = _norm(text)
        if normalised == title_norm or normalised in FURNITURE_HEADINGS:
            drop.add(text)

    if not drop:
        return list(blocks)
    return [b for b in blocks if not (b.kind == BLOCK_HEADING and b.text.strip() in drop)]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# --------------------------------------------------------------------------- #
# Book discovery
# --------------------------------------------------------------------------- #


def slugify(blob: str) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", blob).strip("_").lower()
    return re.sub(r"_{2,}", "_", out)


def folder_to_book_id(rel_dir: str, stem: str) -> str:
    """Stable, filesystem-safe ``book_id`` from a relative folder + filename."""
    return slugify(f"{rel_dir}/{stem}" if rel_dir else stem)


def discover_books(raw_root: Path) -> list[BookSpec]:
    """One book per source file, annotated with its top-level folder.

    Default (no-manifest) behaviour. Because the hard rule is "never join text
    from different books", the conservative mapping is *one file == one book*.
    Use :func:`load_manifest` to declare genuine multi-file books.
    """
    from .extract import SUPPORTED_EXTENSIONS, discover_sources

    files = discover_sources(raw_root, SUPPORTED_EXTENSIONS, recursive=True)
    specs: list[BookSpec] = []
    for order, path in enumerate(files):
        rel = path.relative_to(raw_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "_root"
        rel_dir = str(rel.parent) if len(rel.parts) > 1 else ""
        specs.append(
            BookSpec(
                book_id=folder_to_book_id(rel_dir, path.stem),
                title=path.stem,
                sources=[path],
                series=group,
                order_hint=order,
                group_key=group,
            )
        )
    return specs


def load_manifest(path: Path, raw_root: Path) -> list[BookSpec]:
    """Load an explicit book-grouping manifest (YAML).

    Shape::

        books:
          - id: fablehaven_01
            title: Fablehaven
            series: Fablehaven and Dragonwatch
            sources:
              - Fablehaven and dragonwatch/Fablehaven-01.epub
              - Fablehaven and dragonwatch/Fablehaven-01-notes.txt
    """
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("books") or []
    specs: list[BookSpec] = []
    for i, entry in enumerate(entries):
        if "id" not in entry or "sources" not in entry:
            raise ValueError(f"manifest {path.name}: every entry needs 'id' and 'sources'")
        srcs = [raw_root / s for s in entry["sources"]]
        missing = [str(s) for s in srcs if not s.is_file()]
        if missing:
            raise FileNotFoundError(
                f"manifest {path.name}: book {entry['id']!r} references missing file(s): {missing}"
            )
        specs.append(
            BookSpec(
                book_id=entry["id"],
                title=entry.get("title") or entry["id"],
                sources=srcs,
                series=entry.get("series"),
                order_hint=int(entry.get("order", i)),
                group_key=entry.get("group") or entry.get("series") or entry["id"],
            )
        )
    return specs


def build_book_from_docs(
    docs: Sequence[ExtractedDoc],
    spec: BookSpec,
    drop_repeated_headings: bool = True,
    source_labels: Optional[Sequence[str]] = None,
) -> Book:
    """Concatenate the blocks of one book's sources, in declared order, then structure.

    ``source_labels`` overrides the recorded filenames, which lets callers store
    paths relative to the raw root instead of absolute ones. Relative labels keep
    the dataset portable and avoid leaking a local directory layout.
    """
    blocks: list[Block] = []
    for doc in docs:
        blocks.extend(doc.blocks)

    title = spec.title
    for doc in docs:
        candidate = (doc.metadata or {}).get("title")
        if candidate and len(docs) == 1:
            title = candidate

    labels = list(source_labels) if source_labels is not None else [str(s) for s in spec.sources]

    return blocks_to_book(
        blocks,
        book_id=spec.book_id,
        book_title=title,
        source_files=labels,
        series=spec.series,
        order_hint=spec.order_hint,
        drop_repeated_headings=drop_repeated_headings,
    )
