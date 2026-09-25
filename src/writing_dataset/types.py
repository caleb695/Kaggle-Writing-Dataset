"""Core data model for the book -> chapter -> scene -> paragraph pipeline.

The hierarchy is deliberately explicit because the chunking rules are stated in
terms of it:

    Book > Chapter > Scene > Paragraph

Structural invariants the rest of the package relies on:

* ``Book.book_id`` is unique and stable (derived from the source path + title).
* ``Book.source_files`` lists every file that contributed text. A ``Book`` is
  never assembled from files that were not explicitly declared to belong to it.
* Paragraph order inside a scene, scene order inside a chapter, and chapter
  order inside a book are all source order.
* ``Scene.break_after`` records why the scene ended, so scene boundaries stay
  auditable.

On headings
-----------
Books routinely stack several chapter-level headings before any prose --
``Part One`` then ``Chapter 1`` -- and sub-headings can stack too. Storing a
single ``title`` string would force a lossy choice, so chapters and scenes hold
an ordered ``headings`` list instead. ``title`` is a derived convenience
property; ``title_block`` is what the chunker actually emits, and it is always
verbatim source text joined by a blank line -- never synthesised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Raw extraction layer
# --------------------------------------------------------------------------- #

BLOCK_HEADING = "heading"
BLOCK_SCENE_BREAK = "scene_break"
BLOCK_PARAGRAPH = "paragraph"


@dataclass
class Block:
    """One structural unit pulled out of a source file, in source order.

    Attributes
    ----------
    kind:
        One of :data:`BLOCK_HEADING`, :data:`BLOCK_SCENE_BREAK`,
        :data:`BLOCK_PARAGRAPH`.
    text:
        Cleaned text. Empty for scene breaks (the break itself is structural).
    level:
        Heading depth for ``BLOCK_HEADING`` (1 = chapter-level, 2 = scene-level);
        ``0`` for everything else.
    origin:
        ``epub:chapter3.xhtml`` style locator, for debugging and provenance.
    """

    kind: str
    text: str = ""
    level: int = 0
    origin: str = ""


# --------------------------------------------------------------------------- #
# Structured layer
# --------------------------------------------------------------------------- #


@dataclass
class Paragraph:
    """An atomic unit. The chunker never splits one of these."""

    index: int
    text: str
    #: Token count of ``text`` alone, filled in by the chunker.
    n_tokens: int = 0
    #: Token count including the separator that precedes it inside a chunk.
    n_tokens_with_sep: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Paragraph.text must be str")


@dataclass
class Scene:
    """A run of paragraphs between two scene breaks."""

    index: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    #: Ordered sub-headings that introduce this scene (verbatim source text).
    headings: list[str] = field(default_factory=list)
    #: Ornament/heading that separated this scene from the next.
    break_after: Optional[str] = None
    origin: str = ""

    @property
    def title(self) -> Optional[str]:
        return self.headings[0] if self.headings else None

    @property
    def title_block(self) -> Optional[str]:
        """Verbatim header text for this scene, or ``None``."""
        return "\n\n".join(self.headings) if self.headings else None

    @property
    def n_tokens(self) -> int:
        return sum(p.n_tokens for p in self.paragraphs)

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def is_empty(self) -> bool:
        return not self.paragraphs and not self.headings


@dataclass
class Chapter:
    """A run of scenes under one or more chapter-level headings."""

    index: int
    #: Ordered chapter-level headings (verbatim source text).
    headings: list[str] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    origin: str = ""

    @property
    def title(self) -> Optional[str]:
        return self.headings[0] if self.headings else None

    @property
    def title_block(self) -> Optional[str]:
        """Verbatim header text for this chapter, or ``None``."""
        return "\n\n".join(self.headings) if self.headings else None

    @property
    def n_tokens(self) -> int:
        return sum(s.n_tokens for s in self.scenes)

    @property
    def n_paragraphs(self) -> int:
        return sum(s.n_paragraphs for s in self.scenes)

    @property
    def n_scenes(self) -> int:
        return len(self.scenes)

    @property
    def has_title(self) -> bool:
        return bool(self.headings)


@dataclass
class Book:
    """One work. Chunks are never packed across two ``Book`` objects."""

    book_id: str
    title: str
    source_files: list[str] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    #: Free-form provenance (series, arc, folder) carried into metadata.
    series: Optional[str] = None
    order_hint: int = 0

    @property
    def n_chapters(self) -> int:
        return len(self.chapters)

    @property
    def n_scenes(self) -> int:
        return sum(c.n_scenes for c in self.chapters)

    @property
    def n_paragraphs(self) -> int:
        return sum(c.n_paragraphs for c in self.chapters)

    @property
    def n_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chapters)


@dataclass
class Chunk:
    """A finished training chunk.

    ``text`` is exactly what gets tokenised, prefixed by ``[BOS]`` and suffixed
    by ``[EOS]`` at training time (the two reserved budget positions).
    """

    chunk_id: str
    book_id: str
    book_title: str
    text: str
    n_tokens: int

    source_files: list[str]
    chapter_index: int
    #: Last chapter this chunk touches. Differs from ``chapter_index`` when whole
    #: chapters were packed together, which the rules allow.
    chapter_index_end: int
    chapter_title: Optional[str]
    scene_index_start: int
    scene_index_end: int
    paragraph_index_start: int
    paragraph_index_end: int
    n_paragraphs: int = 0
    #: Paragraph indices inside ``[start, end]`` that this chunk does NOT carry.
    #: Non-empty only when an oversize paragraph was skipped mid-range.
    paragraph_index_gaps: list[int] = field(default_factory=list)

    #: ``True`` when the chunk opens a chapter (its heading is attached).
    starts_chapter: bool = False
    #: ``True`` when the chunk opens a new scene.
    starts_scene: bool = False
    #: ``True`` when the chunk ends exactly on a scene boundary.
    ends_scene: bool = False
    #: ``True`` when the chunk resumes a scene that overflowed the budget.
    continues_scene: bool = False
    #: ``True`` when the chunk covers several whole scenes packed together.
    spans_multiple_scenes: bool = False
    #: Position of this chunk within its book (0-based, source order).
    order_in_book: int = 0

    def as_row(self) -> dict:
        """Serialise to a JSON-lines friendly dict (schema documented in README)."""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "source_files": list(self.source_files),
            "chapter_index": self.chapter_index,
            "chapter_index_end": self.chapter_index_end,
            "chapter_title": self.chapter_title,
            "scene_index_start": self.scene_index_start,
            "scene_index_end": self.scene_index_end,
            "paragraph_index_start": self.paragraph_index_start,
            "paragraph_index_end": self.paragraph_index_end,
            "n_paragraphs": self.n_paragraphs,
            "paragraph_index_gaps": list(self.paragraph_index_gaps),
            "order_in_book": self.order_in_book,
            "flags": {
                "starts_chapter": self.starts_chapter,
                "starts_scene": self.starts_scene,
                "ends_scene": self.ends_scene,
                "continues_scene": self.continues_scene,
                "spans_multiple_scenes": self.spans_multiple_scenes,
            },
        }
