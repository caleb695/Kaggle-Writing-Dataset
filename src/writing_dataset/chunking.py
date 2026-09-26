"""Boundary-respecting greedy chunker.

This module implements the training-spec rules literally. Each rule, and where
it is enforced:

==================================================================  ==========================================
Rule                                                                 Enforcement
==================================================================  ==========================================
8192 total tokens, 2 reserved for boundary tokens                    ``ChunkConfig.content_budget`` = 8190
Chunk on chapter / scene / paragraph boundaries                      ``_BookState._place_scene``
Never split a paragraph to fill a chunk                              a ``Piece`` is a whole paragraph
Never split a scene to fill a chunk                                  a scene that does not fit opens a
                                                                     new chunk instead of being cut
Chapter title stays with its content                                 ``_open_if_needed`` attaches headings to
                                                                     the first chunk carrying the chapter
Never join text from different books                                 one call per :class:`Book`; no state
                                                                     crosses books
Never join unrelated source files                                    a ``Book`` only spans files a manifest
                                                                     explicitly groups, in declared order
Prefer ending a chunk before a scene boundary                        a non-fitting scene triggers a flush
Preserve source order within each book                               pieces are appended forward only
No overlapping chunks that duplicate prose                           each paragraph index is emitted once;
                                                                     see :func:`assert_no_duplicates`
No synthetic text between chunks                                     chunk text is verbatim pieces joined by
                                                                     a separator; no markers are inserted
Oversize paragraph/scene -> record, never split silently             ``oversize_blocks.jsonl``
==================================================================  ==========================================

Exactness
---------
Packing uses cached per-piece token counts for speed, then every candidate
chunk is re-tokenised before it is emitted (``_shrink_to_fit``). If the exact
encode disagrees with the arithmetic, only whole trailing paragraphs are
deferred -- they return to the buffer and land in the next chunk, so no prose is
ever silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from .tokenizer import MAX_SEQ_LEN, RESERVED_BOUNDARY_TOKENS, TokenCounter
from .types import Book, Chapter, Chunk, Scene

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OVERSIZE_SCENE_SPLIT = "split"      # split at paragraph boundaries, and log it
OVERSIZE_SCENE_EXCLUDE = "exclude"  # drop the whole scene, and log it

OversizeHook = Callable[[dict], None]

CHAPTER_HEADING = "chapter_heading"
SCENE_HEADING = "scene_heading"
PARAGRAPH = "paragraph"


@dataclass
class ChunkConfig:
    """Everything the chunker needs to know."""

    max_seq_len: int = MAX_SEQ_LEN
    reserved_boundary_tokens: int = RESERVED_BOUNDARY_TOKENS

    #: Inserted between blocks inside a chunk. A formatting character only --
    #: never a marker or any other synthetic text.
    block_separator: str = "\n\n"

    #: Emit the chapter heading block at the head of a chapter's first chunk.
    include_chapter_headings: bool = True
    #: Emit a scene sub-heading when the source had one.
    include_scene_headings: bool = True

    #: Re-emit the chapter heading at the top of continuation chunks. OFF by
    #: default: it duplicates heading text into the corpus.
    repeat_chapter_title: bool = False

    #: Merge a chunk smaller than this into the previous chunk when the combined
    #: text still fits. 0 disables the merge pass entirely.
    min_chunk_tokens: int = 0
    #: Allow that merge pass to cross a chapter boundary.
    merge_across_chapters: bool = False

    #: What to do when a *scene* is larger than the entire content budget.
    #: ``split`` cuts at paragraph boundaries -- the only legal cut points --
    #: and records the event. ``exclude`` drops the scene entirely.
    oversize_scene_policy: str = OVERSIZE_SCENE_SPLIT

    def __post_init__(self) -> None:
        if self.reserved_boundary_tokens < 0:
            raise ValueError("reserved_boundary_tokens must be >= 0")
        if self.max_seq_len <= self.reserved_boundary_tokens:
            raise ValueError("max_seq_len must exceed reserved_boundary_tokens")
        if self.oversize_scene_policy not in (OVERSIZE_SCENE_SPLIT, OVERSIZE_SCENE_EXCLUDE):
            raise ValueError(f"unknown oversize_scene_policy {self.oversize_scene_policy!r}")

    @property
    def content_budget(self) -> int:
        """Usable tokens per chunk: 8192 - 2 = 8190."""
        return self.max_seq_len - self.reserved_boundary_tokens


# --------------------------------------------------------------------------- #
# Internal piece model
# --------------------------------------------------------------------------- #


@dataclass
class Piece:
    """The smallest thing the chunker will place. A paragraph is never cut."""

    kind: str
    text: str
    n_tokens: int
    scene_index: int
    para_index: Optional[int] = None
    chapter_index: Optional[int] = None


class _Buffer:
    """Accumulates pieces and tracks the running token cost."""

    __slots__ = ("pieces", "tokens", "sep_tokens")

    def __init__(self, sep_tokens: int) -> None:
        self.pieces: list[Piece] = []
        self.tokens = 0
        self.sep_tokens = sep_tokens

    def fits(self, n_tokens: int, budget: int) -> bool:
        extra = n_tokens + (self.sep_tokens if self.pieces else 0)
        return self.tokens + extra <= budget

    def add(self, piece: Piece) -> None:
        if self.pieces:
            self.tokens += self.sep_tokens
        self.pieces.append(piece)
        self.tokens += piece.n_tokens

    def extend(self, pieces: Iterable[Piece]) -> None:
        for p in pieces:
            self.add(p)

    @property
    def is_empty(self) -> bool:
        return not self.pieces

    def render(self, sep: str) -> str:
        return sep.join(p.text for p in self.pieces)

    def clear(self) -> None:
        self.pieces = []
        self.tokens = 0


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def chunk_book(
    book: Book,
    counter: TokenCounter,
    cfg: Optional[ChunkConfig] = None,
    oversize_hook: Optional[OversizeHook] = None,
) -> tuple[list[Chunk], list[dict]]:
    """Chunk one book. Returns ``(chunks, oversize_records)``.

    Called once per book and keeps no state between books, which is what makes
    "never join text from different books" structural rather than something a
    caller has to remember.
    """
    cfg = cfg or ChunkConfig()
    sep = cfg.block_separator
    sep_tokens = counter.count(sep)

    chunks: list[Chunk] = []
    oversize: list[dict] = []

    def record_oversize(rec: dict) -> None:
        oversize.append(rec)
        if oversize_hook is not None:
            oversize_hook(rec)

    state = _BookState(book, counter, cfg, chunks, record_oversize, sep, sep_tokens)

    for chapter in book.chapters:
        state.run_chapter(chapter)

    # Drain: normally one flush empties the buffer, but a deferred tail from
    # exact verification can survive one round. Bounded so a pathological input
    # can never spin.
    for _ in range(8):
        if state.buf.is_empty:
            break
        before = len(chunks)
        state.flush()
        if len(chunks) == before and state.buf.is_empty is False:
            break
    if not state.buf.is_empty:
        state.record_leftover()

    _assign_order(chunks)
    mark_scene_boundaries(chunks, book)

    if cfg.min_chunk_tokens > 0:
        chunks = _merge_small_chunks(chunks, counter, cfg, cfg.content_budget)

    return chunks, oversize


def _assign_order(chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks):
        c.order_in_book = i


# --------------------------------------------------------------------------- #
# Per-book state machine
# --------------------------------------------------------------------------- #


class _BookState:
    """Holds the running buffer while walking one book."""

    def __init__(
        self,
        book: Book,
        counter: TokenCounter,
        cfg: ChunkConfig,
        out: list[Chunk],
        record_oversize: Callable[[dict], None],
        sep: str,
        sep_tokens: int,
        budget: Optional[int] = None,
    ) -> None:
        self.book = book
        self.counter = counter
        self.cfg = cfg
        self.out = out
        self.record_oversize = record_oversize
        self.sep = sep
        self.sep_tokens = sep_tokens
        self.budget = budget if budget is not None else cfg.content_budget

        self.buf = _Buffer(sep_tokens)
        self.chapter: Optional[Chapter] = None
        self.chapter_opened = False
        self.chunk_seq = 0
        #: Paragraph indices already emitted, for the duplicate-prose check.
        self.seen_paragraphs: set[int] = set()

    # -- chapter headings -------------------------------------------------- #

    def _heading_texts(self) -> list[str]:
        if self.chapter is None:
            return []
        return [h for h in self.chapter.headings if h and h.strip()]

    def _chapter_overhead(self) -> int:
        """Tokens the chapter heading block would add to the current buffer."""
        return self._pieces_cost(self._pending_heading_pieces(), self.sep_tokens)

    def _pending_heading_pieces(self) -> list[Piece]:
        """Chapter heading pieces not yet attached to a chunk (empty if already done)."""
        if self.chapter_opened or self.chapter is None:
            return []
        if not self.cfg.include_chapter_headings:
            return []
        chapter = self.chapter.index if self.chapter else None
        return [
            Piece(CHAPTER_HEADING, text, self.counter.count(text), scene_index=-1,
                  chapter_index=chapter)
            for text in self._heading_texts()
        ]

    def _commit_heading(self) -> None:
        """Attach this chapter's heading block to the buffer, then mark it opened.

        Called immediately before the chapter's first content is added, so the
        heading and that content always land in the same chunk.
        """
        if self.chapter_opened or self.chapter is None:
            return
        for piece in self._pending_heading_pieces():
            if not self.buf.fits(piece.n_tokens, self.budget):
                # A heading too long for the budget on its own: record it rather
                # than dropping it silently.
                self.record_oversize(
                    {
                        "kind": "oversize_heading",
                        "book_id": self.book.book_id,
                        "book_title": self.book.title,
                        "source": (self.book.source_files or [None])[0],
                        "chapter": self.chapter.index,
                        "chapter_title": self.chapter.title,
                        "scene": None,
                        "paragraph": None,
                        "n_tokens": piece.n_tokens,
                        "budget": self.budget,
                        "reason": "chapter heading alone exceeds the usable content budget",
                        "action": "excluded",
                    }
                )
                continue
            self.buf.add(piece)
        self.chapter_opened = True

    # -- chunk lifecycle --------------------------------------------------- #

    def flush(self) -> Optional[Chunk]:
        """Finalise the current buffer into a :class:`Chunk`.

        Any trailing paragraphs that exact verification says would overflow are
        *deferred* back into the buffer -- they will be emitted in the next
        chunk instead of being lost.
        """
        if self.buf.is_empty:
            return None

        candidates = list(self.buf.pieces)
        text, kept = self._shrink_to_fit(candidates)
        if not kept:
            self.record_leftover()
            return None

        # A chunk made only of headings would strand a chapter title away from
        # its content. Defer everything and let the next flush try again.
        if not any(p.kind == PARAGRAPH for p in kept):
            return None

        deferred = candidates[len(kept):]

        para_indices = [p.para_index for p in kept if p.para_index is not None]
        scene_indices = [p.scene_index for p in kept if p.scene_index >= 0]
        chapter_indices = [p.chapter_index for p in kept if p.chapter_index is not None]
        paras = [p for p in kept if p.kind == PARAGRAPH]

        for p in paras:
            if p.para_index in self.seen_paragraphs:
                raise AssertionError(
                    f"paragraph {p.para_index} emitted twice in {self.book.book_id}"
                )
            self.seen_paragraphs.add(p.para_index)

        first = kept[0]
        chunk = Chunk(
            chunk_id=f"{self.book.book_id}::{self.chunk_seq:05d}",
            book_id=self.book.book_id,
            book_title=self.book.title,
            text=text,
            n_tokens=self.counter.count(text),
            source_files=list(self.book.source_files),
            chapter_index=min(chapter_indices) if chapter_indices
            else (self.chapter.index if self.chapter else 0),
            chapter_index_end=max(chapter_indices) if chapter_indices
            else (self.chapter.index if self.chapter else 0),
            chapter_title=self.chapter.title if self.chapter else None,
            scene_index_start=min(scene_indices) if scene_indices else -1,
            scene_index_end=max(scene_indices) if scene_indices else -1,
            paragraph_index_start=min(para_indices) if para_indices else -1,
            paragraph_index_end=max(para_indices) if para_indices else -1,
            n_paragraphs=len(paras),
            starts_chapter=first.kind == CHAPTER_HEADING,
            starts_scene=first.kind == SCENE_HEADING,
            ends_scene=False,  # set by mark_scene_boundaries
            continues_scene=False,
            spans_multiple_scenes=len(set(scene_indices)) > 1,
        )
        chunk.paragraph_index_gaps = _gaps(para_indices)

        self.chunk_seq += 1
        self.out.append(chunk)

        self.buf.clear()
        self.buf.extend(deferred)
        return chunk

    def record_leftover(self) -> None:
        """Log whatever is still buffered and drop it, so nothing vanishes quietly."""
        if self.buf.is_empty:
            return
        leftover = list(self.buf.pieces)
        n = self.buf.tokens
        self.record_oversize(
            {
                "kind": "unplaceable_tail",
                "book_id": self.book.book_id,
                "book_title": self.book.title,
                "source": (self.book.source_files or [None])[0],
                "chapter": self.chapter.index if self.chapter else None,
                "chapter_title": self.chapter.title if self.chapter else None,
                "scene": leftover[0].scene_index if leftover[0].scene_index >= 0 else None,
                "paragraph": next((p.para_index for p in leftover if p.para_index is not None), None),
                "n_tokens": n,
                "budget": self.budget,
                "n_paragraphs": sum(1 for p in leftover if p.kind == PARAGRAPH),
                "reason": (
                    "buffer could not be emitted within the content budget after exact "
                    "verification, even when placed alone; excluded to protect the budget"
                ),
                "action": "excluded",
            }
        )
        self.buf.clear()

    def _shrink_to_fit(self, pieces: list[Piece]) -> tuple[str, list[Piece]]:
        """Guarantee the budget by re-tokenising, dropping only whole paragraphs."""
        working = list(pieces)
        while working:
            text = self.sep.join(p.text for p in working)
            if self.counter.count(text) <= self.budget:
                return text, working
            drop_idx = None
            for i in range(len(working) - 1, -1, -1):
                if working[i].kind == PARAGRAPH:
                    drop_idx = i
                    break
            if drop_idx is None:
                return "", []
            working = working[:drop_idx]
        return "", []

    # -- chapter driver ---------------------------------------------------- #

    def run_chapter(self, chapter: Chapter) -> None:
        """Move to a new chapter and place its scenes.

        The buffer is deliberately *not* flushed here. Nothing in the rules
        forbids two whole chapters sharing a chunk -- they only forbid splitting
        a scene to fill one, and require a chapter's title to stay with its
        content. Flushing per chapter would leave every chunk about the size of
        a single chapter, roughly halving token utilisation for no benefit. The
        buffer is flushed when the next scene genuinely does not fit, which
        still ends the chunk on a scene/chapter boundary.
        """
        self.chapter = chapter
        self.chapter_opened = False

        for scene in chapter.scenes:
            if not (scene.paragraphs or scene.headings):
                continue
            self._place_scene(scene)

    # -- scene placement --------------------------------------------------- #

    def _scene_pieces(self, scene: Scene) -> list[Piece]:
        pieces: list[Piece] = []
        chapter = self.chapter.index if self.chapter else None
        if self.cfg.include_scene_headings:
            for h in scene.headings:
                if h and h.strip():
                    pieces.append(
                        Piece(SCENE_HEADING, h, self.counter.count(h), scene_index=scene.index,
                              chapter_index=chapter)
                    )
        for para in scene.paragraphs:
            pieces.append(
                Piece(
                    PARAGRAPH,
                    para.text,
                    self.counter.count(para.text),
                    scene_index=scene.index,
                    para_index=para.index,
                    chapter_index=chapter,
                )
            )
        return pieces

    @staticmethod
    def _pieces_cost(pieces: Sequence[Piece], sep_tokens: int) -> int:
        if not pieces:
            return 0
        return sum(p.n_tokens for p in pieces) + sep_tokens * (len(pieces) - 1)

    def _place_scene(self, scene: Scene) -> None:
        """Place one whole scene, or fall through to the paragraph path.

        The scene's cost is evaluated together with any chapter heading that
        still has to ride along with it, because the heading must not be
        separated from the chapter's first content.
        """
        scene_pieces = self._scene_pieces(scene)
        if not scene_pieces:
            return

        heading_pieces = self._pending_heading_pieces()
        incoming = heading_pieces + scene_pieces
        incoming_cost = self._pieces_cost(incoming, self.sep_tokens)

        if self.cfg.oversize_scene_policy == OVERSIZE_SCENE_EXCLUDE:
            if incoming_cost > self.budget:
                self.record_oversize(
                    self._scene_record(scene, self._pieces_cost(scene_pieces, self.sep_tokens),
                                       "excluded")
                )
                self.chapter_opened = True
                return

        # 1. Does the whole scene (heading included) fit after what is buffered?
        extra_sep = self.sep_tokens if self.buf.pieces else 0
        if self.buf.tokens + extra_sep + incoming_cost <= self.budget:
            self._commit_heading()
            self.buf.extend(scene_pieces)
            return

        # 2. Does it fit in a chunk of its own?
        if incoming_cost <= self.budget:
            # Break BEFORE the scene. This is what satisfies both "prefer ending
            # a chunk before a scene boundary" and "never split a scene merely
            # to fill a chunk".
            self.flush()
            self._commit_heading()
            self.buf.extend(scene_pieces)
            return

        # 3. The scene alone exceeds the whole budget, so paragraph-level cuts
        #    are the only legal option. Log it rather than cutting silently.
        self.flush()
        self._place_oversize_scene(
            scene, scene_pieces, self._pieces_cost(scene_pieces, self.sep_tokens)
        )

    def _place_oversize_scene(self, scene: Scene, pieces: list[Piece], scene_cost: int) -> None:
        produced: list[str] = []
        fragments = 0

        def close() -> None:
            nonlocal fragments
            chunk = self.flush()
            if chunk is None:
                return
            produced.append(chunk.chunk_id)
            if fragments > 0:
                # A later fragment of the same scene: it resumes mid-scene.
                chunk.continues_scene = True
                chunk.starts_scene = False
                chunk.starts_chapter = False
            fragments += 1

        self._commit_heading()
        if self.cfg.include_scene_headings:
            for hp in pieces:
                if hp.kind != SCENE_HEADING:
                    continue
                if self.buf.fits(hp.n_tokens, self.budget):
                    self.buf.add(hp)

        for piece in pieces:
            if piece.kind != PARAGRAPH:
                continue

            if piece.n_tokens > self.budget:
                self.record_oversize(
                    {
                        "kind": "oversize_paragraph",
                        "book_id": self.book.book_id,
                        "book_title": self.book.title,
                        "source": (self.book.source_files or [None])[0],
                        "chapter": self.chapter.index if self.chapter else None,
                        "chapter_title": self.chapter.title if self.chapter else None,
                        "scene": scene.index,
                        "paragraph": piece.para_index,
                        "n_tokens": piece.n_tokens,
                        "budget": self.budget,
                        "reason": (
                            f"paragraph exceeds the usable content budget "
                            f"({piece.n_tokens} > {self.budget}); splitting a paragraph is "
                            "disallowed, so it is excluded from training chunks"
                        ),
                        "action": "excluded",
                    }
                )
                continue

            if not self.buf.fits(piece.n_tokens, self.budget):
                close()

            self.buf.add(piece)

        close()
        self.record_oversize(
            self._scene_record(
                scene, scene_cost, "split_at_paragraph_boundaries", chunk_ids=produced
            )
        )

    def _scene_record(
        self, scene: Scene, scene_cost: int, action: str, chunk_ids: Optional[list[str]] = None
    ) -> dict:
        rec = {
            "kind": "oversize_scene",
            "book_id": self.book.book_id,
            "book_title": self.book.title,
            "source": (self.book.source_files or [None])[0],
            "chapter": self.chapter.index if self.chapter else None,
            "chapter_title": self.chapter.title if self.chapter else None,
            "scene": scene.index,
            "paragraph": None,
            "n_tokens": scene_cost,
            "budget": self.budget,
            "n_paragraphs": scene.n_paragraphs,
            "reason": f"scene exceeds the usable content budget ({scene_cost} > {self.budget})",
            "action": (
                "split_at_paragraph_boundaries"
                if action == "split_at_paragraph_boundaries"
                else "excluded"
            ),
        }
        if chunk_ids:
            rec["chunk_ids"] = chunk_ids
        return rec


# --------------------------------------------------------------------------- #
# Post-passes
# --------------------------------------------------------------------------- #


def _gaps(indices: Sequence[int]) -> list[int]:
    """Paragraph indices missing from an otherwise contiguous range."""
    if not indices:
        return []
    present = set(indices)
    return [i for i in range(min(indices), max(indices) + 1) if i not in present]


def mark_scene_boundaries(chunks: Sequence[Chunk], book: Book) -> None:
    """Set ``ends_scene`` for chunks that finish exactly at a scene boundary.

    A chunk ends on a scene boundary when its last paragraph is the final
    paragraph of the scene it belongs to. Deferred or excluded paragraphs do not
    count, which is why the check is driven by the book structure rather than
    arithmetic on indices.
    """
    scene_last_para: dict[int, int] = {}
    for chapter in book.chapters:
        for scene in chapter.scenes:
            if scene.paragraphs:
                scene_last_para[scene.paragraphs[-1].index] = 1

    for chunk in chunks:
        if chunk.paragraph_index_end < 0:
            continue
        chunk.ends_scene = chunk.paragraph_index_end in scene_last_para


def _merge_small_chunks(
    chunks: list[Chunk],
    counter: TokenCounter,
    cfg: ChunkConfig,
    budget: int,
) -> list[Chunk]:
    """Fold a too-small chunk into its predecessor of the same book/chapter.

    Order is preserved, nothing is split, and the combined text must still fit
    the budget. The merged chunk simply stops ending on a scene boundary, which
    the "prefer" wording allows.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and chunk.n_tokens < cfg.min_chunk_tokens
            and prev.book_id == chunk.book_id
            and (cfg.merge_across_chapters or prev.chapter_index == chunk.chapter_index)
        ):
            text = f"{prev.text}{cfg.block_separator}{chunk.text}"
            n = counter.count(text)
            if n <= budget:
                prev.text = text
                prev.n_tokens = n
                prev.scene_index_start = min(prev.scene_index_start, chunk.scene_index_start)
                prev.scene_index_end = max(prev.scene_index_end, chunk.scene_index_end)
                prev.chapter_index_end = max(prev.chapter_index_end, chunk.chapter_index_end)
                prev.paragraph_index_start = min(
                    prev.paragraph_index_start, chunk.paragraph_index_start
                )
                prev.paragraph_index_end = max(prev.paragraph_index_end, chunk.paragraph_index_end)
                prev.n_paragraphs += chunk.n_paragraphs
                prev.ends_scene = chunk.ends_scene
                prev.spans_multiple_scenes = (
                    prev.spans_multiple_scenes
                    or chunk.spans_multiple_scenes
                    or prev.scene_index_end != prev.scene_index_start
                )
                prev.paragraph_index_gaps = sorted(
                    set(prev.paragraph_index_gaps) | set(chunk.paragraph_index_gaps)
                )
                continue
        merged.append(chunk)

    for i, c in enumerate(merged):
        c.order_in_book = i
    return merged


# --------------------------------------------------------------------------- #
# Invariant checks
# --------------------------------------------------------------------------- #


def assert_no_duplicates(chunks: Sequence[Chunk]) -> None:
    """Verify the 'no overlapping chunks that duplicate prose' rule."""
    per_book: dict[str, set[int]] = {}
    for chunk in chunks:
        if chunk.paragraph_index_start < 0:
            continue
        seen = per_book.setdefault(chunk.book_id, set())
        gaps = set(chunk.paragraph_index_gaps)
        for idx in range(chunk.paragraph_index_start, chunk.paragraph_index_end + 1):
            if idx in gaps:
                continue
            if idx in seen:
                raise AssertionError(
                    f"paragraph {idx} appears in more than one chunk of {chunk.book_id}"
                )
            seen.add(idx)


def assert_budget(chunks: Sequence[Chunk], budget: int) -> None:
    for chunk in chunks:
        if chunk.n_tokens > budget:
            raise AssertionError(
                f"chunk {chunk.chunk_id} has {chunk.n_tokens} tokens > budget {budget}"
            )


def assert_order(chunks: Sequence[Chunk]) -> None:
    """Within a book, chunks must appear in increasing source order."""
    last: dict[str, int] = {}
    for chunk in chunks:
        prev = last.get(chunk.book_id)
        if prev is not None and chunk.paragraph_index_start >= 0 and prev > chunk.paragraph_index_start:
            raise AssertionError(
                f"book {chunk.book_id}: chunk order {chunk.order_in_book} is out of source order"
            )
        if chunk.paragraph_index_start >= 0:
            last[chunk.book_id] = chunk.paragraph_index_start


def assert_single_book_per_chunk(chunks: Sequence[Chunk]) -> None:
    for chunk in chunks:
        files = chunk.source_files
        if len(set(files)) != len(files):
            raise AssertionError(f"chunk {chunk.chunk_id} lists a duplicate source file")
