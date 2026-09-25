"""End-to-end build: raw sources -> validated training-ready dataset.

Pipeline
--------
1. discover sources under ``raw_root`` (or read a grouping manifest);
2. extract each file to structural blocks (:mod:`writing_dataset.extract`);
3. fold blocks into Book/Chapter/Scene/Paragraph (:mod:`writing_dataset.structure`);
4. chunk with the boundary-respecting greedy packer (:mod:`writing_dataset.chunking`);
5. verify every invariant the spec demands;
6. split train/val by book or chapter;
7. write ``chunks.jsonl``, ``train.jsonl``, ``val.jsonl``, ``oversize_blocks.jsonl``,
   ``stats.json`` and ``REPORT.md``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .chunking import (
    ChunkConfig,
    OVERSIZE_SCENE_SPLIT,
    assert_budget,
    assert_no_duplicates,
    assert_order,
    chunk_book,
)
from .extract import ExtractedDoc, extract_file
from .structure import BookSpec, build_book_from_docs, discover_books, load_manifest, slugify
from .tokenizer import TokenCounter, TokenizerInfo, load_tokenizer, summarise_lengths
from .types import Book, Chunk

#: Folders/series excluded from training by default (the Wings of Fire arcs and Onyx).
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "wings of fire*",
    "*wings of fire*",
    "onyx",
    "onyx*",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class BuildConfig:
    raw_root: Path = Path("data/raw")
    out_root: Path = Path("data/out")

    run_name: str = "ministral3-14b-base-8192"

    # --- tokenizer / budget ---
    max_seq_len: int = 8192
    reserved_boundary_tokens: int = 2
    tekken_path: Optional[str] = None

    # --- chunking ---
    block_separator: str = "\n\n"
    include_chapter_headings: bool = True
    include_scene_headings: bool = True
    repeat_chapter_title: bool = False
    min_chunk_tokens: int = 0
    merge_across_chapters: bool = False
    oversize_scene_policy: str = OVERSIZE_SCENE_SPLIT

    # --- source handling ---
    manifest: Optional[Path] = None
    split_omnibuses: bool = True
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS
    aggressive_breaks: bool = False
    numbered_breaks: bool = False
    strip_page_numbers: bool = False
    drop_repeated_headings: bool = True

    # --- splits ---
    val_fraction: float = 0.10
    split_by: str = "book"           # "book" | "chapter" | "none"
    val_books: tuple[str, ...] = ()  # explicit names win over val_fraction

    # --- review gates ---
    oversize_warn_fraction: float = 0.005
    oversize_stop_fraction: float = 0.02
    fail_on_oversize_stop: bool = False

    # --- outputs ---
    emit_tokenized: bool = False
    emit_hf_dataset: bool = False
    write_chunks: bool = True

    dry_run: bool = False

    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root)
        self.out_root = Path(self.out_root)
        if self.split_by not in ("book", "chapter", "none"):
            raise ValueError("split_by must be 'book', 'chapter' or 'none'")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")

    @property
    def content_budget(self) -> int:
        return self.max_seq_len - self.reserved_boundary_tokens

    def chunk_config(self) -> ChunkConfig:
        return ChunkConfig(
            max_seq_len=self.max_seq_len,
            reserved_boundary_tokens=self.reserved_boundary_tokens,
            block_separator=self.block_separator,
            include_chapter_headings=self.include_chapter_headings,
            include_scene_headings=self.include_scene_headings,
            repeat_chapter_title=self.repeat_chapter_title,
            min_chunk_tokens=self.min_chunk_tokens,
            merge_across_chapters=self.merge_across_chapters,
            oversize_scene_policy=self.oversize_scene_policy,
        )


# --------------------------------------------------------------------------- #
# Exclusion
# --------------------------------------------------------------------------- #


def is_excluded(spec: BookSpec, patterns: Sequence[str]) -> tuple[bool, Optional[str]]:
    """Return ``(excluded, matched_pattern)`` for a book against exclusion globs.

    Matching is case-insensitive against the book id, its title, the source
    filenames and the top-level folder. A pattern with no wildcard also matches
    as a substring, so ``onyx`` excludes ``Onyx``, ``onyx_book_1`` and
    ``series/onyx`` alike.
    """
    candidates = {
        spec.book_id.lower(),
        spec.title.lower(),
        (spec.series or "").lower(),
        (spec.group_key or "").lower(),
    }
    for src in spec.sources:
        candidates.add(src.name.lower())
        candidates.add(src.stem.lower())
        if len(src.parts) >= 2:
            candidates.add(src.parts[-2].lower())

    for pattern in patterns:
        p = pattern.lower().strip()
        if not p:
            continue
        for cand in candidates:
            if fnmatch.fnmatch(cand, p):
                return True, pattern
            if not any(ch in p for ch in "*?[") and p in cand:
                return True, pattern
    return False, None


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def _stable_unit(book_id: str) -> float:
    """Deterministic [0,1) value for a book id (hash() is salted; sha1 is not)."""
    digest = hashlib.sha1(book_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def choose_val_books(
    book_ids: Sequence[str],
    val_fraction: float,
    explicit: Sequence[str] = (),
) -> set[str]:
    """Hold out *whole books* for validation, so no book leaks across the split."""
    if explicit:
        wanted = {b.lower() for b in explicit}
        chosen = {b for b in book_ids if b.lower() in wanted}
        missing = wanted - {b.lower() for b in book_ids}
        if missing:
            raise ValueError(f"--val-books names unknown book(s): {sorted(missing)}")
        return chosen

    if val_fraction <= 0:
        return set()

    ordered = sorted(book_ids)
    n_val = max(1, int(round(len(ordered) * val_fraction))) if ordered else 0
    if n_val >= len(ordered):
        n_val = max(0, len(ordered) - 1)
    ranked = sorted(ordered, key=lambda b: (_stable_unit(b), b))
    return set(ranked[:n_val])


def split_chunks(chunks: Sequence[Chunk], cfg: BuildConfig, val_books: set[str]) -> tuple[list, list]:
    """Partition chunks into (train, val) without ever splitting a book."""
    if cfg.split_by == "none":
        return list(chunks), []

    train, val = [], []
    for chunk in chunks:
        if cfg.split_by == "book":
            (val if chunk.book_id in val_books else train).append(chunk)
        else:  # chapter-level holdout, still never mid-book-contiguous
            unit = f"{chunk.book_id}::{chunk.chapter_index}"
            (val if _stable_unit(unit) < cfg.val_fraction else train).append(chunk)
    return train, val


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class BuildResult:
    chunks: list[Chunk] = field(default_factory=list)
    train: list[Chunk] = field(default_factory=list)
    val: list[Chunk] = field(default_factory=list)
    oversize: list[dict] = field(default_factory=list)
    books: list[Book] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    out_dir: Optional[Path] = None
    tokenizer_info: Optional[TokenizerInfo] = None

    @property
    def review_required(self) -> bool:
        return bool(self.stats.get("oversize", {}).get("review_required"))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def run_build(
    cfg: BuildConfig,
    progress: bool = True,
) -> BuildResult:
    """Execute the full pipeline. Returns everything needed to write a report."""
    started = time.time()
    counter = load_tokenizer(cfg.tekken_path, max_seq_len=cfg.max_seq_len)
    if counter.info.content_budget != cfg.content_budget:
        # Keep the CLI's budget authoritative but flag the mismatch.
        counter.info.max_seq_len = cfg.max_seq_len
        counter.info.reserved_boundary_tokens = cfg.reserved_boundary_tokens
        counter.info.content_budget = cfg.content_budget

    chunk_cfg = cfg.chunk_config()
    result = BuildResult(tokenizer_info=counter.info)

    if not cfg.raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {cfg.raw_root}")

    specs = (
        load_manifest(cfg.manifest, cfg.raw_root)
        if cfg.manifest is not None
        else discover_books(cfg.raw_root)
    )
    if not specs:
        raise FileNotFoundError(
            f"no supported source files found under {cfg.raw_root} "
            f"(expected .epub/.txt/.md/.docx/.html)"
        )

    kept: list[BookSpec] = []
    for spec in specs:
        excluded, pattern = is_excluded(spec, cfg.exclude_patterns)
        if excluded:
            result.excluded.append(
                {
                    "book_id": spec.book_id,
                    "title": spec.title,
                    "series": spec.series,
                    "matched_pattern": pattern,
                    "sources": [str(s) for s in spec.sources],
                }
            )
        else:
            kept.append(spec)

    iterator = kept
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(kept, desc="building books", unit="book")
        except ImportError:  # pragma: no cover
            iterator = kept

    for spec in iterator:
        try:
            per_source: list[list[ExtractedDoc]] = [
                extract_file(
                    src,
                    aggressive_breaks=cfg.aggressive_breaks,
                    numbered_breaks=cfg.numbered_breaks,
                    strip_pages=cfg.strip_page_numbers,
                    split_omnibus=cfg.split_omnibuses,
                )
                for src in spec.sources
            ]

            # A single source that yielded several documents is an omnibus: each
            # novel inside it becomes its own Book, so no chunk can ever span two
            # novels. A book declared across several files stays one Book.
            if len(spec.sources) == 1 and len(per_source[0]) > 1:
                groups = [([spec.sources[0]], [doc], True) for doc in per_source[0]]
            else:
                groups = [(
                    list(spec.sources),
                    [doc for docs in per_source for doc in docs],
                    False,
                )]

            for sources, docs, is_part in groups:
                sub_spec, book_id = _sub_spec(spec, docs, sources, is_part)
                labels = [_relative_label(src, cfg.raw_root) for src in sources]
                book = build_book_from_docs(
                    docs,
                    sub_spec,
                    drop_repeated_headings=cfg.drop_repeated_headings,
                    source_labels=labels,
                )
                if book.n_paragraphs == 0:
                    result.errors.append(
                        {"book_id": book_id, "error": "no paragraphs extracted",
                         "sources": [str(s) for s in sources]}
                    )
                    continue

                # Warm the token cache for this book, then chunk it.
                _price_book(book, counter)
                chunks, _ = chunk_book(book, counter, chunk_cfg, result.oversize.append)
                result.books.append(book)
                result.chunks.extend(chunks)
        except Exception as exc:  # keep going; one bad file must not kill the run
            result.errors.append(
                {
                    "book_id": spec.book_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "sources": [str(s) for s in spec.sources],
                }
            )

    # ---- invariant verification ------------------------------------------ #
    assert_budget(result.chunks, cfg.content_budget)
    assert_no_duplicates(result.chunks)
    assert_order(result.chunks)

    # ---- splits ----------------------------------------------------------- #
    val_books = choose_val_books(
        [b.book_id for b in result.books], cfg.val_fraction, cfg.val_books
    ) if cfg.split_by == "book" else set()
    result.train, result.val = split_chunks(result.chunks, cfg, val_books)

    # ---- statistics ------------------------------------------------------- #
    result.stats = compute_stats(cfg, result, val_books, counter)
    result.stats["elapsed_seconds"] = round(time.time() - started, 1)

    frac = result.stats["oversize"]["excluded_fraction"]
    result.stats["oversize"]["review_required"] = frac >= cfg.oversize_warn_fraction
    result.stats["oversize"]["stop"] = frac >= cfg.oversize_stop_fraction

    return result


def _sub_spec(
    spec: BookSpec,
    docs: Sequence[ExtractedDoc],
    sources: Sequence[Path],
    is_part: bool,
):
    """Give each novel split out of an omnibus its own book id, title and order.

    ``is_part`` must be passed explicitly rather than inferred from list
    lengths: a single-part split and an unsplit single-file book look identical
    by length, and conflating them made every novel in an omnibus share one
    ``book_id`` -- which silently merged five books into one entity.
    """
    if not is_part:
        return spec, spec.book_id

    meta = (docs[0].metadata or {}) if docs else {}
    title = meta.get("title")
    part = meta.get("part_index")

    book_id = (
        f"{spec.book_id}__{slugify(str(title))[:60]}"
        if title
        else f"{spec.book_id}__part{part}"
    )

    sub = BookSpec(
        book_id=book_id,
        title=str(title or spec.title),
        sources=list(sources),
        series=spec.series,
        order_hint=spec.order_hint * 100 + (int(part) if part else 0),
        group_key=spec.group_key,
    )
    return sub, book_id


def _relative_label(path: Path, raw_root: Path) -> str:
    """Path relative to the raw root when possible, else the name as given."""
    try:
        return str(Path(path).resolve().relative_to(Path(raw_root).resolve()))
    except (ValueError, OSError):
        return str(path)


def _price_book(book: Book, counter: TokenCounter) -> int:
    """Populate per-paragraph token counts (cached, so the chunker reuses them)."""
    total = 0
    for chapter in book.chapters:
        for scene in chapter.scenes:
            for para in scene.paragraphs:
                n = counter.count(para.text)
                para.n_tokens = n
                total += n
    return total


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def compute_stats(
    cfg: BuildConfig,
    result: BuildResult,
    val_books: set[str],
    counter: TokenCounter,
) -> dict:
    chunks = result.chunks
    lengths = [c.n_tokens for c in chunks]
    content_tokens = sum(lengths)
    boundary_tokens = cfg.reserved_boundary_tokens * len(chunks)

    book_stats = []
    for book in result.books:
        book_chunks = [c for c in chunks if c.book_id == book.book_id]
        book_stats.append(
            {
                "book_id": book.book_id,
                "title": book.title,
                "series": book.series,
                "source_files": book.source_files,
                "chapters": book.n_chapters,
                "scenes": book.n_scenes,
                "paragraphs": book.n_paragraphs,
                "chunks": len(book_chunks),
                "content_tokens": sum(c.n_tokens for c in book_chunks),
                "in_val_split": book.book_id in val_books,
            }
        )
    book_stats.sort(key=lambda b: b["book_id"])

    oversize_by_kind: dict[str, dict] = {}
    for rec in result.oversize:
        slot = oversize_by_kind.setdefault(rec["kind"], {"count": 0, "tokens": 0, "actions": {}})
        slot["count"] += 1
        slot["tokens"] += int(rec.get("n_tokens") or 0)
        slot["actions"][rec.get("action", "unknown")] = (
            slot["actions"].get(rec.get("action", "unknown"), 0) + 1
        )

    excluded_tokens = sum(
        int(r.get("n_tokens") or 0) for r in result.oversize if r.get("action") == "excluded"
    )
    denom = content_tokens + excluded_tokens
    excluded_fraction = (excluded_tokens / denom) if denom else 0.0

    n_chunks = len(chunks) or 1
    chapters_started = sum(1 for c in chunks if c.starts_chapter)
    scenes_started = sum(1 for c in chunks if c.starts_scene)
    ended_on_scene = sum(1 for c in chunks if c.ends_scene)
    continuations = sum(1 for c in chunks if c.continues_scene)

    return {
        "run_name": cfg.run_name,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "name": "Ministral-3-14B-Base-2512",
            "hf_repo": "mistralai/Ministral-3-14B-Base-2512",
            "vocab_size": 131072,
            "tokenizer": "tekken",
        },
        "budget": {
            "max_seq_len": cfg.max_seq_len,
            "reserved_boundary_tokens": cfg.reserved_boundary_tokens,
            "content_budget": cfg.content_budget,
            "layout": "[BOS] + content + [EOS]",
            "block_separator": repr(cfg.block_separator),
        },
        "tokenizer": counter.info.as_dict(),
        "corpus": {
            "books_total": len(result.books) + len(result.excluded),
            "books_included": len(result.books),
            "books_excluded": len(result.excluded),
            "chapters": sum(b.n_chapters for b in result.books),
            "scenes": sum(b.n_scenes for b in result.books),
            "paragraphs": sum(b.n_paragraphs for b in result.books),
            "words": sum(
                len(p.text.split()) for b in result.books for c in b.chapters
                for s in c.scenes for p in s.paragraphs
            ),
        },
        "dataset": {
            "chunks": len(chunks),
            "content_tokens": content_tokens,
            "boundary_tokens": boundary_tokens,
            "training_tokens_including_boundaries": content_tokens + boundary_tokens,
            "chunk_tokens": summarise_lengths(lengths),
            "train_chunks": len(result.train),
            "val_chunks": len(result.val),
            "train_tokens": sum(c.n_tokens for c in result.train),
            "val_tokens": sum(c.n_tokens for c in result.val),
            "budget_utilisation": round(
                (sum(lengths) / (cfg.content_budget * len(chunks))) if chunks else 0.0, 4
            ),
        },
        "boundaries": {
            "chunks_opening_a_chapter": chapters_started,
            "chunks_opening_a_scene": scenes_started,
            "chunks_ending_on_a_scene": ended_on_scene,
            "chunks_resuming_a_scene": continuations,
            "pct_ending_on_a_scene": round(100.0 * ended_on_scene / n_chunks, 2),
            "pct_opening_at_a_boundary": round(
                100.0 * sum(1 for c in chunks if c.starts_chapter or c.starts_scene) / n_chunks, 2
            ),
        },
        "oversize": {
            "records": len(result.oversize),
            "by_kind": oversize_by_kind,
            "excluded_tokens": excluded_tokens,
            "excluded_fraction": round(excluded_fraction, 6),
            "warn_fraction": cfg.oversize_warn_fraction,
            "stop_fraction": cfg.oversize_stop_fraction,
            "review_required": False,
            "stop": False,
        },
        "splits": {
            "strategy": cfg.split_by,
            "val_fraction": cfg.val_fraction,
            "val_books": sorted(val_books),
        },
        "exclusions": {
            "patterns": list(cfg.exclude_patterns),
            "matched": result.excluded,
        },
        "provenance": {
            "raw_root": str(cfg.raw_root),
            "manifest": str(cfg.manifest) if cfg.manifest else None,
            "books": book_stats,
        },
        "errors": result.errors,
    }


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def write_outputs(result: BuildResult, cfg: BuildConfig) -> Path:
    """Write every artifact into ``<out_root>/<run_name>/``."""
    out_dir = cfg.out_root / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result.out_dir = out_dir

    if cfg.write_chunks:
        _write_jsonl(out_dir / "chunks.jsonl", (c.as_row() for c in result.chunks))
    _write_jsonl(out_dir / "train.jsonl", (c.as_row() for c in result.train))
    _write_jsonl(out_dir / "val.jsonl", (c.as_row() for c in result.val))
    _write_jsonl(out_dir / "oversize_blocks.jsonl", result.oversize)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as fh:
        json.dump(result.stats, fh, indent=2, ensure_ascii=False)

    if result.tokenizer_info is not None:
        with (out_dir / "tokenizer_info.json").open("w", encoding="utf-8") as fh:
            json.dump(result.tokenizer_info.as_dict(), fh, indent=2)

    if cfg.emit_tokenized:
        write_tokenized(result, out_dir, cfg)

    if cfg.emit_hf_dataset:
        write_hf_dataset(result, out_dir)

    from .report import render_report

    (out_dir / "REPORT.md").write_text(render_report(result, cfg), encoding="utf-8")
    return out_dir


# --------------------------------------------------------------------------- #
# Optional heavy outputs
# --------------------------------------------------------------------------- #


def write_tokenized(result: BuildResult, out_dir: Path, cfg: BuildConfig) -> None:
    """Memory-mappable token stream: ``input_ids.npy`` + ``offsets.npy``.

    Each chunk is stored as ``[BOS] ids... [EOS]``, length ``n_tokens + 2`` at
    most. ``offsets`` is an ``(n_chunks, 3)`` int64 array of
    ``(start, end, seq_len)`` so a training loader can slice without re-tokenising.
    """
    import numpy as np

    counter = load_tokenizer(cfg.tekken_path, max_seq_len=cfg.max_seq_len)
    bos = counter.info.bos_id if counter.info.bos_id is not None else 1
    eos = counter.info.eos_id if counter.info.eos_id is not None else 2

    total = 0
    encoded: list[list[int]] = []
    for chunk in result.chunks:
        ids = [bos, *counter.encode(chunk.text), eos]
        if len(ids) > cfg.max_seq_len:
            raise AssertionError(
                f"chunk {chunk.chunk_id} is {len(ids)} tokens with boundaries > {cfg.max_seq_len}"
            )
        encoded.append(ids)
        total += len(ids)

    flat = np.empty(total, dtype=np.uint32)
    offsets = np.empty((len(encoded), 3), dtype=np.int64)
    cursor = 0
    for i, ids in enumerate(encoded):
        flat[cursor:cursor + len(ids)] = ids
        offsets[i] = (cursor, cursor + len(ids), len(ids))
        cursor += len(ids)

    np.save(out_dir / "input_ids.npy", flat)
    np.save(out_dir / "offsets.npy", offsets)

    with (out_dir / "tokenized_index.jsonl").open("w", encoding="utf-8") as fh:
        for chunk, off in zip(result.chunks, offsets):
            fh.write(json.dumps({
                "id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "offset": int(off[0]),
                "length_with_boundaries": int(off[2]),
                "content_tokens": chunk.n_tokens,
            }, ensure_ascii=False) + "\n")


def write_hf_dataset(result: BuildResult, out_dir: Path) -> None:
    """Optional ``datasets`` export (requires the ``datasets`` extra)."""
    from datasets import Dataset, DatasetDict

    def to_ds(chunks):
        return Dataset.from_dict({
            "text": [c.text for c in chunks],
            "id": [c.chunk_id for c in chunks],
            "n_tokens": [c.n_tokens for c in chunks],
            "book_id": [c.book_id for c in chunks],
            "chapter_index": [c.chapter_index for c in chunks],
        })

    DatasetDict({"train": to_ds(result.train), "validation": to_ds(result.val)}).save_to_disk(
        str(out_dir / "hf_dataset")
    )
