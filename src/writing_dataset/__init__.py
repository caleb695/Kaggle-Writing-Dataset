"""Boundary-respecting manuscript chunker for Ministral-3-14B-Base continued pretraining.

Quick start::

    from writing_dataset import BuildConfig, run_build, write_outputs

    cfg = BuildConfig(raw_root="data/raw", out_root="data/out")
    result = run_build(cfg)
    write_outputs(result, cfg)

Everything the training spec requires -- the 8192-token sequence budget, the two
reserved boundary positions, and the paragraph/scene/chapter integrity rules --
is enforced inside :func:`writing_dataset.chunking.chunk_book`.
"""

from .build import BuildConfig, BuildResult, run_build, write_outputs
from .chunking import (
    ChunkConfig,
    assert_budget,
    assert_no_duplicates,
    assert_order,
    chunk_book,
    mark_scene_boundaries,
)
from .structure import BookSpec, blocks_to_book
from .tokenizer import (
    CONTENT_BUDGET,
    MAX_SEQ_LEN,
    RESERVED_BOUNDARY_TOKENS,
    TokenCounter,
    TokenizerInfo,
    load_tokenizer,
)
from .types import Block, Book, Chapter, Chunk, Paragraph, Scene

__all__ = [
    "BuildConfig",
    "BuildResult",
    "run_build",
    "write_outputs",
    "ChunkConfig",
    "chunk_book",
    "mark_scene_boundaries",
    "assert_budget",
    "assert_no_duplicates",
    "assert_order",
    "BookSpec",
    "blocks_to_book",
    "load_tokenizer",
    "TokenCounter",
    "TokenizerInfo",
    "MAX_SEQ_LEN",
    "CONTENT_BUDGET",
    "RESERVED_BOUNDARY_TOKENS",
    "Block",
    "Book",
    "Chapter",
    "Chunk",
    "Paragraph",
    "Scene",
]

__version__ = "0.1.0"
