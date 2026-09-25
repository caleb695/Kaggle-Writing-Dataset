# Kaggle-Writing-Dataset

Boundary-respecting chunker that turns a library of manuscripts into a
continued-pretraining corpus for **`mistralai/Ministral-3-14B-Base-2512`**
with an **8192-token** sequence budget.

The chunking rules are enforced in code, not by convention — see
[Guarantees](#guarantees).

---

## Target model

Facts taken from the model's own `config.json` and repository file listing:

| Property | Value |
| --- | --- |
| Architecture | `Mistral3ForConditionalGeneration` (`mistral3`), dense |
| Language layers | 40 · hidden 5120 · FFN 16384 · 32 Q / 8 KV heads (GQA) · head_dim 128 |
| Total parameters | 14B (13.5B language + 0.4B vision) |
| Vocabulary | **131,072** |
| Tokenizer | **Tekken** (`tekken.json`, 16.8 MB, ships in the model repo) |
| Special tokens | `BOS` = 1, `EOS` = 2, `unk` = 0, `pad` = 11, `image_token_index` = 10 |
| Context window | 262,144 |
| `original_max_position_embeddings` | **16,384** (YaRN, factor 16, theta 1e9) |
| License | Apache 2.0 |

8192 sits comfortably inside the 16,384 native window, so training at this
length needs no RoPE/YaRN change.

### Sequence budget

```
8192 total
  -2  reserved for boundary tokens  ([BOS] id 1, [EOS] id 2)
-----
8190  usable content tokens per chunk
```

Every emitted chunk is laid out at train time as `[BOS] <content> [EOS]`.
`--emit-tokenized` writes that packing directly as `input_ids.npy` +
`offsets.npy`.

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# 0. optional: get the exact tokenizer from the model snapshot (see below)
python scripts/fetch_tekken.py --out data/tokenizer/tekken.json
python -m writing_dataset.cli --check-tokenizer --tekken data/tokenizer/tekken.json

# 1. build the dataset
python -m writing_dataset.cli \
  --raw-root data/raw \
  --out-root data/out \
  --run-name ministral3-14b-base-8192 \
  --emit-tokenized

# ...or drive it from the shipped recipe (explicit flags still override it)
python -m writing_dataset.cli \
  --config configs/ministral3-14b-base-8192.yaml \
  --raw-root data/raw --out-root data/out

# 2. read the report
less data/out/ministral3-14b-base-8192/REPORT.md
```

Or from Python:

```python
from writing_dataset import BuildConfig, run_build, write_outputs

cfg = BuildConfig(raw_root="data/raw", out_root="data/out", emit_tokenized=True)
result = run_build(cfg)
write_outputs(result, cfg)
print(result.stats["dataset"]["training_tokens_including_boundaries"])
```

A synthetic corpus is included so you can exercise everything before pointing
the builder at real manuscripts:

```bash
python scripts/make_demo_corpus.py --root data/demo/raw
python -m writing_dataset.cli --raw-root data/demo/raw --out-root data/demo/out --run-name demo
```

---

## Input layout

```
data/raw/
├── Fablehaven and dragonwatch/
│   ├── Fablehaven.epub
│   └── Dragonwatch.epub
├── Harry Potter/
│   └── Philosophers Stone.epub
├── five kingdoms and beyonders/
│   └── Sky Raiders.txt
└── Wings of Fire Arc 1/          <- excluded by default
    └── The Dragonet Prophecy.epub
```

Supported: `.epub`, `.mobi`, `.azw`, `.azw3`, `.prc`, `.txt`, `.md`, `.docx`,
`.html`/`.xhtml`.

By default **one source file == one book**, because the rules forbid joining
text from different books. Use `--manifest configs/books_manifest.example.yaml`
only when a single book is genuinely split across several files; every file
listed for a book is then concatenated in the declared order and no book is ever
mixed with another.

`.pdf` is rejected with a clear message rather than silently mis-parsed — PDF has
no reliable paragraph structure, so export to DOCX or EPUB first.

### Kindle files and omnibuses

`.mobi`/`.azw`/`.azw3` are read natively (`writing_dataset/mobi.py`): the Palm
Database header, the PalmDOC header and the MOBI header are parsed directly and
the text records are decompressed with the PalmDOC LZ77 codec. This handles
files that general-purpose unpackers reject — KindleUnpack crashes with an
`IndexError` on one of the sample files because an `<img recindex>` points past
the end of the resource list, which is common in older conversions.

A "Complete Series" file often holds **five novels in one file**. Left alone,
that would put the end of one novel and the start of the next inside the same
training chunk, and would make a whole collection one train/val unit. So a
multi-novel file is split at the byte offsets its own table of contents
declares, and **each novel becomes its own `Book`**:

* a candidate novel start is a TOC entry whose target opens a fresh front-matter
  block (a "Contents" page within the next few kilobytes);
* glossaries and indexes produce dense false positives, so candidates closer
  together than 50 kB are clustered and only the last of each cluster is kept —
  that is the novel's title page;
* the first segment always starts at byte 0, so the collection's own front
  matter is folded into the first novel and the segments tile the file exactly,
  with no gaps and no overlap.

On the sample files this recovers all five Percy Jackson novels and all five
Heroes of Olympus novels, with correct titles, from two files. Disable it with
`--no-split-omnibuses` if a file legitimately contains one book.

Reading is validated by requiring the decompressed, trailing-data-stripped byte
stream to equal the length the file itself declares — the strongest available
check, since any header-offset error shows up as drift.

Huff/CDIC compression and KF8/AZW3 containers are delegated to the optional
`mobi` package when it is installed, and otherwise reported clearly.

---

## Guarantees

Each rule from the training brief and the code that enforces it:

| Rule | Enforced by |
| --- | --- |
| 8192 total tokens, 2 reserved for boundaries | `ChunkConfig.content_budget` = 8190 |
| Chunk on chapter / scene / paragraph boundaries | `_BookState._place_scene` |
| Never split a paragraph to fill a chunk | a `Piece` is always a whole paragraph |
| Never split a scene to fill a chunk | a scene that does not fit opens a new chunk instead of being cut |
| Chapter title stays with its content | `_open_if_needed` attaches headings to the first chunk carrying the chapter |
| Never join text from different books | one `chunk_book()` call per `Book`; no state crosses books |
| Never join unrelated source files | a `Book` spans only manifest-declared files |
| Prefer ending a chunk before a scene boundary | a non-fitting scene triggers a flush |
| Preserve source order within each book | pieces are appended forward only; `assert_order` |
| No overlapping chunks that duplicate prose | each paragraph index emitted once; `assert_no_duplicates` |
| No synthetic text between chunks | text is verbatim pieces joined by `\n\n`; no markers inserted |
| Oversize content is recorded, never silently split | `oversize_blocks.jsonl` |

`assert_budget`, `assert_no_duplicates` and `assert_order` run on every build,
so a regression fails the run rather than shipping quietly.

### Oversize handling

If a **paragraph** exceeds the whole content budget it is recorded in
`oversize_blocks.jsonl` and excluded — splitting a paragraph is disallowed.

If a **scene** exceeds the whole content budget it is cut only at paragraph
boundaries (the one legal cut point) and recorded, with the `chunk_ids` it
became. `--oversize-scene-policy exclude` drops such scenes instead.

The build warns above `--oversize-warn-fraction` (default 0.5%) of content and
**stops for review** above `--oversize-stop-fraction` (default 2%), exiting with
code 3 unless `--force` is passed.

### A note on budget utilisation

Honouring "never split a scene to fill a chunk" leaves gaps: when the next scene
does not fit, the chunk ends early. The remaining space is padding. That is the
cost of the rule, not a bug.

Measured on the bundled corpora:

| Corpus | Chunks | Utilisation | p50 tokens | Chunks ending on a scene |
| --- | --- | --- | --- | --- |
| Demo (13 synthetic books) | 206 | 82.3% | — | 100% |
| Greek/Roman (10 novels, 1.05M words) | 243 | 72.2% | 6,261 | 97.5% |

Two whole chapters may legally share a chunk — nothing in the rules forbids it,
and a chapter's title still stays with its own content. Flushing at every
chapter end instead would roughly halve utilisation for no gain, so the buffer
is only flushed when the next scene genuinely does not fit.

`--min-chunk-tokens N` additionally merges small chunks when the combined text
still fits. It helps little when preceding chunks are already near-full, so it
is off by default: the strict rules win.

---

## Tokenizer accuracy

`mistral-common` bundles Tekken files with their **special tokens stripped**:
ranks 0–999 are raw control bytes, so `"<s>"` encodes to three ordinary tokens
instead of id 1.

For ordinary prose the bundled file agrees with the model's own `tekken.json`,
because the BPE merge table for ranks ≥ 1000 is shared, and a stripped file can
only *over*-count. So it is a safe conservative upper bound: a chunk verified at
≤ 8190 tokens with the bundled tokenizer also fits under the real one.

For counts that are exact rather than merely conservative, fetch the model's own
file and pass it in:

```bash
python scripts/fetch_tekken.py --out data/tokenizer/tekken.json
python -m writing_dataset.cli --check-tokenizer --tekken data/tokenizer/tekken.json
# expect: real specials: True, BOS 1, EOS 2
```

The resolved tokenizer's SHA-256 is recorded in `stats.json` and
`tokenizer_info.json` so a run is reproducible.

---

## Outputs

Everything lands in `<out_root>/<run_name>/`:

| File | Contents |
| --- | --- |
| `chunks.jsonl` | every chunk with full provenance (disable with `--no-chunks-jsonl`) |
| `train.jsonl` | training split |
| `val.jsonl` | validation split |
| `oversize_blocks.jsonl` | every oversize paragraph/scene, with source, chapter, scene, paragraph, token count, reason, action |
| `stats.json` | full machine-readable build statistics |
| `tokenizer_info.json` | tokenizer provenance and hash |
| `REPORT.md` | the human-readable report |
| `input_ids.npy` + `offsets.npy` | with `--emit-tokenized`; boundary tokens applied, `(start, end, seq_len)` per chunk |
| `hf_dataset/` | with `--emit-hf-dataset` |

### Chunk schema

```json
{
  "id": "harry_potter_philosophers_stone::00007",
  "text": "Chapter 3\n\nThe gate opened at dusk. ...",
  "n_tokens": 6183,
  "book_id": "harry_potter_philosophers_stone",
  "book_title": "Philosophers Stone",
  "source_files": ["Harry Potter/Philosophers Stone.epub"],
  "chapter_index": 2,
  "chapter_title": "Chapter 3",
  "scene_index_start": 1,
  "scene_index_end": 3,
  "paragraph_index_start": 88,
  "paragraph_index_end": 141,
  "n_paragraphs": 54,
  "paragraph_index_gaps": [],
  "order_in_book": 7,
  "flags": {
    "starts_chapter": true,
    "starts_scene": false,
    "ends_scene": true,
    "continues_scene": false,
    "spans_multiple_scenes": true
  }
}
```

`paragraph_index_gaps` lists paragraph indices inside
`[paragraph_index_start, paragraph_index_end]` that this chunk does **not**
carry — non-empty only where an oversize paragraph was skipped mid-range, so
downstream analysis can tell a genuine gap from a bug.

### Splits

Default is **whole books** held out for validation (`--split-by book`,
`--val-fraction 0.10`), assigned by a SHA-1 of the `book_id` so the split is
stable across machines. A book never appears in both splits. `--split-by chapter`
and `--split-by none` are also available, as is `--val-books <id> ...`.

---

## Reporting bugs in a chunk

`python -m writing_dataset.cli --dry-run --json` prints `stats.json` to stdout
without writing anything, which is the fastest way to check a rule is holding
before a long run.

`tests/` covers each rule by name — e.g.
`test_scene_not_split_to_fill_a_chunk`,
`test_chapter_title_stays_with_its_content`,
`test_never_joins_text_from_different_books` — plus regression tests for two
extraction bugs found during development (a scene-break regex whose character
class accidentally matched every letter, and inline HTML markup leaking into
prose).

```bash
python -m pytest tests/ -q
```

---

## Layout

```
src/writing_dataset/
├── types.py        Book / Chapter / Scene / Paragraph / Chunk data model
├── tokenizer.py    tokenizer resolution, budget constants, verification
├── extract.py      epub / txt / md / docx / html  ->  Block list
├── structure.py    Block list -> Book/Chapter/Scene hierarchy
├── chunking.py     the greedy packer (all the rules live here)
├── build.py        orchestration, splits, statistics, writers
├── report.py       console summary + REPORT.md
└── cli.py          command line entry point
configs/            model recipe and manifest template
scripts/            fetch_tekken.py, download_from_drive.sh, make_demo_corpus.py
tests/              rule-by-rule invariant tests
```
