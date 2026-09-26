"""Human-readable reporting: console summary and ``REPORT.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .build import BuildConfig, BuildResult


def _fmt(n: int) -> str:
    return f"{n:,}"


def render_console_summary(result, cfg) -> str:
    """Compact summary for the terminal."""
    s = result.stats
    d, c, o = s["dataset"], s["corpus"], s["oversize"]
    b = s["boundaries"]

    lines = [
        "",
        "=" * 72,
        f"  {s['run_name']}  ->  {result.out_dir}",
        "=" * 72,
        f"  budget            {s['budget']['max_seq_len']} tokens "
        f"({s['budget']['content_budget']} content + "
        f"{s['budget']['reserved_boundary_tokens']} boundary)",
        f"  tokenizer         {s['tokenizer']['name']}",
        f"  books             {c['books_included']} included, "
        f"{c['books_excluded']} excluded",
        f"  structure         {_fmt(c['chapters'])} chapters / {_fmt(c['scenes'])} scenes / "
        f"{_fmt(c['paragraphs'])} paragraphs / {_fmt(c['words'])} words",
        "",
        f"  chunks            {_fmt(d['chunks'])}  "
        f"(train {_fmt(d['train_chunks'])} / val {_fmt(d['val_chunks'])})",
        f"  training tokens   {_fmt(d['training_tokens_including_boundaries'])} "
        f"(incl. boundaries)",
        f"  chunk tokens      min {d['chunk_tokens'].get('min', 0)}  "
        f"p50 {d['chunk_tokens'].get('p50', 0)}  "
        f"p95 {d['chunk_tokens'].get('p95', 0)}  "
        f"max {d['chunk_tokens'].get('max', 0)}",
        f"  budget use        {d['budget_utilisation'] * 100:.1f}% of the "
        f"{s['budget']['content_budget']}-token ceiling",
        "",
        f"  boundary health   {b['pct_ending_on_a_scene']}% of chunks end on a scene break, "
        f"{b['pct_opening_at_a_boundary']}% open at one",
        f"  oversize          {_fmt(o['records'])} record(s), "
        f"{_fmt(o['excluded_tokens'])} tokens excluded "
        f"({o['excluded_fraction'] * 100:.3f}%)",
    ]

    if result.errors:
        lines.append(f"  errors            {len(result.errors)} file(s) failed (see REPORT.md)")

    if o.get("stop"):
        lines += [
            "",
            "  !! OVERSIZE STOP THRESHOLD REACHED -- review before training.",
            f"     excluded {o['excluded_fraction'] * 100:.3f}% >= stop "
            f"{o['stop_fraction'] * 100:.3f}%",
        ]
    elif o.get("review_required"):
        lines += [
            "",
            "  ~  oversize content above the review threshold; check oversize_blocks.jsonl",
        ]

    lines.append("=" * 72)
    return "\n".join(lines)


def render_report(result, cfg) -> str:
    """Full Markdown report written next to the dataset."""
    s = result.stats
    d, c, o, b = s["dataset"], s["corpus"], s["oversize"], s["boundaries"]
    tok = s["tokenizer"]
    chunks = s["chunk_tokens"] if "chunk_tokens" in s else d["chunk_tokens"]

    md: list[str] = []
    a = md.append

    a(f"# Dataset report — {s['run_name']}")
    a("")
    a(f"Generated {s['generated_at_utc']} · build time {s.get('elapsed_seconds', '?')}s")
    a("")
    a("## 1. Target model and sequence budget")
    a("")
    a(f"| Field | Value |")
    a(f"| --- | --- |")
    a(f"| Model | `{s['model']['name']}` |")
    a(f"| HF repo | `{s['model']['hf_repo']}` |")
    a(f"| Vocabulary | {_fmt(s['model']['vocab_size'])} (Tekken) |")
    a(f"| Total sequence budget | **{s['budget']['max_seq_len']}** tokens |")
    a(f"| Reserved for boundary tokens | {s['budget']['reserved_boundary_tokens']} "
      f"(`[BOS]` = {tok['bos_id']}, `[EOS]` = {tok['eos_id']}) |")
    a(f"| Usable content budget | **{s['budget']['content_budget']}** tokens |")
    a(f"| Chunk layout | `{s['budget']['layout']}` |")
    a(f"| Block separator | `{s['budget']['block_separator']}` |")
    a("")
    if tok.get("caution"):
        a(f"> **Tokenizer caution.** {tok['caution']}")
        a("")

    a("## 2. Corpus")
    a("")
    a(f"| Metric | Value |")
    a(f"| --- | --- |")
    a(f"| Books included | {c['books_included']} |")
    a(f"| Books excluded | {c['books_excluded']} |")
    a(f"| Chapters | {_fmt(c['chapters'])} |")
    a(f"| Scenes | {_fmt(c['scenes'])} |")
    a(f"| Paragraphs | {_fmt(c['paragraphs'])} |")
    a(f"| Words | {_fmt(c['words'])} |")
    a("")

    a("## 3. Chunking result")
    a("")
    a(f"| Metric | Value |")
    a(f"| --- | --- |")
    a(f"| Chunks | {_fmt(d['chunks'])} |")
    a(f"| Content tokens | {_fmt(d['content_tokens'])} |")
    a(f"| Boundary tokens | {_fmt(d['boundary_tokens'])} |")
    a(f"| **Training tokens (content + boundaries)** | **{_fmt(d['training_tokens_including_boundaries'])}** |")
    a(f"| Train / val chunks | {_fmt(d['train_chunks'])} / {_fmt(d['val_chunks'])} |")
    a(f"| Train / val tokens | {_fmt(d['train_tokens'])} / {_fmt(d['val_tokens'])} |")
    a(f"| Mean chunk tokens | {chunks.get('mean', 0)} |")
    a(f"| Chunk token percentiles (5/25/50/75/95) | "
      f"{chunks.get('p05', 0)} / {chunks.get('p25', 0)} / {chunks.get('p50', 0)} / "
      f"{chunks.get('p75', 0)} / {chunks.get('p95', 0)} |")
    a(f"| Min / max chunk tokens | {chunks.get('min', 0)} / {chunks.get('max', 0)} |")
    a(f"| Budget utilisation | {d['budget_utilisation'] * 100:.1f}% |")
    a("")

    a("## 4. Boundary conformance")
    a("")
    a("How well the chunks line up with the structure the rules care about.")
    a("")
    a(f"| Check | Value |")
    a(f"| --- | --- |")
    a(f"| Chunks opening a chapter (heading attached) | {_fmt(b['chunks_opening_a_chapter'])} |")
    a(f"| Chunks opening a scene | {_fmt(b['chunks_opening_a_scene'])} |")
    a(f"| Chunks ending exactly on a scene break | {_fmt(b['chunks_ending_on_a_scene'])} "
      f"({b['pct_ending_on_a_scene']}%) |")
    a(f"| Chunks resuming an oversize scene | {_fmt(b['chunks_resuming_a_scene'])} |")
    a(f"| Chunks opening at any boundary | {b['pct_opening_at_a_boundary']}% |")
    a("")
    a("Rules that are structural rather than statistical — never split a "
      "paragraph, never split a scene to fill a chunk, never join two books, no "
      "duplicated prose, source order preserved — are enforced by construction "
      "and re-checked by `assert_budget`, `assert_no_duplicates` and "
      "`assert_order` on every build.")
    a("")

    a("## 5. Oversize blocks")
    a("")
    if not o["records"]:
        a("No oversize blocks. Every paragraph and scene fitted inside the content budget.")
        a("")
    else:
        a(f"{_fmt(o['records'])} record(s). Excluded content: "
          f"{_fmt(o['excluded_tokens'])} tokens ({o['excluded_fraction'] * 100:.4f}% of "
          f"available content).")
        a("")
        a(f"| Kind | Count | Tokens | Actions |")
        a(f"| --- | --- | --- | --- |")
        for kind, slot in sorted(o["by_kind"].items()):
            actions = ", ".join(f"{k}={v}" for k, v in sorted(slot["actions"].items()))
            a(f"| `{kind}` | {slot['count']} | {_fmt(slot['tokens'])} | {actions} |")
        a("")
        a(f"Review threshold {o['warn_fraction'] * 100:.3f}% · "
          f"stop threshold {o['stop_fraction'] * 100:.3f}%")
        a("")
        if o.get("stop"):
            a("> **STOP FOR REVIEW.** Excluded content is above the stop threshold. "
              "Inspect `oversize_blocks.jsonl` before training.")
        elif o.get("review_required"):
            a("> **Review suggested.** Excluded content is above the warn threshold.")
        a("")
        a("Full detail: `oversize_blocks.jsonl` (source, chapter, scene, paragraph, "
          "token count, reason, action).")
        a("")
        if o["by_kind"].get("oversize_scene", {}).get("actions", {}).get(
            "split_at_paragraph_boundaries"
        ):
            a("Scenes marked `split_at_paragraph_boundaries` were cut only at paragraph "
              "boundaries — the one legal cut point — because the scene exceeded the "
              "entire content budget. Each affected scene lists the `chunk_ids` it became.")
            a("")

    a("## 6. Splits")
    a("")
    a(f"Strategy: **{s['splits']['strategy']}**"
      + (f", validation fraction {s['splits']['val_fraction']:.0%}"
         if s["splits"]["strategy"] != "none" else ""))
    a("")
    if s["splits"]["val_books"]:
        a("Held-out books (whole books, so no book leaks across the split):")
        a("")
        for bid in s["splits"]["val_books"]:
            a(f"- `{bid}`")
    elif s["splits"]["strategy"] == "none":
        a("No validation split requested.")
    else:
        a("Validation is held out at chapter granularity.")
    a("")

    a("## 7. Exclusions")
    a("")
    a("Patterns applied: " + ", ".join(f"`{p}`" for p in s["exclusions"]["patterns"]))
    a("")
    if s["exclusions"]["matched"]:
        a(f"| Excluded book | Series | Matched pattern | Sources |")
        a(f"| --- | --- | --- | --- |")
        for rec in s["exclusions"]["matched"]:
            n = len(rec["sources"])
            a(f"| `{rec['book_id']}` | {rec.get('series') or ''} | "
              f"`{rec['matched_pattern']}` | {n} |")
    else:
        a("Nothing matched the exclusion patterns.")
    a("")

    a("## 8. Books")
    a("")
    a("| Book | Ch | Scenes | Paragraphs | Chunks | Tokens | Split |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for bk in s["provenance"]["books"]:
        a(f"| `{bk['book_id']}` | {bk['chapters']} | {_fmt(bk['scenes'])} | "
          f"{_fmt(bk['paragraphs'])} | {_fmt(bk['chunks'])} | "
          f"{_fmt(bk['content_tokens'])} | {'val' if bk['in_val_split'] else 'train'} |")
    a("")

    if result.errors:
        a("## 9. Errors")
        a("")
        a(f"{len(result.errors)} source file(s) failed to build:")
        a("")
        for err in result.errors:
            a(f"- `{err['book_id']}`: {err['error']} — {', '.join(err.get('sources', []))}")
        a("")

    a("---")
    a("")
    a("## Training hand-off")
    a("")
    a("Every chunk in `train.jsonl` / `val.jsonl` already fits "
      f"`{s['budget']['content_budget']}` content tokens. At load time prepend "
      f"`[BOS]` (id {tok['bos_id']}) and append `[EOS]` (id {tok['eos_id']}) to reach "
      f"the full {s['budget']['max_seq_len']}-token budget.")
    a("")
    a("If you enabled `--emit-tokenized`, `input_ids.npy` already contains the "
      "boundary tokens and `offsets.npy` gives `(start, end, seq_len)` per chunk.")
    a("")
    return "\n".join(md)
