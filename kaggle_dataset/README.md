# Ministral 14B writing CPT dataset

Packed 8192-token chunks for continued pretraining of
`mistralai/Ministral-3-14B-Base-2512`.

See `KAGGLE.md` in the repo root for how to import this folder as a Kaggle
dataset and train.

| File | What |
| --- | --- |
| `input_ids.npy` | uint32 token stream, each chunk is `[BOS=1] … [EOS=2]` |
| `train_offsets.npy` / `val_offsets.npy` | `(start, end, seq_len)` int64 |
| `train.jsonl` / `val.jsonl` | UTF-8 text + provenance |
| `stats.json` / `REPORT.md` | build report |
| `manifest.json` | budget, token ids, counts |

Whole books are held out for validation. Chunks never split a paragraph to
fill a sequence, never join two books, and never overlap.
