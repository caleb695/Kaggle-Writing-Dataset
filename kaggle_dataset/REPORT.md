# Dataset report — ministral3-14b-base-8192

Generated 2026-09-26T17:52:54Z · build time 54.4s

## 1. Target model and sequence budget

| Field | Value |
| --- | --- |
| Model | `Ministral-3-14B-Base-2512` |
| HF repo | `mistralai/Ministral-3-14B-Base-2512` |
| Vocabulary | 131,072 (Tekken) |
| Total sequence budget | **8192** tokens |
| Reserved for boundary tokens | 2 (`[BOS]` = 1, `[EOS]` = 2) |
| Usable content budget | **8190** tokens |
| Chunk layout | `[BOS] + content + [EOS]` |
| Block separator | `'\n\n'` |

> **Tokenizer caution.** Bundled Tekken has its special tokens stripped, so counts are a conservative UPPER BOUND vs the model's own tekken.json: a chunk verified at <= budget here also fits under the real tokenizer. Pass --tekken /path/to/tekken.json from the model snapshot for exact counts.

## 2. Corpus

| Metric | Value |
| --- | --- |
| Books included | 60 |
| Books excluded | 0 |
| Chapters | 1,891 |
| Scenes | 2,935 |
| Paragraphs | 264,922 |
| Words | 6,740,169 |

## 3. Chunking result

| Metric | Value |
| --- | --- |
| Chunks | 1,532 |
| Content tokens | 9,153,174 |
| Boundary tokens | 3,064 |
| **Training tokens (content + boundaries)** | **9,156,238** |
| Train / val chunks | 1,398 / 134 |
| Train / val tokens | 8,325,829 / 827,345 |
| Mean chunk tokens | 5974.7 |
| Chunk token percentiles (5/25/50/75/95) | 1322 / 4907 / 6599 / 7610 / 7981 |
| Min / max chunk tokens | 30 / 8102 |
| Budget utilisation | 73.0% |

## 4. Boundary conformance

How well the chunks line up with the structure the rules care about.

| Check | Value |
| --- | --- |
| Chunks opening a chapter (heading attached) | 848 |
| Chunks opening a scene | 43 |
| Chunks ending exactly on a scene break | 1,294 (84.46%) |
| Chunks resuming an oversize scene | 238 |
| Chunks opening at any boundary | 58.16% |

Rules that are structural rather than statistical — never split a paragraph, never split a scene to fill a chunk, never join two books, no duplicated prose, source order preserved — are enforced by construction and re-checked by `assert_budget`, `assert_no_duplicates` and `assert_order` on every build.

## 5. Oversize blocks

138 record(s). Excluded content: 8,560 tokens (0.0934% of available content).

| Kind | Count | Tokens | Actions |
| --- | --- | --- | --- |
| `oversize_paragraph` | 1 | 8,560 | excluded=1 |
| `oversize_scene` | 137 | 2,201,606 | split_at_paragraph_boundaries=137 |

Review threshold 0.500% · stop threshold 2.000%


Full detail: `oversize_blocks.jsonl` (source, chapter, scene, paragraph, token count, reason, action).

Scenes marked `split_at_paragraph_boundaries` were cut only at paragraph boundaries — the one legal cut point — because the scene exceeded the entire content budget. Each affected scene lists the `chunk_ids` it became.

## 6. Splits

Strategy: **book**, validation fraction 10%

Held-out books (whole books, so no book leaks across the split):

- `fablehaven_and_dragonwatch_4`
- `fablehaven_and_dragonwatch_5`
- `five_kingdoms_and_beyonders_2`
- `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_sea_of_monsters`
- `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_house_of_hades`
- `micheal_vey_9`

## 7. Exclusions

Patterns applied: `wings of fire*`, `*wings of fire*`, `onyx`, `onyx*`

Nothing matched the exclusion patterns.

## 8. Books

| Book | Ch | Scenes | Paragraphs | Chunks | Tokens | Split |
| --- | --- | --- | --- | --- | --- | --- |
| `eragon_eragon__brisingr` | 1 | 66 | 5,886 | 61 | 343,459 | train |
| `eragon_eragon__eldest` | 1 | 82 | 6,487 | 52 | 307,161 | train |
| `eragon_eragon__eragon` | 1 | 69 | 4,265 | 37 | 214,421 | train |
| `eragon_eragon__inheritance` | 83 | 92 | 7,814 | 67 | 374,118 | train |
| `fablehaven_and_dragonwatch_1__fablehaven` | 22 | 46 | 3,240 | 17 | 102,159 | train |
| `fablehaven_and_dragonwatch_1__fablehaven_grip_of_the_shadow_plague` | 26 | 42 | 3,951 | 25 | 150,106 | train |
| `fablehaven_and_dragonwatch_1__fablehaven_keys_to_the_demon_prison` | 34 | 38 | 5,328 | 33 | 189,803 | train |
| `fablehaven_and_dragonwatch_1__fablehaven_rise_of_the_evening_star` | 23 | 55 | 3,663 | 22 | 137,290 | train |
| `fablehaven_and_dragonwatch_1__fablehaven_secrets_of_the_dragon_sanctuary` | 30 | 55 | 4,466 | 29 | 167,489 | train |
| `fablehaven_and_dragonwatch_2` | 28 | 28 | 4,050 | 19 | 112,765 | train |
| `fablehaven_and_dragonwatch_3` | 40 | 40 | 4,718 | 20 | 122,353 | train |
| `fablehaven_and_dragonwatch_4` | 2 | 2 | 5,384 | 21 | 151,282 | val |
| `fablehaven_and_dragonwatch_5` | 1 | 52 | 5,778 | 27 | 160,023 | val |
| `fablehaven_and_dragonwatch_6` | 1 | 1 | 6,584 | 24 | 185,225 | train |
| `five_kingdoms_and_beyonders_1` | 36 | 37 | 4,459 | 23 | 137,134 | train |
| `five_kingdoms_and_beyonders_2` | 1 | 49 | 4,812 | 27 | 152,381 | val |
| `five_kingdoms_and_beyonders_3` | 1 | 1 | 5,047 | 20 | 153,177 | train |
| `five_kingdoms_and_beyonders_4` | 40 | 48 | 5,502 | 28 | 157,977 | train |
| `five_kingdoms_and_beyonders_5` | 38 | 46 | 5,216 | 23 | 134,805 | train |
| `five_kingdoms_and_beyonders_brandon_mulls_beyonders_trilogy_mull_brandon_z_library__a_world_without_heroes` | 33 | 67 | 4,950 | 26 | 169,369 | train |
| `five_kingdoms_and_beyonders_brandon_mulls_beyonders_trilogy_mull_brandon_z_library__chasing_the_prophecy` | 42 | 60 | 4,537 | 29 | 187,535 | train |
| `five_kingdoms_and_beyonders_brandon_mulls_beyonders_trilogy_mull_brandon_z_library__seeds_of_rebellion` | 31 | 31 | 4,972 | 34 | 178,958 | train |
| `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_battle_of_the_labyrinth` | 21 | 21 | 3,777 | 22 | 118,677 | train |
| `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_last_olympian` | 27 | 31 | 3,874 | 21 | 123,974 | train |
| `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_lightning_thief` | 27 | 37 | 4,000 | 21 | 120,259 | train |
| `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_sea_of_monsters` | 22 | 28 | 2,812 | 14 | 89,088 | val |
| `greek_and_roman_mythology_percy_jackson__percy_jackson_and_the_titan_s_curse` | 21 | 21 | 3,421 | 19 | 100,841 | train |
| `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_blood_of_olympus` | 69 | 69 | 5,063 | 27 | 167,917 | train |
| `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_house_of_hades` | 79 | 79 | 5,585 | 30 | 185,353 | val |
| `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_lost_hero` | 65 | 70 | 4,969 | 28 | 177,914 | train |
| `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_mark_of_athena` | 53 | 53 | 5,311 | 33 | 187,045 | train |
| `greek_and_roman_mythology_the_heroes_of_olympus_the_complete_series__heroes_of_olympus_the_son_of_neptune` | 53 | 53 | 4,647 | 28 | 166,170 | train |
| `harry_potter_1` | 30 | 46 | 3,214 | 20 | 112,431 | train |
| `harry_potter_2` | 19 | 20 | 3,186 | 19 | 120,377 | train |
| `harry_potter_3` | 31 | 53 | 4,358 | 29 | 159,632 | train |
| `harry_potter_4` | 38 | 38 | 6,170 | 51 | 266,690 | train |
| `harry_potter_5` | 37 | 37 | 9,317 | 68 | 359,298 | train |
| `harry_potter_6` | 32 | 32 | 5,483 | 43 | 229,558 | train |
| `harry_potter_harry_potter_and_the_deathly_hallows_book_7_j_k_rowling_2012_2007_pottermore_limited_thorndike_press_4b62cd3d3f4f75aba11ec1625c9a2f54_anna_s_archive` | 19 | 55 | 6,932 | 53 | 275,019 | train |
| `i_must_betray_you_i_must_betray_you` | 122 | 134 | 3,749 | 13 | 87,550 | train |
| `micheal_vey_1` | 1 | 52 | 4,940 | 19 | 115,005 | train |
| `micheal_vey_10` | 57 | 67 | 4,763 | 19 | 121,462 | train |
| `micheal_vey_2` | 1 | 90 | 4,630 | 19 | 118,931 | train |
| `micheal_vey_3` | 5 | 50 | 4,433 | 21 | 107,009 | train |
| `micheal_vey_4` | 4 | 40 | 4,379 | 20 | 105,203 | train |
| `micheal_vey_5` | 6 | 23 | 3,349 | 16 | 87,312 | train |
| `micheal_vey_6` | 12 | 79 | 3,808 | 18 | 101,283 | train |
| `micheal_vey_7` | 58 | 99 | 3,733 | 15 | 100,385 | train |
| `micheal_vey_8` | 81 | 88 | 4,890 | 18 | 115,867 | train |
| `micheal_vey_9` | 56 | 56 | 3,594 | 15 | 89,218 | val |
| `the_false_prince_1` | 1 | 1 | 2,794 | 13 | 99,991 | train |
| `the_false_prince_2` | 1 | 1 | 2,720 | 13 | 100,193 | train |
| `the_false_prince_3` | 3 | 3 | 2,183 | 14 | 100,360 | train |
| `unwanteds_1` | 49 | 49 | 2,010 | 15 | 91,357 | train |
| `unwanteds_2` | 1 | 1 | 2,144 | 12 | 93,865 | train |
| `unwanteds_3` | 78 | 79 | 2,411 | 15 | 105,737 | train |
| `unwanteds_4` | 1 | 77 | 2,794 | 16 | 112,200 | train |
| `unwanteds_5` | 65 | 65 | 2,491 | 15 | 106,260 | train |
| `unwanteds_6` | 65 | 65 | 2,978 | 18 | 120,935 | train |
| `unwanteds_7` | 66 | 66 | 2,901 | 18 | 123,818 | train |

---

## Training hand-off

Every chunk in `train.jsonl` / `val.jsonl` already fits `8190` content tokens. At load time prepend `[BOS]` (id 1) and append `[EOS]` (id 2) to reach the full 8192-token budget.

If you enabled `--emit-tokenized`, `input_ids.npy` already contains the boundary tokens and `offsets.npy` gives `(start, end, seq_len)` per chunk.
