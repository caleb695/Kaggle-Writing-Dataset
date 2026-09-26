# Kaggle: dataset via GitHub URL, then train Ministral 14B

Two separate Kaggle objects:

1. **Dataset** — the packed 8192-token chunks (`kaggle_dataset/` in this repo).
2. **Notebook** — QLoRA continued pretraining of `Ministral-3-14B-Base-2512`.

Do **not** point Kaggle at branch `main`. `main` is the raw manuscripts. The
training package lives on `arena/01a0d5fc-kaggle-writing-dataset`.

---

## 1. Publish the dataset from GitHub

Kaggle's GitHub connector clones **one branch** of a public repo into dataset
storage. It will not mix with local file uploads.

### Preferred: GitHub URL (what you asked for)

1. On GitHub, set the default branch to
   `arena/01a0d5fc-kaggle-writing-dataset`
   (`Settings → General → Default branch`).
   If you skip this, Kaggle will ingest `main` and you will get the raw
   `.epub` / `.mobi` files instead of `input_ids.npy`.
2. Make the repo **public** (Kaggle cannot clone a private repo this way).
3. Kaggle → **Datasets → New Dataset**.
4. Sidebar: **GitHub** icon.
5. Paste `https://github.com/caleb695/Kaggle-Writing-Dataset`.
6. Title e.g. `Ministral 14B Writing CPT 8192`. Keep it **Private** unless
   you intend to publish.
7. Create. Wait until processing finishes.
8. Optional: Settings → enable periodic GitHub sync.

The files you care about after import are under `kaggle_dataset/`:

| File | Role |
| --- | --- |
| `input_ids.npy` | packed `[BOS] … [EOS]` token stream (uint32) |
| `train_offsets.npy` / `val_offsets.npy` | `(start, end, seq_len)` per chunk |
| `train.jsonl` / `val.jsonl` | same splits as UTF-8 text |
| `stats.json`, `REPORT.md` | build report |
| `manifest.json` | ids, budget, counts |

### Alternative: Kaggle API (no default-branch change)

```bash
# on your machine, with kaggle.json in ~/.kaggle/
pip install kaggle
# edit kaggle_dataset/dataset-metadata.json and put YOUR kaggle username
#   "id": "yourname/ministral-writing-cpt-8192"
kaggle datasets create -p kaggle_dataset --dir-mode zip
```

Later versions:

```bash
kaggle datasets version -p kaggle_dataset -m "rebuild" --dir-mode zip
```

---

## 2. Train Ministral 14B Base (QLoRA CPT)

14B in bf16 is ~28 GB. Kaggle GPUs are 16 GB, so this is **4-bit QLoRA**,
not full-weight training. 8192-token sequences + gradient checkpointing +
batch size 1 fit; pick **GPU T4 x2** if you have the choice.

### Notebook setup

1. Kaggle → **Code → New Notebook**.
2. Settings:
   - Accelerator: **GPU T4 x2** (fallback: GPU P100).
   - Internet: **On** (needed once to `pip install` a new transformers).
3. **Add data** → your dataset from step 1.
4. **Add models** (optional but recommended) → `mistral-ai/ministral-3`
   so the base weights come from Kaggle Models and you do not need a
   Hugging Face token.
5. Upload `notebooks/kaggle_train_ministral14b_cpt.py` or paste it.

If you uploaded the `.py` as a dataset file it will be under
`/kaggle/input/...`. Easiest path: paste the script into one cell, or:

```python
%run /kaggle/input/<your-dataset-slug>/notebooks/kaggle_train_ministral14b_cpt.py
```

(The GitHub importer includes the whole repo, so the script is in the
dataset if you used the GitHub URL.)

### What the script does

- Loads `input_ids.npy` + `train_offsets.npy` (no re-tokenization).
- Loads Ministral 3 14B Base in NF4 4-bit.
- Attaches LoRA (`r=16`) on `q/k/v/o/gate/up/down_proj`.
- Causal LM continued pretraining, 1 epoch, lr `1e-4`, batch 1 × accum 8.
- Writes the adapter to `/kaggle/working/ministral14b-cpt-lora`.

Useful env overrides:

```text
EPOCHS=2
MAX_STEPS=50          # smoke run
LORA_R=8              # if you OOM
MAX_SEQ_LEN=8192
LR=1e-4
```

A smoke run (`MAX_STEPS=20`) should finish in a few minutes and is the
right first check that VRAM holds at 8192.

### After training

`/kaggle/working/ministral14b-cpt-lora` is the PEFT adapter. Save it:

- **Save Version** → Advanced → always save output, or
- **File → Download**, or
- Create a new dataset from the notebook output.

Load later:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model = PeftModel.from_pretrained(base, "/kaggle/working/ministral14b-cpt-lora")
```

---

## 3. Things that will not work

| Idea | Why not |
| --- | --- |
| Full-weight 14B on Kaggle GPU | ~28 GB bf16, GPUs are 16 GB |
| Sequence 8192, batch > 1, no checkpointing | OOM |
| GitHub importer pointed at `main` | raw books, no `input_ids.npy` |
| Mixing GitHub + local files in one dataset | Kaggle forbids mixed sources |
| Private GitHub repo via the GitHub connector | Kaggle cannot clone it |

---

## 4. Rebuild the package after new books

```bash
python -m writing_dataset.cli \
  --config configs/ministral3-14b-base-8192.yaml \
  --raw-root data/raw --out-root data/out --emit-tokenized
python scripts/package_kaggle_dataset.py
git add kaggle_dataset && git commit && git push
```

Then hit **Update** on the Kaggle dataset (or wait for the scheduled sync).
