#!/usr/bin/env python3
"""Continued-pretraining LoRA for Ministral-3-14B-Base-2512 on Kaggle.

Hardware this is sized for
--------------------------
Kaggle GPU **T4 x2** (2x16 GB) is the comfortable choice. A single P100/T4
16 GB can work at batch=1 + 4-bit + gradient checkpointing, but 8192-token
sequences leave little headroom.

The dataset is already packed as ``[BOS] content [EOS]`` in ``input_ids.npy``.
This script does **not** re-tokenize, so it does not need the Tekken file.

Run on Kaggle
-------------
1. New Notebook → GPU T4 x2 → Internet ON.
2. Add Data → the GitHub-imported dataset.
3. Add Models → ``mistral-ai/ministral-3`` (14B Base) if you want kagglehub.
4. File → Upload this script, or paste it into a cell and run.

Environment knobs (optional)::

    DATA_DIR, MODEL_ID, OUTPUT_DIR, MAX_SEQ_LEN, LORA_R, EPOCHS,
    BATCH_SIZE, GRAD_ACCUM, LR, MAX_STEPS
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "8192"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
EPOCHS = float(os.environ.get("EPOCHS", "1"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))  # 0 = full epoch(s)
SEED = int(os.environ.get("SEED", "3407"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/ministral14b-cpt-lora"))
BOS_ID, EOS_ID, PAD_ID = 1, 2, 11

KAGGLE_MODEL = "mistral-ai/ministral-3/Transformers/ministral-3-14b-base-2512"
HF_MODEL = "mistralai/Ministral-3-14B-Base-2512"


def _is_kaggle() -> bool:
    return Path("/kaggle").exists()


def find_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        p = Path(env)
        if (p / "input_ids.npy").exists() or (p / "train.jsonl").exists():
            return p
    candidates = []
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").rglob("input_ids.npy"))
        candidates.extend(Path("/kaggle/input").rglob("train.jsonl"))
    candidates.extend(Path("kaggle_dataset").glob("input_ids.npy"))
    candidates.extend(Path("data/out").rglob("input_ids.npy"))
    for hit in candidates:
        return hit.parent
    raise FileNotFoundError(
        "Could not find input_ids.npy or train.jsonl. "
        "Add the dataset to the notebook (Add Data) or set DATA_DIR."
    )


def find_model_dir() -> str:
    env = os.environ.get("MODEL_ID")
    if env:
        return env
    # Kaggle Models via kagglehub (no Hugging Face token).
    try:
        import kagglehub

        print("downloading base model via kagglehub…")
        return kagglehub.model_download(KAGGLE_MODEL)
    except Exception as exc:
        print(f"kagglehub unavailable ({type(exc).__name__}: {exc}); falling back to {HF_MODEL}")
        return HF_MODEL


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def load_packed_dataset(data_dir: Path, split: str):
    import numpy as np
    from torch.utils.data import Dataset

    ids_path = data_dir / "input_ids.npy"
    off_path = data_dir / f"{split}_offsets.npy"
    if not off_path.exists():
        off_path = data_dir / "offsets.npy"
    if not ids_path.exists():
        raise FileNotFoundError(f"{ids_path} missing — rebuild with --emit-tokenized")

    input_ids = np.load(ids_path, mmap_mode="r")
    offsets = np.load(off_path)

    class PackedChunks(Dataset):
        def __len__(self):
            return int(offsets.shape[0])

        def __getitem__(self, i):
            start, end, length = (int(x) for x in offsets[i])
            if length > MAX_SEQ_LEN:
                end = start + MAX_SEQ_LEN
                length = MAX_SEQ_LEN
            ids = np.asarray(input_ids[start:end], dtype=np.int64)
            return {
                "input_ids": ids,
                "labels": ids.copy(),
                "attention_mask": np.ones(length, dtype=np.int64),
            }

    return PackedChunks()


def make_collator():
    import torch

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        # pad to multiple of 8 for tensor cores
        pad_to = ((max_len + 7) // 8) * 8
        bsz = len(features)
        input_ids = torch.full((bsz, pad_to), PAD_ID, dtype=torch.long)
        labels = torch.full((bsz, pad_to), -100, dtype=torch.long)
        attention_mask = torch.zeros((bsz, pad_to), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.as_tensor(f["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.as_tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_model_and_peft(model_id: str):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    # Ministral 3 is registered as Mistral3ForConditionalGeneration. Auto*
    # resolves it once transformers>=5.0.0rc0 is installed.
    print(f"loading {model_id} 4-bit ({dtype})…")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, bf16


def main() -> int:
    if _is_kaggle():
        # Kaggle images ship an older transformers. Ministral 3 needs v5.
        os.system(
            f"{sys.executable} -m pip install -q -U "
            "'transformers>=4.57.0' peft bitsandbytes accelerate "
            "'mistral-common>=1.8.6' trl"
        )

    import torch
    from transformers import Trainer, TrainingArguments

    data_dir = find_data_dir()
    print("data_dir =", data_dir)
    if (data_dir / "manifest.json").exists():
        print(json.dumps(json.loads((data_dir / "manifest.json").read_text()), indent=2))

    train_ds = load_packed_dataset(data_dir, "train")
    val_ds = load_packed_dataset(data_dir, "val")
    print(f"train chunks={len(train_ds)}  val chunks={len(val_ds)}")

    model_id = find_model_dir()
    model, bf16 = load_model_and_peft(model_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps" if len(val_ds) else "no",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) else None,
        data_collator=make_collator(),
    )

    print(
        f"start CPT  seq={MAX_SEQ_LEN}  r={LORA_R}  "
        f"bs={BATCH_SIZE}x{GRAD_ACCUM}  epochs={EPOCHS}"
    )
    t0 = time.time()
    trainer.train()
    print(f"train wall {time.time() - t0:.1f}s")

    trainer.save_model(str(OUTPUT_DIR))
    (OUTPUT_DIR / "cpt_config.json").write_text(
        json.dumps(
            {
                "base_model": str(model_id),
                "max_seq_len": MAX_SEQ_LEN,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lr": LR,
                "epochs": EPOCHS,
                "data_dir": str(data_dir),
                "bos_id": BOS_ID,
                "eos_id": EOS_ID,
                "pad_id": PAD_ID,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("saved adapter to", OUTPUT_DIR)

    # Tiny smoke generate so you can see the adapter loaded.
    if torch.cuda.is_available():
        model.eval()
        prompt = torch.tensor([[BOS_ID]], device=model.device)
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=40, do_sample=True, temperature=0.8)
        print("sample ids", out[0].tolist()[:48])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
