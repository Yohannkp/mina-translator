#!/usr/bin/env python3
"""
Minimal training loop - step by step to avoid segfault
"""
import gc
import json
import time
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from loguru import logger
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LEN = 128
LEARNING_RATE = 2e-4
LORA_R = 8

# Output
OUTPUT_DIR = Path("models/mina-translator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.add(OUTPUT_DIR / "training.log", rotation="10 MB")

print("=" * 60)
print("MINA TRANSLATOR - TRAINING")
print("=" * 60)

# Step 1: Check GPU
print("\n[1/6] Checking GPU...")
if not torch.cuda.is_available():
    print("ERROR: No CUDA!")
    sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Step 2: Load corpus
print("\n[2/6] Loading corpus...")
corpus_file = Path("c:/Ce PC/Projet_python/IA traduction Français Mina/data/corpus/dataset_mina.jsonl")
data = []
with open(corpus_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            item = json.loads(line)
            if "input" in item and "output" in item:
                text = f"Français: {item['input']}\nMina: {item['output']}"
                data.append(text)

print(f"Loaded {len(data)} examples")

# Step 3: Load tokenizer
print("\n[3/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Tokenizer ready")

# Step 4: Load model
print("\n[4/6] Loading model...")
gc.collect()
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model loaded: {total_params:.1f}M params")

# Step 5: Apply LoRA
print("\n[5/6] Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_R * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
frozen_params = total_params - trainable_params
print(f"Trainable: {trainable_params:.2f}M | Frozen: {frozen_params:.2f}M")

# Step 6: Training loop
print("\n[6/6] Starting training...")
print("=" * 60)

device = "cuda"
model.train()

# Optimizer
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

# Scheduler
num_training_steps = EPOCHS * len(data) // BATCH_SIZE
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(num_training_steps * 0.1),
    num_training_steps=num_training_steps,
)

global_step = 0

for epoch in range(EPOCHS):
    print(f"\n--- EPOCH {epoch + 1}/{EPOCHS} ---")
    epoch_loss = 0.0
    optimizer.zero_grad()

    # Shuffle
    indices = torch.randperm(len(data))

    pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch+1}")
    for i in pbar:
        # Get batch
        batch_texts = [data[idx] for idx in indices[i:i + BATCH_SIZE]]

        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss / GRAD_ACCUM
        epoch_loss += loss.item() * GRAD_ACCUM

        # Backward
        loss.backward()

        # Step
        if (i // BATCH_SIZE + 1) % GRAD_ACCUM == 0 or i + BATCH_SIZE >= len(data):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        # Progress
        pbar.set_postfix({
            "loss": f"{loss.item() * GRAD_ACCUM:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    # Epoch summary
    avg_loss = epoch_loss / (len(data) / BATCH_SIZE)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_dir = OUTPUT_DIR / f"checkpoint-epoch-{epoch+1}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved: {ckpt_dir}")

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

# Final save
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

final_dir = OUTPUT_DIR / "final"
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Model saved: {final_dir}")