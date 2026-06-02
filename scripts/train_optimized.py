#!/usr/bin/env python3
"""
Entraînement Mina Translator avec boucle optimisée
Résout les problèmes de stability du Trainer HuggingFace
"""
import gc
import os
import sys
import time
import json
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

# Configuration
CONFIG = {
    "base_model": "Qwen/Qwen2-0.5B-Instruct",
    "epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "grad_accum": 4,
    "max_len": 128,
    "lora_r": 8,
    "lora_alpha": 16,
}

OUTPUT_DIR = Path("models/mina-translator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.add(OUTPUT_DIR / "training.log", rotation="10 MB", level="INFO")

def check_gpu():
    """Vérifier le GPU"""
    if not torch.cuda.is_available():
        logger.error("Aucun GPU détecté!")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Mémoire: {gpu_mem:.1f} GB")
    return True

def load_corpus():
    """Charger le corpus Mina"""
    from datasets import Dataset

    # Chemins possibles pour le corpus
    possible_paths = [
        Path("data/corpus/dataset_mina.jsonl"),
        Path("data/corpus/corpus_mina_5000.jsonl"),
        Path("data/mina_translation_corpus.jsonl"),
    ]

    corpus_file = None
    for path in possible_paths:
        if path.exists():
            corpus_file = path
            break

    if corpus_file is None:
        logger.error(f"Fichier corpus introuvable. Cherché: {possible_paths}")
        sys.exit(1)

    logger.info(f"Chargement du corpus: {corpus_file}")

    data = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # Format: {"instruction": ..., "input": ..., "output": ...}
                    if "input" in item and "output" in item:
                        text = f"Français: {item['input']}\nMina: {item['output']}"
                        data.append({"text": text})
                except json.JSONDecodeError:
                    continue

    logger.info(f"Corpus chargé: {len(data)} exemples")
    return Dataset.from_list(data)

def setup_model():
    """Charger et configurer le modèle"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Chargement du modèle: {CONFIG['base_model']}")

    # Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Modèle: {params:.1f}M paramètres")

    # Appliquer LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Trainable (LoRA): {trainable:.1f}M")

    return model, tokenizer

def tokenize_batch(texts, tokenizer, max_len=128):
    """Tokeniser un batch de textes"""
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt"
    )
    return encodings

def training_loop(model, tokenizer, dataset):
    """Boucle d'entraînement optimisée"""
    from torch.optim import AdamW

    device = "cuda"
    model.train()

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01,
    )

    # Scheduler
    total_steps = CONFIG["epochs"] * len(dataset) // CONFIG["batch_size"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info(f"Dataset: {len(dataset)} exemples")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info(f"Batch size: {CONFIG['batch_size']}")
    logger.info(f"Gradient accumulation: {CONFIG['grad_accum']}")
    logger.info(f"Steps total: {total_steps}")
    logger.info("=" * 60)

    texts = [item["text"] for item in dataset]
    global_step = 0

    for epoch in range(CONFIG["epochs"]):
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch + 1}/{CONFIG['epochs']}")
        logger.info(f"{'='*40}")

        # Shuffle data
        indices = torch.randperm(len(texts))
        epoch_loss = 0.0
        optimizer.zero_grad()

        # Process batches
        batch_start = time.time()
        for i in tqdm(range(0, len(texts), CONFIG["batch_size"]), desc=f"Epoch {epoch+1}"):
            batch_texts = [texts[idx] for idx in indices[i:i + CONFIG["batch_size"]]]

            # Tokenize
            encodings = tokenize_batch(batch_texts, tokenizer, CONFIG["max_len"])
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # Causal LM
            )

            loss = outputs.loss / CONFIG["grad_accum"]
            epoch_loss += loss.item() * CONFIG["grad_accum"]

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (i + CONFIG["batch_size"]) % CONFIG["grad_accum"] == 0 or i + CONFIG["batch_size"] >= len(texts):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    elapsed = time.time() - batch_start
                    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    logger.info(
                        f"Step {global_step}: loss={loss.item()*CONFIG['grad_accum']:.4f} "
                        f"| mem={mem:.1f}GB | lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        # Fin epoch
        avg_loss = epoch_loss / (len(texts) / CONFIG["batch_size"])
        logger.info(f"Epoch {epoch + 1} terminé - Loss moyen: {avg_loss:.4f}")

        # Sauvegarder checkpoint
        checkpoint_dir = OUTPUT_DIR / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint: {checkpoint_dir}")

        # Reset epoch stats
        gc.collect()
        torch.cuda.empty_cache()

    # Sauvegarder modèle final
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"ENTRAÎNEMENT TERMINÉ!")
    logger.info(f"Modèle sauvegardé: {final_dir}")
    logger.info(f"{'='*60}")

    return model, tokenizer

def main():
    logger.info("=" * 60)
    logger.info("MINA TRANSLATOR - ENTRAÎNEMENT")
    logger.info("=" * 60)

    # Vérifier GPU
    if not check_gpu():
        logger.error("GPU requis pour l'entraînement!")
        return 1

    # Charger corpus
    dataset = load_corpus()

    # Charger modèle
    model, tokenizer = setup_model()

    # Entraînement
    model, tokenizer = training_loop(model, tokenizer, dataset)

    return 0

if __name__ == "__main__":
    sys.exit(main())