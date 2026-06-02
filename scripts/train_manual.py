#!/usr/bin/env python3
"""
Entraînement manuel (sans Trainer) pour Mina Translator
Évite les bugs potentiels du Trainer HuggingFace
"""
import gc
import os
import sys
import torch
from pathlib import Path
from loguru import logger

# Configuration
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LEN = 128
OUTPUT_DIR = Path("models/mina-translator/manual")

# Logging setup
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.add(OUTPUT_DIR / "training.log", rotation="10 MB", level="INFO")

def check_gpu():
    """Vérifier le GPU"""
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    logger.warning("Aucun GPU détecté - entraînement sur CPU (lent)")
    return False

def load_corpus():
    """Charger le corpus Mina"""
    from datasets import Dataset
    from pathlib import Path

    corpus_file = Path("data/mina_translation_corpus.jsonl")
    if not corpus_file.exists():
        logger.error(f"Fichier corpus introuvable: {corpus_file}")
        sys.exit(1)

    import json
    data = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
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

    logger.info(f"Chargement du modèle: {BASE_MODEL}")

    # Nettoyer la mémoire
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Charger tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important pour génération

    # Charger modèle
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Modèle chargé: {params:.1f}M paramètres")

    # Appliquer LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
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
    logger.info(f"Paramètres entraînables (LoRA): {trainable:.1f}M ({trainable/params*100:.2f}%)")

    return model, tokenizer

def tokenize_data(dataset, tokenizer):
    """Tokeniser le dataset"""
    def preprocess(batch):
        # Encoder le texte
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors=None,
        )
        # Labels = mêmes que input_ids (pour language modeling)
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info("Tokenisation du dataset...")
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenisation"
    )

    logger.info(f"Dataset tokenisé: {len(tokenized)} exemples")
    return tokenized

def manual_train_loop(model, tokenizer, dataset, use_gpu=True):
    """Boucle d'entraînement manuelle"""
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT MANUEL")
    logger.info("=" * 60)

    device = "cuda" if use_gpu else "cpu"

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.tensor([d["input_ids"] for d in x], dtype=torch.long),
            "attention_mask": torch.tensor([d["attention_mask"] for d in x], dtype=torch.long),
            "labels": torch.tensor([d["labels"] for d in x], dtype=torch.long),
        }
    )

    model.train()
    global_step = 0
    total_loss = 0

    for epoch in range(EPOCHS):
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch + 1}/{EPOCHS}")
        logger.info(f"{'='*40}")

        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            # Déplacer sur device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            epoch_loss += loss.item() * GRAD_ACCUM

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, "
                        f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB"
                    )

        # Fin d'epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} terminé - loss={avg_epoch_loss:.4f}")

        # Sauvegarder checkpoint
        checkpoint_dir = OUTPUT_DIR / f"checkpoint-epoch-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint sauvegardé: {checkpoint_dir}")

    # Sauvegarder le modèle final
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"\n{'='*60}")
    logger.info(f"ENTRAÎNEMENT TERMINÉ - Modèle sauvegardé dans {final_dir}")
    logger.info(f"{'='*60}")

    return model, tokenizer

def main():
    """Point d'entrée principal"""
    logger.info("=" * 60)
    logger.info("MINA TRANSLATOR - ENTRAÎNEMENT MANUEL")
    logger.info("=" * 60)

    # Vérifier GPU
    use_gpu = check_gpu()

    # Charger corpus
    dataset = load_corpus()

    # Charger modèle
    model, tokenizer = setup_model()

    # Tokeniser
    tokenized = tokenize_data(dataset, tokenizer)

    # Entraînement manuel
    model, tokenizer = manual_train_loop(model, tokenizer, tokenized, use_gpu)

    return 0

if __name__ == "__main__":
    sys.exit(main())
