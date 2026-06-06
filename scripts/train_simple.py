#!/usr/bin/env python
"""
scripts/train_simple.py - Fine-tuning simple et robuste pour Mina-Translator
=============================================================================

Version simplifiée sans gradient checkpointing pour éviter les segfaults.
Compatible RTX 4060 (8 Go VRAM).

Usage:
    py scripts/train_simple.py              # Dry run (50 steps)
    py scripts/train_simple.py --full       # Entraînement complet
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # Plus petit, plus stable
DATASET_PATH = PROJECT_ROOT / "data" / "corpus" / "dataset_mina.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mina-translator"

# =============================================================================
# DATASET
# =============================================================================

class MinaTranslationDataset(Dataset):
    """Dataset pour la traduction FR -> Mina"""

    def __init__(self, data_path: Path, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Charger les données
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        print(f"Dataset chargé: {len(self.data)} exemples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        instruction = sample.get("instruction", "Traduis du français vers le mina.")
        french = sample.get("input", "")
        mina = sample.get("output", "")

        # Format du prompt
        text = f"{instruction}\n\nFrançais: {french}\n\nMina: {mina}"

        # Tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


# =============================================================================
# TRAINER
# =============================================================================

def train_model(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 50,
    learning_rate: float = 2e-4,
    seed: int = 42,
):
    """Fine-tune le modèle avec LoRA"""

    set_seed(seed)

    print("=" * 60)
    print("FINE-TUNING MINA TRANSLATOR (Qwen2-0.5B)")
    print("=" * 60)
    print(f"Modèle: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Steps: {max_steps}")
    print("=" * 60)

    # CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go")

    # Charger tokenizer
    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Charger modèle (float32 pour plus de stabilité)
    print("Chargement du modèle (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Afficher les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Config LoRA
    print("\nConfiguration LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Créer dataset
    print("\nPréparation du dataset...")
    dataset = MinaTranslationDataset(dataset_path, tokenizer, max_length=128)

    # Data collator simple
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Arguments d'entraînement
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=25,
        fp16=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Eviter les problèmes Windows
    )

    # Créer trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Entraîner
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60 + "\n")

    trainer.train()

    # Sauvegarder
    print("\nSauvegarde du modèle...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print(f"\n[OK] Model saved: {output_dir / 'final'}")

    # Test rapide
    test_translation(model, tokenizer)

    return model, tokenizer


def test_translation(model, tokenizer):
    """Test rapide du modèle"""
    print("\n" + "=" * 60)
    print("TEST DE TRADUCTION")
    print("=" * 60)

    model.eval()

    test_phrases = [
        "La santé est importante.",
        "Je suis malade.",
    ]

    for phrase in test_phrases:
        prompt = f"Traduis du français vers le mina.\n\nFrançais: {phrase}\n\nMina:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = result.split("Mina:")[-1].strip()

        print(f"FR: {phrase}")
        print(f"MI: {translation}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.full:
        args.steps = 500

    train_model(
        model_name=args.model,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        max_steps=args.steps,
        learning_rate=args.lr,
    )