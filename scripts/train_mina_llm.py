"""
scripts/train_mina_llm.py - Fine-tune Qwen2 sur le corpus Mina-Français
========================================================================

Fine-tune Qwen2-0.5B-Instruct avec QLoRA sur le corpus parallèle Mina→Français.

Usage:
    python scripts/train_mina_llm.py --sample 500      # Test avec 500 phrases
    python scripts/train_mina_llm.py --epochs 3       # Entraînement complet
    python scripts/train_mina_llm.py --dry-run        # Test 50 steps

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import os
import sys
import json
import gc
import io
from pathlib import Path
from datetime import datetime

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset as HFDataset

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mina-translator-trained"

# Use merged corpus if it exists, otherwise fall back to parallel_corpus
MERGED_CORPUS = PROJECT_ROOT / "data" / "corpus" / "corpus_merged.jsonl"
PARALLEL_CORPUS = PROJECT_ROOT / "data" / "corpus" / "parallel_corpus.jsonl"
CORPUS_FILE = MERGED_CORPUS if MERGED_CORPUS.exists() else PARALLEL_CORPUS

# Base model
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# Training config - optimisé pour RTX 4060 (8GB VRAM)
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "batch_size": 2,
    "gradient_accumulation": 4,  # effective batch = 8
    "max_seq_length": 256,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "default_epochs": 5,  # Better convergence with more epochs
}


# =============================================================================
# DATASET
# =============================================================================

class MinaTranslationDataset(Dataset):
    """Dataset pour l'entraînement translation Mina↔Français"""

    def __init__(self, corpus_file: Path, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Charger le corpus
        self.data = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("french") and len(entry["french"]) > 1:
                    self.data.append(entry)

        print(f"Dataset: {len(self.data)} paires Mina↔Français")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Format instruction-tuning
        texts = [
            # Mina -> Français
            {
                "instruction": "Traduis cette phrase de l'Ewe (Mina du Togo) en Français.",
                "input": entry["mina"],
                "output": entry["french"],
            },
            # Français -> Mina
            {
                "instruction": "Traduis cette phrase du Français en Ewe (Mina du Togo).",
                "input": entry["french"],
                "output": entry["mina"],
            },
        ]

        # Choisir aléatoirement la direction
        import random
        sample = random.choice(texts)

        # Construire le prompt
        prompt = f"<|im_start|>user\n{sample['instruction']}\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n{sample['output']}<|im_end|>"

        # Tokeniser
        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        # Labels = input_ids (le modèle prédit le prochain token)
        labels = tokens["input_ids"].copy()

        # Masker le prompt (ne pas prédire l'instruction, seulement la réponse)
        prompt_len = len(self.tokenizer(
            f"<|im_start|>user\n{sample['instruction']}\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n",
            truncation=True,
            max_length=self.max_length,
        )["input_ids"])

        # Les tokens du prompt sont masqués
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(tokens["input_ids"]),
            "attention_mask": torch.tensor(tokens["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def create_training_dataset(corpus_file: Path, tokenizer, max_length: int = 256):
    """Crée un dataset HuggingFace depuis le corpus"""

    data = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            # Handle both "french" and "fr" keys
            french_text = entry.get("french") or entry.get("fr", "")
            mina_text = entry.get("mina", "")
            if french_text and len(french_text) > 1 and mina_text:
                data.append({"french": french_text, "mina": mina_text})

    print(f"Dataset: {len(data)} paires Mina↔Français")

    # Créer des exemples bidirectionnels
    examples = []
    for entry in data:
        # Mina -> Français
        examples.append({
            "text": f"<|im_start|>user\nTraduis en Français: {entry['mina']}<|im_end|>\n<|im_start|>assistant\n{entry['french']}<|im_end|>"
        })
        # Français -> Mina
        examples.append({
            "text": f"<|im_start|>user\nTraduis en Ewe: {entry['french']}<|im_end|>\n<|im_start|>assistant\n{entry['mina']}<|im_end|>"
        })

    # Tokenizer function
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    # Créer le dataset HF
    dataset = HFDataset.from_list(examples)
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format("torch")

    return dataset


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    corpus_file: Path = CORPUS_FILE,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_steps: int = -1,
    dry_run: bool = False,
):
    """Fine-tune Qwen2 sur le corpus Mina"""

    print("=" * 60)
    print("FINE-TUNING QWEN2-0.5B FOR MINA TRANSLATION")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Corpus: {corpus_file}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Vérifier que le corpus existe
    if not corpus_file.exists():
        print(f"ERREUR: Corpus non trouve: {corpus_file}")
        print("Lance d'abord: python scripts/build_parallel_corpus.py --sample 500")
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory:.1f} GB")

    # Charger le tokenizer
    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Charger le modèle avec QLoRA
    print("\nChargement du modele avec QLoRA...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # Préparer pour QLoRA
    model = prepare_model_for_kbit_training(model)

    # Config LoRA
    lora_config = LoraConfig(
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Créer le dataset
    print("\nCreation du dataset...")
    dataset = create_training_dataset(
        corpus_file,
        tokenizer,
        max_length=TRAINING_CONFIG["max_seq_length"],
    )

    # Split train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Training arguments
    output_dir = str(MODEL_DIR)

    # Optimizer: paged_adamw_8bit sur GPU, adamw_torch sur CPU
    if device == "cuda":
        optim_str = "paged_adamw_8bit"
    else:
        optim_str = "adamw_torch"  # CPU-compatible

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        warmup_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        bf16=device == "cuda",  # RTX 4060 supports bf16, but CPU does not
        fp16=device == "cpu",   # Use fp16 on CPU if bf16 not available
        gradient_checkpointing=True,
        optim=optim_str,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Entraîner
    print("\n" + "=" * 60)
    print("DEBUT DE L'ENTRAINEMENT")
    print("=" * 60)

    trainer.train()

    # Sauvegarder
    print("\nSauvegarde du modele...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("\n" + "=" * 60)
    print("ENTRAINEMENT TERMINE!")
    print(f"Modele sauvegarde: {MODEL_DIR}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2 for Mina translation")

    parser.add_argument("--corpus", type=str, help="Chemin vers le corpus JSONL")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["default_epochs"], help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max steps (-1 = based on epochs)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (50 steps)")
    parser.add_argument("--output", type=str, help="Dossier de sortie")

    args = parser.parse_args()

    corpus_file = Path(args.corpus) if args.corpus else CORPUS_FILE
    output_dir = Path(args.output) if args.output else None

    if args.dry_run:
        print("MODE DRY-RUN: 50 steps")
        train_model(
            corpus_file=corpus_file,
            epochs=1,
            batch_size=args.batch_size,
            max_steps=50,
        )
    else:
        train_model(
            corpus_file=corpus_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()