#!/usr/bin/env python
"""
scripts/train_mini_llm.py - Fine-tuning QLoRA pour Mina-Translator
==================================================================

Script d'entraînement optimisé pour RTX 4060 (8 Go VRAM).
Utilise QLoRA 4-bit pour fine-tuner un petit LLM sur les données Mina.

Compatible avec: Phi-3-mini, Llama-3.2-1B, Qwen2-0.5B

Usage:
    python scripts/train_mini_llm.py                    # Dry run (50 steps)
    python scripts/train_mini_llm.py --full            # Entraînement complet
    python scripts/train_mini_llm.py --model phi3-mini  # Utiliser Phi-3-mini

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Modules PyTorch
import torch
from torch.utils.data import DataLoader

# Transformers & PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import BitsAndBytesConfig

# Logging
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = PROJECT_ROOT / "data" / "corpus" / "dataset_mina.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mina-translator"

# Config LoRA optimisée pour 8 Go VRAM
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
}

# Config entraînement optimisée
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 10,
    "max_steps": 50,  # dry-run
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 25,
    "output_dir": str(OUTPUT_DIR),
    "report_to": "none",
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "seed": 42,
    "remove_unused_columns": False,
}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    """Configure le logging avec loguru"""
    logger.remove()

    log_file = PROJECT_ROOT / "logs" / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    return logger


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_dataset(path: Path) -> list:
    """
    Charge le dataset depuis le fichier JSONL

    Format attendu:
    {
        "instruction": "Traduis en mina...",
        "input": "phrase français",
        "output": "phrase mina"
    }
    """
    logger.info(f"Chargement du dataset: {path}")

    if not path.exists():
        logger.error(f"Dataset non trouvé: {path}")
        sys.exit(1)

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info(f"Dataset chargé: {len(data)} exemples")
    return data


def format_instruction(sample: dict) -> str:
    """Formate un exemple pour l'entraînement"""
    instruction = sample.get("instruction", "Traduis cette phrase du français vers le mina.")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    # Format pour modèle conversationnel
    return f"{instruction}\n\nFrançais: {input_text}\n\nMina: {output_text}"


def tokenize_function(examples, tokenizer, max_length: int = 512):
    """Tokenise les exemples pour l'entraînement"""
    # Concaténer instruction + input + output pour former le texte complet
    texts = []

    for i in range(len(examples["input"])):
        instruction = examples.get("instruction", ["Traduis cette phrase du français vers le mina."])[i]
        input_text = examples["input"][i]
        output_text = examples["output"][i]

        text = f"{instruction}\n\nFrançais: {input_text}\n\nMina: {output_text}"
        texts.append(text)

    # Tokenizer avec truncation
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Pour CAusal LM: les labels sont les mêmes que les input_ids,
    # mais on masque les tokens d'entrée
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# =============================================================================
# CRÉATION DU MODÈLE AVEC QLoRA
# =============================================================================

def load_model_and_tokenizer(
    model_name: str,
    use_quantization: bool = True,
    device: str = "cuda"
):
    """
    Charge le modèle avec QLoRA 4-bit

    Args:
        model_name: Nom HF du modèle
        use_quantization: Utiliser 4-bit quantization
        device: cuda/cpu

    Returns:
        Tuple (model, tokenizer)
    """
    logger.info("=" * 60)
    logger.info("CHARGEMENT DU MODÈLE")
    logger.info("=" * 60)
    logger.info(f"Modèle: {model_name}")

    # Vérifier CUDA
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA non disponible, utilisation CPU")
        device = "cpu"

    # Charger le tokenizer
    logger.info("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Config padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Config chargement
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if device == "cuda":
        # Utiliser FP16 pour de meilleures performances
        load_kwargs["torch_dtype"] = torch.float16
        logger.info("Mode: FP16")

        if use_quantization:
            logger.info("Quantization: 4-bit NF4")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    else:
        load_kwargs["torch_dtype"] = torch.float32

    # Charger le modèle
    logger.info("Chargement du modèle (patientez...)")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        logger.error(f"Erreur chargement: {e}")
        logger.info("Tentative sans quantification...")
        load_kwargs.pop("quantization_config", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # Préparer pour k-bit training
    model = prepare_model_for_kbit_training(model)

    # Appliquer LoRA
    logger.info("Configuration LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Info VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"VRAM: {allocated:.2f} Go allouée, {reserved:.2f} Go réservée")

    logger.info("=" * 60)
    logger.info("MODÈLE PRÊT")
    logger.info("=" * 60)

    return model, tokenizer


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train(
    model,
    tokenizer,
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 50,
    per_device_batch: int = 2,
    learning_rate: float = 2e-4,
    seed: int = 42,
):
    """
    Entraîne le modèle avec QLoRA

    Args:
        model: Modèle PEFT
        tokenizer: Tokenizer
        dataset_path: Chemin vers le dataset
        output_dir: Dossier de sortie
        max_steps: Nombre max de steps (50 = dry-run)
        per_device_batch: Batch size
        learning_rate: Taux d'apprentissage
        seed: Graine aléatoire
    """
    setup_logging()

    logger.info("=" * 60)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT")
    logger.info("=" * 60)
    logger.info(f"Steps: {max_steps}")
    logger.info(f"Batch size: {per_device_batch}")
    logger.info(f"Learning rate: {learning_rate}")

    set_seed(seed)

    # Charger dataset
    raw_data = load_dataset(dataset_path)

    # Transformer en format pour datasets
    from datasets import Dataset

    data_dict = {
        "instruction": [d.get("instruction", "") for d in raw_data],
        "input": [d.get("input", "") for d in raw_data],
        "output": [d.get("output", "") for d in raw_data],
    }

    dataset = Dataset.from_dict(data_dict)

    # Tokenizer function
    def tokenize_fn(examples):
        return tokenize_function(examples, tokenizer, max_length=512)

    # Tokenizer le dataset
    logger.info("Tokenization du dataset...")
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["instruction"],
        desc="Tokenization",
    )

    logger.info(f"Dataset tokenisé: {len(dataset)} exemples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )

    # Config entraînement
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=25,
        fp16=True,
        optim="paged_adamw_8bit",
        seed=seed,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Créer le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Lancer l'entraînement
    logger.info("")
    logger.info("▶ Lancement de l'entraînement...")
    logger.info("")

    try:
        trainer.train()

        # Sauvegarder le modèle
        logger.info("")
        logger.info("Sauvegarde du modèle...")

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info(f"✅ Modèle sauvegardé: {output_dir}")

        # Test d'inférence rapide
        test_inference(model, tokenizer)

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        raise


def test_inference(model, tokenizer, num_tests: int = 3):
    """Teste rapidement le modèle après entraînement"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST D'INFÉRENCE POST-ENTRAÎNEMENT")
    logger.info("=" * 60)

    test_phrases = [
        "Bonjour, comment allez-vous ?",
        "Je voudrais un rendez-vous.",
        "Où est la pharmacie ?",
    ]

    model.eval()

    for phrase in test_phrases[:num_tests]:
        prompt = f"Traduis cette phrase du français vers le mina.\n\nFrançais: {phrase}\n\nMina:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = response.split("Mina:")[-1].strip()

        logger.info(f"FR: {phrase}")
        logger.info(f"MI: {translation}")
        logger.info("")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA pour Mina-Translator")

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Modèle HuggingFace à fine-tuner",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Chemin vers le dataset JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Dossier de sortie du modèle",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Nombre de steps (50 = dry-run, 500+ = full)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size par GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Entraînement complet (1000 steps)",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Désactiver la quantification 4-bit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire",
    )

    args = parser.parse_args()

    # Mode full
    if args.full:
        args.steps = 1000

    # Charger modèle
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        use_quantization=not args.no_quant,
    )

    # Entraîner
    train(
        model=model,
        tokenizer=tokenizer,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output),
        max_steps=args.steps,
        per_device_batch=args.batch,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()