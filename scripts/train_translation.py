"""
Script de Fine-tuning QLoRA pour traduction Mina-Français
=========================================================

Usage:
    python scripts/train_translation.py

Configuration pour RTX 4060 (8Go VRAM):
    - QLoRA 4-bit avec NF4
    - Gradient checkpointing
    - Batch size 4, gradient accumulation 4
    - LoRA r=16, alpha=32
"""
import os
import sys
import json
import torch
from pathlib import Path
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
settings = get_settings()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Modèle de base - Qwen2-0.5B (ouvert, testé, fonctionne sur RTX 4060)
# Alternative: TinyLlama-1.1B ou microsoft/TinyLlama-1.1B-Chat-v1.0
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# Chemins
DATASET_PATH = settings.CORPUS_DIR / "dataset_mina.jsonl"
OUTPUT_DIR = settings.MODELS_DIR / "mina-translator"

# Paramètres d'entraînement optimisés pour RTX 4060 (8Go VRAM)
TRAINING_CONFIG = {
    # Pas de quantification - Qwen2-0.5B tient en FP16 sur RTX 4060
    "use_quantization": False,

    # LoRA
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },

    # Training
    "training_arguments": {
        "output_dir": str(OUTPUT_DIR),
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,  # Stabilité maximale RTX 4060
        "gradient_accumulation_steps": 8,  # Effective batch = 8 (1*8)
        "gradient_checkpointing": True,
        "optim": "paged_adamw_32bit",
        "learning_rate": 2e-4,
        "weight_decay": 0.001,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "report_to": "none",
    },

    # Dataset
    "dataset_config": {
        "max_length": 256,
        "train_split": 0.9,
        "shuffle": True,
    }
}


def check_gpu():
    """Vérifie la disponibilité et les specs GPU"""
    if not torch.cuda.is_available():
        logger.error("CUDA non disponible!")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Mémoire: {gpu_memory:.1f} Go")

    if gpu_memory < 6:
        logger.warning("Mémoire GPU insuffisante pour QLoRA")

    return True


def format_instruction(example):
    """Formate un exemple pour l'instruction-tuning"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Format pour modèle Phi/Llama
    if "Phi" in BASE_MODEL:
        return f"<|user|>\n{instruction}: {input_text}\n<|assistant|>\n{output}<|end|>"
    else:
        return f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"


def load_dataset():
    """Charge et formate le dataset"""
    if not DATASET_PATH.exists():
        logger.error(f"Dataset non trouvé: {DATASET_PATH}")
        logger.info("Lance d'abord: python scripts/generate_dataset.py")
        return None

    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    logger.info(f"Dataset chargé: {len(data)} exemples")

    # Formater les exemples
    formatted = []
    for item in data:
        formatted.append({
            "text": format_instruction(item)
        })

    return formatted


def train():
    """Lance le fine-tuning"""

    logger.info("=" * 60)
    logger.info("FINE-TUNING QLoRA MINA-TRANSLATOR")
    logger.info("=" * 60)

    # Vérifier GPU
    if not check_gpu():
        return

    # Vérifier dataset
    dataset = load_dataset()
    if not dataset:
        return

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset

        logger.info(f"Modèle: {BASE_MODEL}")

        # Nettoyer la mémoire GPU avant de charger le modèle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Chargement modèle SANS quantification (Qwen2-0.5B tient en FP16)
        logger.info("Chargement du modèle (FP16, plus stable)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Modèle chargé: {sum(p.numel() for p in model.parameters())/1e6:.1f}M paramètres")

        # Configuration LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Appliquer LoRA
        logger.info("Configuration LoRA...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Préparer dataset
        ds = Dataset.from_list(dataset)

        def tokenize(batch):
            result = tokenizer(
                batch["text"],
                truncation=True,
                max_length=256,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result

        logger.info("Tokenisation...")
        ds = ds.map(tokenize, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Stabilité maximale
            gradient_accumulation_steps=8,  # Effective batch = 8
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            data_collator=data_collator,
        )

        # Nettoyer le cache GPU avant l'entraînement
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Entraînement
        logger.info("=" * 60)
        logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT")
        logger.info("=" * 60)
        logger.info("Cela peut prendre 30-90 minutes selon le modèle")
        logger.info("Appuyez sur Ctrl+C pour arrêter")

        trainer.train()

        # Sauvegarde
        logger.info("Sauvegarde du modèle...")
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)

        logger.info("=" * 60)
        logger.info("FINE-TUNING TERMINÉ!")
        logger.info(f"Modèle sauvegardé dans: {OUTPUT_DIR}")
        logger.info("=" * 60)

        # Test rapide
        logger.info("\nTest du modèle...")
        test_phrase = "Bonjour"
        inputs = tokenizer(f"Traduis en mina: {test_phrase}", return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test - Input: {test_phrase}")
        logger.info(f"Test - Output: {result}")

    except ImportError as e:
        logger.error(f"Bibliothèque manquante: {e}")
        logger.info("\nInstalle les dépendances:")
        logger.info("  pip install transformers peft bitsandbytes accelerate datasets")
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train()