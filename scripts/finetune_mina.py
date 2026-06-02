"""
Script Complet de Fine-tuning QLoRA pour Mina-Translator
========================================================

Contexte machine:
- GPU: RTX 4060 Laptop (8 Go VRAM)
- RAM: 32 Go DDR5
- CPU: i7-13650HX (14 cœurs)
- Stockage: 1 To SSD NVMe

Modèles recommandés pour 8 Go VRAM:
- Phi-3-mini (3.8B params, optimal)
- Mistral-7B-Instruct (QLoRA 4-bit)
- Llama-3.2-3B-Instruct

Usage:
    python scripts/finetune_mina.py --model phi3-mini --epochs 3
    python scripts/finetune_mina.py --model llama3.2-3b --epochs 5 --early-stop
"""

import os
import sys
import json
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings

settings = get_settings()

# =============================================================================
# CONFIGURATION MODÈLES
# =============================================================================

MODELS_CONFIG = {
    "phi3-mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "params": "3.8B",
        "vram_4bit": "~4.5 Go",
        "recommended": True,
        "context_length": 4096,
        "special_tokens": {
            "user": "<|user|>",
            "assistant": "<|assistant|>",
            "end": "<|end|>",
        }
    },
    "llama3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "params": "1.3B",
        "vram_4bit": "~2.5 Go",
        "recommended": True,
        "context_length": 128K,
        "special_tokens": None  # Utilise le format standard
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "params": "7B",
        "vram_4bit": "~6.5 Go",
        "recommended": False,  # Risqué pour 8Go VRAM
        "context_length": 32768,
        "special_tokens": None
    },
    "llama3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "params": "3B",
        "vram_4bit": "~3.5 Go",
        "recommended": True,
        "context_length": 128K,
        "special_tokens": None
    },
}

# =============================================================================
# PARAMÈTRES QLoRA OPTIMISÉS POUR RTX 4060 (8Go VRAM)
# =============================================================================

DEFAULT_TRAINING_CONFIG = {
    # --- Quantification 4-bit ---
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",

    # --- LoRA ---
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",

    # --- Training ---
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "save_strategy": "steps",
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 0.3,

    # --- Dataset ---
    "max_length": 256,
    "train_split": 0.9,

    # --- Early Stopping ---
    "early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.01,

    # --- Évaluation ---
    "eval_steps": 100,
    "eval_batch_size": 8,
}

# =============================================================================
# ESTIMATION DURÉE PAR EPOCH (RTX 4060 Laptop)
# =============================================================================

EPOCH_ESTIMATIONS = {
    "phi3-mini": {
        "1000_samples": "12-18 min/epoch",
        "5000_samples": "60-90 min/epoch",
        "10000_samples": "120-180 min/epoch",
    },
    "llama3.2-1b": {
        "1000_samples": "8-12 min/epoch",
        "5000_samples": "40-60 min/epoch",
        "10000_samples": "80-120 min/epoch",
    },
    "llama3.2-3b": {
        "1000_samples": "15-22 min/epoch",
        "5000_samples": "75-110 min/epoch",
        "10000_samples": "150-220 min/epoch",
    },
    "mistral-7b": {
        "1000_samples": "25-40 min/epoch",
        "5000_samples": "125-200 min/epoch",
        "10000_samples": "250-400 min/epoch",
    },
}


def get_epoch_estimate(model_name: str, num_samples: int) -> str:
    """Estime la durée par epoch en fonction du modèle et nombre d'échantillons."""
    key = f"{num_samples}_samples"
    model_key = None

    for k in EPOCH_ESTIMATIONS:
        if k in model_name.lower():
            model_key = k
            break

    if model_key and key in EPOCH_ESTIMATIONS[model_key]:
        return EPOCH_ESTIMATIONS[model_key][key]
    return "Variable selon le modèle"


# =============================================================================
# CLASSE PRINCIPALE DE FINE-TUNING
# =============================================================================

class MinaFineTuner:
    """
    Classe complète pour le fine-tuning QLoRA du modèle Mina-Translator.
    """

    def __init__(
        self,
        model_key: str = "phi3-mini",
        config: Optional[Dict] = None,
        output_dir: Optional[Path] = None,
    ):
        self.model_key = model_key
        self.config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
        self.model_info = MODELS_CONFIG.get(model_key, MODELS_CONFIG["phi3-mini"])
        self.base_model = self.model_info["name"]

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = settings.MODELS_DIR / f"mina-translator-{model_key}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = None
        self.tokenizer = None
        self.model = None
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_steps": 0,
            "epoch_times": [],
            "best_loss": float("inf"),
            "early_stopped": False,
        }

    # ==========================================================================
    # VÉRIFICATIONS SYSTÈME
    # ==========================================================================

    def check_system(self) -> bool:
        """Vérifie la configuration système."""
        logger.info("=" * 60)
        logger.info("VÉRIFICATION SYSTÈME")
        logger.info("=" * 60)

        # CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA non disponible - GPU requis pour QLoRA")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"GPU: {gpu_name}")
        logger.info(f"VRAM: {gpu_memory:.1f} Go")
        logger.info(f"RAM: {os.cpu_count()} cœurs CPU")

        # Estimer la mémoire nécessaire
        estimated_vram = self._estimate_vram_usage()
        logger.info(f"VRAM estimée pour QLoRA: {estimated_vram:.1f} Go")

        if gpu_memory < estimated_vram:
            logger.warning(f"Attention: VRAM potentiellement insuffisante")
            logger.warning("Réduisez batch_size ou utilisez un modèle plus léger")

        return True

    def _estimate_vram_usage(self) -> float:
        """Estime la VRAM nécessaire selon le modèle."""
        params_map = {
            "phi3-mini": 4.5,
            "llama3.2-1b": 2.5,
            "llama3.2-3b": 3.5,
            "mistral-7b": 6.5,
        }
        base = params_map.get(self.model_key, 4.0)

        # Ajouter overhead pour batch et gradients
        batch_size = self.config.get("per_device_train_batch_size", 4)
        overhead = batch_size * 0.3

        return base + overhead + 1.0  # +1 Go pour activations

    # ==========================================================================
    # PRÉPARATION DATASET
    # ==========================================================================

    def load_and_prepare_dataset(self) -> "Dataset":
        """Charge et prépare le dataset pour l'instruction-tuning."""
        from datasets import Dataset

        dataset_path = settings.CORPUS_DIR / "dataset_mina.jsonl"

        if not dataset_path.exists():
            logger.error(f"Dataset non trouvé: {dataset_path}")
            logger.info("Lance d'abord: python scripts/generate_dataset.py")
            return None

        # Charger les données
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Filtrer les exemples validés
                if item.get("validated", False):
                    data.append(item)

        logger.info(f"Dataset chargé: {len(data)} exemples validés")

        # Formater pour instruction-tuning
        formatted = [self._format_instruction(item) for item in data]

        ds = Dataset.from_list(formatted)

        # Mélanger si configuré
        if self.config.get("shuffle", True):
            ds = ds.shuffle(seed=42)

        # Séparer train/eval
        train_split = self.config.get("train_split", 0.9)
        split_idx = int(len(ds) * train_split)

        train_ds = ds.select(range(split_idx))
        eval_ds = ds.select(range(split_idx, len(ds)))

        logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

        return {"train": train_ds, "eval": eval_ds, "full": ds}

    def _format_instruction(self, example: Dict) -> Dict:
        """Formate un exemple pour l'instruction-tuning selon le modèle."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        special_tokens = self.model_info.get("special_tokens")

        if special_tokens:
            # Format Phi-3
            text = (
                f"{special_tokens['user']}\n{instruction}: {input_text}\n"
                f"{special_tokens['assistant']}\n{output}{special_tokens['end']}"
            )
        else:
            # Format standard Llama/Mistral
            text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{instruction}: {input_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{output}<|eot_id|>"
            )

        return {"text": text}

    # ==========================================================================
    # CONFIGURATION QLoRA
    # ==========================================================================

    def setup_quantization(self):
        """Configure la quantification 4-bit."""
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=self.config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=self.config.get("bnb_4bit_compute_dtype", torch.float16),
            bnb_4bit_use_double_quant=self.config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=self.config.get("bnb_4bit_quant_type", "nf4"),
        )

    def setup_lora(self):
        """Configure les adapters LoRA."""
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            target_modules=self.config.get("target_modules"),
            bias=self.config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

    def load_model_and_tokenizer(self):
        """Charge le modèle et le tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("=" * 60)
        logger.info("CHARGEMENT DU MODÈLE")
        logger.info("=" * 60)
        logger.info(f"Modèle: {self.base_model}")
        logger.info(f"Paramètres: {self.model_info['params']}")

        # Quantification
        bnb_config = self.setup_quantization()

        # Chargement du modèle
        logger.info("Chargement avec QLoRA 4-bit...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Chargement tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Appliquer LoRA
        logger.info("Configuration des adapters LoRA...")
        lora_config = self.setup_lora()
        self.model = self.get_peft_model(self.model, lora_config)

        # Afficher les paramètres entraînables
        trainable, total = self._count_parameters()
        logger.info(f"Paramètres entraînables: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
                   f"({100*trainable/total:.2f}%)")

        return self.model, self.tokenizer

    def get_peft_model(self, model, lora_config):
        """Wrap pour peft avec fallback."""
        try:
            from peft import get_peft_model
            return get_peft_model(model, lora_config)
        except ImportError:
            logger.error("peft non installé: pip install peft")
            raise

    def _count_parameters(self) -> Tuple[int, int]:
        """Compte les paramètres entraînables vs total."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    # ==========================================================================
    # TOKENIZATION
    # ==========================================================================

    def tokenize_dataset(self, dataset: "Dataset") -> "Dataset":
        """Tokenise le dataset."""
        max_length = self.config.get("max_length", 256)

        def tokenize_fn(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            result["labels"] = result["input_ids"].copy()
            return result

        return dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenisation",
        )

    # ==========================================================================
    # CONFIGURATION TRAINING
    # ==========================================================================

    def setup_training_arguments(self, output_dir: Path):
        """Configure les arguments d'entraînement."""
        from transformers import TrainingArguments

        eval_strategy = "steps" if self.config.get("eval_steps") else "no"

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 8),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "paged_adamw_32bit"),
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.001),
            fp16=self.config.get("fp16", True),
            max_grad_norm=self.config.get("max_grad_norm", 0.3),
            logging_steps=self.config.get("logging_steps", 10),
            eval_steps=self.config.get("eval_steps", 100),
            save_steps=self.config.get("save_steps", 100),
            save_strategy=self.config.get("save_strategy", "steps"),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            report_to="none",
            load_best_model_at_end=self.config.get("early_stopping", False),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def setup_early_stopping_callback(self):
        """Configure l'early stopping."""
        from transformers import EarlyStoppingCallback

        return EarlyStoppingCallback(
            early_stopping_patience=self.config.get("early_stopping_patience", 3),
            early_stopping_threshold=self.config.get("early_stopping_threshold", 0.01),
        )

    # ==========================================================================
    # ENTRAÎNEMENT
    # ==========================================================================

    def train(self, dataset_dict: Dict) -> bool:
        """Lance l'entraînement complet."""
        from transformers import Trainer, DataCollatorForLanguageModeling

        self.training_stats["start_time"] = datetime.now()

        logger.info("=" * 60)
        logger.info("DÉMARRAGE DU FINE-TUNING")
        logger.info("=" * 60)
        logger.info(f"Modèle: {self.base_model}")
        logger.info(f"Epochs: {self.config['num_train_epochs']}")
        logger.info(f"Batch size: {self.config['per_device_train_batch_size']}")
        logger.info(f"Learning rate: {self.config['learning_rate']}")
        logger.info(f"LoRA r={self.config['lora_r']}, alpha={self.config['lora_alpha']}")

        # Estimer durée totale
        num_samples = len(dataset_dict["train"])
        epoch_estimate = get_epoch_estimate(self.base_model, num_samples)
        total_estimate_hours = (
            self.config["num_train_epochs"] * num_samples / 1000 * 0.25
        )  # Approximation
        logger.info(f"Estimation: {epoch_estimate} par epoch")

        # Préparer les données
        train_ds = self.tokenize_dataset(dataset_dict["train"])
        eval_ds = self.tokenize_dataset(dataset_dict["eval"]) if dataset_dict.get("eval") else None

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = self.setup_training_arguments(self.output_dir)

        # Callbacks
        callbacks = []
        if self.config.get("early_stopping", False) and eval_ds:
            callbacks.append(self.setup_early_stopping_callback())

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

        # Hook pour tracking
        self.trainer.add_callback(TrainingStatsCallback(self.training_stats))

        try:
            logger.info("=" * 60)
            logger.info("ENTRAÎNEMENT EN COURS...")
            logger.info("=" * 60)

            epoch_start = time.time()
            self.trainer.train()

            self.training_stats["end_time"] = datetime.now()
            self.training_stats["total_steps"] = self.trainer.state.global_step

            # Calculer durée par epoch
            total_time = (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds()
            num_epochs = self.config.get("num_train_epochs", 3)
            avg_epoch_time = total_time / num_epochs

            logger.info("=" * 60)
            logger.info("FINE-TUNING TERMINÉ")
            logger.info("=" * 60)
            logger.info(f"Durée totale: {total_time/60:.1f} minutes")
            logger.info(f"Moyenne par epoch: {avg_epoch_time/60:.1f} minutes")
            logger.info(f"Steps totaux: {self.training_stats['total_steps']}")

            return True

        except KeyboardInterrupt:
            logger.warning("Entraînement interrompu par l'utilisateur")
            self.training_stats["early_stopped"] = True
            return False
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ==========================================================================
    # SAUVEGARDE ET ÉVALUATION
    # ==========================================================================

    def save_model(self):
        """Sauvegarde le modèle fine-tuné."""
        logger.info("Sauvegarde du modèle...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder avec SafeTensors
        self.trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

        # Sauvegarder la config LoRA séparément
        lora_config_path = self.output_dir / "lora_config.json"
        with open(lora_config_path, "w") as f:
            json.dump({
                "model_key": self.model_key,
                "base_model": self.base_model,
                "training_config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                                   for k, v in self.config.items()},
                "training_stats": {
                    "total_steps": self.training_stats["total_steps"],
                    "best_loss": self.training_stats["best_loss"],
                    "early_stopped": self.training_stats["early_stopped"],
                    "duration_minutes": (
                        (self.training_stats["end_time"] - self.training_stats["start_time"]).total_seconds() / 60
                        if self.training_stats["end_time"] else None
                    ),
                }
            }, f, indent=2, default=str)

        logger.info(f"Modèle sauvegardé: {self.output_dir}")

    def evaluate(self, test_samples: int = 10) -> Dict:
        """Évalue le modèle sur des exemples de test."""
        logger.info("=" * 60)
        logger.info("ÉVALUATION POST-FINE-TUNING")
        logger.info("=" * 60)

        if not self.model or not self.tokenizer:
            logger.error("Modèle non chargé")
            return {}

        # Charger quelques exemples de test
        dataset_path = settings.CORPUS_DIR / "dataset_mina.jsonl"
        test_data = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("validated", False):
                    test_data.append(item)

        # Prendre les derniers exemples comme test
        test_data = test_data[-test_samples:]

        results = []
        correct = 0

        for item in test_data:
            input_text = item["input"]
            expected = item["output"]

            # Générer
            prompt = f"Traduis en mina cette phrase en français: {input_text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extraire la traduction (apres la question)
            if ":" in generated:
                translation = generated.split(":", 1)[1].strip()
            else:
                translation = generated.strip()

            # Vérifier similarité simple (contient les mots clés)
            expected_words = set(expected.split())
            translation_words = set(translation.split())
            similarity = len(expected_words & translation_words) / max(len(expected_words), 1)

            is_correct = similarity > 0.5
            if is_correct:
                correct += 1

            results.append({
                "input": input_text,
                "expected": expected,
                "generated": translation,
                "similarity": similarity,
                "correct": is_correct,
            })

            logger.info(f"Input: {input_text}")
            logger.info(f"Attendu: {expected}")
            logger.info(f"Généré: {translation}")
            logger.info(f"Similarité: {similarity:.2f}")
            logger.info("-" * 40)

        accuracy = correct / len(test_data) if test_data else 0

        evaluation_results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_data),
            "samples": results,
        }

        logger.info("=" * 60)
        logger.info(f"RÉSULTATS: {correct}/{len(test_data)} ({accuracy*100:.1f}%)")
        logger.info("=" * 60)

        return evaluation_results

    # ==========================================================================
    # NETTOYAGE
    # ==========================================================================

    def cleanup(self):
        """Libère la mémoire GPU."""
        if self.model:
            del self.model
        if self.trainer:
            del self.trainer

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Mémoire GPU libérée")


# =============================================================================
# CALLBACKS
# =============================================================================

class TrainingStatsCallback:
    """Callback pour tracker les statistiques d'entraînement."""

    def __init__(self, stats: Dict):
        self.stats = stats
        self.epoch_start = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", None)
            if loss and loss < self.stats.get("best_loss", float("inf")):
                self.stats["best_loss"] = loss

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start:
            epoch_time = time.time() - self.epoch_start
            self.stats["epoch_times"].append(epoch_time)


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def print_config_summary(model_key: str, config: Dict, num_samples: int):
    """Affiche un résumé de la configuration."""
    logger.info("=" * 60)
    logger.info("RÉSUMÉ DE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Modèle: {model_key}")
    logger.info(f"Base model: {MODELS_CONFIG[model_key]['name']}")
    logger.info(f"Quantification: 4-bit QLoRA (NF4)")
    logger.info(f"LoRA r={config['lora_r']}, alpha={config['lora_alpha']}")
    logger.info(f"Epochs: {config['num_train_epochs']}")
    logger.info(f"Batch size: {config['per_device_train_batch_size']}")
    logger.info(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Max length: {config['max_length']}")
    logger.info(f"Early stopping: {config.get('early_stopping', False)}")
    logger.info(f"Échantillons: {num_samples}")
    logger.info(f"Estimation/epoch: {get_epoch_estimate(MODELS_CONFIG[model_key]['name'], num_samples)}")
    logger.info("=" * 60)


def generate_synthetic_test_cases() -> List[Dict]:
    """Génère des cas de test pour évaluation."""
    return [
        {"french": "Bonjour", "mina": "Akpe ɖe o"},
        {"french": "Merci", "mina": "Mele go ye"},
        {"french": "Au revoir", "mina": "Yovo"},
        {"french": "Comment ça va?", "mina": "Efya?"},
        {"french": "Je comprends", "mina": "Mele di"},
        {"french": "Je ne comprends pas", "mina": "Mele mede di o"},
        {"french": "Oui", "mina": "Ee"},
        {"french": "Non", "mina": "Ao"},
        {"french": "S'il vous plaît", "mina": "Afiafi"},
        {"french": "Merci beaucoup", "mina": "Mele gokpɔ wo"},
    ]


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA pour Mina-Translator")

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="phi3-mini",
        choices=list(MODELS_CONFIG.keys()),
        help="Modèle à fine-tuner"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Nombre d'epochs (défaut: 3)"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size par GPU (défaut: 4)"
    )

    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=2e-4,
        help="Learning rate (défaut: 2e-4)"
    )

    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=256,
        help="Longueur max de séquence (défaut: 256)"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Rang LoRA (défaut: 16)"
    )

    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Activer l'early stopping"
    )

    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Désactiver l'early stopping"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Répertoire de sortie"
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Évaluer uniquement (sans entraînement)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher la config sans lancer l'entraînement"
    )

    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()

    # Config
    config = {
        **DEFAULT_TRAINING_CONFIG,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "early_stopping": args.early_stop and not args.no_early_stop,
    }

    # Output dir
    output_dir = Path(args.output) if args.output else None

    # Créer le fine-tuner
    finetuner = MinaFineTuner(
        model_key=args.model,
        config=config,
        output_dir=output_dir,
    )

    # Dry run - afficher la config
    if args.dry_run:
        dataset = finetuner.load_and_prepare_dataset()
        num_samples = len(dataset["train"]) if dataset else 0
        print_config_summary(args.model, config, num_samples)
        return

    # Vérifier le système
    if not finetuner.check_system():
        logger.error("Vérification système échouée")
        return

    # Évaluation uniquement
    if args.evaluate_only:
        logger.info("Mode évaluation uniquement")
        finetuner.load_model_and_tokenizer()
        results = finetuner.evaluate()
        finetuner.cleanup()
        return

    # Charger le dataset
    dataset = finetuner.load_and_prepare_dataset()
    if not dataset:
        return

    num_samples = len(dataset["train"])

    # Afficher config
    print_config_summary(args.model, config, num_samples)

    # Confirmer le lancement
    logger.info("\nAppuyez sur Ctrl+C pour annuler...")
    try:
        input("Appuyez sur Entrée pour continuer...")
    except KeyboardInterrupt:
        logger.info("Annulé")
        return

    # Charger modèle
    finetuner.load_model_and_tokenizer()

    # Entraîner
    success = finetuner.train(dataset)

    if success:
        # Sauvegarder
        finetuner.save_model()

        # Évaluer
        evaluation = finetuner.evaluate()

        # Sauvegarder les résultats
        results_path = finetuner.output_dir / "evaluation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        logger.info(f"\nRésultats sauvegardés: {results_path}")

    # Cleanup
    finetuner.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("TERMINÉ")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()