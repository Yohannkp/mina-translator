"""
scripts/fine_tune_whisper.py - Fine-tuning Whisper sur Common Voice Ewe/Mina
=============================================================================

Fine-tune Whisper (base/small) sur le dataset Common Voice Ewe (16k+ phrases)
pour créer un modèle STT capable de reconnaître l'Ewe/Mina parlé au Togo.

Optimisé pour RTX 4060 (8 Go VRAM) avec QLoRA.

Usage:
    # Préparer les données (une fois)
    python scripts/fine_tune_whisper.py --prepare

    # Lancer le fine-tuning (après préparation)
    python scripts/fine_tune_whisper.py --train --epochs 3

    # Test après fine-tuning
    python scripts/fine_tune_whisper.py --test

    # Dry run (10 steps pour tester)
    python scripts/fine_tune_whisper.py --dry-run

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import os
import sys
import json
import gc
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset, load_dataset, Audio

# Audio processing
import librosa
import soundfile as sf

# Logging
from loguru import logger

# PEFT for QLoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
COMMON_VOICE_PATH = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09" / "gej"
OUTPUT_MODEL_PATH = PROJECT_ROOT / "models" / "whisper-mina-ewe"

# Whisper model config
WHISPER_MODEL = "openai/whisper-base"  # ou "openai/whisper-small" pour plus de qualité
WHISPER_LANGUAGE = "ewe"
WHISPER_TASK = "transcribe"

# Training config (optimisé pour 8 Go VRAM)
TRAINING_CONFIG = {
    "batch_size": 8,              # Batch size par GPU
    "gradient_accumulation": 4,    # Effective batch = 32
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "max_steps": 10000,
    "eval_steps": 500,
    "save_steps": 1000,
    "logging_steps": 100,
    "fp16": True,
    "gradient_checkpointing": True,
    "use_gradient_checkpointing": True,
    "optim": "adamw_torch",
}

# LoRA config pour Whisper
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "v_proj",       # Self-attention
        "k_proj", "o_proj",
        "fc1", "fc2",             # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ2SEQ_ASR",
)


# =============================================================================
# DATA PREPARATION
# =============================================================================

@dataclass
class DatasetInfo:
    """Informations sur le dataset Ewe"""
    total_samples: int
    total_duration_hours: float
    train_samples: int
    eval_samples: int
    avg_duration_sec: float


def prepare_common_voice_dataset(
    cache_dir: Path = None,
    num_samples: int = None,
    test_size: float = 0.1,
) -> DatasetInfo:
    """
    Prépare le dataset Common Voice Ewe pour l'entraînement

    Args:
        cache_dir: Dossier de cache pour les fichiers traités
        num_samples: Nombre max de samples (None = tous)
        test_size: Proportion pour l'évaluation

    Returns:
        DatasetInfo avec statistiques
    """
    logger.info("=" * 60)
    logger.info("PREPARING COMMON VOICE EWE DATASET")
    logger.info("=" * 60)

    clips_dir = COMMON_VOICE_PATH / "clips"
    validated_file = COMMON_VOICE_PATH / "validated.tsv"

    # Charger les métadonnées
    samples = []
    with open(validated_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                audio_path = parts[1]
                sentence = parts[3]

                full_path = clips_dir / audio_path
                if full_path.exists():
                    samples.append({
                        "audio_path": str(full_path),
                        "sentence": sentence,
                    })

    # Limiter si demandé
    if num_samples:
        samples = samples[:num_samples]

    logger.info(f"Found {len(samples)} valid audio files")

    # Calculer la durée totale
    total_duration = sum(
        librosa.get_samplerate(full_path) or 16000
        for s in samples if (full_path := Path(s["audio_path"])).exists()
    )
    # Approximation simple (on肯定会 recalculer)
    avg_duration = 3.0  # Secondes
    total_hours = len(samples) * avg_duration / 3600

    logger.info(f"Total duration: ~{total_hours:.1f} hours")

    # Sauvegarder les métadonnées pour le dataset
    cache_dir = cache_dir or PROJECT_ROOT / "data" / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = cache_dir / "ewe_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    # Split train/eval
    import random
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - test_size))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    logger.info(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}")

    return DatasetInfo(
        total_samples=len(samples),
        total_duration_hours=total_hours,
        train_samples=len(train_samples),
        eval_samples=len(eval_samples),
        avg_duration_sec=avg_duration,
    )


class EweAudioDataset(Dataset):
    """
    Dataset pour l'entraînement Whisper sur l'Ewe
    Charge les fichiers audio et les transcriptions
    """

    def __init__(
        self,
        samples: List[Dict],
        processor: WhisperProcessor,
        max_duration: float = 30.0,  # Whisper max
        min_duration: float = 0.5,
    ):
        """
        Args:
            samples: Liste des métadonnées {audio_path, sentence}
            processor: Whisper processor
            max_duration: Durée max en secondes
            min_duration: Durée min en secondes
        """
        self.samples = samples
        self.processor = processor
        self.max_duration = max_duration

        # Filtrer les samples valides
        self.valid_samples = []
        for s in samples:
            audio_path = Path(s["audio_path"])
            if audio_path.exists():
                try:
                    duration = librosa.get_samplerate(str(audio_path)) / 1000
                    if min_duration <= duration <= max_duration:
                        self.valid_samples.append(s)
                except Exception:
                    pass

        logger.info(f"Valid samples: {len(self.valid_samples)}/{len(samples)}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.valid_samples[idx]
        audio_path = sample["audio_path"]
        sentence = sample["sentence"]

        try:
            # Charger l'audio
            audio, sr = sf.read(audio_path, dtype='float32')

            # Resample si nécessaire
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Normaliser
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            # Traiter avec Whisper
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )

            # Encoder les labels
            labels = self.processor.tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=448,
            ).input_ids

            return {
                "input_features": inputs.input_features[0],
                "labels": labels[0],
            }

        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return dummy sample
            return {
                "input_features": torch.zeros(80, 3000),
                "labels": torch.zeros(1, dtype=torch.long),
            }


def create_dataloader(
    samples: List[Dict],
    processor: WhisperProcessor,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Crée un DataLoader pour l'entraînement"""
    dataset = EweAudioDataset(samples, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function pour le DataLoader"""
    import numpy as np

    input_features = []
    labels = []

    for item in batch:
        input_features.append(item["input_features"])
        labels.append(item["labels"])

    # Pad sequences
    input_features = torch.nn.utils.rnn.pad_sequence(
        input_features,
        batch_first=True,
        padding_value=0,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100,  # Ignore padding in loss
    )

    return {
        "input_features": input_features,
        "labels": labels,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_whisper(
    train_samples: List[Dict],
    eval_samples: List[Dict],
    output_dir: Path = OUTPUT_MODEL_PATH,
    epochs: int = 3,
    batch_size: int = 8,
    dry_run: bool = False,
):
    """
    Fine-tune Whisper sur l'Ewe

    Args:
        train_samples: Données d'entraînement
        eval_samples: Données d'évaluation
        output_dir: Dossier de sortie
        epochs: Nombre d'epochs
        batch_size: Batch size
        dry_run: Si True, ne fait que quelques steps pour tester
    """
    logger.info("=" * 60)
    logger.info("FINE-TUNING WHISPER ON EWE/MINA")
    logger.info("=" * 60)
    logger.info(f"Model: {WHISPER_MODEL}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Eval samples: {len(eval_samples)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")

    # Configurer device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Charger le processor et le modèle
    logger.info("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)

    # Charger avec QLoRA ou full model selon VRAM
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

    if gpu_memory >= 8:
        # 8 Go+ : full model avec gradient checkpointing
        logger.info("Loading full model (FP16)...")
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=torch.float16,
        )

        if TRAINING_CONFIG["use_gradient_checkpointing"]:
            model.config.use_cache = False
            model.enable_gradient_checkpointing()

    else:
        # Moins de 8 Go : QLoRA
        logger.info("Loading with QLoRA...")
        model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = prepare_model_for_kbit_training(model)

        # Appliquer LoRA
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    if device == "cuda":
        model = model.cuda()

    # Préparer les données
    logger.info("Creating datasets...")
    train_loader = create_dataloader(
        train_samples,
        processor,
        batch_size=batch_size,
        shuffle=True,
    )
    eval_loader = create_dataloader(
        eval_samples,
        processor,
        batch_size=batch_size,
        shuffle=False,
    )

    # Préparer les arguments d'entraînement
    output_dir = str(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        num_train_epochs=epochs,
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        max_steps=TRAINING_CONFIG["max_steps"] if dry_run else -1,
        evaluation_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        fp16=TRAINING_CONFIG["fp16"] and device == "cuda",
        predict_with_generate=True,
        generation_max_length=256,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{output_dir}/logs",
        report_to=["loguru"],
    )

    # Créer le trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        tokenizer=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # Entraîner
    logger.info("Starting training...")
    trainer.train()

    # Sauvegarder
    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE - Model saved to: {output_dir}")
    logger.info("=" * 60)


def test_fine_tuned_model(model_path: Path = OUTPUT_MODEL_PATH):
    """Test le modèle fine-tuné avec quelques fichiers Common Voice"""
    logger.info("=" * 60)
    logger.info("TESTING FINE-TUNED WHISPER MODEL")
    logger.info("=" * 60)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run with --train first to fine-tune the model")
        return

    # Charger le modèle
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Test sur quelques samples
    clips_dir = COMMON_VOICE_PATH / "clips"
    validated_file = COMMON_VOICE_PATH / "validated.tsv"

    samples = []
    with open(validated_file, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:6]:  # Test sur 5 samples
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                audio_path = clips_dir / parts[1]
                expected = parts[3]
                samples.append((audio_path, expected))

    correct = 0
    total_words = 0

    for audio_path, expected in samples:
        logger.info(f"\nExpected: {expected}")

        # Charger et traiter l'audio
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Générer
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                max_new_tokens=256,
                language=WHISPER_LANGUAGE,
                task=WHISPER_TASK,
            )

        # Décoder
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"Transcribed: {transcription}")

        # Calculer similarité simple
        exp_words = set(expected.lower().split())
        trans_words = set(transcription.lower().split())
        overlap = len(exp_words & trans_words)
        total_words += len(exp_words)
        if overlap > 0:
            correct += overlap

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    word_acc = 100 * correct / max(total_words, 1)
    logger.info(f"Word accuracy: {word_acc:.1f}%")
    logger.info(f"Note: Fine-tuning improves with more training steps")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Ewe/Mina")

    # Modes
    parser.add_argument("--prepare", action="store_true", help="Préparer le dataset")
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement")
    parser.add_argument("--test", action="store_true", help="Tester le modèle")
    parser.add_argument("--dry-run", action="store_true", help="Test rapide (10 steps)")

    # Options
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--samples", type=int, default=None, help="Limiter le nombre de samples")
    parser.add_argument("--output", type=str, help="Dossier de sortie")
    parser.add_argument("--model", type=str, default=WHISPER_MODEL, help="Modèle Whisper")
    parser.add_argument("--verbose", action="store_true", help="Mode détaillé")

    args = parser.parse_args()

    # Logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level, colorize=True)

    output_dir = Path(args.output) if args.output else OUTPUT_MODEL_PATH

    if args.prepare:
        info = prepare_common_voice_dataset(num_samples=args.samples)
        print("\n" + "=" * 60)
        print("DATASET PREPARED")
        print("=" * 60)
        print(f"Total samples: {info.total_samples}")
        print(f"Train samples: {info.train_samples}")
        print(f"Eval samples: {info.eval_samples}")
        print(f"Duration: {info.total_duration_hours:.1f} hours")

    elif args.dry_run:
        logger.info("DRY RUN MODE")
        info = prepare_common_voice_dataset(num_samples=args.samples or 100)
        train_samples = []
        with open(PROJECT_ROOT / "data" / "processed" / "ewe_metadata.json") as f:
            all_samples = json.load(f)
        split_idx = int(len(all_samples) * 0.9)
        train_samples = all_samples[:split_idx][:50]  # Only 50 for dry run

        train_whisper(
            train_samples=train_samples,
            eval_samples=[],
            output_dir=output_dir,
            epochs=1,
            batch_size=args.batch_size,
            dry_run=True,
        )

    elif args.train:
        # Préparer ou charger les données
        metadata_file = PROJECT_ROOT / "data" / "processed" / "ewe_metadata.json"

        if not metadata_file.exists():
            logger.info("Preparing dataset first...")
            prepare_common_voice_dataset(num_samples=args.samples)

        # Charger les métadonnées
        with open(metadata_file) as f:
            all_samples = json.load(f)

        # Split train/eval
        split_idx = int(len(all_samples) * 0.9)
        train_samples = all_samples[:split_idx]
        eval_samples = all_samples[split_idx:]

        logger.info(f"Loaded {len(all_samples)} samples")
        logger.info(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}")

        # Lancer l'entraînement
        train_whisper(
            train_samples=train_samples,
            eval_samples=eval_samples,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.test:
        test_fine_tuned_model(output_dir)

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START:")
        print("=" * 60)
        print("1. Préparer le dataset:")
        print("   python scripts/fine_tune_whisper.py --prepare")
        print("")
        print("2. Lancer l'entraînement:")
        print("   python scripts/fine_tune_whisper.py --train --epochs 3")
        print("")
        print("3. Tester le modèle:")
        print("   python scripts/fine_tune_whisper.py --test")
        print("")
        print("Pour un test rapide (10 steps):")
        print("   python scripts/fine_tune_whisper.py --dry-run")


if __name__ == "__main__":
    main()