"""
scripts/train_whisper.py - Fine-tuning de Whisper-small pour le Mina
====================================================================

Fine-tuning de openai/whisper-small sur le dataset Mina (gej) optimisé
pour RTX 4060 avec 8 Go VRAM.

Optimisations VRAM:
    - Quantification 8-bit avec bitsandbytes
    - Gradient checkpointing
    - Gradient accumulation (16 steps)
    - Mixed precision (fp16)
    - Optimisation de la longueur de аудио

Usage:
    python scripts/train_whisper.py [--config config.yaml]

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

# Vérification GPU
import torch

# Deep Learning
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset, DatasetDict
import evaluate

# Logging
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_FILE = DATA_DIR / "corpus" / "mina_full_dataset.jsonl"
CV_DATASET_DIR = DATA_DIR / "cv-corpus-25.0-2026-03-09"
CLIPS_DIR = CV_DATASET_DIR / "gej" / "clips"

MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mina-whisper-v1"

# Configuration des logs
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "training.log"

# Paramètres d'entraînement optimisés pour 8 Go VRAM
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,  # effectif: 1 * 16 = 16
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "max_steps": 10000,
    "gradient_checkpointing": True,
    "fp16": True,  # Mixed precision
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 500,
    "save_total_limit": 3,
    "predict_with_generate": True,
    "generation_max_length": 225,
    "remove_unused_columns": False,
    "optim": "adamw_bnb_8bit",  # Optimiseur 8-bit
}

# Langue target
TARGET_LANGUAGE = "French"  # Mina -> Français (ou "Mina" pour Français -> Mina)
LANGUAGE_CODE = "fr"  # Code ISO 639-1
TASK = "translate"  # ou "transcribe"


# =============================================================================
# CONFIGURATION DU LOGGING
# =============================================================================

def setup_logging():
    """Configure le logging vers fichier et console"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration Loguru
    logger.remove()

    # Log vers fichier (avec rotation)
    logger.add(
        LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
    )

    # Log vers console
    logger.add(
        sys.stderr,
        format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    return logger


# =============================================================================
# VÉRIFICATIONS INITIALES
# =============================================================================

def check_gpu():
    """Vérifie la disponibilité et les capacités du GPU"""
    if not torch.cuda.is_available():
        logger.error("Aucun GPU détecté! Ce script nécessite un GPU NVIDIA avec CUDA.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_version = torch.version.cuda

    logger.info("=" * 60)
    logger.info("GPU DÉTECTÉ")
    logger.info("=" * 60)
    logger.info(f"Modèle: {gpu_name}")
    logger.info(f"Mémoire VRAM: {gpu_memory:.1f} Go")
    logger.info(f"CUDA Version: {cuda_version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("=" * 60)

    if gpu_memory < 6:
        logger.warning(f"VRAM ({gpu_memory:.1f} Go) inférieure à 6 Go - utilisation de paramètres très conservatives")
        TRAINING_CONFIG["gradient_accumulation_steps"] = 32
        TRAINING_CONFIG["per_device_train_batch_size"] = 1

    return gpu_memory


def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    missing = []

    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes")

    try:
        import peft
    except ImportError:
        missing.append("peft")

    try:
        from datasets import load_dataset
    except ImportError:
        missing.append("datasets")

    try:
        import evaluate
    except ImportError:
        missing.append("evaluate")

    if missing:
        logger.error(f"Dépendances manquantes: {missing}")
        logger.info("Installez-les avec: pip install " + " ".join(missing))
        sys.exit(1)

    logger.info("Toutes les dépendances sont disponibles")


# =============================================================================
# PRÉPARATION DU DATASET
# =============================================================================

def load_mina_dataset(corpus_file: Path, clips_dir: Path, streaming: bool = True):
    """
    Charge le dataset Mina en mode streaming pour éviter de saturer la RAM

    Args:
        corpus_file: Chemin vers le fichier JSONL
        clips_dir: Répertoire contenant les fichiers audio
        streaming: Mode streaming (True) ou chargement complet (False)

    Returns:
        DatasetDict avec les splits train/dev/test
    """
    logger.info(f"Chargement du dataset depuis {corpus_file}")
    logger.info(f"Mode streaming: {streaming}")

    # Charger en streaming
    raw_dataset = load_dataset(
        "json",
        data_files=str(corpus_file),
        split="train",
        streaming=streaming,
    )

    # Diviser en train/test
    if streaming:
        # Pour le mode streaming, on fait une division simple
        dataset_list = list(raw_dataset)

        # Séparer selon le split
        train_data = [d for d in dataset_list if d.get("split") in ["train", "validated"]]
        test_data = [d for d in dataset_list if d.get("split") in ["test"]]
        dev_data = [d for d in dataset_list if d.get("split") == ["dev"]]

        # Pour le dev, on prend les derniers éléments du train
        if len(dev_data) == 0 and len(train_data) > 100:
            dev_size = min(500, len(train_data) // 10)
            dev_data = train_data[-dev_size:]
            train_data = train_data[:-dev_size]

        from datasets import Dataset

        dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "test": Dataset.from_list(test_data) if test_data else Dataset.from_list(train_data[-500:]),
            "dev": Dataset.from_list(dev_data) if dev_data else Dataset.from_list(train_data[-500:]),
        })
    else:
        dataset_dict = raw_dataset.train_test_split(test_size=0.1)

    logger.info(f"Dataset chargé: {len(dataset_dict['train'])} train, {len(dataset_dict.get('test', 0))} test")

    return dataset_dict


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Prépare un batch pour l'entraînement Whisper

    - Charge le fichier audio
    - Extrait les features
    - Tokenise le texte

    Args:
        batch: Exemple du dataset
        feature_extractor: WhisperFeatureExtractor
        tokenizer: WhisperTokenizer

    Returns:
        Batch préparer pour l'entraînement
    """
    # Construire le chemin complet vers l'audio
    audio_path = str(CV_DATASET_DIR / batch["audio_path"])

    # Charger et ré-échantillonner l'audio à 16kHz
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        logger.warning(f"Erreur chargement audio {audio_path}: {e}")
        return {"labels": [], "input_features": []}

    # Extraire les features
    input_features = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features[0]

    # Tokeniser le texte
    labels = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="max_length",
        max_length=448,
        truncation=True,
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels,
    }


def prepare_dataset_batch(batch, processor):
    """
    Prépare un batch complet avec le processor Whisper
    Plus efficace pour l'entraînement
    """
    from transformers import WhisperProcessor

    # Charger l'audio
    audio_path = str(CV_DATASET_DIR / batch["audio_path"])

    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        logger.warning(f"Erreur chargement audio {audio_path}: {e}")
        return {"labels": [], "input_features": [], "example": batch["text"]}

    # Traiter avec le processor
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features[0]

    # Tokeniser
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="max_length",
        max_length=448,
        truncation=True,
    ).input_ids[0]

    return {
        "input_features": input_features,
        "labels": labels,
        "text": batch["text"],  # Garder le texte original pour le WER
    }


# =============================================================================
# CALCUL DU WER
# =============================================================================

def compute_metrics(pred, tokenizer, metric):
    """
    Calcule le Word Error Rate (WER) pour la génération

    Args:
        pred: Prédiction du modèle
        tokenizer: Tokenizer pour décoder
        metric: Métrique evaluate WER

    Returns:
        Dictionary avec le WER
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplacer -100 par le padding token
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Décoder
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Filtrer les prédictions vides
    pred_str = [p if p else "[EMPTY]" for p in pred_str]
    label_str = [l if l else "[EMPTY]" for l in label_str]

    # Calculer le WER
    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# =============================================================================
# CONFIGURATION DU MODÈLE
# =============================================================================

def load_quantized_model(model_name: str, gradient_checkpointing: bool = True):
    """
    Charge le modèle Whisper avec quantification 8-bit

    Args:
        model_name: Nom du modèle Hugging Face
        gradient_checkpointing: Activer le gradient checkpointing

    Returns:
        Modèle prêt pour l'entraînement
    """
    logger.info(f"Chargement de {model_name} avec quantification 8-bit...")

    # Configuration de quantification
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Quantification 8-bit
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Charger le modèle avec quantification
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Activer le gradient checkpointing pour экономить VRAM
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing activé")

    # Congeler les paramètres non nécessaires (optionnel)
    # for name, param in model.named_parameters():
    #     if "encoder" not in name:
    #         param.requires_grad = False

    logger.info(f"Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres")

    return model


def load_model_with_fp16(model_name: str, gradient_checkpointing: bool = True):
    """
    Charge le modèle Whisper sans quantification (FP16)
    Alternative si bitsandbytes pose des problèmes

    Args:
        model_name: Nom du modèle Hugging Face
        gradient_checkpointing: Activer le gradient checkpointing

    Returns:
        Modèle prêt pour l'entraînement
    """
    logger.info(f"Chargement de {model_name} en FP16...")

    # Charger le modèle en FP16
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Activer le gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing activé")

    logger.info(f"Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres")

    return model


def load_processor(model_name: str):
    """
    Charge le processor Whisper

    Args:
        model_name: Nom du modèle

    Returns:
        Processor complet (feature extractor + tokenizer)
    """
    logger.info(f"Chargement du processor {model_name}...")

    processor = WhisperProcessor.from_pretrained(model_name)

    logger.info(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")
    logger.info(f"Feature extractor: {processor.feature_extractor.__class__.__name__}")

    return processor


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_whisper(
    model_name: str = MODEL_NAME,
    output_dir: Path = OUTPUT_DIR,
    use_quantization: bool = True,
    max_steps: int = 10000,
    learning_rate: float = 1e-4,
    warmup_steps: int = 500,
):
    """
    Lance l'entraînement complet de Whisper

    Args:
        model_name: Modèle de base
        output_dir: Répertoire de sauvegarde
        use_quantization: Utiliser la quantification 8-bit
        max_steps: Nombre maximum de steps
        learning_rate: Taux d'apprentissage
        warmup_steps: Steps de warmup
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU FINE-TUNING WHISPER MINA")
    logger.info("=" * 60)
    logger.info(f"Modèle: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Quantization: {'8-bit' if use_quantization else 'FP16'}")
    logger.info("=" * 60)

    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le processor
    processor = load_processor(model_name)

    # Charger le modèle
    if use_quantization:
        try:
            model = load_quantized_model(model_name, gradient_checkpointing=True)
        except Exception as e:
            logger.warning(f"Quantization 8-bit échouée: {e}")
            logger.info("Fallback vers FP16...")
            model = load_model_with_fp16(model_name, gradient_checkpointing=True)
    else:
        model = load_model_with_fp16(model_name, gradient_checkpointing=True)

    # Charger le dataset
    dataset = load_mina_dataset(CORPUS_FILE, CLIPS_DIR, streaming=False)

    # Préparer le dataset
    logger.info("Préparation du dataset...")

    # Utiliser une fonction de prétraitement plus légère
    def prepare_batch(batch):
        audio_path = str(CV_DATASET_DIR / batch["audio_path"])

        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            logger.debug(f"Erreur audio: {e}")
            batch["input_features"] = None
            batch["labels"] = [0]
            return batch

        input_features = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        labels = processor.tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=448,
            truncation=True,
        ).input_ids[0]

        batch["input_features"] = input_features
        batch["labels"] = labels

        return batch

    # Appliquer le prétraitement
    logger.info("Prétraitement du dataset (ça peut prendre du temps)...")

    # Traiter les splits
    dataset = dataset.map(
        prepare_batch,
        remove_columns=["audio_path", "split", "validated", "text"],
        num_proc=4,
    )

    # Filtrer les entrées problématiques
    dataset["train"] = dataset["train"].filter(
        lambda x: x["input_features"] is not None,
        num_proc=4,
    )

    # Charger la métrique WER
    logger.info("Chargement de la métrique WER...")
    wer_metric = evaluate.load("wer")

    # Configurer les arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        fp16=TRAINING_CONFIG["fp16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        predict_with_generate=TRAINING_CONFIG["predict_with_generate"],
        generation_max_length=TRAINING_CONFIG["generation_max_length"],
        remove_unused_columns=TRAINING_CONFIG["remove_unused_columns"],
        optim=TRAINING_CONFIG["optim"],
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # Créer le trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", dataset["train"]),
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer, wer_metric),
    )

    # Lancer l'entraînement
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'ENTRAÎNEMENT")
    logger.info("=" * 60)

    try:
        trainer.train()

        # Sauvegarder le modèle final
        logger.info("Sauvegarde du modèle final...")
        trainer.save_model(output_dir / "final")
        processor.save_pretrained(output_dir / "final")

        # Temps d'exécution
        elapsed = datetime.now() - start_time
        logger.info("=" * 60)
        logger.info(f"ENTRAÎNEMENT TERMINÉ (durée: {elapsed})")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        raise

    # Nettoyer
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Fine-tuning Whisper pour le Mina")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Modèle de base (défaut: openai/whisper-small)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Désactiver la quantification 8-bit",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Nombre maximum de steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Taux d'apprentissage",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Steps de warmup",
    )

    args = parser.parse_args()

    # Configurer le logging
    setup_logging()

    # Vérifications
    gpu_memory = check_gpu()
    check_dependencies()

    # Lancer l'entraînement
    train_whisper(
        model_name=args.model,
        output_dir=Path(args.output),
        use_quantization=not args.no_quantize,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
    )


if __name__ == "__main__":
    main()