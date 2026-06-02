"""
Fine-tuning Whisper pour la reconnaissance vocale Mina
Optimisé pour RTX 4060 8GB VRAM

Usage:
    python scripts/train_whisper.py --data data/corpus/audio/ --epochs 5
"""
import os
import sys
from pathlib import Path
import argparse
import torch
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# DataCollator peut etre importe differemment selon la version
try:
    from transformers import DataCollatorSpeechSeq2SeqWithPadding
except ImportError:
    from transformers import DataCollatorForSeq2Seq as DataCollatorSpeechSeq2SeqWithPadding

from datasets import load_dataset, Audio
import evaluate


# Configuration
MODEL_NAME = "openai/whisper-small"  # 244M params - optimal pour 8GB VRAM


def prepare_dataset(dataset_path: Path, language: str = "mina"):
    """
    Prépare le dataset pour l'entraînement Whisper

    Attend une structure:
        dataset_path/
            audio1.wav  audio1.txt
            audio2.wav  audio2.txt
            ...
    """
    logger.info(f"Preparation dataset depuis {dataset_path}")

    # Collecter les fichiers audio et texte
    audio_files = list(dataset_path.glob("*.wav")) + list(dataset_path.glob("*.mp3"))

    if not audio_files:
        logger.warning(f"Aucun fichier audio trouve dans {dataset_path}")
        logger.info("Creation d'un dataset dummy pour demo")
        return create_demo_dataset()

    data_items = []
    for audio_file in audio_files:
        text_file = audio_file.with_suffix(".txt")
        if text_file.exists():
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            data_items.append({
                "audio_path": str(audio_file),
                "text": text,
            })

    logger.info(f"Dataset pret: {len(data_items)} exemples")

    # Charger le processor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # Créer le dataset HF
    from datasets import Dataset

    def prepare_example(example):
        # Charger l'audio
        audio, sr = librosa_load(example["audio_path"], sampling_rate=16000)
        input_features = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        # Tokeniser le texte
        labels = processor.tokenizer(example["text"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    # Note: En production, utiliser load_dataset avec Audio pour gérer le resampling
    dataset_dict = {
        "audio_path": [item["audio_path"] for item in data_items],
        "text": [item["text"] for item in data_items],
    }

    dataset = Dataset.from_dict(dataset_dict)

    # Prétraitement
    def preprocess_function(examples):
        # Charger et traiter audio
        audio_arrays = []
        for path in examples["audio_path"]:
            try:
                import librosa
                audio, _ = librosa.load(path, sr=16000)
                audio_arrays.append(audio)
            except Exception as e:
                logger.warning(f"Erreur chargement {path}: {e}")
                audio_arrays.append(torch.zeros(16000))

        # Feature extraction
        input_features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
        ).input_features

        # Tokenisation
        labels = processor.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=448,
        )

        return {
            "input_features": input_features,
            "labels": labels.input_ids,
        }

    # Appliquer le prétraitement
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names,
    )

    return dataset


def create_demo_dataset():
    """Crée un dataset de démo pour tester le pipeline"""
    logger.info("Creation dataset demo...")

    from datasets import Dataset

    # Dataset minimal pour tester
    demo_data = [
        {"text": "Mɛlɔ", "audio_path": "demo"},
        {"text": "Ntsɛ nyabi?", "audio_path": "demo"},
        {"text": "Nyametsɛ", "audio_path": "demo"},
    ]

    dataset = Dataset.from_list(demo_data)
    return dataset


def compute_metrics(pred):
    """Calcule les métriques WER et CER"""
    metric = evaluate.load("wer")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplacer -100 par pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Décoder
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_whisper(
    dataset_path: Path = None,
    output_dir: Path = None,
    language: str = "mina",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
):
    """
    Fine-tune Whisper sur le dataset Mina

    Args:
        dataset_path: Chemin vers les données audio
        output_dir: Répertoire de sortie pour le modèle
        language: Code langue (mina)
        epochs: Nombre d'époques
        batch_size: Taille de batch (ajuster selon VRAM)
        learning_rate: Taux d'apprentissage
    """
    logger.info("=" * 60)
    logger.info("FINE-TUNING WHISPER POUR LE MINA")
    logger.info("=" * 60)

    # Configuration GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM disponible: {vram_gb:.1f} GB")

    # Charger le modèle
    logger.info(f"Chargement {MODEL_NAME}...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Forcer la langue
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )
    model.config.suppress_tokens = []

    # Préparer les données
    if dataset_path and dataset_path.exists():
        dataset = prepare_dataset(dataset_path, language)
    else:
        logger.warning("Dataset non trouve - utilisation mode demo")
        dataset = create_demo_dataset()

    # Split train/val
    if len(dataset) > 2:
        train_test = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test["train"]
        eval_dataset = train_test["test"]
    else:
        train_dataset = dataset
        eval_dataset = dataset

    logger.info(f"Train: {len(train_dataset)} exemples")
    logger.info(f"Eval: {len(eval_dataset)} exemples")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        model=model,
    )

    # Arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir or "models/whisper-mina"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Équivalent batch_size * 4
        learning_rate=learning_rate,
        warmup_steps=100,
        max_steps=len(train_dataset) * epochs // batch_size,
        num_train_epochs=epochs,
        fp16=device == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=448,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        resume_from_checkpoint=True,
        report_to="none",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
        tokenizer=processor.feature_extractor,
    )

    # Entraîner
    logger.info("Debut de l'entrainement...")
    trainer.train()

    # Sauvegarder
    final_model_path = output_dir / "final" if output_dir else Path("models/whisper-mina/final")
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    logger.info(f"Modele sauvegarde: {final_model_path}")
    logger.info("Fine-tuning termine!")

    return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Mina")
    parser.add_argument("--data", type=str, default="data/corpus/audio",
                        help="Chemin vers les donnees audio")
    parser.add_argument("--output", type=str, default="models/whisper-mina",
                        help="Repertoire de sortie")
    parser.add_argument("--language", type=str, default="mina",
                        help="Code langue")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Nombre d'epoques")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Taille de batch")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Taux d'apprentissage")

    args = parser.parse_args()

    # Entraîner
    train_whisper(
        dataset_path=Path(args.data) if args.data else None,
        output_dir=Path(args.output),
        language=args.language,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()