"""
scripts/train_whisper_simple.py - Fine-tuning Whisper simplifié pour Mina
=========================================================================

Version simplifiée avec moins de dépendances multiprocessing.
Utilise les fichiers audio pré-convertis.

Usage:
    python scripts/train_whisper_simple.py

Auteur: Claude Opus 4.8
Date: 2026-06-05
"""

import sys
import io
from pathlib import Path

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = PROJECT_ROOT / "models" / "mina-whisper-v1"
DATA_DIR = PROJECT_ROOT / "data" / "whisper_data"
CV_DIR = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09"
AUDIO_DIR = DATA_DIR / "audio"

def load_dataset():
    """Charge le dataset depuis les manifestes JSONL"""
    import json

    train_data = []
    val_data = []

    # Charger train
    train_file = DATA_DIR / "train.jsonl"
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                train_data.append(json.loads(line))

    # Charger validation
    val_file = DATA_DIR / "validation.jsonl"
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            for line in f:
                val_data.append(json.loads(line))

    return train_data, val_data

def prepare_audio(audio_filepath, processor):
    """Prépare un fichier audio pour Whisper"""
    import librosa

    if not Path(audio_filepath).exists():
        return None

    try:
        audio, sr = librosa.load(audio_filepath, sr=16000)
        input_features = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        return input_features
    except Exception as e:
        print(f"Erreur lecture audio {audio_filepath}: {e}")
        return None

def main():
    print("=" * 60)
    print("FINE-TUNING WHISPER POUR MINA (SIMPLIFIÉ)")
    print("=" * 60)

    # Vérifier GPU
    if not torch.cuda.is_available():
        print("ERREUR: GPU NVIDIA requis!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} Go VRAM)")

    # Charger modèle
    print("\nChargement du modèle Whisper-small...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.cuda()

    # Charger données
    print("\nChargement des données...")
    train_data, val_data = load_dataset()
    print(f"Train: {len(train_data)} clips")
    print(f"Validation: {len(val_data)} clips")

    # Limiter pour test rapide
    max_train = min(500, len(train_data))
    max_val = min(100, len(val_data))
    train_data = train_data[:max_train]
    val_data = val_data[:max_val]

    # Préparer les features (sans multiprocessing)
    print("\nPréparation des features audio...")
    train_features = []
    train_labels_list = []

    for i, item in enumerate(train_data):
        if i % 100 == 0:
            print(f"  Train: {i}/{len(train_data)}")

        features = prepare_audio(item["audio_filepath"], processor)
        if features is not None:
            train_features.append(features)
            labels = processor.tokenizer(
                item["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=448,
                truncation=True,
            ).input_ids[0]
            train_labels_list.append(labels)

    print(f"\nFeatures préparés: {len(train_features)}/{len(train_data)}")

    # Créer dataset PyTorch
    class WhisperDataset:
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return {
                "input_features": torch.tensor(self.features[idx]),
                "labels": self.labels[idx],
            }

    train_dataset = WhisperDataset(train_features, train_labels_list)

    # Préparer validation
    val_features = []
    val_labels_list = []
    for item in val_data:
        features = prepare_audio(item["audio_filepath"], processor)
        if features is not None:
            val_features.append(features)
            labels = processor.tokenizer(
                item["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=448,
                truncation=True,
            ).input_ids[0]
            val_labels_list.append(labels)

    val_dataset = WhisperDataset(val_features, val_labels_list)

    # Charger métrique WER
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Remplacer -100 par pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_score = wer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_score}

    # Config entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=500,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Entraîner
    print("\n" + "=" * 60)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 60)
    trainer.train()

    # Sauvegarder
    print("\nSauvegarde du modèle...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    processor.save_pretrained(str(OUTPUT_DIR / "final"))

    print("\n" + "=" * 60)
    print("FINE-TUNING TERMINÉ!")
    print("=" * 60)

if __name__ == "__main__":
    main()