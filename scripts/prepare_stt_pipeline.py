"""
scripts/prepare_stt_pipeline.py - Pipeline complet STT + LLM pour Mina
======================================================================

Ce script prépare le pipeline complet pour:
1. STT: Fine-tune Whisper sur les audio Mina (Common Voice)
2. LLM: Fine-tune Qwen2-0.5B sur les paires FR→Mina
3. API: API finale avec support audio et traduction

Architecture du pipeline:
    Audio Mina → Whisper (STT) → Texte Mina → Qwen2 (traduction) → Texte FR
    Audio FR → Whisper (STT) → Texte FR → Qwen2 (traduction) → Texte Mina

Usage:
    python scripts/prepare_stt_pipeline.py --step prepare
    python scripts/prepare_stt_pipeline.py --step stt
    python scripts/prepare_stt_pipeline.py --step llm
    python scripts/prepare_stt_pipeline.py --step all

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
DATA_DIR = PROJECT_DIR / "data"
CV_DIR = DATA_DIR / "cv-corpus-25.0-2026-03-09" / "gej"
CORPUS_DIR = DATA_DIR / "corpus"
MODELS_DIR = PROJECT_DIR / "models"

# Audio data
AUDIO_CLIPS_DIR = CV_DIR / "clips"
TRAIN_TSV = CV_DIR / "train.tsv"
DEV_TSV = CV_DIR / "dev.tsv"
TEST_TSV = CV_DIR / "test.tsv"
VALIDATED_TSV = CV_DIR / "validated.tsv"

# Text corpus
FR_MINA_CORPUS = CORPUS_DIR / "dataset_mina.jsonl"

# Output
PREPARED_DATA_DIR = DATA_DIR / "prepared"
STT_OUTPUT_DIR = MODELS_DIR / "whisper-mina"
LLM_OUTPUT_DIR = MODELS_DIR / "qwen-mina-translator"

# =============================================================================
# UTILITAIRES
# =============================================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"   {title}")
    print(f"{'='*70}\n")

def count_files(directory: Path, extension: str = None) -> int:
    """Compte les fichiers dans un répertoire."""
    if not directory.exists():
        return 0
    if extension:
        return len(list(directory.glob(f"*.{extension}")))
    return len(list(directory.rglob("*")))

def read_tsv(tsv_path: Path) -> List[Dict]:
    """Lit un fichier TSV et retourne une liste de dictionnaires."""
    if not tsv_path.exists():
        return []

    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return []

    # Header
    headers = lines[0].strip().split('\t')

    for line in lines[1:]:
        values = line.strip().split('\t')
        if len(values) >= len(headers):
            entry = {headers[i]: values[i] for i in range(len(headers))}
            data.append(entry)

    return data


# =============================================================================
# ÉTAPE 1: PRÉPARER LES DONNÉES
# =============================================================================

def prepare_data():
    """Prépare les données pour l'entraînement."""
    print_section("PRÉPARATION DES DONNÉES STT")

    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Préparer les données audio STT
    print("📂 Analyse des données Common Voice Mina...")

    # Charger les métadonnées
    train_data = read_tsv(TRAIN_TSV)
    dev_data = read_tsv(DEV_TSV)
    test_data = read_tsv(TEST_TSV)
    validated_data = read_tsv(VALIDATED_TSV)

    print(f"   Train: {len(train_data)} samples")
    print(f"   Dev: {len(dev_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    print(f"   Validated: {len(validated_data)} samples")

    # Créer les fichiers JSON pour chaque split
    splits = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data,
    }

    # Ajouter les données validées au train si pas assez
    if len(train_data) < 5000:
        splits["train"] = validated_data[:int(len(validated_data) * 0.8)]
        splits["dev"] = validated_data[int(len(validated_data) * 0.8):int(len(validated_data) * 0.9)]
        splits["test"] = validated_data[int(len(validated_data) * 0.9):]

    for split_name, split_data in splits.items():
        output_file = PREPARED_DATA_DIR / f"stt_{split_name}.json"

        # Formater pour Whisper
        whisper_data = []
        for entry in split_data:
            audio_path = AUDIO_CLIPS_DIR / entry.get("path", "")
            sentence = entry.get("sentence", "")

            if sentence and audio_path.exists():
                whisper_data.append({
                    "audio": str(audio_path),
                    "text": sentence,
                    "client_id": entry.get("client_id", ""),
                })

        # Sauvegarder
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(whisper_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ {split_name}: {len(whisper_data)} samples → {output_file.name}")

    # 2. Préparer le corpus FR→Mina pour le LLM
    print("\n📚 Préparation du corpus FR→Mina pour le LLM...")

    if FR_MINA_CORPUS.exists():
        with open(FR_MINA_CORPUS, 'r', encoding='utf-8') as f:
            fr_mina_data = [json.loads(line) for line in f if line.strip()]

        # Formater pour Qwen2
        llm_data = []
        for entry in fr_mina_data:
            llm_data.append({
                "instruction": entry.get("instruction", "Traduis cette phrase française en mina (ewé du Togo)."),
                "input": entry.get("input", ""),
                "output": entry.get("output", ""),
            })

        # Sauvegarder
        llm_file = PREPARED_DATA_DIR / "llm_training.json"
        with open(llm_file, 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, ensure_ascii=False, indent=2)

        print(f"   ✅ LLM: {len(llm_data)} paires FR→Mina → {llm_file.name}")

    # 3. Vérifier les fichiers audio
    audio_count = count_files(AUDIO_CLIPS_DIR, "mp3")
    print(f"\n🎵 Fichiers audio disponibles: {audio_count}")

    # 4. Créer le fichier de configuration
    config = {
        "project": "Mina-Translator",
        "version": "1.0",
        "date": datetime.now().isoformat(),
        "data": {
            "stt": {
                "train_samples": len(splits["train"]),
                "dev_samples": len(splits["dev"]),
                "test_samples": len(splits["test"]),
                "audio_dir": str(AUDIO_CLIPS_DIR),
            },
            "llm": {
                "train_samples": len(llm_data) if FR_MINA_CORPUS.exists() else 0,
                "corpus_file": str(FR_MINA_CORPUS),
            }
        },
        "models": {
            "stt": {
                "base_model": "openai/whisper-small",
                "output_dir": str(STT_OUTPUT_DIR),
            },
            "llm": {
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "output_dir": str(LLM_OUTPUT_DIR),
            }
        }
    }

    config_file = PREPARED_DATA_DIR / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Configuration sauvegardée: {config_file}")
    print("\n📋 Résumé:")
    print(f"   STT: {len(splits['train'])} samples pour l'entraînement")
    print(f"   LLM: {len(llm_data)} paires FR→Mina")
    print(f"   Audio: {audio_count} fichiers mp3")


# =============================================================================
# ÉTAPE 2: FINE-TUNING WHISPER (STT)
# =============================================================================

def train_whisper(sample: int = 100, epochs: int = 3):
    """Fine-tune Whisper sur les données Mina."""
    print_section("FINE-TUNING WHISPER (STT)")

    # Vérifier les dépendances
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
        from datasets import load_dataset, Audio
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("   pip install transformers datasets librosa torch torchaudio")
        return False

    # Vérifier GPU
    if not torch.cuda.is_available():
        print("❌ GPU requis pour l'entraînement Whisper")
        return False

    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Charger les données préparées
    train_file = PREPARED_DATA_DIR / "stt_train.json"
    if not train_file.exists():
        print("❌ Données STT non préparées")
        print("   Lancez d'abord: python scripts/prepare_stt_pipeline.py --step prepare")
        return False

    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # Limiter le nombre d'échantillons si nécessaire
    if sample > 0 and len(train_data) > sample:
        train_data = train_data[:sample]
        print(f"⚠️  Mode test: {len(train_data)} samples seulement")

    print(f"\n📚 Données d'entraînement: {len(train_data)} samples")

    # Créer le dataset pour HuggingFace
    print("\n⏳ Création du dataset...")
    from datasets import Dataset

    # Formater pour load_dataset
    audio_paths = [d["audio"] for d in train_data]
    texts = [d["text"] for d in train_data]

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": texts,
    })

    # Cast audio column
    dataset = dataset.cast_column("audio", Audio())

    # Charger le processeur Whisper
    model_name = "openai/whisper-small"  # 244M params, idéal pour 8GB VRAM
    print(f"\n🔄 Chargement de Whisper Small ({model_name})...")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"   ✅ Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres")

    # Fonction de préparation
    def prepare_dataset(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features[0]

        batch["input_features"] = input_features
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    print("\n⏳ Prétraitement des données (peut prendre du temps)...")
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    # Entraînement
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(STT_OUTPUT_DIR),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # Effective: 16
        learning_rate=1e-4,
        warmup_steps=10,
        max_steps=epochs * len(dataset) // 16,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        data_collator=lambda x: {
            "input_features": torch.stack([f for f in x["input_features"]]),
            "labels": torch.tensor(x["labels"]),
        },
    )

    print("\n🚀 Début de l'entraînement STT...")
    trainer.train()

    # Sauvegarder
    print("\n💾 Sauvegarde du modèle STT...")
    model.save_pretrained(str(STT_OUTPUT_DIR / "final"))
    processor.save_pretrained(str(STT_OUTPUT_DIR / "final"))

    print("✅ Entraînement STT terminé!")
    return True


# =============================================================================
# ÉTAPE 3: FINE-TUNING QWEN2 (LLM)
# =============================================================================

def train_llm(epochs: int = 3):
    """Fine-tune Qwen2-0.5B sur les paires FR→Mina."""
    print_section("FINE-TUNING QWEN2 (LLM)")

    # Vérifier les dépendances
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        return False

    if not torch.cuda.is_available():
        print("❌ GPU requis pour l'entraînement LLM")
        return False

    # Charger le corpus
    llm_file = PREPARED_DATA_DIR / "llm_training.json"
    if not llm_file.exists():
        print("❌ Corpus LLM non préparé")
        return False

    with open(llm_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)

    print(f"📚 Paires FR→Mina: {len(training_data)}")

    # Formater les données
    def format_entry(entry):
        return f"""Below is an instruction that describes a task, paired with an input. Write a response.

### Instruction:
{entry['instruction']}

### Input:
{entry['input']}

### Response:
{entry['output']}"""

    formatted_data = [format_entry(e) for e in training_data]

    # Tokenizer
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    print(f"\n🔄 Chargement de {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    dataset = Dataset.from_dict({"text": formatted_data})

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.train_test_split(test_size=0.1)

    # Charger le modèle avec QLoRA
    print("🔄 Chargement du modèle avec QLoRA 4-bit...")

    from bitsandbytes.optimization import Adam8bit

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Entraînement
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(LLM_OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        fp16=True,
        logging_steps=10,
        save_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=lambda x: {
            "input_ids": torch.tensor([f["input_ids"] for f in x]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in x]),
            "labels": torch.tensor([f["input_ids"] for f in x]),
        },
    )

    print("\n🚀 Début de l'entraînement LLM...")
    trainer.train()

    # Sauvegarder
    print("\n💾 Sauvegarde du modèle LLM...")
    model.save_pretrained(str(LLM_OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(LLM_OUTPUT_DIR / "final"))

    print("✅ Entraînement LLM terminé!")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline STT + LLM pour Mina")
    parser.add_argument("--step", "-s", choices=["prepare", "stt", "llm", "all"],
                      default="all", help="Étape à exécuter")
    parser.add_argument("--sample", type=int, default=100,
                      help="Nombre d'échantillons audio pour test (0=tous)")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                      help="Nombre d'epochs")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("   🔧 PIPELINE COMPLET MINA-TRANSLATOR")
    print("   📚 STT (Whisper) + LLM (Qwen2) + API")
    print("=" * 70)

    if args.step == "prepare":
        prepare_data()
    elif args.step == "stt":
        train_whisper(sample=args.sample, epochs=args.epochs)
    elif args.step == "llm":
        train_llm(epochs=args.epochs)
    elif args.step == "all":
        prepare_data()
        if input("\n🚀 Lancer l'entraînement STT? (o/n): ").lower() == 'o':
            train_whisper(sample=args.sample, epochs=args.epochs)
        if input("🚀 Lancer l'entraînement LLM? (o/n): ").lower() == 'o':
            train_llm(epochs=args.epochs)
        print("\n" + "=" * 70)
        print("   ✅ PIPELINE TERMINÉ")
        print("=" * 70)
        print("\n📋 Prochaine étape: Lancer l'API")
        print("   python -m uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()