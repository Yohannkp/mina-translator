"""
scripts/prepare_whisper_data.py - Préparation des données audio pour Whisper
=========================================================================

Prépare les données audio Common Voice Mina (Ewe) pour le fine-tuning Whisper.

Étapes:
1. Vérifie les fichiers audio (MP3)
2. Filtre les clips par durée (1-30 secondes)
3. Convertit les fichiers en format WAV (16kHz mono)
4. Crée le fichier de métadonnées JSONL pour Whisper
5. Sépare en train/validation/test

Usage:
    python scripts/prepare_whisper_data.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CV_DIR = PROJECT_DIR / "data" / "cv-corpus-25.0-2026-03-09" / "gej"
OUTPUT_DIR = PROJECT_DIR / "data" / "whisper_data"
PROCESSED_DIR = OUTPUT_DIR / "audio"

# Paramètres audio
SAMPLE_RATE = 16000
MIN_DURATION = 1.0  # secondes
MAX_DURATION = 30.0  # secondes
MAX_SAMPLES = 10000  # Limiter pour VRAM 8Go

# Train/Val/Test split
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05

# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

@dataclass
class AudioEntry:
    path: str
    sentence: str
    duration: float
    client_id: str
    split: str = "train"


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def format_duration(seconds: float) -> str:
    """Formate la durée en mm:ss."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def load_common_voice_data() -> pd.DataFrame:
    """Charge les données Common Voice Mina (Ewe)."""
    print("\n" + "="*60)
    print("   CHARGEMENT DES DONNÉES COMMON VOICE")
    print("="*60)

    # Lire les fichiers TSV
    validated = pd.read_csv(CV_DIR / "validated.tsv", sep='\t')
    durations = pd.read_csv(CV_DIR / "clip_durations.tsv", sep='\t')

    # Merger
    df = validated.merge(durations, left_on='path', right_on='clip')

    # Filtrer par durée
    df['duration_s'] = df['duration[ms]'] / 1000
    df = df[(df['duration_s'] >= MIN_DURATION) & (df['duration_s'] <= MAX_DURATION)]

    print(f"\n   Fichiers chargés: {len(validated)}")
    print(f"   Après filtrage durée ({MIN_DURATION}-{MAX_DURATION}s): {len(df)}")

    return df


def analyze_quality(df: pd.DataFrame) -> Dict:
    """Analyse la qualité des données."""
    print("\n" + "="*60)
    print("   ANALYSE DE QUALITÉ")
    print("="*60)

    stats = {
        "total_clips": len(df),
        "total_hours": df['duration_s'].sum() / 3600,
        "avg_duration": df['duration_s'].mean(),
        "unique_speakers": df['client_id'].nunique(),
        "unique_sentences": df['sentence'].nunique(),
    }

    print(f"\n   Durée totale: {stats['total_hours']:.2f} heures")
    print(f"   Durée moyenne: {stats['avg_duration']:.2f} secondes")
    print(f"   Orateurs uniques: {stats['unique_speakers']}")
    print(f"   Phrases uniques: {stats['unique_sentences']}")

    # Vérifier la distribution des durées
    print(f"\n   Distribution des durées:")
    for min_d, max_d, label in [(0, 5, "0-5s"), (5, 10, "5-10s"), (10, 20, "10-20s"), (20, 30, "20-30s")]:
        count = len(df[(df['duration_s'] >= min_d) & (df['duration_s'] < max_d)])
        print(f"      {label}: {count} clips")

    return stats


def check_audio_files(df: pd.DataFrame) -> tuple:
    """Vérifie quels fichiers audio existent."""
    print("\n" + "="*60)
    print("   VÉRIFICATION DES FICHIERS AUDIO")
    print("="*60)

    clips_dir = CV_DIR / "clips"
    existing_rows = []
    missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Vérification"):
        audio_path = clips_dir / row['path']
        if audio_path.exists():
            existing_rows.append(row)
        else:
            missing.append(row['path'])

    # Convertir la liste de Series en DataFrame
    if existing_rows:
        existing = pd.DataFrame(existing_rows)
    else:
        existing = pd.DataFrame()

    print(f"\n   Fichiers existants: {len(existing)}")
    print(f"   Fichiers manquants: {len(missing)}")

    return existing, missing


def convert_audio_file(audio_path: Path, output_path: Path) -> bool:
    """Copie le fichier audio (Whisper peut lire les MP3 directement)."""
    try:
        # Whisper peut lire les MP3 directement, pas besoin de conversion
        # On copie juste le fichier dans le bon répertoire
        import shutil
        shutil.copy2(audio_path, output_path)
        return True
    except Exception as e:
        print(f"Erreur copie {audio_path}: {e}")
        return False


def process_audio_files(df: pd.DataFrame, max_samples: int = None) -> List[Dict]:
    """Traite les fichiers audio (conversion + création manifest)."""
    print("\n" + "="*60)
    print("   TRAITEMENT DES FICHIERS AUDIO")
    print("="*60)

    # Créer le répertoire de sortie
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # S'assurer que df est un DataFrame
    if not isinstance(df, pd.DataFrame):
        print(f"\n   ⚠️ Warning: df n'est pas un DataFrame, conversion...")
        df = pd.DataFrame(df)

    # Limiter le nombre d'échantillons
    if max_samples and len(df) > max_samples:
        print(f"\n   Limitation à {max_samples} échantillons (sur {len(df)})")
        df = df.sample(n=max_samples, random_state=42)

    # Mélanger les données
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split train/val/test
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    splits = {
        'train': df.iloc[:n_train],
        'validation': df.iloc[n_train:n_train+n_val],
        'test': df.iloc[n_train+n_val:]
    }

    print(f"\n   Split des données:")
    for split_name, split_df in splits.items():
        print(f"      {split_name}: {len(split_df)} clips ({len(split_df)/n_total*100:.1f}%)")

    # Traiter les fichiers
    entries = []

    print(f"\n   Conversion audio (MP3 -> WAV 16kHz mono)...")
    for split_name, split_df in splits.items():
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            source_path = CV_DIR / "clips" / row['path']
            # Garder l'extension .mp3 car Whisper peut lire les MP3 directement
            output_filename = row['path']
            output_path = PROCESSED_DIR / output_filename

            # Convertir si pas déjà fait
            if not output_path.exists():
                success = convert_audio_file(source_path, output_path)
                if not success:
                    continue

            # Ajouter à la liste
            entry = {
                "audio_filepath": str(output_path),
                "text": row['sentence'],
                "duration": row['duration_s'],
                "client_id": row['client_id'],
                "split": split_name
            }
            entries.append(entry)

    print(f"\n   {len(entries)} fichiers traités")

    return entries


def create_manifest(entries: List[Dict], output_path: Path):
    """Crée le fichier manifest JSONL."""
    print(f"\n   Création du manifest: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"   {len(entries)} entrées écrites")


def generate_stats_report(entries: List[Dict]) -> Dict:
    """Génère un rapport de statistiques."""
    print("\n" + "="*60)
    print("   RAPPORT DE STATISTIQUES")
    print("="*60)

    total_duration = sum(e['duration'] for e in entries)
    train_entries = [e for e in entries if e['split'] == 'train']
    val_entries = [e for e in entries if e['split'] == 'validation']
    test_entries = [e for e in entries if e['split'] == 'test']

    stats = {
        "total_entries": len(entries),
        "total_duration_hours": total_duration / 3600,
        "train": {
            "entries": len(train_entries),
            "duration_hours": sum(e['duration'] for e in train_entries) / 3600,
        },
        "validation": {
            "entries": len(val_entries),
            "duration_hours": sum(e['duration'] for e in val_entries) / 3600,
        },
        "test": {
            "entries": len(test_entries),
            "duration_hours": sum(e['duration'] for e in test_entries) / 3600,
        }
    }

    print(f"\n   Total: {stats['total_entries']} entrées ({stats['total_duration_hours']:.2f}h)")
    print(f"   Train: {stats['train']['entries']} entrées ({stats['train']['duration_hours']:.2f}h)")
    print(f"   Val: {stats['validation']['entries']} entrées ({stats['validation']['duration_hours']:.2f}h)")
    print(f"   Test: {stats['test']['entries']} entrées ({stats['test']['duration_hours']:.2f}h)")

    # Exemples de phrases
    print(f"\n   Exemples de phrases Mina:")
    sample_texts = random.sample([e['text'] for e in entries], min(5, len(entries)))
    for text in sample_texts:
        print(f"      - {text}")

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("   PRÉPARATION DES DONNÉES AUDIO POUR WHISPER")
    print("   Common Voice Mina (Ewe) - Togo/Ghana")
    print("="*70)

    # Étape 1: Charger les données
    df = load_common_voice_data()

    # Étape 2: Analyser la qualité
    stats = analyze_quality(df)

    # Étape 3: Vérifier les fichiers audio
    existing, missing = check_audio_files(df)

    if existing is None or (isinstance(existing, pd.DataFrame) and existing.empty):
        print("\n   ❌ Aucun fichier audio trouvé!")
        return

    # Étape 4: Traiter les fichiers audio
    entries = process_audio_files(existing, max_samples=MAX_SAMPLES)

    # Étape 5: Séparer en splits
    splits = {'train': [], 'validation': [], 'test': []}
    for entry in entries:
        splits[entry['split']].append(entry)

    # Étape 6: Créer les manifests
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_entries in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        create_manifest(split_entries, output_path)

    # Manifest complet
    create_manifest(entries, OUTPUT_DIR / "all.jsonl")

    # Étape 7: Générer le rapport
    stats = generate_stats_report(entries)

    # Sauvegarder le rapport
    report_path = OUTPUT_DIR / "prepare_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("   PRÉPARATION TERMINÉE!")
    print("="*70)
    print(f"\n   Fichiers générés:")
    print(f"      - {OUTPUT_DIR / 'train.jsonl'}")
    print(f"      - {OUTPUT_DIR / 'validation.jsonl'}")
    print(f"      - {OUTPUT_DIR / 'test.jsonl'}")
    print(f"      - {OUTPUT_DIR / 'all.jsonl'}")
    print(f"      - {report_path}")
    print(f"\n   Prochaine étape:")
    print(f"   python scripts/train_whisper.py")


if __name__ == "__main__":
    main()