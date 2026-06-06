# -*- coding: utf-8 -*-
"""
scripts/analyze_data.py - Analyse des données Mina disponibles
===============================================================

Analyse et affiche un rapport complet des données disponibles:
- Corpus FR→Mina (360 paires)
- Audio Common Voice Mina (~20k samples)
- Vocabulaire Mina

Usage:
    python scripts/analyze_data.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
DATA_DIR = PROJECT_DIR / "data"
CV_DIR = DATA_DIR / "cv-corpus-25.0-2026-03-09" / "gej"
CORPUS_DIR = DATA_DIR / "corpus"

# =============================================================================
# ANALYSE DU CORPUS FR→MINA
# =============================================================================

def analyze_fr_mina_corpus():
    """Analyse le corpus de traduction FR→Mina."""
    print("\n" + "=" * 70)
    print("   📚 CORPUS FRANÇAIS → MINA")
    print("=" * 70)

    corpus_files = [
        CORPUS_DIR / "dataset_mina.jsonl",
        CORPUS_DIR / "corpus_mina_5000.jsonl",
        CORPUS_DIR / "final_clean_dataset.jsonl",
        CORPUS_DIR / "mina_full_dataset.jsonl",
    ]

    for file in corpus_files:
        if file.exists():
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            print(f"\n📄 {file.name}")
            print(f"   Lignes: {len(lines)}")

            # Analyser quelques entrées
            entries = []
            for line in lines[:5]:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass

            # Afficher les domaines/themes
            domains = Counter()
            samples = []

            for line in lines[:100]:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        domain = entry.get('domain', entry.get('theme', 'unknown'))
                        if isinstance(domain, str):
                            domains[domain] += 1
                        samples.append(entry)
                    except:
                        pass

            print(f"   Domaines: {dict(domains)}")
            print(f"   Exemple:")
            if samples:
                s = samples[0]
                fr = s.get('input', s.get('fr', ''))
                mina = s.get('output', s.get('mina', ''))
                print(f"     FR: {fr[:60]}...")
                print(f"     MINA: {mina[:60]}...")

    return len(lines) if lines else 0


# =============================================================================
# ANALYSE DES DONNÉES AUDIO COMMON VOICE
# =============================================================================

def analyze_audio_data():
    """Analyse les données audio Common Voice Mina."""
    print("\n" + "=" * 70)
    print("   🎵 DONNÉES AUDIO COMMON VOICE (MINA)")
    print("=" * 70)

    if not CV_DIR.exists():
        print("\n❌ Dossier Common Voice non trouvé")
        return 0

    # Lire les fichiers TSV
    splits = {
        "train": CV_DIR / "train.tsv",
        "dev": CV_DIR / "dev.tsv",
        "test": CV_DIR / "test.tsv",
        "validated": CV_DIR / "validated.tsv",
    }

    total_audio = 0
    total_hours = 0

    for name, path in splits.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header

            count = len(lines)

            # Estimer la durée
            durations_file = CV_DIR / "clip_durations.tsv"
            total_duration = 0

            if durations_file.exists():
                with open(durations_file, 'r', encoding='utf-8') as f:
                    durations = f.readlines()[1:]
                    total_duration = sum(float(d.split('\t')[1]) for d in durations if len(d.split('\t')) > 1)

            hours = total_duration / 3600

            print(f"\n📊 Split: {name}")
            print(f"   Samples: {count:,}")
            print(f"   Durée estimée: {hours:.1f} heures")

            total_audio += count
            total_hours += hours

            # Exemples de phrases
            if name == "train" and lines:
                print(f"\n   📝 Exemples de phrases Mina:")
                for line in lines[:3]:
                    parts = line.strip().split('\t')
                    if len(parts) > 3:
                        sentence = parts[3]
                        print(f"     - {sentence[:70]}...")

    # Compter les fichiers audio
    clips_dir = CV_DIR / "clips"
    if clips_dir.exists():
        mp3_files = list(clips_dir.glob("*.mp3"))
        print(f"\n📁 Fichiers audio MP3: {len(mp3_files):,}")

    print(f"\n📈 TOTAL:")
    print(f"   Audio samples: {total_audio:,}")
    print(f"   Durée totale: {total_hours:.1f} heures")

    return total_audio


# =============================================================================
# ANALYSE DU VOCABULAIRE
# =============================================================================

def analyze_vocabulary():
    """Analyse le vocabulaire Mina."""
    print("\n" + "=" * 70)
    print("   📖 VOCABULAIRE MINA")
    print("=" * 70)

    vocab_file = CORPUS_DIR / "vocabulaire_mina.json"

    if not vocab_file.exists():
        print("❌ Vocabulaire non trouvé")
        return 0

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    print(f"\n📁 Fichier: {vocab_file.name}")
    print(f"   Version: {vocab.get('metadata', {}).get('version', 'N/A')}")
    print(f"   Description: {vocab.get('metadata', {}).get('description', 'N/A')}")

    # Catégories
    categories = vocab.get("vocabulary", {})
    print(f"\n📚 Catégories: {len(categories)}")

    total_terms = 0
    for category, terms in categories.items():
        if isinstance(terms, dict):
            count = len(terms)
            total_terms += count
            print(f"   - {category}: {count} termes")

    # Expressions communes
    expressions = vocab.get("expressions_communes", {})
    print(f"\n💬 Expressions communes: {len(expressions)}")

    # Règles de syntaxe
    rules = vocab.get("regles_syntaxe", {})
    if rules:
        print(f"\n📐 Règles de syntaxe:")
        for rule, value in rules.items():
            print(f"   - {rule}: {value}")

    return total_terms


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

def print_summary(fr_mina_count: int, audio_count: int, vocab_count: int):
    """Affiche le résumé final."""
    print("\n" + "=" * 70)
    print("   📊 RÉSUMÉ DES DONNÉES MINA-AVAILABLE")
    print("=" * 70)

    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│  TYPE DE DONNÉES              │  QUANTITÉ        │  STATUT       │
├────────────────────────────────────────────────────────────────────┤
│  Paires FR → MINA (texte)     │  {fr_mina_count:>10,}        │  ✅ Prêt    │
│  Audio Mina (Common Voice)     │  {audio_count:>10,}        │  ✅ Prêt    │
│  Vocabulaire Mina              │  {vocab_count:>10,}         │  ✅ Prêt    │
├────────────────────────────────────────────────────────────────────┤
│  TOTAL                         │  ~{fr_mina_count + audio_count:,}        │               │
└────────────────────────────────────────────────────────────────────┘

📋 DONNÉES UTILISABLES POUR:

   1️⃣  FINE-TUNING LLM (Qwen2)
       └─> 360 paires FR→Mina pour traduction texte
       └─> Corpus: data/corpus/dataset_mina.jsonl

   2️⃣  FINE-TUNING STT (Whisper)
       └─> ~19,000 fichiers audio Mina avec transcriptions
       └─> Dossier: data/cv-corpus-25.0-2026-03-09/gej/

   3️⃣  VOCABULAIRE MINA
       └─> ~{vocab_count} termes Mina avec équivalents FR
       └─> Fichier: data/corpus/vocabulaire_mina.json

🚀 PROCHAINES ÉTAPES RECOMMANDÉES:

   1. Entraîner Whisper sur les audio Mina:
      python scripts/prepare_stt_pipeline.py --step stt

   2. Entraîner Qwen2 sur les paires FR→Mina:
      python scripts/train_mina_translator.py --train

   3. Lancer l'API complète:
      python -m uvicorn api.mina_api:app --reload --port 8000

   ⚠️  ATTENTION: Les entraînements nécessitent GPU avec CUDA
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("   🔍 ANALYSE DES DONNÉES MINA-TRANSLATOR")
    print("   " + "=" * 70)

    # Analyser chaque type de données
    fr_mina_count = analyze_fr_mina_corpus()
    audio_count = analyze_audio_data()
    vocab_count = analyze_vocabulary()

    # Résumé
    print_summary(fr_mina_count, audio_count, vocab_count)


if __name__ == "__main__":
    main()