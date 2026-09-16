"""
scripts/merge_local_db.py - Fusionne une base téléchargée avec la base locale
============================================================================

Usage:
    python scripts/merge_local_db.py
    python scripts/merge_local_db.py --input path/to/downloaded.db

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DB = PROJECT_ROOT / "data" / "mina_crowdsource.db"
CROWD_DB = PROJECT_ROOT / "data" / "mina_crowdsource.db"  # Même fichier
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "crowdsourced_translations.jsonl"


def load_translations(db_path):
    """Charge les traductions depuis une base SQLite"""

    if not db_path or not Path(db_path).exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT audio_id, mina_text, french_text, created_at
        FROM translations
        WHERE french_text IS NOT NULL
        AND french_text != ''
        AND LENGTH(french_text) > 1
        ORDER BY created_at DESC
    """)

    translations = {}
    for row in cursor.fetchall():
        key = row['audio_id']
        translations[key] = {
            'mina': row['mina_text'],
            'french': row['french_text'],
            'date': row['created_at']
        }

    conn.close()
    return translations


def merge_and_save(local_translations, crowd_translations):
    """Fusionne et sauvegarde"""

    # Fusionner (crowd prend le pas sur local)
    all_translations = {}

    # D'abord local
    for audio_id, data in local_translations.items():
        all_translations[audio_id] = data

    # Puis crowd (écrase si doublon)
    for audio_id, data in crowd_translations.items():
        all_translations[audio_id] = data

    return all_translations


def save_to_database(translations, output_path):
    """Sauvegarde dans la base SQLite"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup
    if output_path.exists():
        backup_path = output_path.with_suffix(f'.db.backup_{datetime.now():%Y%m%d_%H%M%S}')
        shutil.copy(output_path, backup_path)
        print(f"📦 Backup créé: {backup_path.name}")

    conn = sqlite3.connect(str(output_path))
    cursor = conn.cursor()

    # Créer les tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL,
            audio_path TEXT,
            mina_text TEXT NOT NULL,
            french_text TEXT,
            translation_type TEXT DEFAULT 'text',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL UNIQUE,
            session_id TEXT,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insérer les traductions
    for audio_id, data in translations.items():
        cursor.execute("""
            INSERT OR REPLACE INTO translations
            (audio_id, mina_text, french_text, translation_type, created_at)
            VALUES (?, ?, ?, 'text', ?)
        """, (
            audio_id,
            data['mina'],
            data['french'],
            data.get('date', datetime.now().isoformat())
        ))

    conn.commit()
    conn.close()


def export_for_training(translations, output_path):
    """Exporte au format JSONL"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for audio_id, data in translations.items():
            entry = {
                "instruction": "Traduis cette phrase de l'Ewe (Mina du Togo) en Français.",
                "input": data['mina'],
                "output": data['french']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    return count


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fusionne les bases de données")
    parser.add_argument('--input', '-i', type=str,
                        help="Chemin vers la base téléchargée (par défaut: data/mina_crowdsource.db)")
    parser.add_argument('--corpus', '-c', type=str,
                        default=str(OUTPUT_FILE),
                        help="Fichier de sortie JSONL")

    args = parser.parse_args()

    print("=" * 60)
    print("🔄 FUSION DES BASES DE DONNÉES")
    print("=" * 60)

    # Charger la base crowd (téléchargée depuis Streamlit Cloud)
    input_db = args.input or str(CROWD_DB)

    print(f"\n📥 Chargement de: {input_db}")
    crowd_translations = load_translations(input_db)
    print(f"   → {len(crowd_translations)} traductions")

    # Charger la base locale
    print(f"\n📥 Chargement de: {LOCAL_DB}")
    local_translations = load_translations(str(LOCAL_DB))
    print(f"   → {len(local_translations)} traductions")

    # Fusionner
    print("\n🔀 Fusion des traductions...")
    merged = merge_and_save(local_translations, crowd_translations)
    print(f"   → Total: {len(merged)} traductions")

    # Sauvegarder
    print(f"\n💾 Sauvegarde...")
    save_to_database(merged, LOCAL_DB)
    print(f"   → Base mise à jour: {LOCAL_DB}")

    # Exporter pour l'entraînement
    print(f"\n📝 Export pour entraînement...")
    count = export_for_training(merged, Path(args.corpus))
    print(f"   → {count} entrées exportées: {args.corpus}")

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"   Crowd (Streamlit): {len(crowd_translations)}")
    print(f"   Local: {len(local_translations)}")
    print(f"   Après fusion: {len(merged)}")
    print("=" * 60)

    print("\n✅ Terminé !")
    print(f"\n💡 Maintenant lancez:")
    print(f"   python scripts/train_mina_llm.py")


if __name__ == "__main__":
    main()