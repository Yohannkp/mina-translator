"""
scripts/sync_from_github.py - Synchronise la base SQLite depuis GitHub
====================================================================

Ce script :
1. Télécharge la dernière base SQLite depuis GitHub
2. Compare avec la base locale
3. Fusionne les nouvelles traductions
4. Met à jour la base locale

Usage:
    python scripts/sync_from_github.py

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import sqlite3
import requests
import json
import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DB = PROJECT_ROOT / "data" / "mina_crowdsource.db"
GITHUB_REPO = "Yohannkp/Api-Fran-ais-a-Mina"
GITHUB_BRANCH = "main"
GITHUB_PATH = "data/mina_crowdsource.db"

# Token GitHub (optionnel, augmente les limites)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# =============================================================================
# FONCTIONS
# =============================================================================

def get_github_file_content(path, repo, branch="main"):
    """Récupère le contenu d'un fichier depuis GitHub"""

    url = f"https://api.github.com/repos/{repo}/contents/{path}"

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    response = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)

    if response.status_code == 200:
        content = response.json()
        if content.get("encoding") == "base64":
            import base64
            return base64.b64decode(content["content"])
    elif response.status_code == 404:
        print(f"❌ Fichier non trouvé sur GitHub: {path}")
        return None
    else:
        print(f"❌ Erreur GitHub: {response.status_code}")

    return None


def get_local_translations():
    """Récupère toutes les traductions locales"""

    if not LOCAL_DB.exists():
        return {}

    conn = sqlite3.connect(str(LOCAL_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT audio_id, mina_text, french_text, created_at
        FROM translations
        ORDER BY created_at
    """)

    translations = {}
    for row in cursor.fetchall():
        key = row['audio_id']
        translations[key] = dict(row)

    conn.close()
    return translations


def get_github_translations(db_bytes):
    """Récupère les traductions depuis la base GitHub"""

    import tempfile
    import os

    if db_bytes is None:
        return {}

    # Sauvegarder temporairement
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.write(db_bytes)
    temp_db.close()

    try:
        conn = sqlite3.connect(temp_db.name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT audio_id, mina_text, french_text, created_at
            FROM translations
            ORDER BY created_at
        """)

        translations = {}
        for row in cursor.fetchall():
            key = row['audio_id']
            translations[key] = dict(row)

        conn.close()

    finally:
        os.unlink(temp_db.name)

    return translations


def merge_translations(local_translations, github_translations):
    """Fusionne les traductions"""

    merged = local_translations.copy()

    new_count = 0
    for audio_id, translation in github_translations.items():
        if audio_id not in merged:
            merged[audio_id] = translation
            new_count += 1

    return merged, new_count


def save_merged_translations(translations, output_path):
    """Sauvegarde les traductions fusionnées"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Ajouter les traductions
    for audio_id, trans in translations.items():
        cursor.execute("""
            INSERT OR IGNORE INTO translations
            (audio_id, mina_text, french_text, translation_type, created_at)
            VALUES (?, ?, ?, 'text', ?)
        """, (
            audio_id,
            trans.get('mina_text', ''),
            trans.get('french_text', ''),
            trans.get('created_at', datetime.now().isoformat())
        ))

    conn.commit()
    conn.close()


def export_for_training(translations, output_path):
    """Exporte les traductions au format JSONL pour l'entraînement"""

    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for audio_id, trans in translations.items():
            if trans.get('mina_text') and trans.get('french_text'):
                entry = {
                    "instruction": "Traduis cette phrase de l'Ewe (Mina du Togo) en Français.",
                    "input": trans['mina_text'],
                    "output": trans['french_text']
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Synchronise la base depuis GitHub"""

    print("=" * 60)
    print("🔄 SYNCHRONISATION DEPUIS GITHUB")
    print("=" * 60)

    # 1. Récupérer les traductions locales
    print("\n📥 Étape 1: Chargement des traductions locales...")
    local_translations = get_local_translations()
    print(f"   → {len(local_translations)} traductions trouvées")

    # 2. Télécharger la base depuis GitHub
    print("\n📥 Étape 2: Téléchargement depuis GitHub...")
    db_bytes = get_github_file_content(GITHUB_PATH, GITHUB_REPO, GITHUB_BRANCH)

    github_translations = {}
    if db_bytes:
        github_translations = get_github_translations(db_bytes)
        print(f"   → {len(github_translations)} traductions sur GitHub")
    else:
        print("   → Aucune base sur GitHub (première sync)")

    # 3. Fusionner
    print("\n🔀 Étape 3: Fusion des traductions...")
    merged, new_count = merge_translations(local_translations, github_translations)
    print(f"   → Total après fusion: {len(merged)} traductions")
    print(f"   → Nouvelles traductions: {new_count}")

    if new_count == 0:
        print("\n✅ Tout est déjà à jour !")
        return

    # 4. Sauvegarder
    print("\n💾 Étape 4: Sauvegarde...")

    # Backup de la base locale
    backup_path = LOCAL_DB.with_suffix(f'.db.backup_{datetime.now():%Y%m%d_%H%M%S}')
    if LOCAL_DB.exists():
        import shutil
        shutil.copy(LOCAL_DB, backup_path)
        print(f"   → Backup créé: {backup_path.name}")

    # Sauvegarder la base fusionnée
    save_merged_translations(merged, LOCAL_DB)
    print(f"   → Base locale mise à jour: {LOCAL_DB}")

    # Exporter pour l'entraînement
    training_file = PROJECT_ROOT / "data" / "corpus" / "crowdsourced_translations.jsonl"
    export_for_training(merged, training_file)
    print(f"   → Export entraînement: {training_file}")

    # 5. Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"   Traductions locales: {len(local_translations)}")
    print(f"   Traductions GitHub: {len(github_translations)}")
    print(f"   Nouvelles: +{new_count}")
    print(f"   Total: {len(merged)}")
    print("=" * 60)

    print("\n✅ Synchronisation terminée !")
    print("\n💡 Prochaine étape:")
    print("   python scripts/train_mina_llm.py")


if __name__ == "__main__":
    main()