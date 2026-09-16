"""
scripts/sync_google_sheets.py - Synchronise les traductions depuis Google Sheets
=============================================================================

Ce script :
1. Se connecte à Google Sheets (base de données crowdsource)
2. Télécharge toutes les traductions
3. Met à jour la base SQLite locale
4. Exporte au format JSONL pour l'entraînement

Usage:
    python scripts/sync_google_sheets.py

Prérequis:
    1. Créer un projet Google Cloud
    2. Activer l'API Google Sheets
    3. Télécharger credentials.json
    4. Partager le Google Sheet avec le email du service account

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import os
import sys
import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DB = PROJECT_ROOT / "data" / "mina_crowdsource.db"
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "crowdsourced_translations.jsonl"

# Google Sheets (à configurer)
SPREADSHEET_ID = os.environ.get("GOOGLE_SPREADSHEET_ID", "")
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"


# =============================================================================
# FONCTIONS GOOGLE SHEETS
# =============================================================================

def get_google_sheets_data():
    """Récupère les données depuis Google Sheets"""

    if not SPREADSHEET_ID:
        print("❌ GOOGLE_SPREADSHEET_ID non configuré")
        return None

    if not CREDENTIALS_FILE.exists():
        print("❌ credentials.json non trouvé")
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        # Connexion
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
        credentials = Credentials.from_service_account_file(
            str(CREDENTIALS_FILE),
            scopes=scopes
        )
        gc = gspread.authorize(credentials)

        # Ouvrir le spreadsheet
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
        sheet = spreadsheet.sheet1

        # Récupérer les données
        data = sheet.get_all_records()

        return data

    except ImportError:
        print("❌ gspread non installé. Lancez: pip install gspread google-auth")
        return None
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return None


def get_local_translations():
    """Récupère les traductions locales"""

    if not LOCAL_DB.exists():
        return {}

    conn = sqlite3.connect(str(LOCAL_DB))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT audio_id, mina_text, french_text, created_at
        FROM translations
    """)

    translations = {}
    for row in cursor.fetchall():
        translations[row['audio_id']] = dict(row)

    conn.close()
    return translations


def merge_and_save(sheet_data, local_translations):
    """Fusionne et sauvegarde"""

    # Créer un dict des traductions Google Sheets
    new_translations = {}
    for row in sheet_data:
        audio_id = row.get('audio_id', '')
        if audio_id:
            new_translations[audio_id] = {
                'mina': row.get('mina_text', ''),
                'french': row.get('french_text', ''),
                'date': row.get('created_at', '')
            }

    # Fusionner (Google Sheets prend le pas)
    all_translations = local_translations.copy()
    all_translations.update(new_translations)

    # Backup de la base locale
    if LOCAL_DB.exists():
        backup = LOCAL_DB.with_suffix(f'.db.backup_{datetime.now():%Y%m%d_%H%M%S}')
        shutil.copy(LOCAL_DB, backup)
        print(f"📦 Backup: {backup.name}")

    # Sauvegarder
    conn = sqlite3.connect(str(LOCAL_DB))
    cursor = conn.cursor()

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

    for audio_id, data in all_translations.items():
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

    return all_translations, len(new_translations)


def export_for_training(translations, output_path):
    """Exporte au format JSONL"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for audio_id, data in translations.items():
            if data['mina'] and data['french']:
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
    print("=" * 60)
    print("🔄 SYNCHRONISATION GOOGLE SHEETS")
    print("=" * 60)

    # Vérifier la config
    if not SPREADSHEET_ID:
        print("\n⚠️ CONFIGURATION REQUISE:")
        print("\n1. Créer un projet Google Cloud:")
        print("   https://console.cloud.google.com/")
        print("\n2. Activer Google Sheets API")
        print("\n3. Créer un Service Account et télécharger credentials.json")
        print("\n4. Créer un Google Sheet et partager avec le service account")
        print("\n5. Configurer la variable d'environnement:")
        print('   set GOOGLE_SPREADSHEET_ID=votre_id_sheet')
        print(f"\n6. Placer credentials.json dans: {CREDENTIALS_FILE}")
        return

    # Récupérer depuis Google Sheets
    print("\n📥 Téléchargement depuis Google Sheets...")
    sheet_data = get_google_sheets_data()

    if sheet_data is None:
        return

    print(f"   → {len(sheet_data)} lignes trouvées")

    if len(sheet_data) == 0:
        print("\n✅ Aucune nouvelle donnée")
        return

    # Récupérer les traductions locales
    print("\n📥 Chargement base locale...")
    local_translations = get_local_translations()
    print(f"   → {len(local_translations)} traductions locales")

    # Fusionner
    print("\n🔀 Fusion...")
    all_translations, new_count = merge_and_save(sheet_data, local_translations)
    print(f"   → Total: {len(all_translations)}")
    print(f"   → Nouvelles: +{new_count}")

    # Exporter
    print(f"\n📝 Export pour entraînement...")
    count = export_for_training(all_translations, OUTPUT_FILE)
    print(f"   → {count} entrées: {OUTPUT_FILE}")

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"   Google Sheets: {len(sheet_data)}")
    print(f"   Locale: {len(local_translations)}")
    print(f"   Nouvelles: +{new_count}")
    print(f"   Total: {len(all_translations)}")
    print("=" * 60)

    print("\n✅ Synchronisation terminée!")


if __name__ == "__main__":
    main()