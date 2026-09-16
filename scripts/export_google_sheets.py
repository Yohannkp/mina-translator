"""
scripts/export_google_sheets.py - Exporte les traductions Google Sheets vers JSONL
================================================================================

Exporte les traductions Mina → Français depuis Google Sheets vers un fichier JSONL
pour l'entraînement du modèle.

Usage:
    python scripts/export_google_sheets.py

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "crowdsource_translations.jsonl"
GOOGLE_SPREADSHEET_ID = os.environ.get("GOOGLE_SPREADSHEET_ID", "")


def export_translations():
    """Exporte les traductions depuis Google Sheets"""

    print("=" * 60)
    print("📤 EXPORT GOOGLE SHEETS → JSONL")
    print("=" * 60)

    # Vérifier les prérequis
    if not CREDENTIALS_FILE.exists():
        print("❌ credentials.json non trouvé!")
        print("   Suivez le guide: docs/SETUP_GOOGLE_SHEETS.md")
        return False

    if not GOOGLE_SPREADSHEET_ID:
        print("❌ GOOGLE_SPREADSHEET_ID non configuré!")
        print("   Exportez: export GOOGLE_SPREADSHEET_ID=votre_id")
        return False

    # Connexion Google Sheets
    print("\n📡 Connexion à Google Sheets...")

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly"
        ]
        credentials = Credentials.from_service_account_file(
            str(CREDENTIALS_FILE),
            scopes=scopes
        )
        gc = gspread.authorize(credentials)

        spreadsheet = gc.open_by_key(GOOGLE_SPREADSHEET_ID)
        sheet = spreadsheet.sheet1

        print("✅ Connexion réussie!")

    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

    # Récupérer les données
    print("\n📥 Récupération des données...")

    try:
        records = sheet.get_all_records()

        if not records:
            print("⚠️ Aucune donnée dans le Google Sheet")
            print("   Commencez par contribuer au crowdsourcing!")
            return False

        print(f"   Trouvé {len(records)} traductions")

        # Filtrer les entrées valides
        valid_records = []
        for record in records:
            mina_text = record.get('mina_text', '')
            french_text = record.get('french_text', '')

            if mina_text and french_text and len(french_text.strip()) > 0:
                valid_records.append({
                    'mina': mina_text.strip(),
                    'french': french_text.strip(),
                    'audio_id': record.get('audio_id', ''),
                    'session_id': record.get('session_id', ''),
                    'created_at': record.get('created_at', ''),
                    'source': 'crowdsource'
                })

        print(f"   {len(valid_records)} traductions valides")

        if not valid_records:
            print("⚠️ Aucune traduction valide trouvée")
            return False

    except Exception as e:
        print(f"❌ Erreur de lecture: {e}")
        return False

    # Créer le répertoire de sortie
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Exporter en JSONL
    print(f"\n💾 Export vers {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in valid_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Export réussi!")
    print(f"   Fichier: {OUTPUT_FILE}")
    print(f"   Traductions: {len(valid_records)}")

    # Afficher un exemple
    print("\n📝 Exemple de traduction:")
    example = valid_records[0]
    print(f"   Mina: {example['mina'][:50]}...")
    print(f"   Français: {example['french'][:50]}...")

    return True


if __name__ == "__main__":
    success = export_translations()
    print("\n" + "=" * 60)
    if success:
        print("✅ Export terminé avec succès!")
    else:
        print("❌ Export échoué. Voir les erreurs ci-dessus.")
    print("=" * 60)