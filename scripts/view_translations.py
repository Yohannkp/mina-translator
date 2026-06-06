"""
scripts/view_translations.py - Affiche les traductions enregistrées
===================================================================

Usage:
    python scripts/view_translations.py

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"


def view_translations():
    """Affiche les traductions dans la base"""

    if not DB_PATH.exists():
        print("❌ Base de données non trouvée!")
        print(f"   Chemin: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Nombre total de traductions
    cursor.execute("SELECT COUNT(*) FROM translations")
    total = cursor.fetchone()[0]

    print("=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    print(f"   Total traductions: {total}")
    print()

    if total == 0:
        print("   ℹ️ Aucune traduction enregistrée pour le moment.")
        print("   Lancez l'application et traduisez quelques phrases!")
    else:
        # Afficher les traductions
        cursor.execute("""
            SELECT id, audio_id, mina_text, french_text, translation_type, created_at
            FROM translations
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()

        print("=" * 70)
        print("📝 TRADUCTIONS ENREGISTRÉES")
        print("=" * 70)

        for row in rows:
            print(f"\n   [{row[0]}] {row[5]}")
            print(f"   Audio: {row[1]}")
            print(f"   Mina:  {row[2]}")
            print(f"   FR:    {row[3]}")
            print(f"   Type:  {row[4]}")

        print("\n" + "=" * 70)

    conn.close()


def export_csv():
    """Exporte les traductions en CSV"""

    import csv

    if not DB_PATH.exists():
        print("❌ Base de données non trouvée!")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT audio_id, mina_text, french_text, translation_type, created_at
        FROM translations
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    csv_path = PROJECT_ROOT / "data" / "translations_export.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_id', 'mina_text', 'french_text', 'type', 'date'])
        writer.writerows(rows)

    print(f"✅ Exporté {len(rows)} traductions vers: {csv_path}")

    conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        export_csv()
    else:
        view_translations()