"""
scripts/inspect_db.py - Inspecte la base SQLite
================================================

Usage:
    python scripts/inspect_db.py
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"


def inspect_database():
    """Affiche la structure et le contenu de la base"""

    print("=" * 70)
    print("🔍 INSPECTION DE LA BASE DE DONNÉES")
    print("=" * 70)

    if not DB_PATH.exists():
        print("❌ Base non trouvée!")
        return

    print(f"\n📁 Fichier: {DB_PATH}")
    print(f"📏 Taille: {DB_PATH.stat().st_size} octets")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ============================================================
    # LISTE DES TABLES
    # ============================================================

    print("\n" + "=" * 70)
    print("📋 TABLES")
    print("=" * 70)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    for table in tables:
        name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        print(f"   • {name}: {count} lignes")

    # ============================================================
    # STRUCTURE DE CHAQUE TABLE
    # ============================================================

    for table in tables:
        name = table[0]

        print("\n" + "=" * 70)
        print(f"📊 Structure de la table: {name}")
        print("=" * 70)

        cursor.execute(f"PRAGMA table_info({name})")
        columns = cursor.fetchall()

        print("\nColonnes:")
        for col in columns:
            print(f"   • {col['name']} ({col['type']})")

        # Contenu
        cursor.execute(f"SELECT * FROM {name} LIMIT 10")
        rows = cursor.fetchall()

        print(f"\nContenu (max 10 lignes):")
        if rows:
            for row in rows:
                print(f"\n   --- Ligne ---")
                for col in columns:
                    val = row[col['name']]
                    # Tronquer les chaînes longues
                    if isinstance(val, str) and len(val) > 50:
                        val = val[:50] + "..."
                    print(f"   {col['name']}: {val}")
        else:
            print("   (vide)")

    # ============================================================
    # STATISTIQUES
    # ============================================================

    print("\n" + "=" * 70)
    print("📈 STATISTIQUES")
    print("=" * 70)

    cursor.execute("SELECT COUNT(*) FROM translations")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT audio_id) FROM translations")
    unique_audios = cursor.fetchone()[0]

    cursor.execute("SELECT translation_type, COUNT(*) FROM translations GROUP BY translation_type")
    types = cursor.fetchall()

    print(f"\n   Total traductions: {total}")
    print(f"   Audios uniques: {unique_audios}")

    if types:
        print("\n   Par type:")
        for t in types:
            print(f"   • {t[0]}: {t[1]}")

    # ============================================================
    # EXEMPLE DE TRADUCTION COMPLETE
    # ============================================================

    print("\n" + "=" * 70)
    print("📝 EXEMPLE DE TRADUCTION")
    print("=" * 70)

    cursor.execute("SELECT * FROM translations LIMIT 2")
    rows = cursor.fetchall()

    if rows:
        for i, row in enumerate(rows, 1):
            print(f"\n   === Traduction #{i} ===")
            for key in row.keys():
                val = row[key]
                if val is None:
                    val = "NULL"
                elif isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                print(f"   {key}: {val}")
    else:
        print("\n   Aucune traduction!")

    conn.close()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    inspect_database()