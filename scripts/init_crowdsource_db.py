"""
scripts/init_crowdsource_db.py - Initialise la base SQLite pour le crowdsourcing
===========================================================================

Usage:
    python scripts/init_crowdsource_db.py

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import sqlite3
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"


def init_database():
    """Crée la base SQLite"""

    print("=" * 50)
    print("INITIALISATION CROWDSOURCE DB")
    print("=" * 50)

    # Créer le dossier
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connexion
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Table des traductions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL,
            audio_path TEXT,
            mina_text TEXT NOT NULL,
            french_text TEXT,
            translation_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT
        )
    """)

    # Table des audios vus
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL UNIQUE,
            session_id TEXT,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print(f"✅ Base créée: {DB_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    init_database()