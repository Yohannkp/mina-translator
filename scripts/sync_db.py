"""
scripts/sync_db.py - Synchronise la base de données depuis GitHub
=================================================================

Ce script :
1. Récupère la base de données depuis GitHub (origin/main)
2. Sauvegarde votre base locale avant tout

Usage:
    python scripts/sync_db.py

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def sync_database():
    """Synchronise la base de données depuis GitHub"""

    print("=" * 60)
    print("🔄 SYNCHRONISATION BASE DE DONNÉES")
    print("=" * 60)

    db_path = PROJECT_ROOT / "data" / "mina_crowdsource.db"

    # Taille avant
    if db_path.exists():
        print(f"\n📂 Base actuelle: {db_path}")
        print(f"   Taille: {db_path.stat().st_size / 1024:.1f} KB")

        # Backup local
        backup = db_path.with_suffix(f'.db.backup_{datetime.now():%Y%m%d_%H%M%S}')
        shutil.copy(db_path, backup)
        print(f"📦 Backup créé: {backup.name}")

    # Git fetch pour être sûr d'avoir les dernières données
    print("\n📥 git fetch origin...")
    result = subprocess.run(
        ["git", "fetch", "origin"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"⚠️ Fetch a échoué: {result.stderr}")
    else:
        print("✅ Fetch OK")

    # Récupérer la base depuis origin/main
    print("\n📥 Téléchargement de la base depuis GitHub...")
    result = subprocess.run(
        ["git", "show", f"origin/main:data/mina_crowdsource.db"],
        cwd=PROJECT_ROOT,
        capture_output=True
    )

    if result.returncode != 0:
        print("❌ Fichier non trouvé sur GitHub")
        print(f"   Erreur: {result.stderr.decode() if result.stderr else 'aucune'}")
        return False

    # Écrire le nouveau fichier
    with open(db_path, 'wb') as f:
        f.write(result.stdout)

    print("✅ Base téléchargée depuis GitHub!")

    # Taille après
    if db_path.exists():
        size = db_path.stat().st_size / 1024
        print(f"\n📂 Nouvelle base: {db_path}")
        print(f"   Taille: {size:.1f} KB")

    print("\n" + "=" * 60)
    print("✅ Synchronisation terminée!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    sync_database()