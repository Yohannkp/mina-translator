"""
scripts/sync_db_github.py - Synchronise la base SQLite avec GitHub
===================================================================

Ce script permet de:
1. Upload la base SQLite vers GitHub (commit)
2. Download la base SQLite depuis GitHub (pull)

Usage:
    python scripts/sync_db_github.py --push    # Upload vers GitHub
    python scripts/sync_db_github.py --pull    # Download depuis GitHub

Prérequis:
    - Installer gh cli: https://cli.github.com/
    - Se connecter: gh auth login
    - Configurer le repo: gh repo clone {user}/{repo}

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import os
import sys
import sqlite3
import requests
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# IMPORTANT: Remplacez ces valeurs avec vos informations GitHub
GITHUB_USER = "TON_USER_GITHUB"
GITHUB_REPO = "TON_REPO"
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Variable d'environnement

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"
DATA_DIR = PROJECT_ROOT / "data"

# URL API GitHub
API_URL = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"

# =============================================================================
# FONCTIONS
# =============================================================================

def get_headers():
    """Retourne les headers pour l'API GitHub"""

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    return headers


def get_file_sha(path):
    """Récupère le SHA d'un fichier existant sur GitHub"""

    url = f"{API_URL}/contents/{path}"

    try:
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            return response.json()["sha"]
    except Exception as e:
        print(f"   Erreur lors de la récupération du SHA: {e}")

    return None


def push_to_github(file_path, remote_path):
    """Upload un fichier vers GitHub"""

    if not file_path.exists():
        print(f"   ❌ Fichier non trouvé: {file_path}")
        return False

    if not GITHUB_TOKEN:
        print("   ⚠️ GITHUB_TOKEN non configuré. Utilisation de gh CLI...")

        # Utiliser gh cli
        os.system(f"gh repo sync -b {GITHUB_BRANCH} || true")

        # Commit via git
        os.system(f"git add {file_path}")
        os.system(f"git commit -m 'Update crowdsource DB: {datetime.now().isoformat()}'")
        os.system(f"git push origin {GITHUB_BRANCH}")

        return True

    # Upload via API
    with open(file_path, "rb") as f:
        content = f.read()

    import base64
    encoded_content = base64.b64encode(content).decode()

    url = f"{API_URL}/contents/{remote_path}"
    data = {
        "message": f"Update crowdsource DB: {datetime.now().isoformat()}",
        "content": encoded_content,
        "branch": GITHUB_BRANCH
    }

    # Ajouter le SHA si le fichier existe
    sha = get_file_sha(remote_path)
    if sha:
        data["sha"] = sha

    try:
        response = requests.put(url, json=data, headers=get_headers())

        if response.status_code in [200, 201]:
            print(f"   ✅ Fichier uploadé: {remote_path}")
            return True
        else:
            print(f"   ❌ Erreur: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def pull_from_github(remote_path, local_path):
    """Download un fichier depuis GitHub"""

    if not GITHUB_TOKEN:
        print("   ⚠️ Utilisation de gh CLI pour download...")
        # Le fichier sera sync via gh repo sync
        return False

    url = f"{API_URL}/contents/{remote_path}"

    try:
        response = requests.get(url, headers=get_headers())

        if response.status_code == 200:
            content = response.json()["content"]

            # Décoder le base64
            import base64
            decoded = base64.b64decode(content)

            # Sauvegarder localement
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(decoded)

            print(f"   ✅ Fichier téléchargé: {local_path}")
            return True

        elif response.status_code == 404:
            print(f"   ℹ️ Fichier non trouvé sur GitHub: {remote_path}")
            return False
        else:
            print(f"   ❌ Erreur: {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def get_db_stats():
    """Affiche les statistiques de la base"""

    if not DB_PATH.exists():
        print("   ℹ️ Base de données non créée")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    print("\n📊 Statistiques de la base:")

    # Total traductions
    cursor.execute("SELECT COUNT(*) FROM translations")
    total = cursor.fetchone()[0]
    print(f"   • Total traductions: {total}")

    # Audios vus
    cursor.execute("SELECT COUNT(*) FROM seen_audios")
    seen = cursor.fetchone()[0]
    print(f"   • Audios vus: {seen}")

    # Traductions validées
    cursor.execute("SELECT COUNT(*) FROM validated_translations")
    validated = cursor.fetchone()[0]
    print(f"   • Traductions validées: {validated}")

    # Taille du fichier
    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"   • Taille fichier: {size_mb:.2f} MB")

    conn.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync SQLite DB avec GitHub pour Mina Crowdsource"
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload la base vers GitHub"
    )

    parser.add_argument(
        "--pull",
        action="store_true",
        help="Download la base depuis GitHub"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques de la base"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MINA CROWDSOURCE - SYNC GITHUB")
    print("=" * 60)

    # Stats
    if args.stats or (not args.push and not args.pull):
        get_db_stats()

    # Push (upload)
    if args.push:
        print("\n📤 Upload vers GitHub...")

        if DB_PATH.exists():
            success = push_to_github(
                DB_PATH,
                f"data/mina_crowdsource.db"
            )

            if success:
                print("\n✅ Upload terminé avec succès !")
            else:
                print("\n❌ Erreur lors de l'upload")
        else:
            print("   ℹ️ La base n'existe pas encore. Créez-la d'abord:")
            print("   python scripts/init_crowdsource_db.py")

    # Pull (download)
    if args.pull:
        print("\n📥 Download depuis GitHub...")

        success = pull_from_github(
            f"data/mina_crowdsource.db",
            DB_PATH
        )

        if success:
            print("\n✅ Download terminé avec succès !")
            get_db_stats()
        else:
            print("\n❌ Erreur lors du download")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()