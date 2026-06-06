"""
scripts/setup_db.py - Initialisation de la base de données B2B
===========================================================

Ce script:
1. Initialise la base SQLite (tables, index)
2. Génère une Master Key sécurisée
3. Configure le fichier .env avec la Master Key

Usage:
    python scripts/setup_db.py

Auteur: Claude Opus 4.8 (1M context)
Date: 2026-06-02
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from api.database import init_db, create_client, create_api_key, get_all_clients
from api.security import generate_master_key, hash_api_key

# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_FILE = project_root / "logs" / "setup.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logger.remove()
logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="INFO")
logger.add(sys.stdout, level="INFO")


def create_env_file(master_key: str) -> None:
    """Crée/met à jour le fichier .env avec la Master Key"""

    env_path = project_root / ".env"

    # Lire l existing content
    existing_content = ""
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8", errors="replace") as f:
            existing_content = f.read()

    # Check if la Master Key est déjà configurée
    if "MINA_MASTER_KEY=" in existing_content:
        logger.warning(".env contient déjà une Master Key")
        reponse = input("Voulez-vous la remplacer? (o/N): ")
        if reponse.lower() != 'o':
            logger.info("保留 l'ancienne configuration")
            return

    # Créer le nouveau fichier .env
    lines = existing_content.split('\n') if existing_content else []

    # Remove existing MIN lignes et ajouter la nouvelle
    new_lines = [line for line in lines if not line.startswith("MINA_")]
    new_lines.append(f"MINA_MASTER_KEY={master_key}")
    new_lines.append("# Configuration API B2B")
    new_lines.append("# NE JAMAIS partager cette clé!")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(new_lines))

    logger.info(f"Master Key enregistrée dans .env")


def main():
    """Point d'entrée principal"""

    print("=" * 60)
    print("MINA-TRANSLATOR - CONFIGURATION B2B")
    print("=" * 60)
    print()

    # =============================================================================
    # ÉTAPE 1: Initialiser la base de données
    # =============================================================================

    print("[1/3] Initialisation de la base de données...")

    try:
        init_db()
        logger.info("Base de données initialisée avec succès")
        print("  ✅ Tables created")

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        print(f"  ❌ Erreur: {e}")
        sys.exit(1)

    # =============================================================================
    # ÉTAPE 2: Générer la Master Key
    # =============================================================================

    print("\n[2/3] Génération de la Master Key...")

    master_key = generate_master_key(length=48)

    print(f"  📋 Master Key générée (longueur: {len(master_key)})")
    print()
    print("  ⚠️  ATTENTION: Copiez cette clé maintenant!")
    print("      Elle ne sera plus jamais affichée.")
    print()
    print(f"  🔑 Master Key: {master_key[:8]}...{master_key[-4:]}")
    print()

    reponse = input("Confirmer (o): ")
    if reponse.lower() != 'o':
        print("Annulé.")
        sys.exit(0)

    # Sauvegarder dans .env
    create_env_file(master_key)
    print("  ✅ Master Key sauvegardée")

    # =============================================================================
    # ÉTAPE 3: Créer un client de test (optionnel)
    # =============================================================================

    print("\n[3/3] Créer un client de test?")

    reponse = input("Creer un client test (o/N): ")
    if reponse.lower() == 'o':
        client_name = input("Nom du client [Test-Client]: ") or "Test-Client"
        client_email = input("Email [test@mina-translator.com]: ") or "test@mina-translator.com"

        try:
            client_id = create_client(
                name=client_name,
                email=client_email,
                company="Test Company",
                plan="basic"
            )
            print(f"  ✅ Client créé (ID: {client_id})")

            # Générer une clé API
            key_info = create_api_key(
                client_id,
                name="Test API Key",
                expires_days=30
            )

            print(f"  📋 Clé API (copiez-la):")
            print(f"     {key_info['full_key']}")

        except Exception as e:
            logger.error(f"Erreur création client: {e}")
            print(f"  ❌ Erreur: {e}")

    # =============================================================================
    # RÉSUMÉ
    # =============================================================================

    print()
    print("=" * 60)
    print("CONFIGURATION TERMINÉE")
    print("=" * 60)
    print()
    print("Étapes suivantes:")
    print("  1. Lancez l'API: uvicorn api.main:app --reload")
    print("  2. Testez /health: curl http://localhost:8000/health")
    print("  3. Testez avec Master Key:")
    print(f"     curl -H 'X-API-Key: {master_key}' http://localhost:8000/admin/clients")
    print()


if __name__ == "__main__":
    main()