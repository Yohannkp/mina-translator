"""
Script d'initialisation pour Mina-Translator B2B
=================================================
Configure la base de données, génère la Master Key,
et crée le premier client admin.

Usage:
    python scripts/init_b2b.py
"""
import os
import sys
import secrets
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_master_key():
    """Génère une Master Key sécurisée"""
    return secrets.token_urlsafe(48)


def create_env_file():
    """Crée le fichier .env avec la Master Key"""
    env_path = Path(__file__).parent.parent / ".env"

    master_key = generate_master_key()

    content = f"""# =============================================================================
# CONFIGURATION MINA-TRANSLATOR B2B
# =============================================================================

# =============================================================================
# MASTER KEY - ACCÈS ADMIN ILLIMITÉ
# =============================================================================
# ⚠️ CRITIQUE: Cette clé donne accès ILLIMITÉ à l'API
# - Contourne tous les quotas
# - N'enregistre PAS dans les logs de facturation
# - Permet les tests sans restriction
# =============================================================================

MINA_MASTER_KEY={master_key}

# =============================================================================
# CONFIGURATION API
# =============================================================================

API_MODE=production
API_PORT=8000
API_HOST=0.0.0.0

# =============================================================================
# CONFIGURATION MODÈLE
# =============================================================================

MODEL_PATH=models/mina-translator
DEVICE=cuda

# =============================================================================
# CONFIGURATION BASE DE DONNÉES
# =============================================================================

DATABASE_PATH=data/mina_api.db

# =============================================================================
# CONFIGURATION LOGGING
# =============================================================================

LOG_PATH=logs/usage_stats.log
LOG_LEVEL=INFO
"""

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    return master_key, env_path


def init_database():
    """Initialise la base de données SQLite"""
    from api.database import init_db, create_client, create_api_key, get_client

    print("\n" + "="*60)
    print("INITIALISATION DE LA BASE DE DONNÉES")
    print("="*60)

    # Initialiser les tables
    init_db()
    print("✅ Tables créées avec succès")

    return True


def create_admin_client(master_key: str):
    """Crée le client administrateur"""
    from api.database import create_client, create_api_key, get_client

    print("\n" + "="*60)
    print("CRÉATION DU CLIENT ADMINISTRATEUR")
    print("="*60)

    # Vérifier si le client admin existe déjà
    existing = get_client(email="admin@mina-translator.tg")

    if existing:
        print(f"⚠️ Client admin déjà existant (ID: {existing['id']})")
        client_id = existing['id']
    else:
        # Créer le client admin
        client_id = create_client(
            name="Administrateur",
            email="admin@mina-translator.tg",
            company="Mina-Translator",
            plan="enterprise"
        )
        print(f"✅ Client administrateur créé (ID: {client_id})")

    # Vérifier si une clé API existe déjà pour admin
    from api.database import get_client_api_keys

    existing_keys = get_client_api_keys(client_id)

    if existing_keys:
        print(f"\n⚠️ {len(existing_keys)} clé(s) API existante(s)")
        for key in existing_keys:
            print(f"   - {key['key_prefix']} (actif: {bool(key['active'])})")
        print("\n   Utilisez une clé existante ou révoquez-les d'abord")
        return None, client_id

    # Créer la clé API admin
    key_info = create_api_key(
        client_id,
        name="Admin Primary Key",
        expires_days=0  # Jamais expiré
    )

    return key_info, client_id


def main():
    """Point d'entrée principal"""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*15 + "MINA-TRANSLATOR B2B" + " "*16 + "║")
    print("║" + " "*10 + "Initialisation du Système" + " "*17 + "║")
    print("╚" + "═"*58 + "╝")

    # 1. Créer le fichier .env avec Master Key
    print("\n[1/4] Configuration de la Master Key...")
    master_key, env_path = create_env_file()
    print(f"   ✅ Fichier .env créé: {env_path}")
    print(f"   🔑 Master Key générée")

    # 2. Initialiser la base de données
    print("\n[2/4] Initialisation de la base de données...")
    init_database()

    # 3. Créer le client admin
    print("\n[3/4] Création du client administrateur...")
    key_info, client_id = create_admin_client(master_key)

    # 4. Résumé
    print("\n[4/4] Résumé de configuration")
    print("\n" + "═"*60)
    print("RÉSUMÉ DE L'INSTALLATION")
    print("═"*60)

    print(f"""
📁 Fichiers créés:
   ├── .env (Master Key configurée)
   └── data/mina_api.db (base de données SQLite)

🔑 Vos identifiants:

   ┌─────────────────────────────────────────────────────────┐
   │  MASTER KEY (accès illimité):                          │
   │  {master_key[:20]}...{master_key[-20:]}  │
   │                                                         │
   │  → Utilisez cette clé pour les endpoints /admin/*       │
   │  → Elle contourne tous les quotas et limitations        │
   └─────────────────────────────────────────────────────────┘
""")

    if key_info:
        print(f"""   ┌─────────────────────────────────────────────────────────┐
   │  CLÉ API CLIENT ADMIN:                                  │
   │  {key_info['full_key']}  │
   │                                                         │
   │  → Utilisez cette clé pour tester /translate           │
   │  → expiration: jamais                                   │
   └─────────────────────────────────────────────────────────┘
""")

    print("""
⚠️  ACTIONS REQUISES:

1. Conservez votre Master Key en lieu sûr!
   Elle ne sera jamais affichée à nouveau.

2. Ajoutez le fichier .env à votre backup/sécurité

3. Lancez l'API:
   uvicorn api.main:app --reload --port 8000

4. Testez l'endpoint admin:
   curl -X GET "http://localhost:8000/admin/clients" \\
        -H "X-API-Key: {master_key}"

═══════════════════════════════════════════════════════════════════════
✅ Installation terminée avec succès!
═══════════════════════════════════════════════════════════════════════════════
""")

    return {
        "master_key": master_key,
        "api_key": key_info['full_key'] if key_info else None,
        "client_id": client_id
    }


if __name__ == "__main__":
    main()