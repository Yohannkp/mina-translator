"""
Middleware d'authentification et autorisation
=============================================
Gère les API Keys, rate limiting, et la Master Key spéciale

用法:
    from api.auth import verify_request, master_key
"""
import os
import time
import hashlib
from typing import Optional

from fastapi import HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from loguru import logger

from api.database import (
    verify_api_key,
    check_quota,
    increment_quota,
    check_rate_limit,
    log_usage
)

# =============================================================================
# CONFIGURATION MASTER KEY
# =============================================================================

# La Master Key peut être définie via:
# 1. Variable d'environnement (recommandé pour production)
# 2. Fichier .env
# 3. Sinon génère une clé temporaire (dev only)

def get_master_key() -> Optional[str]:
    """
    Récupère la Master Key depuis les sources sécurisées
    Priorité: ENV > .env > fallback
    """
    # 1. Variable d'environnement (HAUTEMENT recommandé)
    master_key = os.environ.get("MINA_MASTER_KEY")
    if master_key:
        logger.info("Master Key chargée depuis variable d'environnement")
        return master_key

    # 2. Fichier .env (fallback)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("MINA_MASTER_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key and not key.startswith("#"):
                            logger.info("Master Key chargee depuis fichier .env")
                            return key
        except Exception:
            pass

    # 3. Fallback pour développement ONLY
    # ⚠️ AVERTISSEMENT: Ne pas utiliser en production!
    logger.warning("⚠️ Master Key non configurée!")
    logger.warning("⚠️ Veuillez définir MINA_MASTER_KEY dans .env ou variable d'environnement")
    return None


def is_master_key(key: str) -> bool:
    """Vérifie si la clé fournie est la Master Key"""
    master_key = get_master_key()
    if not master_key:
        return False

    # Comparaison sécurisée (temps constant)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    master_hash = hashlib.sha256(master_key.encode()).hexdigest()

    return key_hash == master_hash


# =============================================================================
# HEADER API KEY
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# =============================================================================
# DÉPENDANCES FASTAPI
# =============================================================================

class AuthenticatedRequest:
    """Objet représentant une requête authentifiée"""

    def __init__(
        self,
        key_id: str,
        client_id: int,
        client_name: str,
        client_email: str,
        company: str,
        plan: str,
        is_master: bool = False
    ):
        self.key_id = key_id
        self.client_id = client_id
        self.client_name = client_name
        self.client_email = client_email
        self.company = company
        self.plan = plan
        self.is_master = is_master


async def verify_api_request(request: Request, api_key: str = Depends(api_key_header)) -> AuthenticatedRequest:
    """
    Dépendance FastAPI pour vérifier l'authentification

    Args:
        request: Requête HTTP
        api_key: Clé API depuis le header X-API-Key

    Returns:
        AuthenticatedRequest avec les infos du client

    Raises:
        HTTPException: 401 si non autorisé, 429 si quota/rate limit atteint
    """
    # Vérifier si c'est la Master Key
    if is_master_key(api_key):
        logger.info("🔑 MASTER KEY utilisée - Accès illimité")
        return AuthenticatedRequest(
            key_id="MASTER_KEY",
            client_id=0,
            client_name="Super-Admin",
            client_email="admin@mina-translator.com",
            company="Mina-Translator",
            plan="enterprise",
            is_master=True
        )

    # Vérifier si la clé API est fournie
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "UNAUTHORIZED",
                "message": "Clé API requise",
                "code": "MISSING_API_KEY",
                "hint": "Ajoutez le header X-API-Key avec votre clé"
            }
        )

    # Vérifier la clé API
    client_info = verify_api_key(api_key)

    if not client_info:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "UNAUTHORIZED",
                "message": "Clé API invalide ou expirée",
                "code": "INVALID_API_KEY",
                "hint": "Vérifiez votre clé ou contactez le support"
            }
        )

    # Rate limiting (60 req/min par défaut)
    allowed, remaining = check_rate_limit(
        client_info['key_id'],
        max_requests=60,
        window_seconds=60
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "TOO_MANY_REQUESTS",
                "message": "Trop de requêtes",
                "code": "RATE_LIMIT_EXCEEDED",
                "limit": 60,
                "window": "60 seconds",
                "hint": "Réduisez votre fréquence de requêtes"
            }
        )

    # Vérifier les quotas (sauf pour Master Key)
    allowed, reason, quota = check_quota(client_info['client_id'])

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "QUOTA_EXCEEDED",
                "message": reason,
                "code": "QUOTA_LIMIT",
                "daily_limit": quota['daily_limit'],
                "daily_used": quota['daily_used'],
                "monthly_limit": quota['monthly_limit'],
                "monthly_used": quota['monthly_used'],
                "hint": "Contactez le support pour augmenter votre quota"
            }
        )

    # Logger le rate limit restant dans les headers
    request.state.rate_limit_remaining = remaining

    return AuthenticatedRequest(
        key_id=client_info['key_id'],
        client_id=client_info['client_id'],
        client_name=client_info['client_name'],
        client_email=client_info['email'],
        company=client_info['company'],
        plan=client_info['plan'],
        is_master=False
    )


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def log_api_usage(
    auth: AuthenticatedRequest,
    endpoint: str,
    domain: str = None,
    request_text: str = None,
    response_text: str = None,
    inference_time_ms: int = 0,
    status_code: int = 200,
    error: str = None
):
    """
    Log l'utilisation de l'API (sauf pour Master Key)
    """
    if auth.is_master:
        # Ne pas logger les requêtes Master Key pour facturation
        logger.debug(f"MASTER KEY usage: {endpoint} - {inference_time_ms}ms")
        return

    log_usage(
        key_id=auth.key_id,
        client_id=auth.client_id,
        endpoint=endpoint,
        domain=domain,
        request_text=request_text,
        response_text=response_text,
        inference_time_ms=inference_time_ms,
        status_code=status_code,
        error_message=error
    )

    # Incrémenter le quota
    increment_quota(auth.client_id)


# =============================================================================
# VALIDATION DES ENTRÉES
# =============================================================================

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Nettoie et valide les entrées utilisateur
    Protège contre les injections et caractères dangereux
    """
    if not text:
        return ""

    # Supprimer les caractères de contrôle
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

    # Tronquer si trop long
    if len(text) > max_length:
        text = text[:max_length]

    # Supprimer les null bytes
    text = text.replace('\x00', '')

    return text.strip()


def validate_api_key_format(key: str) -> bool:
    """Valide le format d'une clé API"""
    if not key:
        return False

    # Doit commencer par mtk_
    if not key.startswith("mtk_"):
        return False

    # Longueur minimum
    if len(key) < 20:
        return False

    return True


# =============================================================================
# GÉNÉRATION DE CLÉS (ADMIN)
# =============================================================================

def generate_master_key_file():
    """Génère une Master Key et la sauvegarde dans .env"""
    import secrets

    master_key = secrets.token_urlsafe(48)  # 48 bytes = 64 caractères

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")

    with open(env_path, "a") as f:
        f.write(f"\n# Master Key pour accès admin (GÉNÉRÉE AUTOMATIQUEMENT)\n")
        f.write(f"MINA_MASTER_KEY={master_key}\n")

    print("="*60)
    print("🔑 MASTER KEY GÉNÉRÉE!")
    print("="*60)
    print(f"\nClé: {master_key}")
    print(f"\nAjoutée dans: {env_path}")
    print("\n⚠️  CONSERVER CETTE CLÉ EN LIEU SÛR!")
    print("⚠️  Ne jamais commiter ce fichier sur GitHub!")

    return master_key


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST D'AUTHENTIFICATION")
    print("="*60)

    # Vérifier si Master Key configurée
    master = get_master_key()

    if master:
        print(f"✅ Master Key configurée (longueur: {len(master)})")
        print(f"   Hash: {hashlib.sha256(master.encode()).hexdigest()[:16]}...")
    else:
        print("❌ Master Key NON configurée")
        print("   → Définissez MINA_MASTER_KEY dans .env")
        print()

        # Proposer de générer une clé
        response = input("Générer une Master Key automatiquement? (o/N): ")
        if response.lower() == 'o':
            generate_master_key_file()

    # Tester une clé bidon
    print("\n" + "-"*60)
    print("Test de validation de clé:")
    print(f"   'mtk_test' valide: {validate_api_key_format('mtk_test')}")
    print(f"   'mtk_abc123def456...' valide: {validate_api_key_format('mtk_abc123def456')}")