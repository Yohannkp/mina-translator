"""
api/security.py - Sécurité API Keys pour Mina-Translator B2B
============================================================

Fournit:
- Hashage bcrypt des clés API
- Vérification rapide des clés
- Génération de clés sécurisées
- Validation du format des clés

Usage:
    from api.security import hash_api_key, verify_api_key, generate_api_key
"""
import os
import re
import secrets
import hashlib
from typing import Optional, Tuple
from datetime import datetime

# bcrypt pour le hashage sécurisé (avec pepper)
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    import hmac

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pepper serveur - doit être dans .env
_SERVER_PEPPER = os.environ.get("MINA_API_PEPPER", "default-pepper-change-me")

# Préfixe des clés API
_API_KEY_PREFIX = "mtk_"

# Longueur de la clé (sans préfixe)
_API_KEY_LENGTH = 32

# Hash des Master Keys autorisées (pour éviter de les stocker en clair)
# Format: {nom: hash_bcrypt}
_MASTER_KEYS_STORE = {}


# =============================================================================
# FONCTIONS DE HASHAGE
# =============================================================================

def _get_peppered_key(key: str) -> str:
    """Ajoute le pepper server à la clé avant hashage"""
    return f"{_SERVER_PEPPER}:{key}"


def hash_api_key(api_key: str) -> str:
    """
    Hashe une clé API avec bcrypt + pepper

    Args:
        api_key: Clé API en clair

    Returns:
        Hash bcrypt encodé en string
    """
    if BCRYPT_AVAILABLE:
        # bcrypt avec pepper
        peppered = _get_peppered_key(api_key).encode('utf-8')
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(peppered, salt)
        return hashed.decode('utf-8')
    else:
        # Fallback: SHA256 + salt (moins sécurisé)
        peppered = _get_peppered_key(api_key).encode('utf-8')
        salt = secrets.token_hex(16)
        salted = salt + peppered.decode('utf-8')
        return f"$sha256${salt}${hashlib.sha256(salted.encode()).hexdigest()}"


def verify_api_key(api_key: str, hashed: str) -> bool:
    """
    Vérifie si une clé API correspond à son hash

    Args:
        api_key: Clé API en clair
        hashed: Hash stocké en base

    Returns:
        True si correspondance, False sinon
    """
    if not api_key or not hashed:
        return False

    if BCRYPT_AVAILABLE:
        try:
            peppered = _get_peppered_key(api_key).encode('utf-8')
            return bcrypt.checkpw(peppered, hashed.encode('utf-8'))
        except Exception:
            return False
    else:
        # Fallback SHA256
        if not hashed.startswith("$sha256$"):
            return False
        try:
            parts = hashed.split("$")
            salt = parts[2]
            expected = parts[3]
            salted = salt + _get_peppered_key(api_key)
            actual = hashlib.sha256(salted.encode()).hexdigest()
            return hmac.compare_digest(expected, actual)
        except Exception:
            return False


def verify_master_key(master_key: str) -> bool:
    """
    Vérifie si une clé est une Master Key valide

    Args:
        master_key: Clé à vérifier

    Returns:
        True si c'est une Master Key valide
    """
    if not master_key:
        return False

    # Méthode 1: Vérifier via variable d'environnement
    env_master = os.environ.get("MINA_MASTER_KEY")
    if env_master and hmac.compare_digest(master_key, env_master):
        return True

    # Méthode 2: Vérifier via .env
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith("MINA_MASTER_KEY="):
                        stored = line.split("=", 1)[1].strip()
                        if stored and hmac.compare_digest(master_key, stored):
                            return True
        except Exception:
            pass

    # Méthode 3: Vérifier les hashes stockés en mémoire
    for stored_hash in _MASTER_KEYS_STORE.values():
        if verify_api_key(master_key, stored_hash):
            return True

    return False


def register_master_key(name: str, master_key: str) -> None:
    """
    Enregistre une Master Key en mémoire (pour le runtime)

    Args:
        name: Nom/identifiant de la Master Key
        master_key: Clé en clair (sera hashée)
    """
    _MASTER_KEYS_STORE[name] = hash_api_key(master_key)


# =============================================================================
# GÉNÉRATION DE CLÉS
# =============================================================================

def generate_api_key(prefix: str = _API_KEY_PREFIX, length: int = _API_KEY_LENGTH) -> str:
    """
    Génère une nouvelle clé API sécurisée

    Args:
        prefix: Préfixe de la clé (défaut: "mtk_")
        length: Longueur de la partie aléatoire

    Returns:
        Nouvelle clé API (ex: "mtk_abc123xyz...")
    """
    random_part = secrets.token_urlsafe(length)[:length]
    return f"{prefix}{random_part}"


def generate_master_key(length: int = 48) -> str:
    """
    Génère une Master Key sécurisée

    Args:
        length: Longueur de la clé (défaut: 48)

    Returns:
        Nouvelle Master Key
    """
    return secrets.token_urlsafe(length)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_api_key_format(key: str) -> bool:
    """
    Valide le format d'une clé API

    Args:
        key: Clé à valider

    Returns:
        True si le format est valide
    """
    if not key:
        return False

    # Doit commencer par le préfixe
    if not key.startswith(_API_KEY_PREFIX):
        return False

    # Longueur minimale (préfixe + quelque chose)
    if len(key) < len(_API_KEY_PREFIX) + 10:
        return False

    # Caractères valides (URL-safe base64)
    pattern = rf'^{_API_KEY_PREFIX}[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, key))


def sanitize_input(text: str, max_length: int = 500) -> str:
    """
    Nettoie et valide une entrée utilisateur

    Args:
        text: Texte à nettoyer
        max_length: Longueur maximale

    Returns:
        Texte nettoyé
    """
    if not text:
        return ""

    # Supprimer les caractères nuls
    text = text.replace('\x00', '')

    # Limiter la longueur
    text = text[:max_length]

    # Normaliser les espaces
    text = ' '.join(text.split())

    return text.strip()


def validate_api_key_header(x_api_key: Optional[str]) -> Tuple[bool, str]:
    """
    Valide le header X-API-Key et retourne un diagnostic

    Args:
        x_api_key: Valeur du header

    Returns:
        Tuple (valide, message_erreur)
    """
    if not x_api_key:
        return False, "Header X-API-Key manquant"

    if len(x_api_key) < 10:
        return False, "Clé API trop courte"

    if len(x_api_key) > 200:
        return False, "Clé API invalide"

    return True, ""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'hash_api_key',
    'verify_api_key',
    'verify_master_key',
    'register_master_key',
    'generate_api_key',
    'generate_master_key',
    'validate_api_key_format',
    'validate_api_key_header',
    'sanitize_input',
]