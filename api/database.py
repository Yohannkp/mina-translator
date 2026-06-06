"""
Base de donnees SQLite pour la gestion B2B
==========================================
Gere les cles API, clients et quotas d'utilisation

Tables:
    - api_keys: Cles API des clients
    - clients: Informations des entreprises clientes
    - usage_logs: Historique d'utilisation pour facturation
    - quotas: Limites et compteurs par client

Usage:
    from api.database import init_db, verify_api_key, get_quota, create_client
"""
import sqlite3
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from loguru import logger

from api.security import hash_api_key

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_PATH = Path(__file__).parent.parent / "data" / "mina_api.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GESTIONNAIRE DE CONNEXION
# =============================================================================

@contextmanager
def get_db_connection():
    """Contexte sécurisé pour les connexions SQLite"""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# =============================================================================
# INITIALISATION DE LA BASE DE DONNÉES
# =============================================================================

def init_db():
    """
    Initialise la base de données avec toutes les tables nécessaires
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Table des clients (entreprises)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                company TEXT,
                phone TEXT,
                plan TEXT DEFAULT 'basic' CHECK(plan IN ('basic', 'pro', 'enterprise')),
                active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table des clés API
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT UNIQUE NOT NULL,
                key_hash TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                client_id INTEGER NOT NULL,
                name TEXT,
                active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used_at TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
            )
        """)

        # Table des quotas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quotas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER NOT NULL UNIQUE,
                daily_limit INTEGER DEFAULT 1000,
                monthly_limit INTEGER DEFAULT 10000,
                daily_used INTEGER DEFAULT 0,
                monthly_used INTEGER DEFAULT 0,
                last_reset_daily TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_reset_monthly TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
            )
        """)

        # Table des logs d'utilisation (pour facturation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL,
                client_id INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                domain TEXT,
                request_text TEXT,
                response_text TEXT,
                inference_time_ms INTEGER,
                tokens_used INTEGER,
                status_code INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
            )
        """)

        # Table des limites de taux (rate limiting)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id TEXT NOT NULL UNIQUE,
                requests_count INTEGER DEFAULT 0,
                window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (key_id) REFERENCES api_keys(key_id) ON DELETE CASCADE
            )
        """)

        # Index pour performances
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_logs_key_id ON usage_logs(key_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON usage_logs(created_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rate_limits_key_id ON rate_limits(key_id)
        """)

        logger.info("Base de données initialisée avec succès")
        return True


# =============================================================================
# GÉNÉRATION ET VÉRIFICATION DES CLÉS API
# =============================================================================

def generate_api_key() -> tuple[str, str, str]:
    """
    Génère une nouvelle clé API sécurisée

    Returns:
        tuple: (full_key, key_hash, key_prefix)
        - full_key: La clé complète à donner au client (ex: mtk_abc123...)
        - key_hash: Hash pour stockage sécurisé
        - key_prefix: Préfixe visible pour identification
    """
    from api.security import generate_api_key as gen_key

    full_key = gen_key()

    # Hasher pour stockage
    key_hash = hash_api_key(full_key)

    # Préfixe visible (16 premiers caractères)
    key_prefix = full_key[:16]

    return full_key, key_hash, key_prefix


def verify_api_key(key: str) -> Optional[dict]:
    """
    Vérifie une clé API et retourne les infos du client

    Args:
        key: Clé API à vérifier (ex: mtk_abc123...)

    Returns:
        dict avec les infos du client ou None si invalide/expirée
    """
    key_hash = hash_api_key(key)

    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                ak.key_id,
                ak.key_prefix,
                ak.active,
                ak.expires_at,
                ak.client_id,
                c.name as client_name,
                c.email,
                c.company,
                c.plan,
                c.active as client_active
            FROM api_keys ak
            JOIN clients c ON ak.client_id = c.id
            WHERE ak.key_hash = ? AND ak.active = 1 AND c.active = 1
        """, (key_hash,))

        row = cursor.fetchone()

        if not row:
            return None

        result = dict(row)

        # Vérifier expiration
        if result['expires_at']:
            expires = datetime.fromisoformat(result['expires_at'])
            if datetime.now() > expires:
                return None

        # Mettre à jour last_used_at
        cursor.execute("""
            UPDATE api_keys
            SET last_used_at = CURRENT_TIMESTAMP
            WHERE key_id = ?
        """, (result['key_id'],))

        return result


# =============================================================================
# GESTION DES QUOTAS
# =============================================================================

def get_quota(client_id: int) -> dict:
    """Récupère les quotas d'un client"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM quotas WHERE client_id = ?
        """, (client_id,))

        row = cursor.fetchone()

        if not row:
            # Créer les quotas par défaut
            cursor.execute("""
                INSERT INTO quotas (client_id) VALUES (?)
            """, (client_id,))
            return get_quota(client_id)

        return dict(row)


def check_quota(client_id: int) -> tuple[bool, str, dict]:
    """
    Vérifie si le client a assez de quota disponible

    Returns:
        tuple: (allowed, reason, quota_info)
    """
    quota = get_quota(client_id)

    # Vérifier reset quotidien
    last_daily = datetime.fromisoformat(quota['last_reset_daily'])
    if datetime.now().date() > last_daily.date():
        reset_daily(client_id)
        quota = get_quota(client_id)

    # Vérifier reset mensuel
    last_monthly = datetime.fromisoformat(quota['last_reset_monthly'])
    if datetime.now().month > last_monthly.month or datetime.now().year > last_monthly.year:
        reset_monthly(client_id)
        quota = get_quota(client_id)

    # Vérifier limites
    if quota['daily_used'] >= quota['daily_limit']:
        return False, "Quota quotidien épuisé", quota

    if quota['monthly_used'] >= quota['monthly_limit']:
        return False, "Quota mensuel épuisé", quota

    return True, "OK", quota


def increment_quota(client_id: int, tokens: int = 1):
    """Incrément le compteur d'utilisation"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE quotas
            SET daily_used = daily_used + 1,
                monthly_used = monthly_used + 1
            WHERE client_id = ?
        """, (client_id,))


def reset_daily(client_id: int):
    """Reset le compteur quotidien"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE quotas
            SET daily_used = 0,
                last_reset_daily = CURRENT_TIMESTAMP
            WHERE client_id = ?
        """, (client_id,))


def reset_monthly(client_id: int):
    """Reset le compteur mensuel"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE quotas
            SET monthly_used = 0,
                last_reset_monthly = CURRENT_TIMESTAMP
            WHERE client_id = ?
        """, (client_id,))


# =============================================================================
# LOGGING D'UTILISATION
# =============================================================================

def log_usage(
    key_id: str,
    client_id: int,
    endpoint: str,
    domain: str = None,
    request_text: str = None,
    response_text: str = None,
    inference_time_ms: int = 0,
    tokens_used: int = 0,
    status_code: int = 200,
    error_message: str = None
):
    """
    Enregistre une requête dans les logs d'utilisation
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO usage_logs (
                key_id, client_id, endpoint, domain,
                request_text, response_text, inference_time_ms,
                tokens_used, status_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key_id, client_id, endpoint, domain,
            request_text[:500] if request_text else None,
            response_text[:500] if response_text else None,
            inference_time_ms, tokens_used, status_code, error_message
        ))


def get_usage_stats(key_id: str = None, client_id: int = None, days: int = 30) -> list:
    """Récupère les statistiques d'utilisation"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = """
            SELECT * FROM usage_logs
            WHERE created_at >= datetime('now', '-' || ? || ' days')
        """
        params = [days]

        if key_id:
            query += " AND key_id = ?"
            params.append(key_id)
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# GESTION DES CLIENTS
# =============================================================================

def create_client(name: str, email: str, company: str = None, phone: str = None, plan: str = 'basic') -> int:
    """Crée un nouveau client"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO clients (name, email, company, phone, plan)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, company, phone, plan))

        client_id = cursor.lastrowid

        # Créer les quotas par défaut selon le plan
        default_quotas = {
            'basic': (500, 5000),
            'pro': (2000, 20000),
            'enterprise': (10000, 100000)
        }
        daily, monthly = default_quotas.get(plan, (500, 5000))

        cursor.execute("""
            INSERT INTO quotas (client_id, daily_limit, monthly_limit)
            VALUES (?, ?, ?)
        """, (client_id, daily, monthly))

        return client_id


def create_api_key(client_id: int, name: str = None, expires_days: int = 365) -> dict:
    """
    Crée une nouvelle clé API pour un client

    Args:
        client_id: ID du client
        name: Nom de la clé (optionnel)
        expires_days: Jours avant expiration (0 = jamais)

    Returns:
        dict avec full_key (à donner au client) et key_id
    """
    full_key, key_hash, key_prefix = generate_api_key()

    expires_at = None
    if expires_days > 0:
        expires_at = (datetime.now() + timedelta(days=expires_days)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO api_keys (key_id, key_hash, key_prefix, client_id, name, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (key_prefix, key_hash, key_prefix, client_id, name, expires_at))

        # Initialiser le rate limit
        cursor.execute("""
            INSERT INTO rate_limits (key_id) VALUES (?)
        """, (key_prefix,))

        return {
            'full_key': full_key,
            'key_id': key_prefix,
            'expires_at': expires_at
        }


def get_client(client_id: int = None, email: str = None) -> Optional[dict]:
    """Récupère les infos d'un client"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if client_id:
            cursor.execute("SELECT * FROM clients WHERE id = ?", (client_id,))
        elif email:
            cursor.execute("SELECT * FROM clients WHERE email = ?", (email,))
        else:
            return None

        row = cursor.fetchone()
        return dict(row) if row else None


def get_client_api_keys(client_id: int) -> list:
    """Récupère toutes les clés API d'un client"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT key_id, key_prefix, name, active, created_at, expires_at, last_used_at
            FROM api_keys WHERE client_id = ?
        """, (client_id,))

        return [dict(row) for row in cursor.fetchall()]


def revoke_api_key(key_id: str):
    """Désactive une clé API"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE api_keys SET active = 0 WHERE key_id = ?
        """, (key_id,))


# =============================================================================
# RATE LIMITING
# =============================================================================

def check_rate_limit(key_id: str, max_requests: int = 60, window_seconds: int = 60) -> tuple[bool, str]:
    """
    Vérifie les limites de taux (rate limiting)

    Args:
        key_id: ID de la clé API
        max_requests: Nombre max de requêtes
        window_seconds: Fenêtre de temps en secondes

    Returns:
        tuple: (allowed, remaining_requests)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT requests_count, window_start FROM rate_limits WHERE key_id = ?
        """, (key_id,))

        row = cursor.fetchone()

        if not row:
            # Première requête
            cursor.execute("""
                INSERT INTO rate_limits (key_id, requests_count) VALUES (?, 1)
            """, (key_id,))
            return True, max_requests - 1

        requests_count = row['requests_count']
        window_start = datetime.fromisoformat(row['window_start'])

        # Vérifier si la fenêtre est expirée
        elapsed = (datetime.now() - window_start).total_seconds()
        if elapsed >= window_seconds:
            # Reset
            cursor.execute("""
                UPDATE rate_limits
                SET requests_count = 1, window_start = CURRENT_TIMESTAMP
                WHERE key_id = ?
            """, (key_id,))
            return True, max_requests - 1

        # Vérifier limite
        if requests_count >= max_requests:
            remaining = max_requests - requests_count
            return False, remaining

        # Incrémenter
        cursor.execute("""
            UPDATE rate_limits SET requests_count = requests_count + 1 WHERE key_id = ?
        """, (key_id,))

        return True, max_requests - requests_count - 1


# =============================================================================
# ADMINISTRATION
# =============================================================================

def get_all_clients() -> list:
    """Récupère tous les clients avec leurs statistiques"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                c.*,
                q.daily_limit, q.daily_used, q.monthly_limit, q.monthly_used,
                (SELECT COUNT(*) FROM api_keys WHERE client_id = c.id) as key_count,
                (SELECT COUNT(*) FROM usage_logs WHERE client_id = c.id AND
                    created_at >= datetime('now', '-7 days')) as requests_week
            FROM clients c
            LEFT JOIN quotas q ON c.id = q.client_id
            ORDER BY c.created_at DESC
        """)

        return [dict(row) for row in cursor.fetchall()]


def get_billing_summary(client_id: int, month: int = None, year: int = None) -> dict:
    """Génère un résumé de facturation pour un client"""
    if not month:
        month = datetime.now().month
    if not year:
        year = datetime.now().year

    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_requests,
                SUM(inference_time_ms) as total_inference_ms,
                SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as failed_requests,
                AVG(inference_time_ms) as avg_inference_ms,
                MIN(inference_time_ms) as min_inference_ms,
                MAX(inference_time_ms) as max_inference_ms,
                COUNT(DISTINCT domain) as domains_used
            FROM usage_logs
            WHERE client_id = ?
            AND strftime('%m', created_at) = ?
            AND strftime('%Y', created_at) = ?
        """, (client_id, f'{month:02d}', str(year)))

        row = cursor.fetchone()

        if row['total_requests'] == 0:
            return {
                'client_id': client_id,
                'month': f'{year}-{month:02d}',
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_inference_ms': 0,
                'total_inference_ms': 0
            }

        return dict(row)


# =============================================================================
# INITIALISATION AUTOMATIQUE
# =============================================================================

if __name__ == "__main__":
    # Initialiser la DB
    init_db()

    # Créer un client de test
    print("\n" + "="*60)
    print("CRÉATION D'UN CLIENT DE TEST")
    print("="*60)

    try:
        client_id = create_client(
            name="Administrateur",
            email="admin@mina-translator.com",
            company="Mina-Translator",
            plan="enterprise"
        )
        print(f"✅ Client créé avec ID: {client_id}")

        # Générer une clé API pour admin
        key_info = create_api_key(client_id, name="Master Key Admin", expires_days=0)
        print(f"\n🔑 Clé API Admin:")
        print(f"   Clé: {key_info['full_key']}")
        print(f"   Key ID: {key_info['key_id']}")
        print(f"   Expiration: Jamais")

        print("\n⚠️  CONSERVER CETTE CLÉ EN LIEU SÛR!")
        print("⚠️  Elle donne accès illimité à l'API.")

    except sqlite3.IntegrityError as e:
        print(f"Client déjà existant: {e}")