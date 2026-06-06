"""
Système de Logging pour Facturation
===============================
Enregistre chaque requête dans logs/usage_stats.log
Format CSV pour analyse facile: Date,ClientID,Domaine,Temps d'inférence

用法:
    from api.logger import log_request, get_usage_report
"""
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "usage_stats.log"

# Assurer que le répertoire existe
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FORMAT DU LOG
# =============================================================================

LOG_FIELDS = [
    "timestamp",           # ISO 8601
    "client_id",         # ID du client
    "client_name",       # Nom du client
    "api_key_id",        # Préfixe de la clé
    "endpoint",          # /translate, /transcribe, etc.
    "domain",            # Domaine/theme de la requête
    "request_text",      # Texte demandé (tronqué à 200 chars)
    "response_text",     # Réponse (tronquée à 200 chars)
    "inference_ms",      # Temps d'inférence en ms
    "tokens_used",       # Tokens utilisés (estimés)
    "status_code",       # Code HTTP
    "error",            # Message d'erreur si échec
    "is_master",        # Si使用的是 Master Key
    "plan"             # Plan du client (basic/pro/enterprise)
]


# =============================================================================
# FONCTIONS DE LOGGING
# =============================================================================

def log_request(
    client_id: int,
    client_name: str,
    api_key_id: str,
    endpoint: str,
    request_text: str = None,
    response_text: str = None,
    inference_ms: int = 0,
    tokens_used: int = 0,
    status_code: int = 200,
    error: str = None,
    domain: str = None,
    is_master: bool = False,
    plan: str = None
):
    """
    Enregistre une requête dans le fichier de log

    Args:
        client_id: ID du client
        client_name: Nom du client
        api_key_id: Préfixe de la clé API
        endpoint: Endpoint utilisé
        request_text: Texte de la requête
        response_text: Texte de la réponse
        inference_ms: Temps d'inférence en millisecondes
        tokens_used: Nombre de tokens
        status_code: Code HTTP de la réponse
        error: Message d'erreur
        domain: Domaine/thème de la requête
        is_master: Si c'est une requête Master Key
        plan: Plan du client
    """
    timestamp = datetime.now().isoformat()

    # Tronquer les texts à 200 caractères
    req_text = (request_text[:200] + "...") if request_text and len(request_text) > 200 else request_text
    resp_text = (response_text[:200] + "...") if response_text and len(response_text) > 200 else response_text

    # Créer la ligne CSV
    row = {
        "timestamp": timestamp,
        "client_id": client_id,
        "client_name": client_name[:50],
        "api_key_id": api_key_id[:16],
        "endpoint": endpoint,
        "domain": domain or "",
        "request_text": req_text or "",
        "response_text": resp_text or "",
        "inference_ms": inference_ms,
        "tokens_used": tokens_used,
        "status_code": status_code,
        "error": error or "",
        "is_master": "YES" if is_master else "NO",
        "plan": plan or ""
    }

    # Écrire dans le fichier CSV
    write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    # Logger aussi dans loguru pour debugging
    if is_master:
        logger.debug(
            f"[MASTER] {endpoint} | {client_name} | {inference_ms}ms | {status_code}"
        )
    else:
        logger.info(
            f"{endpoint} | Client:{client_id} | {domain} | {inference_ms}ms | {status_code}"
        )

    return row


def get_usage_report(
    client_id: int = None,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 1000
) -> list:
    """
    Génère un rapport d'utilisation

    Args:
        client_id: ID du client (None = tous)
        start_date: Date de début
        end_date: Date de fin
        limit: Nombre maximum de lignes

    Returns:
        list de dictionnaires avec les données
    """
    if not LOG_FILE.exists():
        return []

    results = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Filtrer par client
            if client_id and int(row["client_id"]) != client_id:
                continue

            # Filtrer par date
            if start_date:
                row_date = datetime.fromisoformat(row["timestamp"])
                if row_date < start_date:
                    continue

            if end_date:
                row_date = datetime.fromisoformat(row["timestamp"])
                if row_date > end_date:
                    continue

            results.append(row)

            if len(results) >= limit:
                break

    return results


def get_billing_summary(client_id: int, month: int = None, year: int = None) -> dict:
    """
    Génère un résumé de facturation pour un client

    Args:
        client_id: ID du client
        month: Mois (1-12), défaut = mois actuel
        year: Année, défaut = année actuelle

    Returns:
        Résumé avec total requêtes, temps moyen, etc.
    """
    if not month:
        month = datetime.now().month
    if not year:
        year = datetime.now().year

    if not LOG_FILE.exists():
        return {
            "client_id": client_id,
            "period": f"{year}-{month:02d}",
            "total_requests": 0,
            "total_inference_ms": 0,
            "avg_inference_ms": 0,
            "success_rate": 0,
        }

    requests = []
    total_inference = 0

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["client_id"]) != client_id:
                continue

            row_date = datetime.fromisoformat(row["timestamp"])

            if row_date.month == month and row_date.year == year:
                requests.append(row)
                total_inference += int(row["inference_ms"])

    if not requests:
        return {
            "client_id": client_id,
            "period": f"{year}-{month:02d}",
            "total_requests": 0,
            "total_inference_ms": 0,
            "avg_inference_ms": 0,
            "success_rate": 0,
        }

    successful = sum(1 for r in requests if r["status_code"] == "200")

    return {
        "client_id": client_id,
        "period": f"{year}-{month:02d}",
        "total_requests": len(requests),
        "total_inference_ms": total_inference,
        "avg_inference_ms": total_inference // len(requests) if requests else 0,
        "success_rate": round(successful / len(requests) * 100, 2),
    }


def get_daily_usage(client_id: int, days: int = 30) -> list:
    """
    Récupère l'utilisation quotidienne

    Args:
        client_id: ID du client
        days: Nombre de jours à récupérer

    Returns:
        Liste de tuples (date, count, total_inference_ms)
    """
    if not LOG_FILE.exists():
        return []

    daily_usage = {}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["client_id"]) != client_id:
                continue

            date = row["timestamp"][:10]  # YYYY-MM-DD

            if date not in daily_usage:
                daily_usage[date] = {"count": 0, "inference_ms": 0}

            daily_usage[date]["count"] += 1
            daily_usage[date]["inference_ms"] += int(row["inference_ms"])

    # Trier par date
    sorted_dates = sorted(daily_usage.items(), key=lambda x: x[0], reverse=True)

    return [(d, v["count"], v["inference_ms"]) for d, v in sorted_dates[:days]]


# =============================================================================
# ANALYSE EN TEMPS RÉEL
# =============================================================================

def get_realtime_stats() -> dict:
    """
    Retourne les statistiques en temps réel
    """
    if not LOG_FILE.exists():
        return {
            "total_requests": 0,
            "active_clients": 0,
            "avg_inference_ms": 0,
        }

    requests_today = 0
    unique_clients = set()
    total_inference = 0

    today = datetime.now().strftime("%Y-%m-%d")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["timestamp"].startswith(today):
                requests_today += 1
                unique_clients.add(row["client_id"])
                total_inference += int(row["inference_ms"])

    return {
        "total_requests": requests_today,
        "active_clients": len(unique_clients),
        "avg_inference_ms": total_inference // requests_today if requests_today > 0 else 0,
    }


# =============================================================================
# EXPORT
# =============================================================================

def export_csv(filename: str = None, client_id: int = None):
    """
    Exporte les données en CSV

    Args:
        filename: Nom du fichier export (défaut: usage_export_YYYYMMDD.csv)
        client_id: ID du client à exporter (None = tous)
    """
    if not filename:
        filename = f"usage_export_{datetime.now().strftime('%Y%m%d')}.csv"

    with open(LOG_FILE, "r", encoding="utf-8") as f_in:
        with open(filename, "w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=LOG_FIELDS)
            writer.writeheader()

            for row in reader:
                if client_id and int(row["client_id"]) != client_id:
                    continue
                writer.writerow(row)

    return filename


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DE LOGGER")
    print("="*60)

    # Logger une requête de test
    log_request(
        client_id=1,
        client_name="Entreprise Test",
        api_key_id="mtk_test123",
        endpoint="/translate",
        request_text="Bonjour, comment allez-vous?",
        response_text="Woezor, ayi?",
        inference_ms=150,
        status_code=200,
        domain="famille",
        plan="pro"
    )

    print(f"\n✅ Log enregistré dans: {LOG_FILE}")

    # Lire le résumé
    print("\n" + "-"*60)
    stats = get_realtime_stats()
    print(f"Requêtes aujourd'hui: {stats['total_requests']}")
    print(f"Clients actifs: {stats['active_clients']}")
    print(f"Temps moyen: {stats['avg_inference_ms']}ms")