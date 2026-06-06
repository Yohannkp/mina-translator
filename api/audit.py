"""
api/audit.py - Logging d'audit pour Mina-Translator B2B
======================================================

Fournit:
- Logging JSON structuré pour audit
- Métriques temps réel
- Export pour facturation

Usage:
    from api.audit import audit_log, get_request_stats, get_audit_trail
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from collections import defaultdict
from loguru import logger
from threading import Lock

# =============================================================================
# CONFIGURATION
# =============================================================================

AUDIT_DIR = Path(__file__).parent.parent / "logs"
AUDIT_FILE = AUDIT_DIR / "audit.log"
AUDIT_DIR.mkdir(exist_ok=True)

# Stats en mémoire (pour requêtes rapides)
_stats_lock = Lock()
_request_stats = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0,
    "errors": 0,
    "last_request": None,
    "clients": set()
})

# Cache des stats (rafraîchit toutes les 60s)
_stats_cache = {"data": {}, "updated": 0}
_CACHE_TTL = 60  # secondes


# =============================================================================
# FONCTIONS DE LOGGING
# =============================================================================

def audit_log(
    event_type: str,
    client_id: int,
    client_name: str,
    endpoint: str,
    method: str = "POST",
    request_text: str = "",
    response_text: str = "",
    inference_ms: int = 0,
    status_code: int = 200,
    api_key_id: str = "",
    domain: str = "",
    plan: str = "",
    error: str = "",
    is_master: bool = False,
    extra: Optional[Dict] = None
) -> None:
    """
    Log une requête dans le fichier d'audit JSON

    Args:
        event_type: Type d'événement (request, error, auth_failure)
        client_id: ID du client (0 pour Master)
        client_name: Nom du client
        endpoint: Endpoint appelé
        method: Méthode HTTP
        request_text: Texte de la requête
        response_text: Texte de la réponse
        inference_ms: Temps d'inférence en ms
        status_code: Code HTTP de réponse
        api_key_id: ID de la clé API
        domain: Domaine thématique
        plan: Plan du client
        error: Message d'erreur (si applicable)
        is_master: Si la requête vient du Master Key
        extra: Données additionnelles
    """
    timestamp = datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "event_type": event_type,
        "client_id": client_id,
        "client_name": client_name,
        "api_key_id": api_key_id,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "inference_ms": inference_ms,
        "response_time_ms": inference_ms,  # Alias
        "domain": domain,
        "plan": plan,
        "is_master": is_master,
        "request_length": len(request_text) if request_text else 0,
        "response_length": len(response_text) if response_text else 0,
    }

    # Ajouter les textes (avec limite de taille pour éviter les logs trop longs)
    if request_text:
        log_entry["request_text"] = request_text[:500]
    if response_text:
        log_entry["response_text"] = response_text[:500]
    if error:
        log_entry["error"] = error[:200]

    # Ajouter les données extras
    if extra:
        log_entry["extra"] = extra

    # Écrire dans le fichier JSON (une ligne par entrée)
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Erreur écriture audit log: {e}")

    # Mettre à jour les stats en mémoire
    _update_stats(endpoint, inference_ms, status_code, client_id, client_name)


def _update_stats(
    endpoint: str,
    inference_ms: int,
    status_code: int,
    client_id: int,
    client_name: str
) -> None:
    """Met à jour les statistiques en mémoire (thread-safe)"""

    with _stats_lock:
        key = f"{endpoint}"

        _request_stats[key]["count"] += 1
        _request_stats[key]["total_ms"] += inference_ms
        if status_code >= 400:
            _request_stats[key]["errors"] += 1
        _request_stats[key]["last_request"] = datetime.now().isoformat()
        _request_stats[key]["clients"].add(client_id)

        # Stats globales
        _request_stats["_global"]["count"] += 1
        _request_stats["_global"]["total_ms"] += inference_ms
        if status_code >= 400:
            _request_stats["_global"]["errors"] += 1
        _request_stats["_global"]["clients"].add(client_id)


# =============================================================================
# FONCTIONS DE STATISTIQUES
# =============================================================================

def get_request_stats(endpoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Récupère les statistiques de requêtes

    Args:
        endpoint: Filtre par endpoint (None = tous)

    Returns:
        Stats formatées
    """
    with _stats_lock:
        stats = dict(_request_stats)

    result = {}
    for key, data in stats.items():
        if endpoint and key != endpoint:
            continue
        if key.startswith("_"):
            continue

        result[key] = {
            "count": data["count"],
            "avg_ms": round(data["total_ms"] / data["count"], 2) if data["count"] > 0 else 0,
            "total_ms": data["total_ms"],
            "errors": data["errors"],
            "success_rate": round((data["count"] - data["errors"]) / data["count"] * 100, 2)
                if data["count"] > 0 else 0,
            "last_request": data["last_request"],
            "unique_clients": len(data["clients"])
        }

    return result


def get_realtime_stats() -> Dict[str, Any]:
    """
    Récupère les stats temps réel globales

    Returns:
        Stats globales du système
    """
    global _stats_cache

    # Retourner le cache si encore valide
    now = time.time()
    if now - _stats_cache["updated"] < _CACHE_TTL:
        return _stats_cache["data"]

    with _stats_lock:
        global_stats = dict(_request_stats.get("_global", {
            "count": 0,
            "total_ms": 0,
            "errors": 0,
            "clients": set()
        }))
        endpoint_stats = {k: v for k, v in _request_stats.items() if not k.startswith("_")}

    result = {
        "total_requests": global_stats["count"],
        "active_clients": len(global_stats["clients"]),
        "avg_inference_ms": round(global_stats["total_ms"] / global_stats["count"], 2)
            if global_stats["count"] > 0 else 0,
        "error_rate": round(global_stats["errors"] / global_stats["count"] * 100, 2)
            if global_stats["count"] > 0 else 0,
        "endpoints": {k: v["count"] for k, v in endpoint_stats.items()}
    }

    _stats_cache = {"data": result, "updated": now}
    return result


def get_audit_trail(
    client_id: Optional[int] = None,
    endpoint: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Récupère l'historique d'audit

    Args:
        client_id: Filtrer par client
        endpoint: Filtrer par endpoint
        start_date: Date de début
        end_date: Date de fin
        limit: Nombre maximum de résultats

    Returns:
        Liste des entrées d'audit
    """
    results = []
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= limit:
                    break

                try:
                    entry = json.loads(line.strip())

                    # Appliquer les filtres
                    if client_id is not None and entry.get("client_id") != client_id:
                        continue
                    if endpoint and entry.get("endpoint") != endpoint:
                        continue

                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue

                    results.append(entry)
                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        pass

    return results


def get_billing_summary(client_id: int, month: int, year: int) -> Dict[str, Any]:
    """
    Génère un résumé de facturation pour un client

    Args:
        client_id: ID du client
        month: Mois (1-12)
        year: Année

    Returns:
        Résumé de facturation
    """
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)

    trail = get_audit_trail(
        client_id=client_id,
        start_date=start_date,
        end_date=end_date,
        limit=10000
    )

    total_requests = len(trail)
    successful = sum(1 for e in trail if e.get("status_code", 200) < 400)
    failed = total_requests - successful

    total_ms = sum(e.get("inference_ms", 0) for e in trail)
    avg_ms = round(total_ms / total_requests, 2) if total_requests > 0 else 0

    return {
        "client_id": client_id,
        "period": f"{year}-{month:02d}",
        "total_requests": total_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "total_inference_ms": total_ms,
        "avg_inference_ms": avg_ms,
        "domains": list(set(e.get("domain", "") for e in trail if e.get("domain")))
    }


def reset_stats() -> None:
    """Réinitialise les statistiques en mémoire"""

    with _stats_lock:
        _request_stats.clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'audit_log',
    'get_request_stats',
    'get_realtime_stats',
    'get_audit_trail',
    'get_billing_summary',
    'reset_stats',
]