"""
scripts/test_api.py - Test de l'API Mina-Translator
===================================================

Teste l'API de traduction avec des phrases inédites.

Usage:
    py scripts/test_api.py

Auteur: Claude Opus 4.8
Date: 2026-06-05
"""

import sys
import io
import time
import json
from pathlib import Path

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests

API_URL = "http://localhost:8000"

def test_health():
    """Test le endpoint /health"""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        data = resp.json()

        print(f"Status: {data['status']}")
        print(f"Model: {data['model_name']}")
        print(f"Device: {data['device']}")
        print(f"Timestamp: {data['timestamp']}")

        return data['status'] == 'healthy'
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_translate_fr_to_mina(phrase: str) -> tuple:
    """Teste la traduction FR → Mina et mesure le temps"""
    start = time.time()

    try:
        resp = requests.post(
            f"{API_URL}/translate",
            json={
                "text": phrase,
                "temperature": 0.3,
                "max_length": 256
            },
            timeout=60
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return data['translation'], elapsed, True
        else:
            return f"Erreur {resp.status_code}", elapsed, False

    except Exception as e:
        return str(e), 0, False

def test_translate_mina_to_fr(phrase: str) -> tuple:
    """Teste la traduction Mina → FR et mesure le temps"""
    start = time.time()

    try:
        resp = requests.post(
            f"{API_URL}/translate_mina",
            json={
                "text": phrase,
                "temperature": 0.3,
                "max_length": 256
            },
            timeout=60
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            return data['translation'], elapsed, True
        else:
            return f"Erreur {resp.status_code}", elapsed, False

    except Exception as e:
        return str(e), 0, False

def test_batch_translate():
    """Teste la traduction par lot"""
    print("\n" + "=" * 60)
    print("TEST: Batch Translation")
    print("=" * 60)

    texts = [
        "La santé est importante pour tous.",
        "J'ai besoin d'un médecin.",
        "Où est la pharmacie?"
    ]

    try:
        resp = requests.post(
            f"{API_URL}/translate/batch",
            json={
                "texts": texts,
                "temperature": 0.3
            },
            timeout=120
        )

        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ Traductions traitées: {data['count']}")

            for i, result in enumerate(data['results']):
                print(f"\n[{i+1}] FR: {result['original']}")
                if 'error' in result:
                    print(f"    Erreur: {result['error']}")
                else:
                    print(f"    Mina: {result['translation']}")
                    print(f"    Temps: {result['inference_time_ms']:.0f} ms")

            return True
        else:
            print(f"❌ Erreur: {resp.status_code}")
            return False

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("MINA-TRANSLATOR API - TESTS")
    print("=" * 60)

    # Test santé
    if not test_health():
        print("\n❌ API non accessible. Lancez d'abord:")
        print("   py -m uvicorn api.main:app --reload --port 8000")
        return

    # Phrases de test FR → Mina (inédites, pas dans le dataset)
    test_phrases_fr = [
        "Le président de la République est arrivé à Lomé ce matin.",
        "J'ai perdu ma carte d'identité nationale.",
        "Mon fils est malade, je cherche un hôpital pour enfants.",
        "Combien coûte un taxi pour aller à Kpalimé?",
        "Je voudrais acheter du poisson au marché de Djidjolé."
    ]

    print("\n" + "=" * 60)
    print("TEST: Traduction FR → Mina (5 phrases inédites)")
    print("=" * 60)

    total_time = 0
    for i, phrase in enumerate(test_phrases_fr):
        print(f"\n[{i+1}/5] Français: {phrase}")

        translation, elapsed, success = test_translate_fr_to_mina(phrase)
        total_time += elapsed

        if success:
            print(f"    Mina: {translation}")
            print(f"    ⏱ Temps: {elapsed:.0f} ms")
        else:
            print(f"    ❌ Erreur: {translation}")

    avg_time = total_time / len(test_phrases_fr)
    print(f"\n📊 Temps moyen: {avg_time:.0f} ms / requête")

    # Test Mina → FR avec quelques traductions
    test_phrases_mina = [
        "Nusɔla va le ta.",
        "Miƒo nuve.",
        "Ema na govie."
    ]

    print("\n" + "=" * 60)
    print("TEST: Traduction Mina → FR")
    print("=" * 60)

    for i, phrase in enumerate(test_phrases_mina):
        print(f"\n[{i+1}/3] Mina: {phrase}")

        translation, elapsed, success = test_translate_mina_to_fr(phrase)

        if success:
            print(f"    Français: {translation}")
            print(f"    ⏱ Temps: {elapsed:.0f} ms")

    # Test batch
    test_batch_translate()

    print("\n" + "=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)

if __name__ == "__main__":
    main()