"""
scripts/translate_with_ollama.py - Traduction via API Ollama locale
=====================================================================

Utilise les modèles Ollama déjà installés sur votre machine pour traduire
du français vers le Mina. Pas besoin de télécharger des modèles séparés.

Usage:
    python scripts/translate_with_ollama.py [--model MODEL] [--auto]

Exemples:
    # Sélection automatique du meilleur modèle
    python scripts/translate_with_ollama.py --auto

    # Avec le modèle recommandé (qwen2.5-coder)
    python scripts/translate_with_ollama.py --model qwen2.5-coder:7b

    # Mode interactif
    python scripts/translate_with_ollama.py --interactive

Prérequis:
    - Ollama installé et en cours d'exécution
    - Modèle disponible

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

# Requêtes HTTP
try:
    import requests
except ImportError:
    print("❌ Module 'requests' non installé.")
    print("   Installez avec: pip install requests")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# URL de l'API Ollama (localhost)
OLLAMA_API = "http://localhost:11434"

# =============================================================================
# SÉLECTION DU MODÈLE OPTIMAL
# =============================================================================

# Critères de sélection pour la traduction FR→Mina
MODEL_SCORES = {
    # Format: "nom_base": (score, taille_go, vitesse, notes)
    # score: 0-100 (qualité pour traduction)
    # taille: GB (mémoire utilisée)
    # vitesse: 1-5 (1=très rapide, 5=lent)
    "qwen2.5-coder": (95, 3.5, 1, "⭐ Recommandé - Rapide et bon en suivi d'instructions"),
    "qwen2.5": (90, 3.5, 1, "Bon - Version standard"),
    "llama3.1": (92, 4.5, 2, "Très bon - Qualité supérieure"),
    "llama3": (88, 4.5, 3, "Bon - Alternative stable"),
    "mistral": (85, 4.0, 2, "Correct - Instruction following"),
    "deepseek-coder": (82, 3.5, 2, "Correct - Moins optimisé pour traduction"),
    "deepseek-r1": (75, 5.0, 5, "⚠️ Lente - Modèle raisonnement (affiche pensée)"),
    "nomic-embed": (0, 0.5, 1, "❌ Embeddings only - Ne pas utiliser pour traduction"),
}

# Modèles à exclure (pas adaptés pour traduction)
EXCLUDED_PATTERNS = ["embed", "embedding", "vision", "visionary"]


def get_model_score(model_name: str) -> tuple[int, int, int, str]:
    """Calcule le score d'un modèle pour la traduction."""
    model_lower = model_name.lower()

    # Vérifier les exclusions
    for pattern in EXCLUDED_PATTERNS:
        if pattern in model_lower:
            return (0, 0, 0, "❌ Exclu - Pas adapté")

    # Rechercher une correspondance
    for base_name, score_info in MODEL_SCORES.items():
        if base_name in model_lower:
            return score_info

    # Modèle non reconnu - score moyen
    return (50, 4.0, 3, "⚠️ Non testé")


def select_best_model(available_models: List[str]) -> tuple[str, int, int, int, str]:
    """
    Sélectionne le meilleur modèle pour la traduction.

    Returns:
        Tuple (nom_model, score, taille_go, vitesse, note)
    """
    candidates = []

    for model in available_models:
        # Extraire le nom de base (sans tag de version)
        base_name = model.split(":")[0] if ":" in model else model

        score, size, speed, note = get_model_score(base_name)

        if score > 0:  # Exclure les modèles non adaptés
            candidates.append((model, score, size, speed, note))

    if not candidates:
        return ("aucun", 0, 0, 0, "❌ Aucun modèle adapté trouvé")

    # Trier par score (desc), puis par vitesse (asc)
    candidates.sort(key=lambda x: (-x[1], x[3]))

    best = candidates[0]
    return best


def analyze_models(available_models: List[str]) -> List[Dict]:
    """Analyse tous les modèles disponibles et retourne un rapport."""
    print("\n" + "=" * 70)
    print("   📊 ANALYSE DES MODÈLES POUR TRADUCTION FR→MINA")
    print("=" * 70)
    print()
    print(f"   {'Modèle':<30} {'Score':<8} {'Taille':<10} {'Vitesse':<8} {'Note'}")
    print("   " + "-" * 70)

    analysis = []
    for model in available_models:
        score, size, speed, note = get_model_score(model)
        speed_label = ["⚡⚡⚡⚡⚡", "⚡⚡⚡⚡", "⚡⚡⚡", "⚡⚡", "⚡"][min(speed-1, 4)] if speed else "?"

        print(f"   {model:<30} {score:<8} {size}Go{'':<5} {speed_label:<8} {note}")
        analysis.append({
            "model": model,
            "score": score,
            "size": size,
            "speed": speed,
            "note": note,
        })

    print()
    return analysis

# =============================================================================
# PROMPTS
# =============================================================================

# Template de génération - prompt enrichi pour le Mina
TRANSLATION_TEMPLATE = """Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.

RÈGLES:
- Traduis EXACTEMENT du français vers le Mina (gen/min) parlé à Lomé et au sud du Togo
- Utilise les expressions locales et l'argot togolais si approprié
- Réponds UNIQUEMENT avec la traduction, sans explanation ni phrase en français
- Caractères spéciaux du Mina: ɔ ɛ ɖ ŋ et tones
- Exemples de termes Mina: wɔ (verbe), na (pour/à), be (que), se (dit)
- Commence directement par la traduction

Traduis cette phrase en Mina:

Français: {phrase}

Mina:"""


# =============================================================================
# FONCTIONS OLLAMA
# =============================================================================

def check_ollama_running() -> bool:
    """Vérifie si Ollama est en cours d'exécution."""
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def list_available_models() -> List[str]:
    """Liste les modèles Ollama disponibles."""
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        return []
    except:
        return []


def clean_mina_output(text: str) -> str:
    """
    Nettoie la sortie du modèle pour extraire uniquement la traduction Mina.
    Gère les modes réflexion (deepseek-r1, etc.) et les artefacts markdown.
    """
    if not text:
        return text

    # Supprimer les balises de réflexion/think blocks
    patterns_to_remove = [
        r'</?think[^>]*>.*?</think>',  # <think>...</think>
        r'<think>.*?</think>',            # <think>...</think>
        r'\[think\][\s\S]*?\[/think\]',  # [think]...[/think]
    ]

    cleaned = text
    for pattern in patterns_to_remove:
        import re
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Supprimer les espaces multiples
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Supprimer les balises markdown残留
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # **bold**
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # *italic*

    return cleaned.strip()


def generate_with_ollama(model: str, prompt: str, temperature: float = 0.3) -> tuple[str, float]:
    """
    Génère du texte avec Ollama.

    Args:
        model: Nom du modèle Ollama
        prompt: Prompt de génération
        temperature: Température (0 = déterministe, 1 = créatif)

    Returns:
        Tuple (texte_généré, temps_exécution_ms)
    """
    url = f"{OLLAMA_API}/api/generate"

    # Payload simple - sans "system" qui peut causer des erreurs
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": 128,
        "top_p": 0.9,
        "stream": False,
    }

    start_time = time.time()

    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            raw_response = result.get("response", "").strip()

            # Nettoyer la sortie
            cleaned = clean_mina_output(raw_response)

            return cleaned, elapsed_ms
        else:
            return f"[Erreur HTTP {response.status_code}: {response.text[:200]}]", elapsed_ms

    except requests.Timeout:
        return "[Timeout - modèle trop lent]", 0
    except Exception as e:
        return f"[Erreur: {type(e).__name__}: {e}]", 0


# =============================================================================
# FONCTIONS DE TRADUCTION
# =============================================================================

def translate_phrase(phrase: str, model: str) -> tuple[str, float]:
    """
    Traduit une phrase du français vers le Mina.

    Args:
        phrase: Texte en français
        model: Modèle Ollama à utiliser

    Returns:
        Tuple (traduction, temps_ms)
    """
    prompt = TRANSLATION_TEMPLATE.format(phrase=phrase)
    return generate_with_ollama(model, prompt, temperature=0.2)


def translate_batch(phrases: List[str], model: str) -> List[dict]:
    """
    Traduit une liste de phrases.

    Args:
        phrases: Liste de phrases en français
        model: Modèle Ollama à utiliser

    Returns:
        Liste de résultats avec traductions et temps
    """
    results = []

    for i, phrase in enumerate(phrases, 1):
        print(f"\n📝 Phrase {i}/{len(phrases)}: {phrase}")

        translation, elapsed_ms = translate_phrase(phrase, model)

        print(f"   ⏱️  Temps: {elapsed_ms:.0f} ms")
        print(f"   ✅ Mina: {translation}")

        results.append({
            "id": i,
            "source": phrase,
            "translation": translation,
            "time_ms": round(elapsed_ms, 1),
            "model": model,
            "timestamp": datetime.now().isoformat(),
        })

    return results


# =============================================================================
# INTERFACE UTILISATEUR
# =============================================================================

def print_header():
    """Affiche l'en-tête du programme."""
    print()
    print("=" * 70)
    print("   🌐 TRADUCTION FRANÇAIS → MINA VIA OLLAMA")
    print("=" * 70)
    print()


def print_status(model: str, available_models: List[str]):
    """Affiche le statut actuel."""
    print(f"   🤖 Modèle: {model}")
    print(f"   🔗 API: {OLLAMA_API}")
    print()
    print("   📦 Modèles disponibles:")
    for m in available_models:
        marker = " ←" if m == model else ""
        print(f"      - {m}{marker}")
    print()


def display_results(results: List[dict], total_time: float):
    """Affiche les résultats de traduction."""
    print()
    print("=" * 70)
    print("   📊 RÉSULTATS")
    print("=" * 70)
    print()

    avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0

    for r in results:
        print(f"  [{r['id']}] {r['source']}")
        print(f"      → {r['translation']}")
        print(f"      ⏱️  {r['time_ms']:.0f} ms")
        print()

    print("-" * 70)
    print(f"   📈 Temps moyen: {avg_time:.0f} ms/phrase")
    print(f"   📈 Total: {total_time:.0f} ms pour {len(results)} phrases")
    print("=" * 70)


def interactive_mode(model: str):
    """Mode interactif pour traductions en continu."""
    print("\n🗣️  Mode interactif - Tapez 'quit' pour quitter")
    print("-" * 50)

    while True:
        try:
            phrase = input("\n💬 Français > ").strip()

            if phrase.lower() in ['quit', 'exit', 'q', 'quitter']:
                print("\n👋 Au revoir !")
                break

            if not phrase:
                continue

            translation, elapsed = translate_phrase(phrase, model)
            print(f"   ⏱️  {elapsed:.0f} ms")
            print(f"   Mina: {translation}")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir !")
            break


# =============================================================================
# TESTS
# =============================================================================

DEFAULT_TEST_PHRASES = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Je voudrais un rendez-vous à l'hôpital demain matin.",
    "Où se trouve la pharmacie la plus proche ?",
    "Je dois payer ma facture d'électricité.",
    "Mon enfant est malade, j'ai besoin d'un médecin.",
]


def run_test(model: str, phrases: List[str] = None):
    """Exécute un test de traduction."""
    if phrases is None:
        phrases = DEFAULT_TEST_PHRASES

    print(f"\n🧪 Test avec {len(phrases)} phrases...")
    print()

    start = time.time()
    results = translate_batch(phrases, model)
    total_time = (time.time() - start) * 1000

    display_results(results, total_time)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Traduction Français → Mina via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Modèle Ollama à utiliser (défaut: sélection automatique)",
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Sélection automatique du meilleur modèle",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Mode interactif",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Lancer le test de traduction",
    )
    parser.add_argument(
        "--phrase", "-p",
        type=str,
        help="Traduire une seule phrase",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Analyser et classer les modèles disponibles",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Comparer tous les modèles disponibles",
    )

    args = parser.parse_args()

    print_header()

    # Vérifier Ollama
    if not check_ollama_running():
        print("❌ Ollama n'est pas en cours d'exécution !")
        print()
        print("   Démarrez Ollama avec:")
        print("   - Windows: Recherche 'Ollama' dans le menu Démarrer")
        print("   - Terminal: ollama serve")
        print()
        sys.exit(1)

    print("✅ Ollama est en cours d'exécution")

    # Lister les modèles
    available = list_available_models()

    if args.compare or args.list:
        analyze_models(available)
        if args.compare:
            print("\n📌 Recommandations:")
            best_model, best_score, best_size, best_speed, best_note = select_best_model(available)
            print(f"\n   🏆 Meilleur modèle: {best_model}")
            print(f"   📊 Score: {best_score}/100 | 💾 Taille: {best_size}Go | ⚡ Vitesse: {best_speed}/5")
            print(f"   📝 {best_note}")
            print()
            print("   Pour lancer avec ce modèle:")
            print(f"   python scripts/translate_with_ollama.py --model {best_model}")
        return

    # Sélection automatique si demandé ou pas de modèle spécifié
    if args.auto or args.model is None:
        best_model, best_score, best_size, best_speed, best_note = select_best_model(available)

        print("\n" + "=" * 70)
        print("   🎯 SÉLECTION AUTOMATIQUE DU MODÈLE")
        print("=" * 70)
        print(f"\n   🏆 Modèle sélectionné: {best_model}")
        print(f"   📊 Score: {best_score}/100 | 💾 Taille: {best_size}Go | ⚡ Vitesse: {best_speed}/5")
        print(f"   📝 {best_note}")

        model = best_model
    else:
        model = args.model

    # Vérifier le modèle
    if model not in available:
        print(f"\n⚠️  Modèle '{model}' non disponible.")
        print("   Modèles disponibles:", ", ".join(available))

        if available:
            # Utiliser le meilleur modèle disponible
            best_model, best_score, best_size, best_speed, best_note = select_best_model(available)
            model = best_model
            print(f"   → Utilisation de: {model} (meilleur alternatif)")
        else:
            print("❌ Aucun modèle disponible. Installez-en un avec:")
            print("   ollama pull qwen2.5-coder")
            sys.exit(1)

    print_status(model, available)

    # Mode interactif
    if args.interactive:
        interactive_mode(model)
        return

    # Mode phrase unique
    if args.phrase:
        print(f"\n🔄 Traduction de: {args.phrase}")
        translation, elapsed = translate_phrase(args.phrase, model)
        print(f"   ⏱️  {elapsed:.0f} ms")
        print(f"   Mina: {translation}")
        return

    # Mode test (par défaut)
    results = run_test(model)

    # Sauvegarder les résultats
    output_file = Path("logs") / f"translation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats sauvegardés: {output_file}")


if __name__ == "__main__":
    main()