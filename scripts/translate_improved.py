"""
scripts/translate_improved.py - Traduction améliorée avec vocabulaire Mina
=========================================================================

Utilise le vocabulaire Mina pour améliorer les traductions.
Inclut un vocabulaire structuré et des expressions courantes.

Usage:
    python scripts/translate_improved.py [--phrase "texte"] [--interactif]

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ Module 'requests' non installé.")
    print("   pip install requests")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_API = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
VOCAB_FILE = Path("data/corpus/vocabulaire_mina.json")

# =============================================================================
# CHARGEMENT DU VOCABULAIRE
# =============================================================================

def load_vocabulary() -> Dict:
    """Charge le vocabulaire Mina depuis le fichier JSON."""
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def build_improved_prompt(phrase: str, vocab: Dict) -> str:
    """
    Construit un prompt amélioré avec le vocabulaire Mina.
    """
    vocab_context = """
=== VOCABULAIRE MINA (Gen/Min) - TERMES CLÉS ===

SALUTATIONS: bonjour=sawo, merci=akɔ, au-revoir=nukɔ, s'il-vous-plaît=meki
SANTÉ: médecin=dɔta, pharmacie=soko aɖa, hôpital=hɔspita, malade=fɛ́e, médicament=aɖa
ADMIN: carte d'identité=kati, passeport=pasepɔ, police=polisi, école=sku
TEMPS: aujourd'hui=ai, demain=ẹmɛ, matin=zinzin, soir=blɔna
LIEUX: maison=se, rue=dzio, ici=ci, là=yi, près=lai, loin=gan
VERBES: être=le, avoir=ena, aller=ka, venir=va, faire=wɔ, dire=ke, voir=kpɔ
QUESTIONS: où=ci, quand=etɛ, oui=awo, non=ai
PRONOMS: je=me, tu=a, il/elle=e, nous=ye, vous=me
ARGOT TOGO: yovo=blanc/étranger, kpo=très, dzro=argent, kaka=tout

RÈGLES SYNTAXE:
- Ordre: Sujet-Verbe-Objet (SVO)
- Négation: "ai" après le verbe (ex: me kpede ai = je ne comprends pas)
- Utilise les termes Mina列表, pas les emprunts français/anglais!

"""

    prompt = f"""{vocab_context}

=== TRADUCTION FRANÇAIS → MINA ===

Traduis cette phrase en Mina en utilisant le vocabulaire ci-dessus:

"{phrase}"

RÈGLES:
1. Utilise les termes Mina du vocabulaire (pas d'emprunts français/anglais)
2. Garde l'ordre des mots: Sujet-Verbe-Objet
3. Réponds UNIQUEMENT avec la traduction Mina
4. Pas d'explications ni de commentaires

Mina:"""

    return prompt


# =============================================================================
# TRADUCTION
# =============================================================================

def translate_with_vocab(phrase: str, model: str, vocab: Dict) -> tuple[str, float]:
    """Traduit avec le vocabulaire Mina intégré au prompt."""
    url = f"{OLLAMA_API}/api/generate"

    prompt = build_improved_prompt(phrase, vocab)

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.2,
        "num_predict": 128,
        "top_p": 0.9,
        "stream": False,
    }

    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed = (time.time() - start) * 1000

        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "").strip()

            # Nettoyer la sortie
            import re
            text = re.sub(r'</?think[^>]*>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'\s+', ' ', text)

            return text, elapsed
        else:
            return f"[Erreur {response.status_code}]", elapsed
    except Exception as e:
        return f"[Erreur: {e}]", 0


def check_ollama() -> bool:
    """Vérifie si Ollama est actif."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """Liste les modèles disponibles."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    return []


# =============================================================================
# TESTS COMPARATIFS
# =============================================================================

TEST_PHRASES = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Je voudrais un rendez-vous à l'hôpital demain matin.",
    "Où se trouve la pharmacie la plus proche ?",
    "Je dois payer ma facture d'électricité.",
    "Mon enfant est malade, j'ai besoin d'un médecin.",
    "Merci beaucoup pour votre aide.",
    "Je ne comprends pas, pouvez-vous répéter ?",
    "Combien ça coûte ?",
    "Appelez la police, c'est urgent !",
    "Où est le marché ?",
]


def run_comparison():
    """Compare les traductions avec et sans vocabulaire."""
    print("\n" + "=" * 70)
    print("   🔬 TEST COMPARATIF: AVEC VOCABULAIRE MINA")
    print("=" * 70)

    if not check_ollama():
        print("\n❌ Ollama n'est pas en cours d'exécution!")
        return

    vocab = load_vocabulary()
    if not vocab:
        print("\n⚠️  Vocabulaire non trouvé, création...")
        vocab = {}

    models = get_available_models()
    model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    if not model:
        print("\n❌ Aucun modèle disponible.")
        return

    print(f"\n🤖 Modèle: {model}")
    print(f"📚 Vocabulaire: {'Chargé' if vocab else 'Non chargé'}")
    print()

    results = []
    for i, phrase in enumerate(TEST_PHRASES, 1):
        print(f"[{i}/{len(TEST_PHRASES)}] {phrase}")

        translation, elapsed = translate_with_vocab(phrase, model, vocab)

        print(f"   ⏱️  {elapsed:.0f} ms")
        print(f"   Mina: {translation}")
        print()

        results.append({
            "phrase": phrase,
            "traduction": translation,
            "temps_ms": round(elapsed, 1),
        })

    # Résumé
    avg_time = sum(r["temps_ms"] for r in results) / len(results)
    print("=" * 70)
    print(f"   📊 RÉSUMÉ")
    print("=" * 70)
    print(f"   Phrases traduites: {len(results)}")
    print(f"   Temps moyen: {avg_time:.0f} ms/phrase")
    print("=" * 70)

    # Sauvegarder
    output = Path("logs") / f"improved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model,
            "vocabulary_loaded": bool(vocab),
            "results": results,
            "avg_time_ms": avg_time,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats: {output}")


# =============================================================================
# MODE INTERACTIF
# =============================================================================

def interactive_mode():
    """Mode interactif pour traductions continues."""
    vocab = load_vocabulary()
    models = get_available_models()
    model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)

    if not model:
        print("❌ Aucun modèle disponible.")
        return

    print(f"\n🗣️  Mode interactif (modèle: {model})")
    print("   Tapez 'quit' ou 'exit' pour quitter")
    print("   Tapez 'vocab' pour voir le vocabulaire")
    print("-" * 50)

    while True:
        try:
            phrase = input("\n💬 Français > ").strip()

            if phrase.lower() in ['quit', 'exit', 'q', 'quitter']:
                print("\n👋 Au revoir!")
                break

            if phrase.lower() == 'vocab':
                print("\n📚 Vocabulaire Mina:")
                if vocab.get("vocabulary"):
                    for cat, words in vocab["vocabulary"].items():
                        print(f"\n  {cat.upper()}:")
                        for fr, mina in list(words.items())[:5]:
                            print(f"    {fr} = {mina}")
                continue

            if not phrase:
                continue

            translation, elapsed = translate_with_vocab(phrase, model, vocab)
            print(f"   ⏱️  {elapsed:.0f} ms")
            print(f"   Mina: {translation}")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Traduction FR→Mina améliorée")
    parser.add_argument("--phrase", "-p", type=str, help="Traduire une phrase")
    parser.add_argument("--interactif", "-i", action="store_true", help="Mode interactif")
    parser.add_argument("--test", "-t", action="store_true", help="Test comparatif")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Modèle Ollama")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("   🌐 TRADUCTION FRANÇAIS → MINA (AMÉLIORÉE)")
    print("   📚 Avec vocabulaire Mina intégré")
    print("=" * 70)

    if not check_ollama():
        print("\n❌ Ollama n'est pas en cours d'exécution!")
        print("   Lancez: ollama serve")
        sys.exit(1)

    vocab = load_vocabulary()
    print(f"\n✅ Ollama actif | 📚 Vocabulaire: {'✓' if vocab else '⚠️'}")

    if args.interactif:
        interactive_mode()
        return

    if args.phrase:
        translation, elapsed = translate_with_vocab(args.phrase, args.model, vocab)
        print(f"\n🔄 {args.phrase}")
        print(f"   ⏱️  {elapsed:.0f} ms")
        print(f"   Mina: {translation}")
        return

    if args.test:
        run_comparison()
        return

    # Demo avec une phrase
    demo = "Bonjour, je voudrais voir le médecin"
    translation, elapsed = translate_with_vocab(demo, args.model, vocab)
    print(f"\n🔄 {demo}")
    print(f"   ⏱️  {elapsed:.0f} ms")
    print(f"   Mina: {translation}")
    print("\n\nOPTIONS:")
    print("  --phrase 'texte'    Traduire une phrase")
    print("  --interactif        Mode interactif")
    print("  --test              Test comparatif (10 phrases)")
    print("  --model <nom>       Choisir le modèle")


if __name__ == "__main__":
    main()