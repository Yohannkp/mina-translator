"""
scripts/test_mina_translator.py - Test du système de traduction Mina
====================================================================

Teste la traduction FR↔Mina avec Ollama et le vocabulaire Mina.

Usage:
    python scripts/test_mina_translator.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import sys
import time
from pathlib import Path

# UTF-8 sur Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_API = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:7b"
PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"

# =============================================================================
# CHARGER LE CORPUS
# =============================================================================

def load_corpus():
    """Charge le corpus FR→Mina."""
    corpus_file = CORPUS_DIR / "dataset_mina.jsonl"
    if not corpus_file.exists():
        return []

    entries = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    return entries


# =============================================================================
# EXEMPLES DU CORPUS (les 30 meilleures paires)
# =============================================================================

CORPUS = load_corpus()
EXAMPLE_PAIRS = [(e['input'], e['output']) for e in CORPUS[:30]]

# Construire le prompt avec exemples
def build_examples_block():
    lines = ["EXEMPLES DE TRADUCTIONS FRANÇAIS -> MINA (Ewe du Togo):"]
    for fr, mina in EXAMPLE_PAIRS[:20]:
        # Simplifier pour Ollama (éviter les caractères spéciaux)
        fr_clean = fr.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
        fr_clean = fr_clean.replace('à', 'a').replace('â', 'a')
        fr_clean = fr_clean.replace('î', 'i').replace('ô', 'o').replace('ù', 'u')
        mina_clean = mina.replace('ɛ', 'e').replace('ɔ', 'o').replace('ɖ', 'd')
        lines.append(f"- {fr_clean} = {mina_clean}")
    return "\n".join(lines)


# =============================================================================
# TRADUCTION
# =============================================================================

def translate(text: str, source_lang: str = "fr", target_lang: str = "mina") -> dict:
    """Traduit le texte via Ollama."""

    start = time.time()

    # Construire le prompt
    if source_lang == "fr" and target_lang == "mina":
        instruction = "Traduis cette phrase française en mina (ewe du Togo)."
        direction = "FRANÇAIS -> MINA"
    else:
        instruction = "Traduis cette phrase en mina vers le français."
        direction = "MINA -> FRANÇAIS"

    # Vocabulaire clé
    vocab = """
VOCABULAIRE MINA CLÉ:
- sawo = bonjour
- akɔ = merci
- me = je
- a = tu
- ku = avoir mal / être malade
- mɛ = négation
- Nusɔla = infirmier/médecin
- hɔspita = hôpital
- va = venir
- ka = aller
- ɖevi = fièvre/chaleur
- susumɔ = dormir
- nuwɔna = santé
"""

    # Exemples du corpus
    examples = build_examples_block()

    # Prompt complet
    prompt = f"""Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.
{direction}

{vocab}

{examples}

{instruction}
{text}

Réponds UNIQUEMENT avec la traduction en mina. Pas d'explication."""

    # Appeler Ollama
    try:
        r = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0.1,
                "num_predict": 100,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=60
        )

        elapsed = (time.time() - start) * 1000

        if r.status_code == 200:
            result = r.json()["response"].strip()
            return {
                "original": text,
                "translation": result,
                "time_ms": round(elapsed, 1),
                "success": True,
            }
        else:
            return {
                "original": text,
                "error": r.text,
                "success": False,
            }
    except Exception as e:
        return {
            "original": text,
            "error": str(e),
            "success": False,
        }


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Lance les tests de traduction."""

    print("\n" + "="*70)
    print("   MINA-TRANSLATOR - TEST DE TRADUCTION")
    print("="*70)

    # Vérifier Ollama
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"\n   Ollama: ✓ Connecté")
            print(f"   Modèle: {OLLAMA_MODEL}")
            print(f"   Corpus: {len(EXAMPLE_PAIRS)} paires FR→Mina")
        else:
            print("\n   ❌ Ollama non accessible")
            return
    except:
        print("\n   ❌ Ollama non accessible")
        print("   Lancez: ollama serve")
        return

    # Phrases de test (jamais vues par le modèle)
    test_phrases = [
        "Je vais bien.",
        "Le medecin est arrive.",
        "Ou est la pharmacie?",
        "J'ai mal a la tete.",
        "L'enfant dort bien.",
    ]

    print("\n" + "-"*70)
    print("   RÉSULTATS DES TESTS")
    print("-"*70)

    results = []
    for i, phrase in enumerate(test_phrases, 1):
        result = translate(phrase)

        # Afficher
        print(f"\n   [{i}/{len(test_phrases)}]")
        print(f"   FR:  {phrase}")

        if result["success"]:
            print(f"   MINA: {result['translation']}")
            print(f"   TEMPS: {result['time_ms']} ms")

            # Comparer avec le corpus si disponible
            for fr, mina in EXAMPLE_PAIRS:
                if phrase.lower() in fr.lower() or fr.lower() in phrase.lower():
                    print(f"   REF:  {mina}")
                    break
        else:
            print(f"   ❌ Erreur: {result.get('error', 'Unknown')}")

        results.append(result)

    # Résumé
    print("\n" + "="*70)
    print("   RÉSUMÉ")
    print("="*70)

    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["time_ms"] for r in results if r["success"]) / max(success_count, 1)

    print(f"\n   Tests réussis: {success_count}/{len(results)}")
    print(f"   Temps moyen: {avg_time:.1f} ms")
    print(f"   Modèle: {OLLAMA_MODEL}")

    if success_count == len(results):
        print("\n   ✅ TOUS LES TESTS RÉUSSIS!")
    else:
        print(f"\n   ⚠️  {len(results) - success_count} tests échoués")

    return results


def interactive_mode():
    """Mode interactif de traduction."""
    print("\n" + "="*70)
    print("   MODE INTERACTIF - Tapez 'quit' pour sortir")
    print("="*70)

    while True:
        try:
            text = input("\n   FRANÇAIS > ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("   Au revoir!")
                break

            if not text:
                continue

            result = translate(text)

            if result["success"]:
                print(f"   MINA   > {result['translation']}")
                print(f"   ({result['time_ms']} ms)")
            else:
                print(f"   ❌ Erreur: {result.get('error', 'Unknown')}")

        except KeyboardInterrupt:
            print("\n   Au revoir!")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Mina Translator")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Mode interactif")
    parser.add_argument("--examples", "-e", action="store_true",
                       help="Afficher les exemples du corpus")
    args = parser.parse_args()

    if args.examples:
        print("\n" + "="*70)
        print("   EXEMPLES DU CORPUS (30 premières paires)")
        print("="*70)
        for i, (fr, mina) in enumerate(EXAMPLE_PAIRS[:30], 1):
            print(f"   {i:2d}. {fr}")
            print(f"       {mina}")
            print()

    if args.interactive:
        interactive_mode()
    else:
        run_tests()


if __name__ == "__main__":
    main()