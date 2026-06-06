"""
scripts/build_parallel_corpus.py - Construit un corpus parallèle Mina→Français
==========================================================================

Utilise Ollama pour générer des traductions françaises du corpus Mina.
Gère les erreurs et reprend si le processus est interrompu.

Usage:
    python scripts/build_parallel_corpus.py --test           # Test 20 phrases
    python scripts/build_parallel_corpus.py --sample 500     # 500 phrases (~2h)
    python scripts/build_parallel_corpus.py --full           # Corpus complet (~12h)

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_FILE = PROJECT_ROOT / "data" / "corpus" / "mina_ewe_corpus.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "parallel_corpus.jsonl"
PROGRESS_FILE = PROJECT_ROOT / "logs" / "translation_progress.json"
ERRORS_FILE = PROJECT_ROOT / "logs" / "translation_errors.json"

OLLAMA_MODEL = "llama3.1:8b"  # Plus stable que deepseek-r1 pour traduction
TIMEOUT = 30


# =============================================================================
# TRANSLATION
# =============================================================================

def translate(mina_text: str) -> Optional[str]:
    """Traduit Mina vers Français avec Ollama"""

    prompt = f"Translate to French: {mina_text}"

    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, "--verbose"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        if result.returncode == 0:
            french = result.stdout.strip()

            # Nettoyer le blabla
            if "done thinking" in french:
                french = french.split("done thinking")[-1].strip()

            # Prendre la première ligne courte
            lines = [l.strip() for l in french.split('\n') if l.strip() and len(l.strip()) < 100]
            for line in lines:
                if line and not any(kw in line.lower() for kw in ['i will', 'the translation', 'this is', 'here is']):
                    return line

            return french if len(french) < 100 else lines[0] if lines else french[:100]

    except subprocess.TimeoutExpired:
        pass
    except UnicodeDecodeError:
        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=prompt,
                capture_output=True,
                timeout=TIMEOUT,
            )
            if result.returncode == 0:
                return result.stdout.decode('utf-8', errors='ignore')[:100].strip()
        except:
            pass
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    return None


def translate_batch(texts: list[str], progress: bool = True) -> list[tuple[str, Optional[str]]]:
    """Traduit un lot de textes"""
    results = []

    for i, text in enumerate(texts):
        french = translate(text)

        if progress and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(texts)}: {text[:30]} → {french[:30] if french else 'FAIL'}...")

        results.append((text, french))

        # Pause pour éviter de surcharger Ollama
        if i > 0 and i % 20 == 0:
            time.sleep(1)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test 20 phrases")
    parser.add_argument("--sample", type=int, help="Nombre de phrases")
    parser.add_argument("--full", action="store_true", help="Corpus complet")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILDING PARALLEL CORPUS MINA -> FRANCAIS")
    print("=" * 60)

    # Charger le corpus Mina
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Corpus Mina: {len(corpus)} phrases")
    print(f"Modèle Ollama: {OLLAMA_MODEL}")
    print()

    # Déterminer le nombre de phrases
    if args.test:
        corpus = corpus[:20]
        print("MODE TEST: 20 phrases")
    elif args.sample:
        corpus = corpus[:args.sample]
        print(f"MODE SAMPLE: {args.sample} phrases")
    else:
        print(f"MODE FULL: {len(corpus)} phrases")

    # Charger la progression existante
    start_idx = 0
    translations = []

    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                start_idx = data.get("processed", 0)
                translations = data.get("translations", [])
            print(f"Reprise depuis: phrase {start_idx}")
        except:
            pass

    print()

    # Traiter
    start_time = time.time()
    total = len(corpus)
    errors = []

    for i in range(start_idx, total):
        mina = corpus[i]["mina"]

        # Traduire
        french = translate(mina)

        # Enregistrer
        entry = {
            "mina": mina,
            "french": french,
            "audio": corpus[i].get("audio", ""),
            "timestamp": datetime.now().isoformat()
        }
        translations.append(entry)

        if not french:
            errors.append({"index": i, "mina": mina})
            print(f"  ✗ [{i+1}/{total}] {mina[:40]}... → ERREUR")
        else:
            print(f"  ✓ [{i+1}/{total}] {mina[:40]}... → {french[:40]}...")

        # Sauvegarder régulièrement
        if (i + 1) % 50 == 0:
            # Sauvegarder le JSONL
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for entry in translations:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Sauvegarder la progression
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({"processed": i + 1, "translations": translations[:100]}, f)

            # Temps restant
            elapsed = time.time() - start_time
            speed = (i + 1 - start_idx) / elapsed
            remaining = (total - i - 1) / speed if speed > 0 else 0
            print(f"\n  ↳ Sauvegarde: {i+1} phrases | ETA: {remaining/60:.1f} min")
            print()

    # Sauvegarde finale
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in translations:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if errors:
        with open(ERRORS_FILE, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False)

    elapsed = time.time() - start_time
    success = sum(1 for t in translations if t["french"])

    print()
    print("=" * 60)
    print("TERMINE!")
    print("=" * 60)
    print(f"Total: {len(translations)} phrases")
    print(f"Réussies: {success} ({100*success/len(translations):.1f}%)")
    print(f"Erreurs: {len(errors)}")
    print(f"Temps: {elapsed/60:.1f} minutes")
    print(f"Fichier: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()