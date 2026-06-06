"""
scripts/validate_corpus.py - Valide les traductions Mina-Français
================================================================

Interface terminal pour valider/corriger les traductions générées.

Usage:
    py scripts/validate_corpus.py                    # Mode interactif
    py scripts/validate_corpus.py --stats           # Affiche les stats

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import json
import sys
import io
from pathlib import Path
from typing import Optional

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_FILE = PROJECT_ROOT / "data" / "corpus" / "parallel_corpus.jsonl"
VALIDATED_FILE = PROJECT_ROOT / "data" / "corpus" / "parallel_corpus_validated.jsonl"


def load_corpus():
    """Charge le corpus des traductions"""
    corpus = []
    if CORPUS_FILE.exists():
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    corpus.append(json.loads(line.strip()))
                except:
                    pass
    return corpus


def load_validated():
    """Charge les phrases déjà validées"""
    validated = set()
    if VALIDATED_FILE.exists():
        with open(VALIDATED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    validated.add(entry.get("mina", ""))
                except:
                    pass
    return validated


def show_stats():
    """Affiche les statistiques du corpus"""
    corpus = load_corpus()
    validated = load_validated()

    total = len(corpus)
    valid = len([e for e in corpus if e.get("validated", False)])
    validated_count = len(validated)

    good = len([e for e in corpus if e.get("quality") == "good"])
    bad = len([e for e in corpus if e.get("quality") == "bad"])

    print("=" * 60)
    print("STATISTIQUES DU CORPUS")
    print("=" * 60)
    print(f"Total phrases: {total}")
    print(f"Validées: {validated_count}")
    print(f"Non validées: {total - validated_count}")
    print()
    print(f"Qualité:")
    print(f"  - Bonnes: {good}")
    print(f"  - Mauvaises: {bad}")
    print(f"  - Non évaluées: {total - good - bad}")
    print()


def interactive_validation():
    """Mode interactif de validation"""
    corpus = load_corpus()
    validated = load_validated()

    print("=" * 60)
    print("VALIDATION DU CORPUS MINA-FRANÇAIS")
    print("=" * 60)
    print(f"Total: {len(corpus)} phrases")
    print()
    print("Commandes:")
    print("  o - Bonne traduction (valide)")
    print("  c - Correction nécessaire")
    print("  s - Skip (passer)")
    print("  q - Quitter et sauvegarder")
    print("  r - Remplacer la traduction française")
    print()

    validated_entries = []
    skipped = 0

    for i, entry in enumerate(corpus):
        mina = entry.get("mina", "")
        french = entry.get("french", "")

        # Skip si déjà validé
        if mina in validated:
            skipped += 1
            continue

        print("-" * 60)
        print(f"[{i+1}/{len(corpus)}] ({skipped} skippés)")
        print(f"  Mina: {mina}")
        print(f"  FR:   {french}")
        print()

        while True:
            choice = input("> ").strip().lower()

            if choice == "o":
                # Valider tel quel
                entry["validated"] = True
                entry["quality"] = "good"
                validated_entries.append(entry)
                break

            elif choice == "c":
                # Marquer pour correction
                entry["validated"] = False
                entry["needs_review"] = True
                validated_entries.append(entry)
                break

            elif choice == "s":
                # Skip
                break

            elif choice == "r":
                # Remplacer la traduction
                new_fr = input("Nouvelle traduction FR: ").strip()
                if new_fr:
                    entry["french_original"] = french
                    entry["french"] = new_fr
                    entry["validated"] = True
                    entry["quality"] = "corrected"
                    validated_entries.append(entry)
                    print(f"  -> Remplacé: {new_fr}")
                break

            elif choice == "q":
                # Sauvegarder et quitter
                print("\nSauvegarde...")
                with open(VALIDATED_FILE, "a", encoding="utf-8") as f:
                    for e in validated_entries:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")

                print(f"{len(validated_entries)} entrées sauvegardées")
                return

            else:
                print("Commande non reconnue. Utilisez: o/c/s/q/r")

    # Fin de la liste
    print("\n" + "=" * 60)
    print("FIN DE LA VALIDATION")
    print("=" * 60)
    print(f"{len(validated_entries)} phrases validées")

    # Sauvegarder
    with open(VALIDATED_FILE, "a", encoding="utf-8") as f:
        for e in validated_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Fichier: {VALIDATED_FILE}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Valide les traductions Mina-Français")
    parser.add_argument("--stats", action="store_true", help="Affiche les statistiques")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        interactive_validation()


if __name__ == "__main__":
    main()