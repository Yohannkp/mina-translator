"""
Script pour peupler le corpus avec le vocabulaire Mina de base
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from data.corpus_manager import CorpusManager
from data.mina_vocabulary import get_corpus_data, get_audio_phrases
from loguru import logger


def populate_corpus():
    """Peuple le corpus avec le vocabulaire Mina"""
    manager = CorpusManager()

    # Obtenir les phrases de base
    corpus_data = get_corpus_data()

    print(f"\n{'='*60}")
    print("PEUPLEMENT DU CORPUS MINA")
    print(f"{'='*60}")
    print(f"Phrases a ajouter: {len(corpus_data)}")

    # Ajouter au corpus
    manager.add_manual_entries(corpus_data)

    # Afficher quelques exemples
    print("\nExemples ajoutes:")
    print("-" * 60)
    for item in corpus_data[:10]:
        print(f"  MINA: {item['text']:25} FR: {item['translation']}")
    if len(corpus_data) > 10:
        print(f"  ... et {len(corpus_data) - 10} autres phrases")

    # Afficher statistiques
    print("\n" + "-" * 60)
    manager.print_stats()

    # Exporter le corpus parallèle
    print("\n" + "-" * 60)
    print("Export du corpus parallele...")
    output = manager.export_for_translation()
    print(f"Exporte vers: {output}")

    return corpus_data


def add_audio_phrases_to_file():
    """Sauve les phrases pour audio dans un fichier"""
    phrases = get_audio_phrases()
    output_file = Path("data/corpus/audio_phrases.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Phrases pour enregistrement audio STT\n")
        f.write("# Choisir des phrases courtes et claires\n\n")
        for i, phrase in enumerate(phrases, 1):
            f.write(f"{i}. {phrase}\n")

    print(f"\nPhrases audio sauvegardees: {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PEUPLEMENT DU CORPUS MINA")
    print("=" * 60)

    # Peuplement
    populate_corpus()

    # Phrases audio
    add_audio_phrases_to_file()

    print("\n" + "=" * 60)
    print("CORPUS PRET POUR FINE-TUNING")
    print("=" * 60)