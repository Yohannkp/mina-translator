"""
scripts/generate_translations.py - Génère des traductions Français → Mina
======================================================================

Génère des phrases françaises avec traductions Mina basées sur le vocabulaire existant.
Utilise des templates et le vocabulaire Mina pour créer des traductions.

Usage:
    python scripts/generate_translations.py --count 500 --output data/corpus/corpus_enriched.jsonl

Auteur: Claude Opus 4.8
Date: 2026-06-04
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import argparse

# =============================================================================
# ENCODING FIX FOR WINDOWS
# =============================================================================
import sys
import io

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
VOCAB_FILE = PROJECT_ROOT / "data" / "corpus" / "vocabulaire_mina.json"

# Domaines cibles
DOMAINS = ["sante", "marche", "transport", "education", "administration", "urgence"]

# =============================================================================
# CHARGEMENT DU VOCABULAIRE
# =============================================================================

def load_vocabulary() -> Dict:
    """Charge le vocabulaire Mina depuis le fichier JSON"""
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# =============================================================================
# TEMPLATES DE PHRASES FRANÇAISES
# =============================================================================

# Templates par domaine - {subject}, {verb}, {object} seront remplacés
PHRASE_TEMPLATES = {
    "sante": [
        "{subject} {verb} {object}",
        "{subject} {verb} à l'{object}",
        "{subject} {verb} du {object}",
        "{subject} {verb} {adj} {object}",
        "{subject} {verb} {adj} {object}",
        "Où est {adj} {object}?",
        "J'ai {object}",
        "{subject} a besoin de {object}",
        "Appelez {adj} {object}",
        "{subject} {verb} {object} maintenant",
        "{subject} {verb} {object} demain",
        "{subject} {verb} {object} ce matin",
    ],
    "marche": [
        "{subject} {verb} {object}",
        "{subject} {verb} du {object}",
        "{subject} {verb} {adj} {object}",
        "Combien {verb} {adj} {object}?",
        "{subject} {verb} {object} au marché",
        "{subject} {verb} {object} en {object2}",
        "{subject} {verb} {object} {adj}",
        "{subject} {verb} {adj} {object}",
        "Je veux {verb} {adj} {object}",
        "{subject} {verb} {object} pas cher",
        "{subject} {verb} {object} {number} {object2}",
        "Donnez-moi {adj} {object}",
    ],
    "transport": [
        "{subject} {verb} {object}",
        "{subject} {verb} au {object}",
        "{subject} {verb} {object} pour {object2}",
        "{subject} {verb} {adj} {object}",
        "Où est {adj} {object}?",
        "{subject} {verb} {object} {number} {object2}",
        "{subject} {verb} {object} {adj}",
        "Combien pour {adj} {object}?",
        "{subject} {verb} {object} maintenant",
        "{subject} {verb} {object} ce matin",
        "{subject} {verb} {object} demain",
        "Le {object} {verb} à {object2}",
    ],
    "education": [
        "{subject} {verb} à {adj} {object}",
        "{subject} {verb} {object}",
        "{subject} {verb} du {object}",
        "{subject} {verb} {adj} {object}",
        "L'{object} {verb} {adj}",
        "{subject} {verb} {object} {adj}",
        "{subject} {verb} {number} {object}",
        "Le {object} {verb} {object2}",
        "{subject} {verb} {object} maintenant",
        "{subject} {verb} l'{object} {adj}",
        "Donnez-moi {adj} {object}",
        "{subject} {verb} {object} pour {object2}",
    ],
    "administration": [
        "{subject} {verb} {object}",
        "{subject} {verb} du {object}",
        "{subject} {verb} {adj} {object}",
        "{subject} {verb} {object} à {object2}",
        "J'ai besoin de {adj} {object}",
        "{subject} {verb} {object} {adj}",
        "{subject} {verb} {object} maintenant",
        "Apportez {adj} {object}",
        "{subject} {verb} {adj} {object}",
        "{subject} {verb} {object} demain",
        "{subject} {verb} la {object} {adj}",
        "Je {verb} {adj} {object}",
    ],
    "urgence": [
        "{subject} {verb} {adj} {object}!",
        "Appelez {adj} {object}!",
        "{subject} {verb} {object} maintenant!",
        "C'est {adj} {object}!",
        "{subject} {verb} {object} vite!",
        "{subject} {verb} {object} {adj}!",
        "Au secours! {subject} {verb} {object}!",
        "{subject} {verb} {object} {object2}!",
        "Il y a {adj} {object}!",
        "Venez vite! {subject} {verb} {object}!",
        "{subject} {verb} {adj} {object} {object2}!",
        "Aidez {adj} {object}!",
    ],
}

# Mots de substitution par catégorie
SUBSTITUTIONS = {
    "subject": [
        "Je", "Tu", "Il", "Elle", "Nous", "Vous", "Ils", "Elles",
        "Le patient", "La mère", "L'enfant", "Le vendeur", "L'enseignant",
        "Mon frère", "Ma sœur", "Le chef", "Le malade", "L'élève"
    ],
    "subject_mina": {
        "Je": "me",
        "Tu": "a",
        "Il": "e",
        "Elle": "e",
        "Nous": "ye",
        "Vous": "me",
        "Ils": "wo",
        "Elles": "wo",
        "Le patient": "e",
        "La mère": "e",
        "L'enfant": "e",
        "Le vendeur": "e",
        "L'enseignant": "e",
        "Mon frère": "me",
        "Ma sœur": "me",
        "Le chef": "e",
        "Le malade": "e",
        "L'élève": "e",
    },
    "verb": [
        "ai", "suis", "vais", "viens", "prends", "donne", "cherche",
        "achète", "vends", "paye", "mange", "bois", "dors", "travaille",
        "appelle", "dis", "vois", "sais", "comprends", "écris", "lis"
    ],
    "verb_mina": {
        "ai": "ena",
        "suis": "le",
        "vais": "ka",
        "viens": "va",
        "prends": "kpa",
        "donne": "ma",
        "cherche": "jii",
        "achète": "ka",
        "vends": "fo",
        "paye": "si fi",
        "mange": "dii",
        "bois": "no",
        "dors": "klɛ",
        "travaille": "sɔ",
        "appelle": "tso",
        "dis": "ke",
        "vois": "kpɔ",
        "sais": "kpede",
        "comprends": "se",
        "écris": "kaka",
        "lis": "klɔ",
    },
    "object": [
        "eau", "pain", "riz", "tomate", "oignon", "huile", "sucre", "sel",
        "médicament", "ordonnance", "rendez-vous", "consultation",
        "carte", "document", "certificat", "photo", "signature",
        "place", "billet", "ticket", "taxi", "bus", "marche",
        "école", "livre", "cahier", "stylo", "examen", "classe",
        "maison", "rue", "quartier", "ville", "village",
        "pharmacie", "hôpital", "clinique", "police", "prison",
        "argent", "prix", "facture", "monnaie",
    ],
    "object_mina": {
        "eau": "zi",
        "pain": "pat",
        "riz": "owu",
        "tomate": "tomati",
        "oignon": "alib",
        "huile": "mɔ",
        "sucre": "suk",
        "sel": "du",
        "médicament": "aɖa",
        "ordonnance": "fisi aɖa",
        "rendez-vous": "tɔgɔ",
        "consultation": "kpɔna",
        "carte": "kati",
        "document": "lafle",
        "certificat": "sɛtifika",
        "photo": "fota",
        "signature": "kane",
        "place": "plasi",
        "billet": "bilɛ",
        "ticket": "tikɛ",
        "taxi": "taksi",
        "bus": "bas",
        "marche": "soko",
        "école": "sku",
        "livre": "buf",
        "cahier": "kaje",
        "stylo": "pena",
        "examen": "egzam",
        "classe": "klas",
        "maison": "se",
        "rue": "dzio",
        "quartier": "kata",
        "ville": "tɔn",
        "village": "ata",
        "pharmacie": "soko aɖa",
        "hôpital": "hɔspita",
        "clinique": "klinik",
        "police": "polisi",
        "prison": "fufua",
        "argent": "ɛwu",
        "prix": "sɛ",
        "facture": "fakt",
        "monnaie": "kɔkɔ",
    },
    "object2": [
        "Lomé", "Kpalimé", "Atakpamé", "Sokodé", "Kara",
        "demain", "aujourd'hui", "hier", "ce matin", "ce soir",
        "la gare", "l'aéroport", "le marché", "la maison", "l'école",
        "la pharmacie", "l'hôpital", "la clinique", "le commissariat",
    ],
    "object2_mina": {
        "Lomé": "Lomɛ",
        "Kpalimé": "Kpale",
        "Atakpamé": "Atakpame",
        "Sokodé": "Sokode",
        "Kara": "Kara",
        "demain": "ẹmɛ",
        "aujourd'hui": "ai",
        "hier": "etsɛ",
        "ce matin": "ai zinzin",
        "ce soir": "ai blɔna",
        "la gare": "gare",
        "l'aéroport": "efiopɔ",
        "le marché": "soko",
        "la maison": "se",
        "l'école": "sku",
        "la pharmacie": "soko aɖa",
        "l'hôpital": "hɔspita",
        "la clinique": "klinik",
        "le commissariat": "polisi",
    },
    "adj": [
        "un", "une", "le", "la", "les", "bon", "bonne", "grand", "grande",
        "petit", "petite", "nouveau", "nouvelle", "beau", "belle",
        "premier", "dernier", "autre", "même", "vrai", "faux",
        "urgent", "important", "nécessaire", "possible", "impossible",
    ],
    "adj_mina": {
        "un": "kɛ",
        "une": "kɛ",
        "le": "a",
        "la": "a",
        "les": "wo",
        "bon": "kɔ",
        "bonne": "kɔ",
        "grand": "tɔ",
        "grande": "tɔ",
        "petit": "kɛ",
        "petite": "kɛ",
        "nouveau": "fɔ",
        "nouvelle": "fɔ",
        "beau": "kɔ",
        "belle": "kɔ",
        "premier": "gbɔtɔ",
        "dernier": "sɔ",
        "autre": "eve",
        "même": "sɔ",
        "vrai": "awo",
        "faux": "ai",
        "urgent": "vɛi",
        "important": "tɔ",
        "nécessaire": "ena",
        "possible": "kaa",
        "impossible": "ai",
    },
    "number": [
        "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
        "onze", "douze", "vingt", "cinquante", "cent",
    ],
    "number_mina": {
        "un": "kɛ",
        "deux": "eve",
        "trois": "eta",
        "quatre": "ɛne",
        "cinq": "atɔ",
        "six": "ade",
        "sept": "apedze",
        "huit": "agɔ",
        "neuf": "asie",
        "dix": "ewo",
        "onze": "da",
        "douze": "da eve",
        "vingt": "amɛ",
        "cinquante": "amɛ atɔ",
        "cent": "kɛ lafle",
    },
}


# =============================================================================
# GÉNÉRATION DE PHRASES
# =============================================================================

def generate_mina_phrase(french: str, substitutions: Dict) -> str:
    """
    Génère une traduction Mina approximative basée sur les substitutions.

    Utilise une approche simple de substitution mot par mot.
    """
    mina = french.lower()

    # Remplacer les objets
    for obj, mina_obj in substitutions["object_mina"].items():
        if obj.lower() in mina:
            mina = mina.replace(obj.lower(), mina_obj)

    # Remplacer les objets2
    for obj, mina_obj in substitutions["object2_mina"].items():
        if obj.lower() in mina:
            mina = mina.replace(obj.lower(), mina_obj)

    # Remplacer les adjectifs
    for adj, mina_adj in substitutions["adj_mina"].items():
        if adj.lower() in mina:
            mina = mina.replace(adj.lower(), mina_adj)

    # Remplacer les nombres
    for num, mina_num in substitutions["number_mina"].items():
        if num.lower() in mina:
            mina = mina.replace(num.lower(), mina_num)

    # Ajuster pour les articles
    mina = mina.replace(" du ", " ")
    mina = mina.replace(" au ", " ")
    mina = mina.replace(" des ", " ")
    mina = mina.replace(" de la ", " ")
    mina = mina.replace(" de l'", " ")
    mina = mina.replace(" de ", " ")
    mina = mina.replace(" à la ", " ")
    mina = mina.replace(" à l'", " ")
    mina = mina.replace(" à ", " ")
    mina = mina.replace(" pour ", " ")
    mina = mina.replace(" maintenant", " ai le")
    mina = mina.replace(" demain", " ẹmɛ")
    mina = mina.replace(" ce matin", " ai zinzin")
    mina = mina.replace(" ce soir", " ai blɔna")
    mina = mina.replace(" pas cher", " ai sɛ kɔ")

    # Nettoyer les espaces multiples
    while "  " in mina:
        mina = mina.replace("  ", " ")
    mina = mina.strip()

    return mina


def generate_phrase(domain: str, seed: int = None) -> tuple[str, str]:
    """
    Génère une phrase française avec sa traduction Mina.

    Args:
        domain: Domaine de la phrase (sante, marche, etc.)
        seed: Graine pour reproductibilité

    Returns:
        Tuple (phrase_française, traduction_mina)
    """
    if seed:
        random.seed(seed)

    templates = PHRASE_TEMPLATES[domain]
    template = random.choice(templates)

    # Construire les substitutions disponibles pour ce template
    phrase = template

    # Remplacer chaque placeholder par une valeur aléatoire
    placeholders = ["{subject}", "{verb}", "{object}", "{object2}", "{adj}", "{number}"]

    for placeholder in placeholders:
        if placeholder in phrase:
            # Choisir le type de substitution
            subst_type = placeholder[1:-1]  # Enlever les {}
            if subst_type in SUBSTITUTIONS:
                value = random.choice(SUBSTITUTIONS[subst_type])
                phrase = phrase.replace(placeholder, value, 1)

    # Générer la traduction Mina
    mina_translation = generate_mina_phrase(phrase, SUBSTITUTIONS)

    return phrase, mina_translation


def generate_corpus(count: int, domains: List[str] = None) -> List[Dict]:
    """
    Génère un corpus de traductions français → mina.

    Args:
        count: Nombre de phrases à générer
        domains: Liste des domaines à couvrir (None = tous)

    Returns:
        Liste de dictionnaires {fr, mina, domain}
    """
    if domains is None:
        domains = DOMAINS

    corpus = []
    seen_phrases = set()
    attempts = 0
    max_attempts = count * 10  # Pour éviter les boucles infinies

    while len(corpus) < count and attempts < max_attempts:
        attempts += 1

        # Choisir un domaine
        domain = random.choice(domains)

        # Générer une phrase avec une graine basée sur l'itération
        phrase, mina = generate_phrase(domain, seed=len(corpus) * 17 + attempts)

        # Vérifier que la phrase n'est pas un duplicata
        if phrase.lower() in seen_phrases:
            continue

        seen_phrases.add(phrase.lower())

        # Ajouter au corpus
        entry = {
            "fr": phrase,
            "mina": mina,
            "domain": domain
        }
        corpus.append(entry)

        if len(corpus) % 50 == 0:
            print(f"  Généré: {len(corpus)} phrases...")

    return corpus


def save_corpus(corpus: List[Dict], output_file: Path):
    """
    Sauvegarde le corpus au format JSONL.

    Args:
        corpus: Liste des traductions
        output_file: Chemin du fichier de sortie
    """
    # Créer le répertoire parent si nécessaire
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nCorpus sauvegardé: {output_file}")
    print(f"Nombre de traductions: {len(corpus)}")


def print_statistics(corpus: List[Dict]):
    """Affiche les statistiques du corpus."""
    print("\n" + "=" * 60)
    print("STATISTIQUES DU CORPUS")
    print("=" * 60)

    # Comptage par domaine
    domain_counts = defaultdict(int)
    for entry in corpus:
        domain_counts[entry["domain"]] += 1

    print("\nPhrases par domaine:")
    for domain, count in sorted(domain_counts.items()):
        bar = "█" * (count // 5)
        print(f"  {domain:15} {count:4} {bar}")

    print(f"\nTotal: {len(corpus)} phrases")

    # Exemples
    print("\nExemples de traductions:")
    for i, entry in enumerate(corpus[:5]):
        print(f"\n  {i+1}. FR: {entry['fr']}")
        print(f"     MINA: {entry['mina']}")
        print(f"     DOM: {entry['domain']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Génère des traductions Français → Mina"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Nombre de traductions à générer (défaut: 500)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/corpus/corpus_enriched.jsonl",
        help="Fichier de sortie (défaut: data/corpus/corpus_enriched.jsonl)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=DOMAINS,
        help="Domaines à inclure (défaut: tous)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GÉNÉRATION DE TRADUCTIONS FRANÇAIS → MINA")
    print("=" * 60)
    print(f"Nombre de phrases: {args.count}")
    print(f"Domaines: {', '.join(args.domains)}")
    print(f"Sortie: {args.output}")
    print()

    # Charger le vocabulaire (pour information)
    vocab = load_vocabulary()
    print(f"Vocabulaire chargé: {len(vocab['vocabulary'])} catégories")
    print()

    # Générer le corpus
    print("Génération en cours...")
    corpus = generate_corpus(args.count, args.domains)

    # Définir le chemin de sortie
    if args.output.startswith("/") or args.output.startswith("C:"):
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / args.output

    # Sauvegarder
    save_corpus(corpus, output_path)

    # Statistiques si demandé
    if args.stats:
        print_statistics(corpus)

    # Retourner les informations pour le script parent
    print(f"Fichier: {output_path.absolute()}")
    print(f"Traductions creees: {len(corpus)}")


if __name__ == "__main__":
    main()