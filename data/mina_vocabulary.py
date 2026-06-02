"""
Vocabulaire Mina de base pour initialiser le corpus
Phrases essentielles pour la traduction FR-Mina
"""
from pathlib import Path

# Phrases de salutation et présentations
SALUTATIONS = [
    ("Mɛlɔ", "Bonjour"),
    ("Mɛlɔ fɛfɛ", "Bonjour (matin)"),
    ("Mɛlɔ tso", "Bonsoir"),
    ("Ntsɛ nyabi?", "Comment allez-vous?"),
    ("Ntsɛ nyabi fɛfɛ?", "Comment allez-vous ce matin?"),
    ("Nyabi mɔ", "Au revoir"),
    ("Mi sɔ na?", "Comment vous appelez-vous?"),
    ("Mɛ sɔ [nom]", "Je m'appelle [nom]"),
    ("Kofi wɔ nunana fefefia", "Kofi habite à Lomé"),
    ("Nyametsɛ", "Merci"),
    ("Nyametsɛ blakpo", "Merci beaucoup"),
]

# Réponses
REPONSES = [
    ("Mɛ nyabi fɛfɛ", "Je vais bien"),
    ("Mɛ nyabi manɔmanɔ", "Je vais très bien"),
    ("Ao", "Non"),
    ("Ai", "Oui"),
    ("Mɛ sɔ nyabi", "Ça va"),
]

# Numéros et quantités
NUMEROS = [
    ("kɔ", "un"),
    ("ɛgbɛ", "deux"),
    ("ɛta", "trois"),
    ("ɛnɛ", "quatre"),
    ("aanɔ", "cinq"),
    ("ado", "six"),
    ("ayɛ", "sept"),
    ("hana", "huit"),
    ("Hetɔ", "neuf"),
    ("gɔ", "dix"),
    ("awu", "vingt"),
    ("agbɛ", "cent"),
    ("akɔ", "mille"),
]

# Jours de la semaine
JOURS = [
    ("Fiafla", "Jour"),
    ("Dedienɔ", "Lundi"),
    ("Edzima", "Mardi"),
    ("Dzemɛ", "Mercredi"),
    ("Dzioʋa", "Jeudi"),
    ("Fiafla", "Vendredi"),
    ("Memleƒe", "Samedi"),
    ("Nidie", "Dimanche"),
]

# mois de l'année
MOIS = [
    ("Dzove", "Janvier"),
    ("Dzododo", "Février"),
    ("Tedoxe", "Mars"),
    ("Dame", "Avril"),
    ("Mɔ", "Mai"),
    ("Mɔasi", "Juin"),
    ("Mɔsi", "Juillet"),
    ("Duda", "Août"),
    ("Anyɔli", "Septembre"),
    ("Kele", "Octobre"),
    ("Damien", "Novembre"),
    ("Dzanɔ", "Décembre"),
]

# Corps humain
CORPS = [
    ("tɔnu", "tête"),
    ("xɔ", "œil/yeux"),
    ("dzise", "nez"),
    ("nu", "bouche"),
    ("kpakpa", "oreille"),
    ("se", "main"),
    ("sra", "pied/jambe"),
    ("fe", "cœur"),
]

# Famille
FAMILLE = [
    ("Fɔfɔ", "Père"),
    ("Na", "Mère"),
    ("Dada", "Grand-mère"),
    ("Fia", "Roi"),
    ("Nyɔnu", "Femme"),
    ("Nyitsu", "Homme/Garçon"),
    ("Mɔ", "Enfant"),
    ("Nɔƒe", "Maison/Famille"),
    ("Nuwoe", "Frères/Sœurs"),
]

# Nature
NATURE = [
    ("Tso", "Soleil"),
    ("Sifa", "Lune"),
    ("Fifia", "Étoile"),
    ("Duke", "Eau"),
    ("Xɔli", "Terre/Sol"),
    ("Kɔli", "Forêt/Arbre"),
    ("Nyɔnu", "Femme"),
    ("Nyitsu", "Homme"),
    ("Nyametsi", "Animal"),
    ("Dzo", "Feu"),
    ("Fufu", "Vent"),
]

# Nourriture et boissons
NOURRITURE = [
    ("Gbetato", "Boire"),
    ("Dzakpɔ", "Manger"),
    ("Fufu", "Fufu (plat traditionnel)"),
    ("Akpɔ", "Riz"),
    ("Dɔli", "Mil"),
    ("Aliha", "Maïs"),
    ("Akɔkɔ", "Piment"),
    ("Tsii", "Sel"),
    ("Ayikpɔ", "Viande"),
    ("Ago", "Poisson"),
    ("Dzi", "Sauce"),
    ("Nunu", "Lait"),
]

# Verbes courants
VERBES = [
    ("Azɔ", "Travailler"),
    ("Xlɛ", "Apprendre/Étudier"),
    ("Dzi", "Voir/Regarder"),
    ("De", "Prendre/Porter"),
    ("Na", "Donner"),
    ("Klɛ", "Écrire"),
    ("Kɔ", "Aller"),
    ("Ba", "Venir"),
    ("Lɔ", "Quitter/Partir"),
    ("Dzɔ", "Être/Situé"),
    ("Yra", "Savoir/Connaître"),
    ("Gblɔ", "Dire/Parler"),
    ("Kpee", "Entendre"),
    ("Dzi nɔ", "Écouter"),
    ("Gbo", "Sentir"),
    ("Kpo", "Toucher"),
    ("Wo", "Soulever/Porter"),
    ("Ka", "Acheter"),
    ("Dɔ", "Vendre"),
]

# Adjectifs courants
ADJECTIFS = [
    ("fɛfɛ", "nouveau/récent"),
    ("kɔ", "grand/élevé"),
    ("dze", "petit"),
    ("sɔ", "beau/joli"),
    ("sra", "mauvais"),
    ("nyɔnyɔ", "bon/acceptable"),
    ("ka", "long/durable"),
    ("dzi", "chaud"),
    ("gbɔ", "froid"),
    ("kplɔ", "noir"),
    ("dzi", "rouge"),
    ("dziwɔ", "blanc"),
    ("dzaka", "jaune"),
    ("blave", "vert"),
]

# Phrases utilitaires
PHRASES = [
    ("Mi sɛ na dɔ?", "Voulez-vous vendre?"),
    ("Mɛ sɛ na ka?", "Je veux acheter"),
    ("Nyɔnu la sɛ na nyametsi", "La femme veut de la viande"),
    ("Xɔle la dɔnyi", "Le prix est élevé"),
    ("Fiafla la ko", "Le jour se lève"),
    ("Tso la ku", "Le soleil se couche"),
    ("Duke la fifi", "L'eau coule"),
    ("Mɔ la dzi nɔ", "L'enfant écoute"),
    ("Fɔfɔ la azɔ", "Le père travaille"),
    ("Na la xlɛ nu", "La mère prépare la nourriture"),
    ("Nɔƒe la kpli", "La famille se rassemble"),
    ("Kpli la fia", "La réunion est chez le chef"),
]

# Lieux
LIEUX = [
    ("Fefia", "Lomé (capitale)"),
    ("Kpalimɛ", "Kpalimé"),
    ("Sokodɛ", "Sokodé"),
    ("Kara", "Kara"),
    ("Tsɛvi", "Cové/Tsévié"),
    ("Baflo", "Baflo"),
    ("Dapa", "Dapa"),
    ("Kumasi", "Kumasi (Ghana)"),
    ("Xɔli", "Togo"),
    ("Kɛli", "Côte ( rivage)"),
    ("Nɔƒe", "Maison"),
    ("Xɔxɔ", "Marché"),
    ("Kplikpli", "Église"),
    ("Fufuwo", "Pharmacie"),
    ("Dzimɛ", "Hôpital"),
]

# Tout regrouper
TOUTES_LES_PHRASES = [
    # Salutations
    ("Mɛlɔ", "Bonjour"),
    ("Mɛlɔ fɛfɛ", "Bonjour (le matin)"),
    ("Mɛlɔ tso", "Bonsoir"),
    ("Ntsɛ nyabi?", "Comment allez-vous?"),
    ("Mɛ nyabi fɛfɛ", "Je vais bien"),
    ("Nyabi mɔ", "Au revoir"),
    ("Aflɔ mi nyabi?", "Au plaisir de vous revoir"),
    ("Nyametsɛ", "Merci"),
    ("Nyametsɛ blakpo", "Merci beaucoup"),
    ("Mɛ gblɔ nu na wɔ", "Je vous en prie (de rien)"),
    ("Sɔ sesẽ", "Pardon/Excusez-moi"),
    ("Mi sɔ na?", "Comment vous appelez-vous?"),
    ("Mɛ sɔ [nom]", "Je m'appelle [nom]"),

    # Basiques
    ("Ai", "Oui"),
    ("Ao", "Non"),
    ("Kɔ si?", "Où?"),
    ("Nka si?", "Quand?"),
    ("Nɛ si?", "Comment?"),
    ("Nka sɛ?", "Pourquoi?"),

    # Corps
    ("tɔnu", "tête"),
    ("xɔ", "œil/yeux"),
    ("dzise", "nez"),
    ("nu", "bouche"),
    ("kpakpa", "oreille"),
    ("se", "main"),
    ("sra", "pied/jambe"),
    ("fe", "cœur"),

    # Nombres
    ("kɔ", "un"),
    ("ɛgbɛ", "deux"),
    ("ɛta", "trois"),
    ("ɛnɛ", "quatre"),
    ("aanɔ", "cinq"),
    ("ado", "six"),
    ("ayɛ", "sept"),
    ("hana", "huit"),
    ("Hetɔ", "neuf"),
    ("gɔ", "dix"),

    # Familles
    ("Fɔfɔ", "Père"),
    ("Na", "Mère"),
    ("Dada", "Grand-mère"),
    ("Nyɔnu", "Femme"),
    ("Nyitsu", "Homme"),
    ("Mɔ", "Enfant"),
    ("Nɔƒe", "Maison/Famille"),

    # Nourriture
    ("Fufu", "Fufu"),
    ("Akpɔ", "Riz"),
    ("Dɔli", "Mil"),
    ("Ayikpɔ", "Viande"),
    ("Ago", "Poisson"),
    ("Dzi", "Sauce"),
    ("Duke", "Eau"),

    # Lieux
    ("Fefia", "Lomé"),
    ("Kpalimɛ", "Kpalimé"),
    ("Sokodɛ", "Sokodé"),
    ("Kara", "Kara"),
    ("Tsɛvi", "Tsévié"),

    # Verbes
    ("Azɔ", "Travailler"),
    ("Xlɛ", "Apprendre"),
    ("Dzi", "Voir"),
    ("Kɔ", "Aller"),
    ("Ba", "Venir"),
    ("Gblɔ", "Parler"),
    ("Klɛ", "Écrire"),
    ("Ka", "Acheter"),
    ("Dɔ", "Vendre"),

    # Phrases courantes
    ("Mɛ kɔ xɔxɔ", "Je vais au marché"),
    ("Xɔxɔ la fifi", "Le marché est loin"),
    ("Duke la dzɔ", "L'eau est prête"),
    ("Tso la ku", "Le soleil se couche"),
    ("Fiafla la fifi", "La journée est terminée"),
    ("Mɛ nyametsi", "J'ai faim"),
    ("Mɛ dzɔ", "J'ai soif"),
    ("Nɔƒe la dzɔ", "La maison est là"),
]


def get_corpus_data():
    """Retourne toutes les phrases sous forme de liste de dictionnaires"""
    corpus = []
    for mina, francais in TOUTES_LES_PHRASES:
        corpus.append({
            "text": mina,
            "lang": "mina",
            "translation": francais,
            "source": "vocabulaire_base",
        })
    return corpus


def get_audio_phrases():
    """Phrases optimisées pour enregistrement audio (courtes)"""
    return [
        "Mɛlɔ",
        "Ntsɛ nyabi?",
        "Nyametsɛ",
        "Ai",
        "Ao",
        "Duke",
        "Fufu",
        "Kɔ",
        "Ba",
    ]


if __name__ == "__main__":
    corpus = get_corpus_data()
    print(f"Corpus de base: {len(corpus)} phrases Mina-Francais")
    print("\nExemples:")
    for item in corpus[:5]:
        print(f"  Mina: {item['text']:20} FR: {item['translation']}")