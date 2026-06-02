"""
Plan d'enrichissement du corpus Mina-Français
Objectif: Passer de 500 à 5000 phrases (10 domaines x 500 phrases)
"""

# =============================================================================
# 10 DOMAINES PRIORITAIRES POUR L'INCLUSION NUMÉRIQUE
# =============================================================================

DOMAINES_PRIORITAIRES = {
    # Domaine 1: Santé (0-499)
    "sante": {
        "priorite": 1,
        "description": "Communications sanitaires, hôpitaux, pharmacies, maladies",
        "cible": 500,
        "sous_themes": [
            "symptomes", "medicaments", "rendez-vous", "urgences",
            "grossesse", "enfants", "prévention", "hygiène"
        ]
    },

    # Domaine 2: Justice (500-999)
    "justice": {
        "priorite": 2,
        "description": "Affaires judiciaires, police, tribunal, droits",
        "cible": 500,
        "sous_themes": [
            "plainte", "police", "tribunal", "avocat", "prison",
            "droits_civiques", "mariage_légal", "succession"
        ]
    },

    # Domaine 3: Marché/Commerce (1000-1499)
    "marche": {
        "priorite": 3,
        "description": "Transactions commerciales, marchés, boutique",
        "cible": 500,
        "sous_themes": [
            "prix", "marchandage", "paiement", "négoce",
            "produits", "poids_mesures", "devises", "crédit"
        ]
    },

    # Domaine 4: Administration (1500-1999)
    "administration": {
        "priorite": 4,
        "description": "Services administratifs, documents, procédures",
        "cible": 500,
        "sous_themes": [
            "carte_identite", "passeport", "acte_naissance", "permis",
            "impots", "mairies", "préfectures", "guichets"
        ]
    },

    # Domaine 5: Éducation (2000-2499)
    "education": {
        "priorite": 5,
        "description": "École, university, formation professionnelle",
        "cible": 500,
        "sous_themes": [
            "inscription", "examens", "diplômes", "cours",
            "parents", "école_primaire", "collège_lycee", "universite"
        ]
    },

    # Domaine 6: Transport (2500-2999)
    "transport": {
        "priorite": 6,
        "description": "Déplacements, taxi, bus, routes",
        "cible": 500,
        "sous_themes": [
            "taxi", "bus", "zemia", "car", "destination",
            "horaires", "ticket", "bagages", "accident"
        ]
    },

    # Domaine 7: Agriculture (3000-3499)
    "agriculture": {
        "priorite": 7,
        "description": "Agriculture, elevage, pêche",
        "cible": 500,
        "sous_themes": [
            "champs", "récolte", "bétail", "pêche",
            "marchandises", "engrais", "saison", "marché_agricole"
        ]
    },

    # Domaine 8: Religion/Culture (3500-3999)
    "religion_culture": {
        "priorite": 8,
        "description": "Traditions, religions, célébrations",
        "cible": 500,
        "sous_themes": [
            "eglise", "prière", "bapteme", "mariage_tradition",
            "funérailles", "fêtes_traditionnelles", "proverbes"
        ]
    },

    # Domaine 9: Technologie/Communication (4000-4499)
    "technologie": {
        "priorite": 9,
        "description": "Téléphone, internet, services numériques",
        "cible": 500,
        "sous_themes": [
            "téléphone", "argent_mobile", "Orange", "T-Money",
            "internet", "réseaux_sociaux", "whatsapp", "appels"
        ]
    },

    # Domaine 10: Urgence/Secours (4500-4999)
    "urgence": {
        "priorite": 10,
        "description": "Situations d'urgence, sécurité, catastrophes",
        "cible": 500,
        "sous_themes": [
            "ambulance", "pompiers", "police_secours",
            "inondation", "accident_routier", "vol", "incendie"
        ]
    }
}


# =============================================================================
# TERMES ADMINISTRATIFS TOGOLAIS
# =============================================================================

TERMES_ADMINISTRATIFS = {
    # Documents officiels
    "cni": ("Carte Nationale d'Identité", "Carte d'Identité Nationale"),
    "extrait_naissance": ("Extrait de naissance", "Copie d'acte de naissance"),
    "acte_mariage": ("Acte de mariage", "Certificat de mariage"),
    "certificat_scolarite": ("Certificat de scolarité", "Attestation scolaire"),
    "attestation_residence": ("Attestation de résidence", "Certificat de domicile"),
    "passeport": ("Passeport", "Laissez-passer"),
    "permis_conduire": ("Permis de conduire", "Licence de conduite"),
    "carte_sejour": ("Carte de séjour", "Permis de résidence"),

    # Lieux administratifs
    "prefecture": ("Préfecture", "Sous-préfecture"),
    "mairie": ("Mairie", "Hôtel de ville"),
    "commissariat": ("Commissariat de police", "Gendarmerie"),
    "greffe": ("Greffe du tribunal", "Tribunal"),
    "parquet": ("Parquet", "Procureur"),
    "cosef": ("COSEF", "Centre des œuvres universitaires"),

    # Procédures
    "depot_plainte": ("Déposer une plainte", "Porter plainte"),
    "timbre": ("Timbre fiscal", "Droit de timbre"),
    "droit": ("Droit administratif", "Taxe"),
    "delai": ("Délai de traitement", "Temps d'attente"),
    "rendez_vous": ("Rendez-vous", "Appointment"),
}


# =============================================================================
# ARGOT DE LOMÉ (K起大 EWO)
# =============================================================================

ARGOT_LOME = {
    # Expressions courantes
    "kpɔ": "voir/comprendre (argot)",
    "dzrɔ": "avoir peur",
    "flɛ": "sortir/partir",
    "gba": "donner/ offrir",
    "kaka": "trop/beaucoup",
    "kwi": "travailler",
    "fla": "manger",
    "dzɔ": "être bien/juste",
    "vɛ": "non (familier)",
    "ʋiʋi": "mensonge",
    "fufu": "chose/combat (figuré)",
    "kpalakpalada": "confusion/chaos",
    "kplɔ": "aider/accompagner",
    "tsi": "dire/parler",
    "wɔ": "avoir/posséder",
    "gbɔ": "courir/s'enfuir",
    "mɛlɔ": "salut (réduit)",
    "yovo": "étranger/occidental",
    "togolais": "personne du pays",

    # Phrases argot
    "kpɔ dzi": "comprendre (bien saisir)",
    "wo kpɔa": "tu comprends",
    "mɛ flɛ": "je pars/m'en vais",
    "gba nyɔ": "donne-moi ça",
    "kaka kaka": "trop trop (exagération)",
    "kwi mɔ": "travaille bien",
    "fla mɔ": "mange bien",
    "vɛ vɛ": "non non (renforcement)",
    "kpɔ sesẽ": "fais attention",
    "gba mɔ": "sois brave",
    "kpalakpalada la wɔ": "c'est le chaos",
    "yovo la ba": "l'occidental arrive",
}


# =============================================================================
# VOCABULAIRE SPÉCIFIQUE PAR DOMAINE
# =============================================================================

VOCABULAIRE_SANTE = [
    ("fufuwo", "pharmacie"),
    ("dzimɛ", "hôpital"),
    ("nyametsi", "maladie"),
    ("dziɖe", "fièvre"),
    ("vɔnɔ", "douleur"),
    ("tsiɖe", "toux"),
    ("ɖu", "maux de tête"),
    ("gbe", "fatigue"),
    ("nuɖu", "mal de gorge"),
    ("xɔɖu", "mal aux yeux"),
    ("srɔ", "vomir"),
    ("dze", "diarrhée"),
    ("nyametsɔ", "guérison"),
    ("kplɔ", "médecin"),
    ("sɔɖa", "infirmier"),
    ("nyɔnu kplɔ", "docteur (femme)"),
    ("nyitsu kplɔ", "docteur (homme)"),
    ("nunya", "ordonnance"),
    ("akpɔ", "médicament"),
    ("kɔsi", "injection"),
    ("dziɖa", "pansement"),
    ("xɔxɔ", "pommade"),
    ("tableti", "comprimés"),
    ("siropi", "sirop"),
    ("ayikpɔ", "viande (nutrition)"),
    ("ago", "poisson"),
    ("aliha", "maïs"),
    ("fufu", "fufu (plat)"),
    ("akpɔ", "riz"),
    ("dukɛ", "eau"),
    ("nyɔnu", "femme"),
    ("mɔ", "enfant"),
    ("nya", "temps/occasion"),
    ("nyɔ", "chose/objet"),
    ("dzɔ", "situation"),
]

VOCABULAIRE_ADMINISTRATION = [
    ("tɔli", "carte"),
    ("pasepɔ", "passeport"),
    ("CNI", "Carte Nationale d'Identité"),
    ("dzɔli", "acte/document"),
    ("dzɔliaɖo", "certificat"),
    ("dzome", "nom"),
    ("dzɔ", "date"),
    ("xɔli", "lieu"),
    ("fiefia", "ville"),
    ("fefia", "Lomé"),
    ("kpalimɛ", "Kpalimé"),
    ("sokodɛ", "Sokodé"),
    ("kara", "Kara"),
    ("tsɛvi", "Tsévié"),
    ("kɔ si", "Où?"),
    ("nka si", "Quand?"),
    ("nyitsukpa", "citoyen"),
    ("xɔliaɖo", "adresse"),
    ("xfɛ", "argent"),
    ("timbrɔ", "timbre fiscal"),
    ("dɔli", "droit/taxe"),
    ("deli", "délai"),
    ("dzɔƒe", "tribunal"),
    ("komesi", "commissariat"),
    ("maʋi", "mairie"),
    ("prefɛ", "préfecture"),
    ("afɔ", "travail/service"),
    ("azɔ", "travail"),
]

VOCABULAIRE_TECHNOLOGIE = [
    ("telefɔ", "téléphone"),
    ("fon", "téléphone portable"),
    ("oranj", "Orange (opérateur)"),
    ("t모니", "T-Money (mobile money)"),
    ("floflo", "application mobile"),
    ("internɛ", "internet"),
    ("wifi", "WiFi"),
    ("kliklik", "clic/appui"),
    ("mesi", "message"),
    ("apɛ", "appels"),
    ("vidio", "vidéo/appel vidéo"),
    ("fotografi", "photo"),
    ("kamerɛ", "caméra"),
    ("ladane", "réseau social"),
    ("whatSapp", "WhatsApp"),
    ("feisbuk", "Facebook"),
    ("instagram", "Instagram"),
    ("yutub", "YouTube"),
    ("twitɛ", "Twitter/X"),
    ("karte", "crédit/recharge"),
    ("kod", "code"),
    ("PIN", "code PIN"),
    ("saldo", "solde"),
    ("transfɛ", "transfert d'argent"),
    ("depot", "dépôt"),
    ("retrait", "retrait"),
    ("paiyɛ", "paiement mobile"),
    ("kɔntak", "contact"),
    ("adres", "adresse email"),
    ("paswɔ", "mot de passe"),
    ("koneksyɔ", "connexion"),
]


# =============================================================================
# STRUCTURE JSONL POUR CHAQUE PHRASE
# =============================================================================

FORMAT_JSONL = {
    "instruction": "Traduis en mina cette phrase en français",
    "input": "<phrase_française>",
    "output": "<phrase_mina>",
    "theme": "<domaine>",
    "sous_theme": "<sous_thème>",
    "validated": false,
    "tonalite": "<haut|moyen|bas>",  # Annotation tonale
    "argot": <bool>,  # True si expression argot
    "source": "enrichissement_plan"
}


# =============================================================================
# COMPTEURS ET STATISTIQUES
# =============================================================================

STATISTIQUES = {
    "total_vise": 5000,
    "total_actuel": 500,  # Estimation
    "ecart": 4500,
    "domaines": 10,
    "phrases_par_domaine": 500,
    "progression": "0%",
}

def generer_plan():
    """Génère le plan d'exécution complet"""
    print("=" * 70)
    print("PLAN D'ENRICHISSEMENT DU CORPUS MINA-FRANÇAIS")
    print("=" * 70)
    print()

    print("RÉSUMÉ:")
    print(f"  - Corpus actuel: ~500 phrases")
    print(f"  - Objectif: 5000 phrases")
    print(f"  - Écart: 4500 phrases à générer")
    print(f"  - 10 domaines x 500 phrases chacun")
    print()

    print("-" * 70)
    print("DOMAINES PRIORITAIRES:")
    print("-" * 70)

    for domaine, info in DOMAINES_PRIORITAIRES.items():
        print(f"\n{info['priorite']:2d}. {domaine.upper()}")
        print(f"    Description: {info['description']}")
        print(f"    Cible: {info['cible']} phrases")
        print(f"    Sous-thèmes: {', '.join(info['sous_themes'][:4])}...")

    print()
    print("-" * 70)
    print("ÉLÉMENTS SPÉCIAUX:")
    print("-" * 70)
    print("  - Argot de Lomé: ~200 phrases")
    print("  - Termes administratifs togolais: ~150 phrases")
    print("  - Annotations tonales: haut/moyen/bas")
    print("  - Validation linguistique requise")

    return DOMAINES_PRIORITAIRES


if __name__ == "__main__":
    plan = generer_plan()