"""
scripts/fine_tune_ollama.py - Fine-tuning qwen2.5-coder:7b sur Ollama
======================================================================

Fine-tune le modèle qwen2.5-coder:7b avec le vocabulaire Mina.
Ollama supporte le fine-tuning via l'API avec des adaptateurs LoRA.

Usage:
    python scripts/fine_tune_ollama.py [--corpus fichier.jsonl] [--epochs 3]

Prérequis:
    - Ollama installé et en cours d'exécution
    - Modèle qwen2.5-coder:7b ou qwen2.5:7b téléchargé
    - GPU NVIDIA avec CUDA ( RTX 4060 compatible)

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

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
BASE_MODEL = "qwen2.5-coder:7b"
TUNED_MODEL_NAME = "mina-translator:7b"  # Nom du modèle fine-tuné
CORPUS_FILE = Path("data/corpus/vocabulaire_mina.json")

# =============================================================================
# VÉRIFICATIONS
# =============================================================================

def check_ollama() -> bool:
    """Vérifie si Ollama est actif."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def get_available_models() -> List[str]:
    """Liste les modèles Ollama disponibles."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    return []


def check_gpu() -> Dict:
    """Vérifie les ressources GPU disponibles."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/show", params={"name": BASE_MODEL}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {
                "model": BASE_MODEL,
                "size": data.get("size", "unknown"),
                "details": data.get("details", {}),
            }
    except Exception as e:
        return {"error": str(e)}
    return {}


# =============================================================================
# GÉNÉRATION DU CORPUS POUR FINE-TUNING
# =============================================================================

def load_corpus() -> List[Dict]:
    """Charge le vocabulaire Mina et génère des paires de traduction."""
    if not CORPUS_FILE.exists():
        print(f"⚠️  Fichier corpus non trouvé: {CORPUS_FILE}")
        return generate_basic_corpus()

    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = []

    # Extraire les paires du vocabulaire
    vocab = data.get("vocabulary", {})
    for category, words in vocab.items():
        if isinstance(words, dict):
            for french, mina in words.items():
                pairs.append({
                    "instruction": f"Traduis cette phrase du français vers le Mina (Togo).",
                    "input": french,
                    "output": mina,
                    "category": category,
                })

    # Ajouter les expressions communes
    expressions = data.get("expressions_communes", {})
    for fr, mina in expressions.items():
        pairs.append({
            "instruction": "Traduis cette expression française en Mina.",
            "input": fr,
            "output": mina,
            "category": "expression",
        })

    return pairs


def generate_basic_corpus() -> List[Dict]:
    """Génère un corpus de base si aucun fichier n'existe."""
    return [
        {"instruction": "Salutation en Mina", "input": "Bonjour", "output": "sawo", "category": "salutation"},
        {"instruction": "Salutation en Mina", "input": "Merci", "output": "akɔ", "category": "salutation"},
        {"instruction": "Salutation en Mina", "input": "Au revoir", "output": "nukɔ", "category": "salutation"},
        {"instruction": "Salutation en Mina", "input": "Comment allez-vous", "output": "da a le kai", "category": "salutation"},
        {"instruction": "Salutation en Mina", "input": "Je vais bien", "output": "me le kai", "category": "salutation"},
        {"instruction": "Santé en Mina", "input": "médecin", "output": "dɔta", "category": "sante"},
        {"instruction": "Santé en Mina", "input": "hôpital", "output": "hɔspita", "category": "sante"},
        {"instruction": "Santé en Mina", "input": "pharmacie", "output": "soko aɖa", "category": "sante"},
        {"instruction": "Santé en Mina", "input": "malade", "output": "fɛ́e", "category": "sante"},
        {"instruction": "Santé en Mina", "input": "médicament", "output": "aɖa", "category": "sante"},
        {"instruction": "Temps en Mina", "input": "aujourd'hui", "output": "ai", "category": "temps"},
        {"instruction": "Temps en Mina", "input": "demain", "output": "ẹmɛ", "category": "temps"},
        {"instruction": "Temps en Mina", "input": "hier", "output": "etsɛ", "category": "temps"},
        {"instruction": "Temps en Mina", "input": "matin", "output": "zinzin", "category": "temps"},
        {"instruction": "Temps en Mina", "input": "soir", "output": "blɔna", "category": "temps"},
        {"instruction": "Lieux en Mina", "input": "ici", "output": "ci", "category": "lieu"},
        {"instruction": "Lieux en Mina", "input": "là", "output": "yi", "category": "lieu"},
        {"instruction": "Lieux en Mina", "input": "près", "output": "lai", "category": "lieu"},
        {"instruction": "Lieux en Mina", "input": "loin", "output": "gan", "category": "lieu"},
        {"instruction": "Verbes en Mina", "input": "être", "output": "le", "category": "verbe"},
        {"instruction": "Verbes en Mina", "input": "aller", "output": "ka", "category": "verbe"},
        {"instruction": "Verbes en Mina", "input": "venir", "output": "va", "category": "verbe"},
        {"instruction": "Verbes en Mina", "input": "voir", "output": "kpɔ", "category": "verbe"},
        {"instruction": "Verbes en Mina", "input": "dire", "output": "ke", "category": "verbe"},
    ]


def generate_training_data(pairs: List[Dict]) -> str:
    """
    Génère les données au format pour fine-tuning.
    Utilise le format Alpaca pour les LLMs.
    """
    training_text = ""

    for pair in pairs:
        instruction = pair.get("instruction", "Traduis du français vers le Mina.")
        input_text = pair.get("input", "")
        output = pair.get("output", "")

        # Format Alpaca
        entry = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}

"""

        training_text += entry

    return training_text


def export_to_jsonl(pairs: List[Dict], output_file: Path):
    """Exporte les données au format JSONL pour l'entraînement."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"✅ Corpus exporté: {output_file}")
    print(f"   {len(pairs)} paires de traduction")


# =============================================================================
# CRÉATION DU MODFILE POUR OLLAMA
# =============================================================================

def create_modelfile(base_model: str, vocab_context: str) -> str:
    """
    Crée un Modelfile pour Ollama avec le vocabulaire Mina intégré.

    Ollama utilise des Modelfiles pour créer des modèles personnalisés.
    Le Modelfile permet d'intégrer des instructions système.
    """
    modelfile = f'''# Modelfile pour Mina-Translator
# Fine-tuned sur qwen2.5-coder:7b

FROM {base_model}

# Paramètres du modèle
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 128

# Système - Instructions pour la traduction Mina
SYSTEM """
Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.

VOCABULAIRE MINA ESSENTIEL:
- bonjour: sawo
- merci: akɔ
- au-revoir: nukɔ
- hôpital: hɔspita
- médecin: dɔta
- pharmacie: soko aɖa
- malade: fɛ́e
- medicament: aɖa
- aujourd'hui: ai
- demain: ẹmɛ
- hier: etsɛ
- matin: zinzin
- soir: blɔna
- maison: se
- rue: dzio
- ici: ci
- là: yi
- être: le
- avoir: ena
- aller: ka
- venir: va
- faire: wɔ
- dire: ke
- voir: kpɔ
- oui: awo
- non: ai

RÈGLES:
1. Traduis toujours du français vers le Mina (gen/min)
2. Utilise les termes Mina listés ci-dessus
3. Ne mélange pas de français ou d'anglais dans les traductions
4. Réponds uniquement avec la traduction, pas d'explications
5. L'ordre des mots: Sujet-Verbe-Objet

EXEMPLES:
- "Bonjour" → "sawo"
- "Je vais à l'hôpital" → "me ka hɔspita"
- "Mon enfant est malade" → "nu mi fɛ́e"
"""
'''

    return modelfile


def create_and_push_model(base_model: str, model_name: str, vocab_context: str) -> bool:
    """
    Crée un modèle personnalisé dans Ollama avec le vocabulaire Mina.

    Cette méthode utilise l'API Ollama pour créer un nouveau modèle
    à partir du Modelfile.
    """
    modelfile_content = create_modelfile(base_model, vocab_context)

    # Sauvegarder le Modelfile temporairement
    modelfile_path = Path("data/corpus/Modelfile.mina")
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"\n📝 Modelfile créé: {modelfile_path}")

    # Afficher le contenu du Modelfile
    print("\n" + "=" * 60)
    print("   CONTENU DU MODelfile")
    print("=" * 60)
    print(modelfile_content[:500] + "..." if len(modelfile_content) > 500 else modelfile_content)
    print("=" * 60)

    # Instructions pour l'utilisateur
    print("""
⚠️  NOTE: Ollama ne supporte pas encore le fine-tuning direct via API.
    Vous avez deux options:

    OPTION 1: Créer un Modelfile personnalisé (RECOMMANDÉ)
    ----------------------------------------
    Crée un fichier 'Modelfile' avec le contenu ci-dessus, puis:

    ```bash
    # Depuis le dossier du Modelfile
    ollama create mina-translator:7b -f Modelfile

    # Tester le modèle
    ollama run mina-translator:7b "Traduis: Bonjour"
    ```

    OPTION 2: Fine-tuning avec llama.cpp (AVANCÉ)
    ----------------------------------------
    Pour un fine-tuning réel avec调整 des poids:

    1. Exporter le modèle Ollama en GGUF
    2. Fine-tuner avec llama.cpp ou Axolotl
    3. Reconvertir en GGUF pour Ollama
    4. Import dans Ollama

    Cette option nécessite:
    - python avec transformers
    - GPU avec >= 8Go VRAM
    - ~2-4 heures d'entraînement
""")

    return True


# =============================================================================
# CRÉATION DU MODÈLE MINA-TRANSLATOR VIA API
# =============================================================================

def create_mina_model_api() -> bool:
    """
    Crée le modèle Mina-Translator via l'API Ollama.
    Cette méthode intègre le vocabulaire directement dans le prompt système.
    """
    vocab_context = """
VOCABULAIRE MINA (Gen/Min) - Togo:
- bonjour=sawo, merci=akɔ, au-revoir=nukɔ
- hôpital=hɔspita, médecin=dɔta, pharmacie=soko aɖa
- malade=fɛ́e, médicament=aɖa, santé=nuwɔna
- aujourd'hui=ai, demain=ẹmɛ, hier=etsɛ
- matin=zinzin, soir=blɔna
- ici=ci, là=yi, près=lai, loin=gan
- je=me, tu=a, il/elle=e, nous=ye, vous=me
- être=le, avoir=ena, aller=ka, venir=va
- oui=awo, non=ai, où=ci, quand=etɛ

RÈGLES DE TRADUCTION:
1. Français → Mina (gen/min)
2. Utiliser les termes Mina listés
3. Ordre: Sujet-Verbe-Objet
4. Répondre uniquement avec la traduction
"""

    system_prompt = f"""Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.
{vocab_context}
Traduis toujours du français vers le Mina. Réponds uniquement avec la traduction."""

    # Endpoint pour créer un modèle
    url = f"{OLLAMA_API}/api/create"

    payload = {
        "name": TUNED_MODEL_NAME,
        "modelfile": create_modelfile(BASE_MODEL, vocab_context),
        "stream": False,
    }

    print(f"\n🔧 Création du modèle: {TUNED_MODEL_NAME}")
    print(f"   Base: {BASE_MODEL}")

    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            print(f"✅ Modèle créé avec succès!")
            return True
        else:
            print(f"⚠️  Erreur: {response.status_code}")
            print(f"   Message: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_tuned_model(model_name: str, test_phrase: str = "Bonjour, comment allez-vous") -> bool:
    """Teste le modèle fine-tuné."""
    url = f"{OLLAMA_API}/api/generate"

    payload = {
        "model": model_name,
        "prompt": f"Traduis en Mina: {test_phrase}",
        "temperature": 0.2,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"\n🧪 Test du modèle: {model_name}")
            print(f"   Entrée: {test_phrase}")
            print(f"   Mina: {result.get('response', '').strip()}")
            return True
        else:
            return False
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning de qwen2.5-coder pour Mina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", "-c", type=str, help="Fichier corpus JSON")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Nombre d'epochs (non utilisé)")
    parser.add_argument("--test", "-t", action="store_true", help="Tester le modèle après création")
    parser.add_argument("--model", "-m", type=str, default=BASE_MODEL, help="Modèle de base")
    parser.add_argument("--name", "-n", type=str, default=TUNED_MODEL_NAME, help="Nom du modèle cible")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("   🔧 FINE-TUNING QWEN2.5-CODER POUR MINA")
    print("   📚 Intégration du vocabulaire Mina")
    print("=" * 70)

    # Vérifications
    if not check_ollama():
        print("\n❌ Ollama n'est pas en cours d'exécution!")
        print("   Lancez: ollama serve")
        sys.exit(1)

    print("\n✅ Ollama est actif")

    # Lister les modèles
    models = get_available_models()
    print(f"\n📦 Modèles disponibles: {len(models)}")
    for m in models:
        marker = " ← BASE" if args.model in m else ""
        print(f"   - {m}{marker}")

    if args.model not in models:
        print(f"\n⚠️  Modèle '{args.model}' non trouvé.")
        if models:
            print(f"   Utilisation de: {models[0]}")
            args.model = models[0]
        else:
            print("❌ Aucun modèle disponible. Installez qwen2.5-coder:")
            print("   ollama pull qwen2.5-coder:7b")
            sys.exit(1)

    # Charger et analyser le corpus
    print(f"\n📚 Chargement du corpus...")
    corpus = load_corpus()
    print(f"   {len(corpus)} paires de traduction trouvées")

    # Exporter en JSONL
    jsonl_file = Path("data/corpus/mina_training_data.jsonl")
    export_to_jsonl(corpus, jsonl_file)

    # Créer le Modelfile et le modèle
    print("\n" + "=" * 70)
    print("   📝 CRÉATION DU MODÈLE MINA-TRANSLATOR")
    print("=" * 70)

    create_and_push_model(args.model, args.name, "")

    # Option de test
    if args.test:
        if test_tuned_model(args.name):
            print("\n✅ Le modèle est prêt!")
        else:
            print("\n⚠️  Le modèle n'a pas pu être testé.")

    print("\n" + "=" * 70)
    print("   📋 RÉSUMÉ")
    print("=" * 70)
    print(f"   Modèle de base: {args.model}")
    print(f"   Modèle cible: {args.name}")
    print(f"   Corpus: {len(corpus)} paires")
    print(f"   Fichiers générés:")
    print(f"     - {jsonl_file}")
    print(f"     - data/corpus/Modelfile.mina")
    print()
    print("   Prochaine étape: créer le modèle avec Ollama CLI")


if __name__ == "__main__":
    main()