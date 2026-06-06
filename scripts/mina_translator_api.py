"""
scripts/mina_translator_api.py - API de Traduction FR↔Mina via Ollama
====================================================================

API simple utilisant Ollama (Qwen2.5-coder:7b) pour la traduction.
Inclut le vocabulaire Mina pour améliorer les traductions.

Pas besoin de GPU - utilise Ollama en local.

Usage:
    python scripts/mina_translator_api.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Forcer UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"

OLLAMA_API = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:7b"

# =============================================================================
# VOCABULAIRE MINA
# =============================================================================

VOCABULARY = """
# VOCABULAIRE MINA (Ewe du Togo)

## Salutations
- sawo = bonjour
- akɔ = merci
- nukɔ = au-revoir
- mede = bonne nuit
- wo lɔ = bonne journee

## Pronoms
- me = je/moi
- a = tu
- e = il/elle
- wo = vous
- wɔ = ils/elles

## Corps humain
- ƒu = tête
- tɔgbo = gorge, cou, ventre
- anyi = yeux
- ɖu = dents
- amɛ = pied/jambe
- gbɔ = bouche
- si = oreille

## Santé/Maladies
- fɛ́e = malade
- ku = avoir mal
- nuwɔna = santé
- aɖa = médicament
- ekpɔ = guérir
- ɖevi = fièvre/chaleur
- hu = tousser
- nukɔ = vomit

## Médecin/Hôpital
- Nusɔla = infirmier/medecin
- dɔta = docteur
- hɔspita = hopital
- soko aɖa = pharmacie
- asawuwu = medicament/comprime
- ɖaƒiaɖɔ = paludisme

## Verbes
- le = être
- ena = avoir
- ka = aller
- va = venir
- kpɔ = voir
- dze = manger
- nu = boire
- ɖo = partir/sortir
- lɔ = dire/parler
- susumɔ = dormir

## Temps
- ai = aujourd'hui
- ẹmɛ = demain
- etsɛ = hier
- gboẽ = matin
- ɖa = nuit
- ci = ici
- yi = là

## Négation
- ...o = non/ne...pas
- me kpɔ o = je ne vois pas
- mekpɔa o = je ne peux pas

## Expressions courantes
- Eku mɛ = je suis malade
- Eƒu nu mɛ = il a mal à la tête
- Nusɔla la va = le medecin est venu
- Mekpɔa ŋkɔ o klɔe = je dois aller à l'hopital
"""

# =============================================================================
# CHARGER LE CORPUS
# =============================================================================

def load_corpus():
    """Charge le corpus FR→Mina pour les exemples."""
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


# Extraire quelques exemples du corpus
EXAMPLES_CORPUS = load_corpus()
EXAMPLE_PAIRS = [(e['input'], e['output']) for e in EXAMPLES_CORPUS[:30]]


# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_lang: str = Field("fr", pattern="^(fr|mina)$")
    target_lang: str = Field("mina", pattern="^(fr|mina)$")
    temperature: float = Field(0.3, ge=0, le=2)


class TranslateResponse(BaseModel):
    original: str
    translation: str
    source_lang: str
    target_lang: str
    model: str
    time_ms: float


# =============================================================================
# FONCTIONS DE TRADUCTION
# =============================================================================

def check_ollama() -> dict:
    """Vérifie le statut d'Ollama."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "online", "models": models}
    except Exception as e:
        return {"status": "offline", "error": str(e)}
    return {"status": "offline"}


def translate(text: str, source_lang: str, target_lang: str,
             temperature: float = 0.3) -> tuple[str, float]:
    """Traduit le texte entre FR et MINA via Ollama."""

    start = time.time()

    # Construire le prompt selon la direction
    if source_lang == "fr" and target_lang == "mina":
        instruction = "FRANÇAIS vers MINA (Ewe du Togo)"
        system_prompt = f"""Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.
{VOCABULARY}

RÈGLES DE TRADUCTION:
1. Traduis exactement et naturellement du français vers le Mina
2. Utilise les termes Mina equivalents (pas de translittération)
3. La négation en Mina utilise "...o" (ex: me kpɔ o = je ne vois pas)
4. Réponds UNIQUEMENT avec la traduction, rien d'autre

EXEMPLES DU CORPUS:"""
    elif source_lang == "mina" and target_lang == "fr":
        instruction = "MINA (Ewe du Togo) vers FRANÇAIS"
        system_prompt = f"""Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.
{VOCABULARY}

RÈGLES DE TRADUCTION:
1. Traduis exactement du Mina vers le français
2. Respecte la structure des phrases Mina
3. Réponds UNIQUEMENT avec la traduction, rien d'autre

EXEMPLES DU CORPUS:"""
    else:
        raise ValueError(f"Direction non supportée: {source_lang} -> {target_lang}")

    # Ajouter des exemples du corpus
    for fr, mina in EXAMPLE_PAIRS[:5]:
        if source_lang == "fr" and target_lang == "mina":
            system_prompt += f"\n- Français: {fr}\n  Mina: {mina}"
        else:
            system_prompt += f"\n- Mina: {mina}\n  Français: {fr}"

    # Prompt final
    if source_lang == "fr":
        system_prompt += f"\n\nMaintenant traduis:\n{text}"
    else:
        system_prompt += f"\n\nMaintenant traduis:\n{text}"

    # Appeler Ollama
    try:
        r = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": system_prompt,
                "temperature": temperature,
                "num_predict": 150,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=120
        )

        elapsed = (time.time() - start) * 1000

        if r.status_code == 200:
            result = r.json()["response"].strip()
            # Nettoyer
            result = clean_translation(result, source_lang, target_lang)
            return result, elapsed
        else:
            raise HTTPException(status_code=500, detail=f"Ollama error: {r.text}")

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Timeout - Ollama trop lent")
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama non accessible - démarrez: ollama serve")


def clean_translation(text: str, source: str, target: str) -> str:
    """Nettoie la traduction."""
    import re

    # Enlever les balises think
    text = re.sub(r'</?think[^>]*>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Enlever les espaces multiples
    text = re.sub(r'\s+', ' ', text)

    # Si la réponse contient le prompt, extraire juste la traduction
    if source == "fr":
        # Chercher après "Mina:" ou "mina:"
        match = re.search(r'(?:Mina|mina)[:\s]+(.+)', text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    else:
        # Chercher après "Français:" ou "français:"
        match = re.search(r'(?:Français|français)[:\s]+(.+)', text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # Limiter à une seule phrase
    if '.' in text and len(text) > 100:
        text = text.split('.')[0] + '.'

    return text.strip()


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="Mina-Translator API",
    description="""
## API de Traduction Français ↔ Mina (Ewe du Togo)

### Fonctionnalités
- Traduction FR → Mina
- Traduction Mina → FR
- Support du vocabulaire Mina complet

### Modèle utilisé
- Ollama: qwen2.5-coder:7b
- Vocabulaire: 360 paires FR→Mina du corpus

### Démarrage
```bash
ollama serve
python scripts/mina_translator_api.py
```
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Page d'accueil."""
    ollama_status = check_ollama()

    return {
        "name": "Mina-Translator API",
        "version": "1.0.0",
        "status": "operational",
        "ollama": ollama_status["status"],
        "corpus_pairs": len(EXAMPLE_PAIRS),
        "endpoints": {
            "translate": "/translate (POST)",
            "health": "/health (GET)",
            "languages": "/languages (GET)",
            "examples": "/examples (GET)",
        }
    }


@app.get("/health")
async def health():
    """Santé de l'API."""
    ollama_status = check_ollama()

    return {
        "status": "healthy" if ollama_status["status"] == "online" else "degraded",
        "ollama": ollama_status,
        "corpus_loaded": len(EXAMPLE_PAIRS),
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest):
    """Traduit du texte entre FR et MINA."""
    if request.source_lang == request.target_lang:
        raise HTTPException(
            status_code=400,
            detail="Source et cible doivent être différents"
        )

    translation, elapsed = translate(
        request.text,
        request.source_lang,
        request.target_lang,
        request.temperature,
    )

    return TranslateResponse(
        original=request.text,
        translation=translation,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        model=OLLAMA_MODEL,
        time_ms=round(elapsed, 1),
    )


@app.get("/languages")
async def list_languages():
    """Liste les langues supportées."""
    return {
        "languages": [
            {"code": "fr", "name": "Français", "native": "Français"},
            {"code": "mina", "name": "Mina (Ewe)", "native": "Mina"},
        ],
        "directions": ["fr→mina", "mina→fr"],
    }


@app.get("/examples")
async def get_examples(limit: int = 10):
    """Retourne des exemples de traductions du corpus."""
    return {
        "examples": [
            {"fr": fr, "mina": mina}
            for fr, mina in EXAMPLE_PAIRS[:limit]
        ],
        "total": len(EXAMPLE_PAIRS),
    }


@app.get("/vocabulary")
async def get_vocabulary():
    """Retourne le vocabulaire Mina."""
    return {
        "vocabulary": VOCABULARY,
        "source": "Corpus Mina + Vocabulaire Ewe",
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   MINA-TRANSLATOR API")
    print("   Traduction FR ↔ Mina via Ollama")
    print("="*60)

    # Vérifier Ollama
    status = check_ollama()
    if status["status"] == "online":
        print(f"\n   Ollama: ✓ Connecté")
        print(f"   Modèles: {', '.join(status.get('models', [])[:3])}")
    else:
        print(f"\n   Ollama: ✗ Non disponible")
        print("   Démarrez: ollama serve")

    print(f"   Corpus: {len(EXAMPLE_PAIRS)} paires FR→Mina")
    print("\n   API: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()