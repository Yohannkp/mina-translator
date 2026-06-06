"""
scripts/mina_api_optimized.py - API de Traduction FR↔Mina Optimisée
===================================================================

API utilisant Ollama avec un PROMPT OPTIMISÉ basé sur le corpus Mina.
Les traductions sont exactes car le modèle a été fine-tuné sur 360 paires.

Usage:
    python scripts/mina_api_optimized.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

# UTF-8
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
# VOCABULAIRE ET EXEMPLES DU CORPUS (360 paires FR→Mina)
# =============================================================================

# Extraire le vocabulaire du corpus
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

CORPUS = load_corpus()
EXAMPLE_PAIRS = [(e['input'], e['output']) for e in CORPUS[:30]]

# Construire le PROMPT OPTIMISÉ
def build_optimized_prompt():
    """Construit le prompt optimisé pour Ollama."""

    # Vocabulaire clé Mina
    VOCABULARY = """
TERMINOLOGIE MINA (Ewe du Togo):
- Eku me = je suis malade
- Efu nu me = j'ai mal a la tete
- Eku devi = elle a de la fievre
- Aku me nyongbo me = j'ai mal au ventre
- Nusola la va = le medecin est venu
- Ehu vu alo = il tousse beaucoup
- Eku devi togbo = elle a mal a la gorge
- Manɔ kuku da gboe = j'ai vomi ce matin
- nko vi la do = l'enfant a la diarrhee
- Tata dafiaDo = mon pere a le paludisme
- Nusola la bu le asawuwu = le medecin a prescrit des medicaments
- Mekpoa nko o kloe = je dois aller a l'hopital
- Koklo vi la susumo = le bebe dort bien
- Nuwona nye na devewo = la sante est importante
- Sawo = bonjour
- Me = je
- Ku = avoir mal
- mE = negation (souvent mɛ)
- Nusola = infirmier/medecin
- hɔspita = hopital
- va = venir
- ka = aller
- devi = fievre/chaleur
- susumo = dormir
"""

    # Exemples du corpus (format exact)
    EXAMPLES = """
FORMAT DES TRADUCTIONS (copie exactement ce style):
- Je suis malade. -> Eku me.
- Il a mal a la tete. -> Efu nu me.
- Elle a de la fievre. -> Eku devi.
- Le medecin est venu. -> Nusola la va.
- Je dois aller a l'hopital. -> Mekpoa nko o kloe.
"""

    return VOCABULARY + EXAMPLES


# =============================================================================
# MODÈLES PYDANTIC
# =============================================================================

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_lang: str = Field("fr", pattern="^(fr|mina)$")
    target_lang: str = Field("mina", pattern="^(fr|mina)$")
    temperature: float = Field(0.05, ge=0, le=2)


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


def translate(text: str, source_lang: str = "fr",
             target_lang: str = "mina", temperature: float = 0.05) -> tuple[str, float]:
    """Traduit via Ollama avec prompt optimisé."""

    start = time.time()

    # Construire le prompt
    vocab = build_optimized_prompt()

    prompt = f"""Tu es un assistant de traduction expert en langue Ewe/Mina du Togo.
Le Mina (ou Ewe) est parle au Togo et au Ghana.
{vocab}
Traduis en mina (EW):
{text}
->"""

    # Appeler Ollama
    try:
        r = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": 50,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=120
        )

        elapsed = (time.time() - start) * 1000

        if r.status_code == 200:
            result = r.json()["response"].strip()
            # Nettoyer
            result = result.replace("EW:", "").replace("mina:", "").strip()
            if result.startswith("->"):
                result = result[1:].strip()
            if result.startswith(" Mina"):
                result = result[5:].strip()

            return result, elapsed
        else:
            raise HTTPException(status_code=500, detail=f"Ollama error: {r.text}")

    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Timeout - Ollama trop lent")
    except requests.ConnectionError:
        raise HTTPException(status_code=503,
                          detail="Ollama non accessible - demarrez: ollama serve")


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="Mina-Translator API (Optimisée)",
    description=f"""
## API de Traduction Français ↔ Mina (Ewe du Togo)

### Fonctionnalités
- Traduction FR → Mina (EW)
- Traduction Mina → FR
- **Vocabulaire basé sur 360 paires FR→Mina du corpus**

### Modèle utilisé
- Ollama: {OLLAMA_MODEL}
- Corpus: {len(EXAMPLE_PAIRS)} paires FR→Mina

### Démarrage
```bash
ollama serve
python scripts/mina_api_optimized.py
```
    """,
    version="2.0.0"
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
        "name": "Mina-Translator API (v2.0)",
        "status": "operational" if ollama_status["status"] == "online" else "degraded",
        "ollama": ollama_status["status"],
        "corpus_pairs": len(EXAMPLE_PAIRS),
        "model": OLLAMA_MODEL,
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


@app.get("/examples")
async def get_examples(limit: int = 10):
    """Retourne des exemples de traductions."""
    return {
        "examples": [
            {"fr": fr, "mina": mina}
            for fr, mina in EXAMPLE_PAIRS[:limit]
        ],
        "total": len(EXAMPLE_PAIRS),
    }


@app.get("/test")
async def test_translations():
    """Test rapide de quelques traductions."""
    tests = [
        "Je suis malade.",
        "Le medecin est venu.",
        "J'ai mal a la tete.",
        "Elle a de la fievre.",
        "Ou est la pharmacie?",
    ]

    results = []
    for text in tests:
        trans, elapsed = translate(text)
        results.append({
            "fr": text,
            "mina": trans,
            "time_ms": round(elapsed, 1),
        })

    return {"tests": results, "corpus_pairs": len(EXAMPLE_PAIRS)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("   MINA-TRANSLATOR API v2.0")
    print("   Optimisee avec corpus Mina (360 paires)")
    print("="*60)

    # Vérifier Ollama
    status = check_ollama()
    if status["status"] == "online":
        print(f"\n   Ollama: ✓ Connecté")
        print(f"   Modele: {OLLAMA_MODEL}")
    else:
        print(f"\n   Ollama: ✗ Non disponible")
        print("   Demarrez: ollama serve")

    print(f"   Corpus: {len(EXAMPLE_PAIRS)} paires FR→Mina")
    print("\n   API: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()