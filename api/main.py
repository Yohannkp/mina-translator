"""
API FastAPI pour traduction Mina-Français
=========================================

Endpoints:
    POST /translate - Traduit du texte
    POST /transcribe - Transcrit de l'audio
    POST /speak - Synthétise la parole
    GET /health - Santé de l'API

Usage:
    uvicorn api.main:app --reload --port 8000
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
settings = get_settings()

# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

class TranslateRequest(BaseModel):
    """Requête de traduction"""
    text: str = Field(..., min_length=1, max_length=1000, description="Texte à traduire")
    source_lang: str = Field(default="fr", description="Langue source: fr ou mina")
    target_lang: str = Field(default="mina", description="Langue cible: fr ou mina")


class TranslateResponse(BaseModel):
    """Réponse de traduction"""
    original: str
    translated: str
    source_lang: str
    target_lang: str
    confidence: Optional[float] = None


class TranscribeRequest(BaseModel):
    """Requête de transcription audio"""
    audio_url: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded


class TranscribeResponse(BaseModel):
    """Réponse de transcription"""
    text: str
    language: str = "mina"
    confidence: Optional[float] = None


class SpeakRequest(BaseModel):
    """Requête de synthèse vocale"""
    text: str = Field(..., min_length=1, max_length=500)
    lang: str = Field(default="mina", description="Langue: mina ou fr")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class SpeakResponse(BaseModel):
    """Réponse de synthèse vocale"""
    audio_url: str
    duration: float


class HealthResponse(BaseModel):
    """Réponse de santé"""
    status: str
    gpu_available: bool
    model_loaded: bool
    gpu_memory_used: float
    gpu_memory_total: float


# =============================================================================
# GESTIONNAIRE DE CYCLE DE VIE
# =============================================================================

# Global state
model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le démarrage et l'arrêt de l'application"""
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE L'API MINA-TRANSLATOR")
    logger.info("=" * 60)

    global model, tokenizer, device

    # Initialiser le device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go")

    # Charger le modèle si disponible
    model_path = settings.MODELS_DIR / "mina-translator"
    if model_path.exists():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            logger.info("Chargement du modèle fine-tuné...")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token

            # Chargement avec quantisation si sur GPU
            if torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)

            logger.info("Modèle chargé avec succès!")

        except Exception as e:
            logger.warning(f"Impossible de charger le modèle fine-tuné: {e}")
            logger.info("L'API utilisera des traductions de base")
            model = None
    else:
        logger.info("Modèle fine-tuné non trouvé, utilisation du mode base")
        logger.info(f"Entraîne le modèle avec: python scripts/train_translation.py")

    yield

    # Nettoyage
    logger.info("Arrêt de l'API...")
    if model:
        del model
    if tokenizer:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# CRÉER L'APPLICATION
# =============================================================================

app = FastAPI(
    title="Mina-Translator API",
    description="API pour la traduction et transcription Mina-Français",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# FONCTIONS DE TRADUCTION
# =============================================================================

def translate_with_model(text: str, source: str, target: str) -> tuple[str, float]:
    """Traduit en utilisant le modèle fine-tuné"""

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Traduction de base uniquement."
        )

    # Formater le prompt selon la direction
    if source == "fr" and target == "mina":
        prompt = f"Traduis en mina: {text}"
    else:
        prompt = f"Traduis en français: {text}"

    # Tokeniser
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Générer
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Décoder
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire la traduction
    if "Traduis en mina:" in result:
        result = result.split("Traduis en mina:")[1].strip()
    elif "Traduis en français:" in result:
        result = result.split("Traduis en français:")[1].strip()

    # Estimation confiance (basée sur la longueur)
    confidence = min(len(result) / 50, 1.0)

    return result.strip(), confidence


def translate_base(text: str, source: str, target: str) -> tuple[str, float]:
    """Traduit en utilisant les données de base"""

    from scripts.generate_dataset import TRANSLATIONS

    text_lower = text.lower().strip()

    # Construire un index bidirectionnel
    fr_to_mina = {}
    mina_to_fr = {}

    for theme_data in TRANSLATIONS.values():
        for fr, mina in theme_data["paires"]:
            fr_lower = fr.lower().strip()
            mina_lower = mina.lower().strip()
            fr_to_mina[fr_lower] = mina
            mina_to_fr[mina_lower] = fr

    # Chercher selon la direction
    if source == "fr" and target == "mina":
        if text_lower in fr_to_mina:
            return fr_to_mina[text_lower], 0.95
    elif source == "mina" and target == "fr":
        if text_lower in mina_to_fr:
            return mina_to_fr[text_lower], 0.95

    # Recherche par mot-clé (si la phrase exacte n'est pas trouvée)
    # Pour FR -> MINA: chercher si les mots sont dans une phrase du corpus
    if source == "fr" and target == "mina":
        for fr_lower, mina in fr_to_mina.items():
            # Si au moins 60% des mots matchent
            text_words = set(text_lower.split())
            fr_words = set(fr_lower.split())
            if len(text_words) > 0 and len(fr_words & text_words) / len(text_words) >= 0.6:
                return mina, 0.7  # Confiance réduite pour match partiel

    # Traduction non trouvée
    raise HTTPException(
        status_code=400,
        detail=f"Traduction non trouvee pour: '{text}'. "
               f"Entraine le modele pour plus de couverture."
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil"""
    return {
        "name": "Mina-Translator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Vérifie l'état de l'API"""

    import torch

    gpu_available = torch.cuda.is_available()
    model_loaded = model is not None

    gpu_memory_used = 0
    gpu_memory_total = 0

    if gpu_available:
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_memory_used = (torch.cuda.get_device_properties(0).total_memory -
                          torch.cuda.memory_reserved(0)) / 1e9

    return HealthResponse(
        status="healthy" if (not gpu_available or gpu_memory_total > 1) else "degraded",
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        gpu_memory_used=round(gpu_memory_used, 2),
        gpu_memory_total=round(gpu_memory_total, 2),
    )


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate(request: TranslateRequest):
    """
    Traduit du texte entre le français et le mina

    - **text**: Texte à traduire
    - **source_lang**: Langue source (fr ou mina)
    - **target_lang**: Langue cible (fr ou mina)
    """

    if request.source_lang not in ["fr", "mina"]:
        raise HTTPException(status_code=400, detail="source_lang doit être 'fr' ou 'mina'")

    if request.target_lang not in ["fr", "mina"]:
        raise HTTPException(status_code=400, detail="target_lang doit être 'fr' ou 'mina'")

    if request.source_lang == request.target_lang:
        raise HTTPException(status_code=400, detail="source et target doivent être différents")

    try:
        # Essayer avec le modèle si disponible
        if model is not None:
            translated, confidence = translate_with_model(
                request.text,
                request.source_lang,
                request.target_lang
            )
        else:
            translated, confidence = translate_base(
                request.text,
                request.source_lang,
                request.target_lang
            )

        return TranslateResponse(
            original=request.text,
            translated=translated,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscribeResponse, tags=["Speech"])
async def transcribe(request: TranscribeRequest):
    """
    Transcrit de l'audio en texte

    Nécessite Whisper pour fonctionner.
    """

    if not request.audio_url and not request.audio_data:
        raise HTTPException(status_code=400, detail="audio_url ou audio_data requis")

    raise HTTPException(
        status_code=501,
        detail="Transcription non implémentée. Installe Whisper: pip install openai-whisper"
    )


@app.post("/speak", response_model=SpeakResponse, tags=["Speech"])
async def speak(request: SpeakRequest):
    """
    Synthétise du texte en parole

    Nécessite un moteur TTS pour fonctionner.
    """

    if len(request.text) > 500:
        raise HTTPException(status_code=400, detail="Texte trop long (max 500 caractères)")

    raise HTTPException(
        status_code=501,
        detail="Synthèse vocale non implémentée. Configure un moteur TTS."
    )


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )