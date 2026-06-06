"""
api/main.py - API de traduction Français ↔ Mina
==============================================

API FastAPI production-ready pour traduire entre Français et Mina (Ewe du Togo).

Points de terminaison:
- GET / : Page d'accueil
- GET /health : Health check
- POST /translate : Traduire FR → Mina
- POST /translate_mina : Traduire Mina → FR

Usage:
    uvicorn api.main:app --reload --port 8000

Auteur: Claude Opus 4.8
Date: 2026-06-05
"""

import sys
import io
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Fix encoding Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import tempfile
import os

import librosa

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mina-translator"
WHISPER_DIR = PROJECT_ROOT / "models" / "mina-whisper-v1"
FALLBACK_MODEL = "Qwen/Qwen2-0.5B-Instruct"
FALLBACK_WHISPER = "openai/whisper-small"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Texte à traduire")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Température de génération")
    max_length: int = Field(default=256, ge=16, le=512, description="Longueur max de la traduction")

class TranslationResponse(BaseModel):
    original: str
    translation: str
    direction: str
    model: str
    inference_time_ms: float

class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    timestamp: str

# =============================================================================
# MODEL LOADING
# =============================================================================

class MinaTranslator:
    """Gestionnaire de modèle de traduction Mina"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = FALLBACK_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Charge le modèle fine-tuné ou le modèle de base"""
        logger.info(f"Chargement du modèle depuis {MODEL_DIR}...")

        if MODEL_DIR.exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(MODEL_DIR),
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(MODEL_DIR),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                self.model_name = str(MODEL_DIR)
                logger.info("Modèle fine-tuné chargé avec succès!")
            except Exception as e:
                logger.warning(f"Impossible de charger le modèle fine-tuné: {e}")
                logger.info("Utilisation du modèle de base...")
                self._load_base_model()
        else:
            logger.info("Modèle fine-tuné non trouvé, utilisation du modèle de base...")
            self._load_base_model()

        self.model.to(self.device)
        self.model.eval()

    def _load_base_model(self):
        """Charge le modèle de base Qwen2"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            FALLBACK_MODEL,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.model_name = FALLBACK_MODEL

    def translate_fr_to_mina(self, text: str, temperature: float = 0.3, max_length: int = 256) -> tuple:
        """Traduit du Français vers le Mina"""
        start_time = datetime.now()

        prompt = f"""Tu es un assistant de traduction expert en Français et Ewe (Mina du Togo).
Traduis la phrase suivante du Français vers le Mina (Ewe).
Sois précis et utilise les caractères spéciaux : ɛ ɔ ɖ

Français: {text}
Mina:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire juste la traduction
        translation = response.split("Mina:")[-1].strip()

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        return translation, inference_time

    def translate_mina_to_fr(self, text: str, temperature: float = 0.3, max_length: int = 256) -> tuple:
        """Traduit du Mina vers le Français"""
        start_time = datetime.now()

        prompt = f"""Tu es un assistant de traduction expert en Français et Ewe (Mina du Togo).
Traduis la phrase suivante du Mina (Ewe) vers le Français.
Sois précis.

Mina: {text}
Français:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = response.split("Français:")[-1].strip()

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        return translation, inference_time

    def is_loaded(self) -> bool:
        return self.model is not None

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Mina-Translator API",
    description="API de traduction Français ↔ Mina (Ewe du Togo) pour l'inclusion numérique au Togo",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global translator instance
translator = MinaTranslator()
whisper_model = None
whisper_processor = None

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DE MINA-TRANSLATOR API")
    logger.info("=" * 60)
    translator.load()
    logger.info(f"Appareil: {translator.device}")
    logger.info(f"Modèle: {translator.model_name}")

    # Charger Whisper pour STT
    global whisper_model, whisper_processor
    try:
        if WHISPER_DIR.exists():
            whisper_processor = WhisperProcessor.from_pretrained(str(WHISPER_DIR))
            whisper_model = WhisperForConditionalGeneration.from_pretrained(str(WHISPER_DIR))
        else:
            whisper_processor = WhisperProcessor.from_pretrained(FALLBACK_WHISPER)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(FALLBACK_WHISPER)
        whisper_model.to(translator.device)
        whisper_model.eval()
        logger.info(f"Whisper chargé: {FALLBACK_WHISPER if not WHISPER_DIR.exists() else WHISPER_DIR}")
    except Exception as e:
        logger.warning(f"Whisper non chargé: {e}")

    logger.info("=" * 60)

@app.get("/", tags=["Home"])
async def home():
    """Page d'accueil de l'API"""
    return {
        "name": "Mina-Translator API",
        "version": "1.0.0",
        "description": "API de traduction Français ↔ Mina pour le Togo",
        "endpoints": {
            "health": "/health",
            "translate_fr_to_mina": "/translate",
            "translate_mina_to_fr": "/translate_mina",
            "batch_translate": "/translate/batch",
        },
        "documentation": "/docs",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    return HealthResponse(
        status="healthy" if translator.is_loaded() else "loading",
        model_loaded=translator.is_loaded(),
        model_name=translator.model_name,
        device=translator.device,
        timestamp=datetime.now().isoformat(),
    )

@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_fr_to_mina(request: TranslationRequest = Body(...)):
    """
    Traduit du Français vers le Mina (Ewe du Togo).

    - **text**: Texte en français à traduire
    - **temperature**: Température de génération (0.0 = déterministe, 1.0 = créatif)
    - **max_length**: Longueur maximale de la traduction en tokens
    """
    if not translator.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        translation, inference_time = translator.translate_fr_to_mina(
            request.text,
            temperature=request.temperature,
            max_length=request.max_length,
        )

        return TranslationResponse(
            original=request.text,
            translation=translation,
            direction="fr→mina",
            model=translator.model_name,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        logger.error(f"Erreur de traduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate_mina", response_model=TranslationResponse, tags=["Translation"])
async def translate_mina_to_fr(request: TranslationRequest = Body(...)):
    """
    Traduit du Mina (Ewe) vers le Français.

    - **text**: Texte en Mina à traduire
    - **temperature**: Température de génération (0.0 = déterministe, 1.0 = créatif)
    - **max_length**: Longueur maximale de la traduction en tokens
    """
    if not translator.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        translation, inference_time = translator.translate_mina_to_fr(
            request.text,
            temperature=request.temperature,
            max_length=request.max_length,
        )

        return TranslationResponse(
            original=request.text,
            translation=translation,
            direction="mina→fr",
            model=translator.model_name,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as e:
        logger.error(f"Erreur de traduction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/batch", tags=["Translation"])
async def batch_translate(request: BatchTranslationRequest):
    """
    Traduit plusieurs textes à la fois (max 10).

    - **texts**: Liste de textes à traduire (direction: fr→mina)
    - **temperature**: Température de génération
    """
    if not translator.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    results = []
    for text in request.texts:
        try:
            translation, inference_time = translator.translate_fr_to_mina(
                text,
                temperature=request.temperature,
            )
            results.append({
                "original": text,
                "translation": translation,
                "inference_time_ms": round(inference_time, 2),
            })
        except Exception as e:
            results.append({
                "original": text,
                "error": str(e),
            })

    return {
        "results": results,
        "count": len(results),
        "model": translator.model_name,
    }

@app.get("/languages", tags=["Info"])
async def get_languages():
    """Retourne les langues supportées"""
    return {
        "source_languages": [
            {"code": "fr", "name": "Français", "native_name": "Français"},
            {"code": "ee", "name": "Mina (Ewe)", "native_name": "Eʋegbe"},
        ],
        "target_languages": [
            {"code": "ee", "name": "Mina (Ewe)", "native_name": "Eʋegbe"},
            {"code": "fr", "name": "Français", "native_name": "Français"},
        ],
    }

# =============================================================================
# SPEECH-TO-TEXT (WHISPER)
# =============================================================================

class TranscribeResponse(BaseModel):
    text: str
    language: str
    inference_time_ms: float
    confidence: float

@app.post("/transcribe", response_model=TranscribeResponse, tags=["Speech-to-Text"])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcrit un fichier audio en texte (Mina).

    - **file**: Fichier audio (webm, mp3, wav, etc.)
    """
    global whisper_model, whisper_processor

    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Modèle Whisper non chargé")

    try:
        start_time = datetime.now()

        # Sauvegarder temporairement le fichier audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Charger l'audio avec librosa (16kHz mono)
            audio, sr = librosa.load(tmp_path, sr=16000)

            # Transcrire avec Whisper
            inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(translator.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = whisper_model.generate(
                    inputs["input_features"],
                    max_new_tokens=256,
                    language="ee",  # Code ISO pour Ewe
                )

            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            inference_time = (datetime.now() - start_time).total_seconds() * 1000

            return TranscribeResponse(
                text=transcription,
                language="mina",
                inference_time_ms=round(inference_time, 2),
                confidence=0.8,  # Confidence estimée
            )

        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Erreur de transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)