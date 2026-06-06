"""
api/mina_api.py - API FastAPI pour Mina-Translator
==================================================

API de traduction/transcription Mina <-> Français
avec support STT (Whisper) et TTS (optionnel)

Usage:
    python -m uvicorn api.mina_api:app --reload --port 8000

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import io
import sys
import os
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, List

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mina-translator-trained"
WHISPER_MODEL = "base"  # ou "small" pour plus de précision

# =============================================================================
# MODELS LOADING (lazy)
# =============================================================================

class MinaTranslator:
    """Traducteur Mina <-> Français"""

    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print("Chargement du modele de traduction Mina...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        # Charger le modèle de base
        BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Charger les adaptateurs LoRA si disponibles
        model_path = MODEL_DIR
        if not model_path.exists():
            # Fallback vers l'ancien modèle
            model_path = PROJECT_ROOT / "models" / "mina-translator"

        if (model_path / "adapter_config.json").exists():
            self._model = PeftModel.from_pretrained(base_model, str(model_path))
            print(f"LoRA loaded from: {model_path}")
        else:
            self._model = base_model
            print("Using base model (no LoRA)")

        self._model.eval()

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print("Traducteur pret!")

    def translate(self, text: str, direction: str = "mina_to_french") -> dict:
        """
        Traduit du texte

        Args:
            text: Texte à traduire
            direction: "mina_to_french" ou "french_to_mina"

        Returns:
            dict avec "translation", "time_ms", etc.
        """
        if direction == "mina_to_french":
            prompt = f"<|im_start|>user\nTraduis en Français: {text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\nTraduis en Ewe: {text}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.time()

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        elapsed_ms = (time.time() - start) * 1000

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = response.split("assistant\n")[-1].strip()

        return {
            "original": text,
            "translation": translation,
            "direction": direction,
            "time_ms": round(elapsed_ms, 1),
        }


class MinaSTT:
    """Reconnaissance vocale Mina avec Whisper"""

    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import whisper
        import soundfile as sf
        import librosa

        print("Chargement du modele Whisper...")

        self._model = whisper.load_model(WHISPER_MODEL)
        self._sf = sf
        self._librosa = librosa

        print("STT pret!")

    def transcribe(self, audio_data: bytes, language: str = "ewe") -> dict:
        """
        Transcrit l'audio en texte Mina

        Args:
            audio_data: Données audio (MP3, WAV, etc.)
            language: Code langue ("ewe" pour Mina)

        Returns:
            dict avec "text", "language", "time_ms", etc.
        """
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Charger l'audio
            audio, sr = self._sf.read(temp_path, dtype="float32")

            # Resample si nécessaire
            if sr != 16000:
                audio = self._librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Normaliser
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            # Transcrire
            start = time.time()

            result = self._model.transcribe(
                audio,
                language=None,  # Auto-détection (le modèle va reconnaitre ce qu'il peut)
                task="transcribe",
            )

            elapsed_ms = (time.time() - start) * 1000

            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "time_ms": round(elapsed_ms, 1),
                "segments": len(result.get("segments", [])),
            }

        finally:
            os.unlink(temp_path)


# =============================================================================
# API
# =============================================================================

app = FastAPI(
    title="Mina-Translator API",
    description="API de traduction/transcription pour le Mina (Ewe) du Togo",
    version="1.0.0",
)

# Modèles de requêtes
class TranslateRequest(BaseModel):
    text: str
    direction: str = "mina_to_french"  # ou "french_to_mina"


class TranslateResponse(BaseModel):
    original: str
    translation: str
    direction: str
    time_ms: float


# Endpoints
@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "name": "Mina-Translator API",
        "version": "1.0.0",
        "description": "Traduction et transcription Mina <-> Français",
        "endpoints": {
            "POST /translate": "Traduire du texte",
            "POST /transcribe": "Transcrire un fichier audio",
            "POST /pipeline": "Audio -> Transcription -> Traduction",
            "GET /health": "Health check",
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "cuda": torch.cuda.is_available(),
        "whisper_model": WHISPER_MODEL,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Traduit du texte entre Mina et Français"""

    try:
        translator = MinaTranslator.get_instance()
        result = translator.translate(request.text, request.direction)

        return TranslateResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcrit un fichier audio en texte Mina"""

    try:
        audio_data = await file.read()

        stt = MinaSTT.get_instance()
        result = stt.transcribe(audio_data)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    """
    Pipeline complet: Audio -> Transcription -> Traduction
    Retourne le texte en Mina ET la traduction Française
    """

    try:
        # 1. Transcrire
        audio_data = await file.read()

        stt = MinaSTT.get_instance()
        transcription = stt.transcribe(audio_data)

        # 2. Traduire
        translator = MinaTranslator.get_instance()
        translation = translator.translate(transcription["text"], "mina_to_french")

        return {
            "audio_transcribed": transcription["text"],
            "transcription_time_ms": transcription["time_ms"],
            "french_translation": translation["translation"],
            "translation_time_ms": translation["time_ms"],
            "total_time_ms": transcription["time_ms"] + translation["time_ms"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("LANCEMENT DE MINA-TRANSLATOR API")
    print("=" * 60)
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)