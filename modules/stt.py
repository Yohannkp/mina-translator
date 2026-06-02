"""
Module STT (Speech-to-Text) avec Whisper
========================================

Utilise OpenAI Whisper pour la transcription audio Mina/Français
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
settings = get_settings()


class SpeechToText:
    """
    Transcription audio avec Whisper
    """

    def __init__(self, model_name: str = "base"):
        """
        Initialise le modèle Whisper

        Args:
            model_name: Size du modèle (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Charge le modèle Whisper"""
        if self.model is not None:
            return

        logger.info(f"Chargement Whisper ({self.model_name})...")

        try:
            import whisper

            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper chargé sur {self.device}")

        except ImportError:
            logger.error("Whisper non installé!")
            logger.info("Instale avec: pip install openai-whisper")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        """
        Transcrit un fichier audio

        Args:
            audio_path: Chemin vers le fichier audio
            language: Code langue (fr, en, ou None pour auto-detect)
            task: "transcribe" ou "translate"

        Returns:
            dict avec:
                - text: texte transcrit
                - language: langue détectée
                - segments: segments avec timestamps
        """
        if self.model is None:
            self.load()

        import whisper

        # Options de transcription
        options = {
            "task": task,
            "verbose": False,
        }

        if language:
            options["language"] = language

        # Transcription
        result = self.model.transcribe(audio_path, **options)

        return {
            "text": result["text"],
            "language": result.get("language", language or "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in result.get("segments", [])
            ],
        }

    def transcribe_microphone(self, duration: int = 5, language: Optional[str] = None) -> str:
        """
        Transcrit depuis le microphone

        Args:
            duration: Durée d'enregistrement en secondes
            language: Langue目标 (fr, mina)

        Returns:
            Texte transcrit
        """
        try:
            import pyaudio
            import whisper
        except ImportError:
            logger.error("pyaudio et/ou whisper non installés!")
            return ""

        # Configuration audio
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        audio = pyaudio.PyAudio()

        # Enregistrement
        logger.info(f"Enregistrement pendant {duration}s...")
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        frames = []
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Conversion en numpy array
        audio_np = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0

        # Transcription
        if self.model is None:
            self.load()

        options = {}
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_np, **options)
        return result["text"]


# =============================================================================
# GESTIONNAIRE GLOBAL
# =============================================================================

_stt_model = None


def get_stt_model(model_name: str = "base") -> SpeechToText:
    """Obtient ou crée le modèle STT global"""
    global _stt_model

    if _stt_model is None:
        _stt_model = SpeechToText(model_name)

    return _stt_model


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def download_model(model_name: str = "base", cache_dir: Optional[str] = None):
    """
    Télécharge un modèle Whisper

    Args:
        model_name: Nom du modèle (tiny, base, small, medium, large)
        cache_dir: Dossier de cache (défaut: ~/.cache/whisper)
    """
    import whisper

    if cache_dir:
        os.environ["WHISPER_CACHE_DIR"] = cache_dir

    logger.info(f"Téléchargement du modèle {model_name}...")
    model = whisper.load_model(model_name)
    logger.info("Modèle téléchargé!")

    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcription Whisper")
    parser.add_argument("--audio", "-a", help="Fichier audio à transcrire")
    parser.add_argument("--mic", "-m", action="store_true", help="Utiliser le microphone")
    parser.add_argument("--model", "-M", default="base", help="Modèle Whisper (tiny/base/small/medium/large)")
    parser.add_argument("--language", "-l", default="fr", help="Langue (fr, en)")

    args = parser.parse_args()

    stt = SpeechToText(model_name=args.model)

    if args.audio:
        # Transcription depuis fichier
        result = stt.transcribe(args.audio, language=args.language)
        print(f"\nTexte transcrit:\n{result['text']}")
        print(f"\nLangue: {result['language']}")

    elif args.mic:
        # Transcription depuis microphone
        print("Enregistrement... Parle maintenant!")
        text = stt.transcribe_microphone(duration=5, language=args.language)
        print(f"\nTu as dit: {text}")

    else:
        print("Spécifie --audio <fichier> ou --mic pour le microphone")