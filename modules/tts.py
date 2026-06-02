"""
Module TTS (Text-to-Speech) pour Mina
=====================================

Synthèse vocale en Mina et Français
"""
import os
import sys
import base64
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
settings = get_settings()


class TextToSpeech:
    """
    Synthèse vocale avec Coqui TTS ou alternatives
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        """
        Initialise le modèle TTS

        Args:
            model_name: Modèle à utiliser
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Charge le modèle TTS"""
        if self.model is not None:
            return

        logger.info(f"Chargement TTS ({self.model_name})...")

        try:
            from TTS.api import TTS

            self.tts = TTS(model_name=self.model_name, gpu=self.device == "cuda")
            logger.info("TTS chargé!")

        except ImportError:
            logger.warning("Coqui TTS non installé!")
            logger.info("Instale avec: pip install TTS")
            logger.info("Alternative: pip install edge-tts")
            self.tts = None

    def synthesize(
        self,
        text: str,
        language: str = "fr",
        output_path: Optional[str] = None,
        speed: float = 1.0,
    ) -> dict:
        """
        Synthétise du texte en audio

        Args:
            text: Texte à synthétiser
            language: Langue (fr, en)
            output_path: Chemin du fichier de sortie
            speed: Vitesse de lecture (0.5 - 2.0)

        Returns:
            dict avec:
                - audio_path: chemin vers le fichier audio
                - duration: durée en secondes
        """
        if self.tts is None:
            self.load()

        if self.tts is None:
            raise RuntimeError("TTS non disponible. Installe Coqui TTS ou edge-tts.")

        if output_path is None:
            output_path = str(settings.TEMP_DIR / "output.wav")

        # Synthèse
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speed=speed,
        )

        # Calculer la durée (approximatif)
        duration = len(text) / (12 * speed)  # ~12 caractères/seconde

        return {
            "audio_path": output_path,
            "duration": duration,
        }

    def synthesize_to_base64(
        self,
        text: str,
        language: str = "fr",
        speed: float = 1.0,
    ) -> str:
        """
        Synthétise et retourne en base64

        Args:
            text: Texte à synthétiser
            language: Langue
            speed: Vitesse

        Returns:
            Audio encodé en base64
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        result = self.synthesize(text, language, temp_path, speed)

        # Lire et encoder
        with open(temp_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()

        # Nettoyer
        os.unlink(temp_path)

        return audio_data


class EdgeTTS:
    """
    Alternative utilisant Edge TTS (Microsoft)
    Plus léger, fonctionne sans GPU
    """

    VOICES = {
        "fr": "fr-FR-DeniseNeural",
        "mina": "fr-TG-Frontline",
        "en": "en-US-JennyNeural",
    }

    def __init__(self):
        self.voices = self.VOICES

    async def synthesize(
        self,
        text: str,
        language: str = "fr",
        output_path: Optional[str] = None,
        speed: float = 1.0,
    ) -> dict:
        """
        Synthétise du texte en audio avec Edge TTS

        Args:
            text: Texte à synthétiser
            language: Langue (fr, mina, en)
            output_path: Chemin du fichier de sortie
            speed: Vitesse de lecture

        Returns:
            dict avec chemin et durée
        """
        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts non installé!")
            logger.info("Instale avec: pip install edge-tts")
            raise

        if output_path is None:
            output_path = str(settings.TEMP_DIR / "output.mp3")

        # Mapper le langage
        voice = self.voices.get(language, self.voices["fr"])

        # Ajuster la vitesse
        rate = f"{int((speed - 1) * 100)}%"

        # Synthèse
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

        # Durée approximative
        duration = len(text) / (12 * speed)

        return {
            "audio_path": output_path,
            "duration": duration,
        }

    async def synthesize_to_base64(
        self,
        text: str,
        language: str = "fr",
        speed: float = 1.0,
    ) -> str:
        """Synthétise et retourne en base64"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        await self.synthesize(text, language, temp_path, speed)

        with open(temp_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()

        os.unlink(temp_path)

        return audio_data


# =============================================================================
# GESTIONNAIRE GLOBAL
# =============================================================================

_tts_model = None


def get_tts_model() -> TextToSpeech:
    """Obtient ou crée le modèle TTS global"""
    global _tts_model

    if _tts_model is None:
        _tts_model = TextToSpeech()

    return _tts_model


def get_edge_tts() -> EdgeTTS:
    """Obtient Edge TTS"""
    return EdgeTTS()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Synthèse vocale")
    parser.add_argument("--text", "-t", required=True, help="Texte à synthétiser")
    parser.add_argument("--output", "-o", help="Fichier de sortie")
    parser.add_argument("--lang", "-l", default="fr", help="Langue (fr, en)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Vitesse (0.5-2.0)")

    args = parser.parse_args()

    async def main():
        tts = get_edge_tts()
        result = await tts.synthesize(
            text=args.text,
            language=args.lang,
            output_path=args.output,
            speed=args.speed,
        )
        print(f"Audio sauvegardé: {result['audio_path']}")
        print(f"Durée: {result['duration']:.1f}s")

    asyncio.run(main())