"""
Pipeline STT (Speech-to-Text) avec Whisper
Optimisé pour RTX 4060 8GB VRAM
"""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from typing import Optional
from loguru import logger


class STTPipeline:
    """Pipeline de reconnaissance vocale Mina-Français"""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        torch_dtype=None,
    ):
        self.model_name = model_name

        # Détection automatique du device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # dtype optimal selon le device
        if torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.float16  # fp16 sur GPU
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        logger.info(f"Chargement Whisper sur {self.device} ({self.torch_dtype})")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)

        # Languages supportés (ne pas passer device à get_prompt_ids)
        self.french_token = self.processor.get_prompt_ids("french")
        self.mina_token = self.processor.get_prompt_ids("mina")

    def transcribe(
        self,
        audio_input,
        language: str = "fr",
        task: str = "transcribe",
        do_sample: bool = False,
        num_beams: int = 5,
    ) -> dict:
        """
        Transcrit un audio en texte.

        Args:
            audio_input: Chemin fichier ou array numpy
            language: Code langue ("fr" ou "mina")
            task: "transcribe" ou "translate"
            do_sample: Utiliser sampling (vs greedy)
            num_beams: Nombre de beams pour beam search

        Returns:
            dict avec "text", "language", "segments"
        """
        # Traitement audio
        if isinstance(audio_input, str):
            import librosa
            audio, sr = librosa.load(audio_input, sr=16000)
        else:
            audio = audio_input
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Tokeniser avec langueforcée
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        # Langue cible
        forced_decoder_ids = None
        if language == "mina":
            # Mina - utilise tokenizer pour obtenir le token
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="mina", task=task
            )
        elif language == "fr":
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="french", task=task
            )

        # Génération
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                do_sample=do_sample,
                num_beams=num_beams,
                max_new_tokens=448,
            )

        # Décodage
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        logger.info(f"Transcription réussie: {len(transcription)} caractères")

        return {
            "text": transcription,
            "language": language,
            "task": task,
        }

    def transcribe_french(self, audio_input) -> dict:
        """Transcrit directement en français"""
        return self.transcribe(audio_input, language="fr")

    def transcribe_mina(self, audio_input) -> dict:
        """Transcrit directement en mina"""
        return self.transcribe(audio_input, language="mina")

    def translate_to_french(self, audio_input) -> dict:
        """Traduit l'audio en français"""
        return self.transcribe(audio_input, language="mina", task="translate")

    def get_vram_usage(self) -> dict:
        """Retourne l'utilisation VRAM actuelle"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
            }
        return {"allocated_gb": 0, "reserved_gb": 0}


def load_stt_pipeline(
    model_name: str = "openai/whisper-small",
) -> STTPipeline:
    """Factory function pour charger le pipeline STT"""
    return STTPipeline(model_name=model_name)


if __name__ == "__main__":
    # Test rapide
    logger.info("Test du pipeline STT")
    pipeline = load_stt_pipeline()

    # Afficher VRAM utilisée
    vram = pipeline.get_vram_usage()
    logger.info(f"VRAM utilisée: {vram}")

    logger.info("STT Pipeline prêt!")