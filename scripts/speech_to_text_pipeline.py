"""
scripts/speech_to_text_pipeline.py - Pipeline complet Audio → Mina → Français
=============================================================================

Pipeline permettant de:
1. Charger un fichier audio (MP3, WAV, OGG, M4A)
2. Transcrire en Mina avec Whisper (fine-tuné si disponible)
3. Traduire en Français avec le modèle Mina-Translator
4. Retourner le texte Mina et la traduction Française

Usage:
    # Transcription + traduction d'un fichier audio
    python scripts/speech_to_text_pipeline.py --audio "mon_audio.mp3"

    # Mode batch - traiter plusieurs fichiers
    python scripts/speech_to_text_pipeline.py --batch --folder "data/corpus/audio"

    # Avec modèle Whisper fine-tuné
    python scripts/speech_to_text_pipeline.py --audio "audio.m4a" --whisper-model "models/whisper-mina"

    # Test avec les données Common Voice
    python scripts/speech_to_text_pipeline.py --test-common-voice --num-samples 10

Auteur: Claude Opus 4.8
Date: 2026-06-03
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

# Audio processing
import torch
import numpy as np

# Speech Recognition
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: openai-whisper not installed. Run: pip install openai-whisper")

# Transcription avec HF Transformers
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Logging
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
COMMON_VOICE_PATH = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09" / "gej"
WHISPER_MODEL_DIR = PROJECT_ROOT / "models" / "whisper-mina"
MINA_TRANSLATOR_PATH = PROJECT_ROOT / "models" / "mina-translator"

# Paramètres audio
AUDIO_CONFIG = {
    "sample_rate": 16000,  # Whisper attend 16kHz
    "chunk_duration": 30,  # Seconds - Whisper limite
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TranscriptionResult:
    """Résultat de transcription audio → texte"""
    original_audio: str
    mina_text: str
    french_translation: str
    confidence: float
    audio_duration_ms: float
    transcription_time_ms: float
    translation_time_ms: float
    stt_model: str
    llm_model: str


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False):
    """Configure le logging"""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    return logger


# =============================================================================
# AUDIO LOADING
# =============================================================================

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Charge un fichier audio et le convertit au format Whisper

    Args:
        audio_path: Chemin vers le fichier audio
        target_sr: Taux d'échantillonnage cible (défaut: 16kHz)

    Returns:
        Tuple (audio_array, duration_seconds)
    """
    try:
        import librosa
    except ImportError:
        logger.error("librosa required: pip install librosa")
        raise

    logger.debug(f"Loading audio: {audio_path}")

    # Charger avec librosa (gère MP3, WAV, OGG, etc.)
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Normaliser
    audio = librosa.util.normalize(audio, norm=np.inf)

    duration = len(audio) / sr

    logger.debug(f"Audio loaded: {duration:.2f}s, {sr}Hz, shape: {audio.shape}")

    return audio, duration


def load_audio_with_soundfile(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Alternative avec soundfile (plus rapide pour certains formats)
    """
    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile required: pip install soundfile")
        raise

    logger.debug(f"Loading audio with soundfile: {audio_path}")

    # Charger le fichier
    audio, sr = sf.read(audio_path, dtype='float32')

    # Convertir en mono si stérééo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resampler si nécessaire
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normaliser
    audio = audio / np.max(np.abs(audio))

    duration = len(audio) / sr

    logger.debug(f"Audio loaded: {duration:.2f}s, {sr}Hz")

    return audio, duration


# =============================================================================
# SPEECH TO TEXT - WHISPER
# =============================================================================

class MinaSpeechRecognizer:
    """
    Reconnaissance vocale pour le Mina utilisant Whisper
    Supporte le modèle de base ou un modèle fine-tuné
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = None,
        compute_type: str = "float16",
    ):
        """
        Initialise le reconnaisseur vocal

        Args:
            model_name: "tiny", "base", "small", "medium", "large" ou chemin vers modèle local
            device: "cuda", "cpu", ou None pour auto-détection
            compute_type: "float16", "float32", "int8"
        """
        self.logger = setup_logging()

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load Whisper model
        self._load_model(model_name, compute_type)

    def _load_model(self, model_name: str, compute_type: str):
        """Charge le modèle Whisper"""

        # Vérifier si c'est un modèle local (fine-tuné)
        local_model_path = Path(model_name)
        if local_model_path.exists():
            self.logger.info(f"Loading local Whisper model: {local_model_path}")
            model_path = local_model_path
        else:
            self.logger.info(f"Loading Whisper '{model_name}' from HuggingFace")
            model_path = model_name

        # Charger avec whisper (API native)
        if WHISPER_AVAILABLE and not local_model_path.exists():
            self.logger.info(f"Using OpenAI Whisper: {model_name}")
            self.model = whisper.load_model(model_name, device=self.device)
            self.use_hf = False
        else:
            # Utiliser HF Transformers pour modèle local ou fine-tuné
            self.logger.info(f"Using HuggingFace Transformers for Whisper")
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

            if self.device == "cuda":
                self.model = self.model.cuda()

            self.model.eval()
            self.use_hf = True

        self.logger.info(f"Whisper model loaded: {model_name}")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = None,  # None = auto-detection
        task: str = "transcribe",
        condition_on_previous_text: bool = True,
    ) -> Dict:
        """
        Transcrit l'audio en texte

        Args:
            audio: Tableau numpy d'audio (16kHz mono)
            language: Code langue ("ewe", "fr", "en")
            task: "transcribe" ou "translate"
            condition_on_previous_text: Pour la continuité

        Returns:
            Dict avec "text", "language", "segments"
        """
        start_time = time.time()

        if self.use_hf:
            # HF Transformers pipeline
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Générer
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language,
                task=task
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=448,
                )

            # Décoder
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        else:
            # Determine language for Whisper
            whisper_lang = language if language else None
            whisper_task = "transcribe" if whisper_lang else "translate"  # Auto-detect → translate to English

            # OpenAI Whisper API
            result = self.model.transcribe(
                audio,
                language=whisper_lang,
                task=whisper_task,
                condition_on_previous_text=condition_on_previous_text,
            )
            transcription = result["text"]

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.debug(f"Transcription done in {elapsed_ms:.0f}ms: {transcription[:50]}...")

        return {
            "text": transcription.strip(),
            "language": language,
            "transcription_time_ms": elapsed_ms,
        }

    def transcribe_file(self, audio_path: str, **kwargs) -> Dict:
        """Transcrit un fichier audio"""
        audio, duration = load_audio_with_soundfile(audio_path)
        result = self.transcribe(audio, **kwargs)
        result["duration_s"] = duration
        result["original_file"] = audio_path
        return result


# =============================================================================
# MINA TO FRENCH TRANSLATION
# =============================================================================

class MinaTranslator:
    """
    Traducteur Mina → Français utilisant le modèle fine-tuné
    """

    SYSTEM_PROMPT = """Tu es un assistant de traduction professionnel.
Tu dois traduire EXACTEMENT et UNIQUEMENT la phrase suivante du Mina (Ewé) vers le Français.
Ne добавляй AUCUN commentaire, excuse ou hésitation.
Réponds UNIQUEMENT avec la traduction, rien d'autre.

Phrase en Mina: """

    def __init__(
        self,
        model_path: Path = MINA_TRANSLATOR_PATH,
        device: str = None,
    ):
        """
        Initialise le traducteur

        Args:
            model_path: Chemin vers le modèle Mina-Translator
            device: "cuda", "cpu", ou None
        """
        self.logger = setup_logging()

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Charge le modèle de traduction"""

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        # Modèle de base (doit correspondre au fine-tuning)
        base_model_id = "Qwen/Qwen2-0.5B-Instruct"

        self.logger.info(f"Loading Mina translator from: {model_path}")

        # Chercher les adaptateurs LoRA
        has_lora = any(
            (model_path / d / "adapter_config.json").exists()
            for d in model_path.iterdir()
            if d.is_dir()
        )

        if has_lora:
            self.logger.info("Loading base model + LoRA adapters")

            # Charger base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            # Trouver le dernier checkpoint
            checkpoints = [
                d for d in model_path.iterdir()
                if d.is_dir() and "checkpoint" in d.name
            ]

            # Extraire le numéro de checkpoint (gère "checkpoint-100", "checkpoint-epoch-1", etc.)
            def get_checkpoint_number(x):
                name = x.name
                for part in name.split("-"):
                    try:
                        return int(part)
                    except ValueError:
                        continue
                return 0

            checkpoints.sort(key=get_checkpoint_number)

            if checkpoints:
                adapter_path = checkpoints[-1]
            else:
                adapter_path = model_path

            # Charger LoRA
            self.model = PeftModel.from_pretrained(
                self.base_model,
                str(adapter_path),
            )
            self.model.eval()

        else:
            # Modèle fusionné
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model.eval()

        # Charger tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Mina translator loaded")

    def translate(self, mina_text: str) -> Tuple[str, float]:
        """
        Traduit du Mina vers le Français

        Args:
            mina_text: Texte en Mina à traduire

        Returns:
            Tuple (french_translation, time_ms)
        """
        start_time = time.time()

        # Créer le prompt
        prompt = self.SYSTEM_PROMPT + mina_text.strip()

        # Tokeniser
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Générer avec paramètres optimisés pour traduction
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=128,
                temperature=0.1,  # Basse température = cohérent
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Décoder
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraire la traduction (après le prompt)
        translation = response[len(prompt):].strip()

        # Nettoyer
        translation = translation.split("\n")[0].strip()

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.debug(f"Translation done in {elapsed_ms:.0f}ms: {translation[:50]}...")

        return translation, elapsed_ms


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

class SpeechToFrenchPipeline:
    """
    Pipeline complet: Audio → Transcription Mina → Traduction Français
    """

    def __init__(
        self,
        whisper_model: str = "base",
        mina_model_path: Path = MINA_TRANSLATOR_PATH,
    ):
        """
        Initialise le pipeline complet

        Args:
            whisper_model: Modèle Whisper à utiliser
            mina_model_path: Chemin vers le modèle Mina-Translator
        """
        self.logger = setup_logging()

        self.logger.info("=" * 60)
        self.logger.info("INITIALISING SPEECH-TO-FRENCH PIPELINE")
        self.logger.info("=" * 60)

        # Initialiser STT
        self.stt = MinaSpeechRecognizer(model_name=whisper_model)

        # Initialiser traduction
        self.translator = MinaTranslator(model_path=mina_model_path)

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE READY")
        self.logger.info("=" * 60)

    def process(
        self,
        audio_path: str,
        language: str = "ewe",
    ) -> TranscriptionResult:
        """
        Traite un fichier audio complet

        Args:
            audio_path: Chemin vers le fichier audio
            language: Langue de reconnaissance

        Returns:
            TranscriptionResult avec Mina et Français
        """
        self.logger.info(f"Processing: {audio_path}")

        # Étape 1: Transcription Mina
        stt_result = self.stt.transcribe_file(audio_path, language=language)
        mina_text = stt_result["text"]
        stt_time = stt_result["transcription_time_ms"]
        duration = stt_result["duration_s"]

        self.logger.info(f"  Mina: {mina_text[:60]}...")

        # Étape 2: Traduction Français
        if mina_text.strip():
            french_text, translate_time = self.translator.translate(mina_text)
        else:
            french_text = "[Pas de transcription]"
            translate_time = 0

        self.logger.info(f"  Francais: {french_text[:60]}...")

        return TranscriptionResult(
            original_audio=audio_path,
            mina_text=mina_text,
            french_translation=french_text,
            confidence=1.0,  # Whisper ne fournit pas de confiance directe
            audio_duration_ms=duration * 1000,
            transcription_time_ms=stt_time,
            translation_time_ms=translate_time,
            stt_model=self.stt.model.__class__.__name__,
            llm_model="Qwen2-0.5B-Mina",
        )

    def process_batch(
        self,
        audio_files: List[str],
        language: str = "ewe",
    ) -> List[TranscriptionResult]:
        """Traite plusieurs fichiers audio"""
        results = []

        for i, audio_path in enumerate(audio_files, 1):
            self.logger.info(f"[{i}/{len(audio_files)}] Processing: {audio_path}")
            try:
                result = self.process(audio_path, language)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {audio_path}: {e}")

        return results


# =============================================================================
# TEST WITH COMMON VOICE
# =============================================================================

def test_with_common_voice(num_samples: int = 10):
    """
    Test le pipeline avec des fichiers Common Voice Ewe

    Args:
        num_samples: Nombre d'échantillons à tester
    """
    logger.info("=" * 60)
    logger.info(f"TESTING WITH COMMON VOICE (n={num_samples})")
    logger.info("=" * 60)

    # Lire le fichier validated.tsv
    clips_dir = COMMON_VOICE_PATH / "clips"
    validated_file = COMMON_VOICE_PATH / "validated.tsv"

    if not validated_file.exists():
        logger.error(f"Validated file not found: {validated_file}")
        return

    # Charger les métadonnées
    samples = []
    with open(validated_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines[:num_samples]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                samples.append({
                    "path": parts[1],  # path
                    "sentence": parts[3],  # sentence
                })

    logger.info(f"Found {len(samples)} samples to test")

    # Initialiser le pipeline
    pipeline = SpeechToFrenchPipeline()

    # Tester chaque sample
    correct = 0
    total_mina_length = 0

    for i, sample in enumerate(samples, 1):
        audio_path = clips_dir / sample["path"]
        expected_mina = sample["sentence"]

        if not audio_path.exists():
            logger.warning(f"File not found: {audio_path}")
            continue

        logger.info(f"\n--- Sample {i}/{len(samples)} ---")
        logger.info(f"Expected Mina: {expected_mina}")

        try:
            # Traiter
            result = pipeline.process(str(audio_path), language=None)

            # Afficher résultats
            logger.info(f"Transcribed Mina: {result.mina_text}")
            logger.info(f"French:         {result.french_translation}")
            logger.info(f"STT time:       {result.transcription_time_ms:.0f}ms")
            logger.info(f"Translate time: {result.translation_time_ms:.0f}ms")

            # Calculer similarité (simple)
            if result.mina_text.lower() == expected_mina.lower():
                correct += 1
                logger.info("[OK] Exact match!")
            else:
                # Similarité simple par mots
                expected_words = set(expected_mina.lower().split())
                transcribed_words = set(result.mina_text.lower().split())
                overlap = len(expected_words & transcribed_words)
                similarity = overlap / max(len(expected_words), 1)
                logger.info(f"[~] Similarity: {similarity:.1%}")

            total_mina_length += len(result.mina_text)

        except Exception as e:
            logger.error(f"Error: {e}")

    # Résumé
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Samples tested: {len(samples)}")
    logger.info(f"Exact matches: {correct} ({100*correct/len(samples):.1f}%)")
    logger.info(f"Avg Mina length: {total_mina_length/len(samples):.0f} chars")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Speech to French Pipeline")

    # Mode de fonctionnement
    parser.add_argument("--audio", type=str, help="Fichier audio à traiter")
    parser.add_argument("--batch", action="store_true", help="Mode batch")
    parser.add_argument("--folder", type=str, help="Dossier contenant les audios")
    parser.add_argument("--test-common-voice", action="store_true", help="Test avec Common Voice")

    # Options
    parser.add_argument("--num-samples", type=int, default=10, help="Nombre d'échantillons pour test")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Modèle Whisper: tiny/base/small/medium/large")
    parser.add_argument("--output", type=str, help="Fichier de sortie JSON")
    parser.add_argument("--verbose", action="store_true", help="Mode détaillé")

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.test_common_voice:
        test_with_common_voice(args.num_samples)

    elif args.audio:
        # Traiter un fichier
        pipeline = SpeechToFrenchPipeline(whisper_model=args.whisper_model)
        result = pipeline.process(args.audio)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Audio:        {result.original_audio}")
        print(f"Duration:     {result.audio_duration_ms/1000:.2f}s")
        print(f"Mina:         {result.mina_text}")
        print(f"Francais:     {result.french_translation}")
        print(f"STT time:     {result.transcription_time_ms:.0f}ms")
        print(f"Translate:    {result.translation_time_ms:.0f}ms")

        # Sauvegarder si demandé
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "original_audio": result.original_audio,
                    "mina_text": result.mina_text,
                    "french_translation": result.french_translation,
                    "audio_duration_ms": result.audio_duration_ms,
                    "transcription_time_ms": result.transcription_time_ms,
                    "translation_time_ms": result.translation_time_ms,
                }, f, ensure_ascii=False, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.batch and args.folder:
        # Traiter un dossier
        folder = Path(args.folder)
        audio_files = list(folder.glob("*.mp3")) + \
                     list(folder.glob("*.wav")) + \
                     list(folder.glob("*.m4a")) + \
                     list(folder.glob("*.ogg"))

        print(f"Found {len(audio_files)} audio files")

        pipeline = SpeechToFrenchPipeline(whisper_model=args.whisper_model)
        results = pipeline.process_batch([str(f) for f in audio_files])

        # Sauvegarder
        if args.output:
            output_data = [
                {
                    "original_audio": r.original_audio,
                    "mina_text": r.mina_text,
                    "french_translation": r.french_translation,
                }
                for r in results
            ]
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nSaved {len(results)} results to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()