"""
Data Augmentation pour corpus Mina
Techniques pour enrichir les données texte et audio
"""
import numpy as np
import random
from typing import List, Dict, Callable
from pathlib import Path
from loguru import logger
import librosa
import soundfile as sf
import os
import re

# Configuration
from config.settings import get_settings
settings = get_settings()


class TextAugmenter:
    """Augmentation de données texte"""

    # Mots Mina courants pour substitution
    MINA_COMMON_WORDS = [
        "mɛlɔ", "kofi", "nyabi", "wɔ", "nunana", "gbɔ",
        "fefia", "lomé", "togo", "anya", "ami", "ata",
    ]

    # Patterns pour transformations
    TRANSFORMATIONS = []

    def __init__(self):
        self.variations = []

    def backtranslate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Simule back-translation (nécessite LLM en réalité)

        Version simplifiée: retourne le texte original avec marqueur
        """
        # En prod: appeler LLM pour générer paraphrase
        logger.debug(f"Back-translation: {source_lang} -> {target_lang} -> {source_lang}")
        return text

    def synonym_replacement(self, text: str, n: int = 3) -> str:
        """Remplace n mots aléatoires par des synonymes"""
        words = text.split()
        if len(words) < n + 1:
            return text

        indices = random.sample(range(len(words)), min(n, len(words)))
        for idx in indices:
            # En prod: utiliser wordnet Mina ou embedding
            words[idx] = words[idx] + "*"  # Marque placeholder

        return " ".join(words)

    def random_insertion(self, text: str, n: int = 2) -> str:
        """Insère aléatoirement n mots"""
        words = text.split()
        filler_words = ["ɛ", "na", "o", "a"]
        for _ in range(n):
            if words:
                pos = random.randint(0, len(words))
                words.insert(pos, random.choice(filler_words))
        return " ".join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Supprime aléatoirement des mots avec probabilité p"""
        words = text.split()
        if len(words) < 3:
            return text

        kept = [w for w in words if random.random() > p]
        return " ".join(kept) if kept else text

    def random_swap(self, text: str, n: int = 2) -> str:
        """Échange aléatoirement n paires de mots"""
        words = text.split()
        if len(words) < 4:
            return text

        for _ in range(n):
            if len(words) >= 4:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def random_char_edit(self, text: str, p: float = 0.05) -> str:
        """Édit aléatoirement des caractères"""
        result = []
        for char in text:
            if char.isalnum() and random.random() < p:
                # Insertion, suppression ou substitution
                op = random.choice(["keep", "sub", "del", "ins"])
                if op == "keep":
                    result.append(char)
                elif op == "sub":
                    result.append(char.upper() if char.islower() else char.lower())
                elif op == "ins":
                    result.append(char + random.choice("aeiou"))
            else:
                result.append(char)
        return "".join(result)

    def augment(self, text: str, methods: List[str] = None) -> List[str]:
        """
        Applique plusieurs méthodes d'augmentation

        Args:
            text: Texte à augmenter
            methods: Liste de méthodes à appliquer
                     ["backtranslate", "synonym", "insert", "delete", "swap", "char"]

        Returns:
            Liste de variations
        """
        if methods is None:
            methods = ["synonym", "insert", "delete", "swap"]

        results = []
        for method in methods:
            try:
                if method == "backtranslate":
                    results.append(self.backtranslate(text, "fr", "mina"))
                elif method == "synonym":
                    results.append(self.synonym_replacement(text))
                elif method == "insert":
                    results.append(self.random_insertion(text))
                elif method == "delete":
                    results.append(self.random_deletion(text))
                elif method == "swap":
                    results.append(self.random_swap(text))
                elif method == "char":
                    results.append(self.random_char_edit(text))
            except Exception as e:
                logger.warning(f"Erreur {method}: {e}")

        return [t for t in results if t and t != text]


class AudioAugmenter:
    """Augmentation de données audio pour STT"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Charge un fichier audio"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio

    def save_audio(self, audio: np.ndarray, output_path: str):
        """Sauvegarde un audio"""
        sf.write(output_path, audio, self.sample_rate)

    def speed_perturbation(self, audio: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Modifie la vitesse de l'audio

        Args:
            audio: Signal audio
            factor: Facteur de vitesse (0.9 = plus lent, 1.1 = plus rapide)
        """
        return librosa.effects.time_stretch(audio, rate=factor)

    def pitch_shift(self, audio: np.ndarray, n_steps: int = 2) -> np.ndarray:
        """Décale le pitch de n demi-tons"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Ajoute du bruit gaussien"""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise

    def add_reverb(self, audio: np.ndarray, room_size: float = 0.5) -> np.ndarray:
        """Ajoute un effet de réverbération simplifié"""
        # Implémentation basique - en prod utiliser convolve avec IR
        echo = np.zeros_like(audio)
        delay = int(self.sample_rate * 0.05 * room_size)
        for i in range(delay, len(audio)):
            echo[i] = audio[i] + 0.3 * audio[i - delay]
        return echo

    def random_crop(self, audio: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """Découpe aléatoirement l'audio"""
        crop_length = int(len(audio) * crop_ratio)
        start = random.randint(0, max(0, len(audio) - crop_length))
        return audio[start:start + crop_length]

    def volume_perturbation(self, audio: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Modifie le volume"""
        return audio * factor

    def augment_audio(
        self,
        audio: np.ndarray,
        methods: List[str] = None,
    ) -> List[np.ndarray]:
        """
        Applique plusieurs augmentations

        Args:
            audio: Signal audio
            methods: ["speed", "pitch", "noise", "reverb", "crop", "volume"]

        Returns:
            Liste d'audios augmentés
        """
        if methods is None:
            methods = ["speed", "noise", "volume"]

        results = []

        for method in methods:
            try:
                if method == "speed":
                    # 3 variantes de vitesse
                    for factor in [0.9, 1.0, 1.1]:
                        results.append(self.speed_perturbation(audio, factor))
                elif method == "pitch":
                    for n_steps in [-2, 2]:
                        results.append(self.pitch_shift(audio, n_steps))
                elif method == "noise":
                    for level in [0.003, 0.005, 0.01]:
                        results.append(self.add_noise(audio, level))
                elif method == "reverb":
                    results.append(self.add_reverb(audio))
                elif method == "crop":
                    results.append(self.random_crop(audio))
                elif method == "volume":
                    for factor in [0.7, 0.85, 1.15]:
                        results.append(self.volume_perturbation(audio, factor))
            except Exception as e:
                logger.warning(f"Erreur augmentation {method}: {e}")

        return results


def augment_corpus(
    texts: List[Dict],
    audio_dir: Path = None,
    output_dir: Path = None,
    num_augmentations: int = 3,
) -> Dict:
    """
    Augmente un corpus complet

    Args:
        texts: Liste de textes {text, lang, source}
        audio_dir: Répertoire audios (optionnel)
        output_dir: Répertoire sortie
        num_augmentations: Nombre d'augmentations par exemple

    Returns:
        Statistiques d'augmentation
    """
    text_augmenter = TextAugmenter()
    audio_augmenter = AudioAugmenter()

    output_dir = output_dir or settings.CORPUS_DIR / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "original_texts": len(texts),
        "augmented_texts": 0,
        "original_audio": 0,
        "augmented_audio": 0,
    }

    augmented_texts = []

    # Augmentation texte
    logger.info("Augmentation textes...")
    for item in texts:
        text = item.get("text", "")
        if text:
            variations = text_augmenter.augment(text)
            for var in variations[:num_augmentations]:
                augmented_texts.append({
                    **item,
                    "text": var,
                    "augmented": True,
                    "original": text,
                })
            augmented_texts.append(item)  # Garder original

    stats["augmented_texts"] = len(augmented_texts)
    logger.info(f"Textes augmentés: {len(augmented_texts)} ({len(texts)} originaux)")

    # Augmentation audio
    if audio_dir and audio_dir.exists():
        logger.info("Augmentation audios...")
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
        stats["original_audio"] = len(audio_files)

        for audio_file in audio_files:
            try:
                audio = audio_augmenter.load_audio(str(audio_file))
                variations = audio_augmenter.augment_audio(audio)

                for i, var_audio in enumerate(variations):
                    output_path = output_dir / f"{audio_file.stem}_aug{i}{audio_file.suffix}"
                    audio_augmenter.save_audio(var_audio, str(output_path))
                    stats["augmented_audio"] += 1

            except Exception as e:
                logger.warning(f"Erreur {audio_file}: {e}")

        logger.info(f"Audios augmentés: {stats['augmented_audio']} ({stats['original_audio']} originaux)")

    return stats


if __name__ == "__main__":
    # Test rapide
    aug = TextAugmenter()
    test_text = "Mɛlɔ, ntsɛ nyabi? Kofi wɔ nunana fefefia."

    variations = aug.augment(test_text)
    print(f"Original: {test_text}")
    print(f"Variations: {len(variations)}")
    for v in variations:
        print(f"  - {v}")