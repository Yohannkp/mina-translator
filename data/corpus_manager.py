"""
Corpus Manager - Gestion centralisée du corpus Mina
Coordonne collecte, annotation, augmentation et export
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger

# Configuration
from config.settings import get_settings
settings = get_settings()

# Imports des modules de collecte
from data.collectors.web_scraper import TogoleseScraper
from data.collectors.dataset_loader import DatasetLoader
from data.augmentation.augmenter import TextAugmenter, AudioAugmenter
from data.annotation.annotator import AnnotationManager


@dataclass
class CorpusStats:
    """Statistiques du corpus"""
    total_texts: int = 0
    mina_texts: int = 0
    french_texts: int = 0
    parallel_pairs: int = 0
    audio_files: int = 0
    validated: int = 0
    pending: int = 0
    augmented: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class CorpusManager:
    """
    Gestionnaire central du corpus Mina-Français

    Fonctionnalités:
    - Collecte de données depuis multiples sources
    - Annotation et validation
    - Augmentation de données
    - Export pour entraînement
    """

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir or settings.CORPUS_DIR
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

        # Sous-répertoires
        self.raw_dir = self.corpus_dir / "raw"
        self.processed_dir = self.corpus_dir / "processed"
        self.audio_dir = self.corpus_dir / "audio"
        self.annotations_dir = self.corpus_dir / "annotations"

        for d in [self.raw_dir, self.processed_dir, self.audio_dir, self.annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Métadonnées du corpus
        self.metadata_file = self.corpus_dir / "corpus_metadata.json"
        self.metadata = self._load_metadata()

        # Modules
        self.scraper = TogoleseScraper(self.raw_dir)
        self.dataset_loader = DatasetLoader(self.raw_dir)
        self.text_augmenter = TextAugmenter()
        self.audio_augmenter = AudioAugmenter()
        self.annotation_manager = AnnotationManager(self.annotations_dir)

        # Cache en mémoire
        self._cache = {}

        logger.info(f"CorpusManager initialisé: {self.corpus_dir}")

    def _load_metadata(self) -> Dict:
        """Charge les métadonnées"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "sources": [],
            "last_update": None,
            "version": "1.0.0",
        }

    def _save_metadata(self):
        """Sauvegarde les métadonnées"""
        self.metadata["last_update"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # === COLLECTE ===

    def collect_from_web(self, max_pages: int = 10) -> List[Dict]:
        """Collecte texte depuis sites togolais"""
        logger.info("Collecte depuis le web...")
        data = self.scraper.scrape_all()

        # Sauvegarder
        output_file = self.raw_dir / f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.metadata["sources"].append({
            "type": "web",
            "file": str(output_file),
            "count": len(data),
            "date": datetime.now().isoformat(),
        })
        self._save_metadata()

        return data

    def collect_from_huggingface(self, dataset_name: str = "ewe") -> List[Dict]:
        """Charge dataset depuis HuggingFace"""
        logger.info(f"Collecte depuis HuggingFace: {dataset_name}")

        if dataset_name == "ewe":
            data = self.dataset_loader.load_masakhane("ewe")
        else:
            data = []

        if data:
            output_file = self.raw_dir / f"hf_{dataset_name}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.metadata["sources"].append({
                "type": "huggingface",
                "dataset": dataset_name,
                "file": str(output_file),
                "count": len(data),
                "date": datetime.now().isoformat(),
            })
            self._save_metadata()

        return data

    def add_manual_entries(self, texts: List[Dict]):
        """Ajoute des entrées manuelles au corpus"""
        self.annotation_manager.add_batch(texts)
        logger.info(f"Ajouté {len(texts)} entrées manuelles")

    # === AUGMENTATION ===

    def augment_texts(
        self,
        texts: List[Dict],
        num_variations: int = 3,
    ) -> List[Dict]:
        """Augmente un ensemble de textes"""
        logger.info(f"Augmentation de {len(texts)} textes...")

        augmented = []
        for item in texts:
            text = item.get("text", "")
            if text:
                variations = self.text_augmenter.augment(text)
                for var in variations[:num_variations]:
                    augmented.append({
                        **item,
                        "text": var,
                        "augmented": True,
                        "original_id": item.get("id"),
                    })
                augmented.append(item)  # Garder original

        logger.info(f"Généré {len(augmented)} textes (augmenté)")

        # Sauvegarder
        output_file = self.processed_dir / f"augmented_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)

        return augmented

    # === EXPORT ===

    def export_for_stt(
        self,
        output_file: Path = None,
        include_audio: bool = True,
    ) -> Path:
        """
        Exporte les données pour fine-tuning STT (Whisper)

        Format attendu par Whisper fine-tuning:
        - audio_path: chemin vers fichier audio
        - text: transcription texte
        """
        if output_file is None:
            output_file = self.corpus_dir / "export" / "stt_training.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        training_data = []

        if include_audio:
            # Avec audio
            audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))
            for audio_file in audio_files:
                # Chercher transcription correspondante
                text_file = self.audio_dir / f"{audio_file.stem}.txt"
                if text_file.exists():
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    training_data.append({
                        "audio_path": str(audio_file),
                        "text": text,
                        "language": "mina",
                    })
        else:
            # Texte uniquement (pour validation)
            for entry in self.annotation_manager.validated:
                if entry.get("text") and entry.get("translation"):
                    training_data.append({
                        "source": entry["text"],
                        "target": entry["translation"],
                        "language": entry.get("lang", "mina"),
                    })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Export STT: {len(training_data)} exemples vers {output_file}")
        return output_file

    def export_for_translation(
        self,
        output_file: Path = None,
        min_quality: str = "medium",
    ) -> Path:
        """
        Exporte les données pour fine-tuning LLM (QLoRA)

        Format: JSONL avec paires FR-Mina
        """
        if output_file is None:
            output_file = self.corpus_dir / "export" / "translation_training.jsonl"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        parallel_data = self.annotation_manager.export_parallel_corpus(
            output_file,
            min_quality=min_quality,
        )

        return parallel_data

    def export_for_tts(
        self,
        output_file: Path = None,
    ) -> Path:
        """
        Exporte les données pour fine-tuning TTS

        Format: JSON avec alignements audio-texte
        """
        if output_file is None:
            output_file = self.corpus_dir / "export" / "tts_training.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        tts_data = []
        audio_files = list(self.audio_dir.glob("*.wav"))

        for audio_file in audio_files:
            text_file = self.audio_dir / f"{audio_file.stem}.txt"
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                tts_data.append({
                    "audio_path": str(audio_file),
                    "text": text,
                    "duration": self._get_audio_duration(audio_file),
                })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tts_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Export TTS: {len(tts_data)} exemples vers {output_file}")
        return output_file

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Retourne la durée d'un fichier audio"""
        try:
            import librosa
            duration = librosa.get_duration(filename=str(audio_path))
            return round(duration, 2)
        except Exception:
            return 0.0

    # === STATISTIQUES ===

    def get_stats(self) -> CorpusStats:
        """Calcule les statistiques du corpus"""
        stats = CorpusStats()

        # Fichiers texte
        text_files = list(self.raw_dir.glob("*.json")) + list(self.processed_dir.glob("*.json"))
        for f in text_files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    stats.total_texts += len(data)
            except Exception:
                pass

        # Fichiers audio
        audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))
        stats.audio_files = len(audio_files)

        # Annotations
        annot_stats = self.annotation_manager.get_statistics()
        stats.validated = annot_stats["validated"]
        stats.pending = annot_stats["pending"]

        return stats

    def print_stats(self):
        """Affiche les statistiques du corpus"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("STATISTIQUES DU CORPUS MINA")
        print("=" * 60)
        print(f"\nTextes:")
        print(f"   - Total: {stats.total_texts}")
        print(f"   - Mina: {stats.mina_texts}")
        print(f"   - Francais: {stats.french_texts}")
        print(f"   - Paires paralleles: {stats.parallel_pairs}")

        print(f"\nAudio:")
        print(f"   - Fichiers: {stats.audio_files}")

        print(f"\nValidation:")
        print(f"   - Valides: {stats.validated}")
        print(f"   - En attente: {stats.pending}")
        print(f"   - Augmentes: {stats.augmented}")

        print(f"\nSources: {len(self.metadata.get('sources', []))}")
        for source in self.metadata.get("sources", [])[-5:]:
            print(f"   - {source.get('type', 'unknown')}: {source.get('count', 0)} items")

        print("=" * 60)

    # === WORKFLOW COMPLET ===

    def initialize_corpus(self):
        """Initialise le corpus avec des données de base"""
        logger.info("Initialisation du corpus Mina...")

        # Phrases de base Mina
        base_mina = [
            {"text": "Mɛlɔ", "lang": "mina", "translation": "Bonjour", "source": "base"},
            {"text": "Ntsɛ nyabi?", "lang": "mina", "translation": "Comment allez-vous?", "source": "base"},
            {"text": "Kofi wɔ nunana fefefia", "lang": "mina", "translation": "Kofi habite à Lomé", "source": "base"},
            {"text": "Nyabi mɔ", "lang": "mina", "translation": "Au revoir", "source": "base"},
            {"text": "Mi sɔ na?", "lang": "mina", "translation": "Comment vous appelez-vous?", "source": "base"},
            {"text": "Mɛ sɔ Kofi", "lang": "mina", "translation": "Je m'appelle Kofi", "source": "base"},
            {"text": "Nyametsɛ", "lang": "mina", "translation": "Merci", "source": "base"},
            {"text": "Ai", "lang": "mina", "translation": "Oui", "source": "base"},
            {"text": "Ao", "lang": "mina", "translation": "Non", "source": "base"},
            {"text": "Duke", "lang": "mina", "translation": "Eau", "source": "base"},
            {"text": "Dzakpɔ", "lang": "mina", "translation": "Manger", "source": "base"},
            {"text": "Fiafla", "lang": "mina", "translation": "Jour", "source": "base"},
            {"text": "Tso", "lang": "mina", "translation": "Soleil", "source": "base"},
            {"text": "Fɔfɔ", "lang": "mina", "translation": "Père", "source": "base"},
            {"text": "Na", "lang": "mina", "translation": "Mère", "source": "base"},
        ]

        self.add_manual_entries(base_mina)

        # Collecter depuis Masakhane (Ewe)
        ewe_data = self.collect_from_huggingface("ewe")
        if ewe_data:
            logger.info(f"Collecté {len(ewe_data)} phrases Ewe depuis HuggingFace")

        self._save_metadata()
        self.print_stats()

        return True

    def full_pipeline(
        self,
        collect_web: bool = False,
        collect_hf: bool = True,
        augment: bool = True,
        export: bool = True,
    ):
        """
        Exécute le pipeline complet de constitution du corpus

        Args:
            collect_web: Collecter depuis le web
            collect_hf: Charger depuis HuggingFace
            augment: Appliquer augmentation
            export: Exporter les données
        """
        logger.info("Lancement pipeline complet")

        # 1. Collecte
        if collect_hf:
            self.collect_from_huggingface("ewe")

        if collect_web:
            self.collect_from_web()

        # 2. Augmentation (si assez de données)
        if augment:
            stats = self.get_stats()
            if stats.total_texts > 10:
                # Charger les textes existants
                all_texts = []
                for f in self.raw_dir.glob("*.json"):
                    with open(f, "r", encoding="utf-8") as fp:
                        all_texts.extend(json.load(fp))

                if all_texts:
                    self.augment_texts(all_texts)

        # 3. Export
        if export:
            logger.info("Export des données...")
            self.export_for_translation()
            self.export_for_stt(include_audio=False)
            self.export_for_tts()

        self.print_stats()
        logger.info("Pipeline termine")


# === CLI ===

def main():
    """Interface CLI pour le CorpusManager"""
    import sys

    manager = CorpusManager()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "init":
            manager.initialize_corpus()

        elif cmd == "stats":
            manager.print_stats()

        elif cmd == "collect-web":
            manager.collect_from_web()

        elif cmd == "collect-hf":
            manager.collect_from_huggingface("ewe")

        elif cmd == "augment":
            # Charger données existantes
            texts = []
            for f in manager.raw_dir.glob("*.json"):
                with open(f, "r", encoding="utf-8") as fp:
                    texts.extend(json.load(fp))
            manager.augment_texts(texts)

        elif cmd == "export":
            manager.export_for_translation()
            manager.export_for_stt(include_audio=False)

        elif cmd == "full":
            manager.full_pipeline(collect_web=False, collect_hf=True, augment=True, export=True)

        else:
            print(f"Commande inconnue: {cmd}")
            print("Commandes: init, stats, collect-web, collect-hf, augment, export, full")

    else:
        # Mode interactif
        print("\n" + "=" * 60)
        print("CORPUS MANAGER - MINA-FRANCAIS")
        print("=" * 60)
        manager.print_stats()
        print("\nCommandes: init, stats, collect-web, collect-hf, augment, export, full")


if __name__ == "__main__":
    main()