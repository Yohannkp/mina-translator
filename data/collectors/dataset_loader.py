"""
Loader pour datasets existants
Masakhane, Common Voice, et autres sources open-source
"""
import requests
from pathlib import Path
from typing import Optional, List, Dict
import zipfile
import tarfile
from loguru import logger
from datasets import load_dataset

# Configuration
from config.settings import get_settings
settings = get_settings()


class DatasetLoader:
    """Chargeur de datasets open-source pour langues africaines"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.CORPUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_masakhane(self, language: str = "ewe") -> List[Dict]:
        """
        Charge un dataset depuis Masakhane

        Args:
            language: Code langue (ewe, hausa, swahili, etc.)

        Returns:
            Liste de dictionnaires {text, lang, source}
        """
        datasets_info = {
            # Textes
            "ewe": "dodziraynard/ewe",
            "ewe_news": "VKAgbesi/Ewe_News_Dataset",
            # Bible (pour TTS)
            "ewe_bible": "abiyo27/BibleTTS_Ewe-Bible",
            "ewe_bible_tts": "worldboss/ewe_bible_v2_tts",
        }

        if language not in datasets_info:
            logger.warning(f"Dataset Masakhane non trouvé: {language}")
            return []

        dataset_name = datasets_info[language]
        logger.info(f"Chargement {dataset_name}")

        try:
            dataset = load_dataset(dataset_name)
            results = []

            for split in ["train", "validation", "test"]:
                if split in dataset:
                    for item in dataset[split]:
                        results.append({
                            "text": item.get("text", item.get("translation", "")),
                            "lang": language,
                            "source": f"masakhane/{dataset_name}",
                            "split": split,
                        })

            logger.info(f"Chargé {len(results)} exemples depuis {dataset_name}")
            return results

        except Exception as e:
            logger.error(f"Erreur chargement {dataset_name}: {e}")
            return []

    def load_common_voice(self, language: str = "ewe") -> List[Dict]:
        """
        Charge dataset Common Voice pour une langue

        Note: Common Voice n'a pas de dataset Mina spécifique,
        mais peut avoir Ewe ou d'autres langues ouest-africaines
        """
        try:
            logger.info(f"Chargement Common Voice {language}")
            dataset = load_dataset(
                "mozilla-foundation/common_voice",
                language,
                split="train",
                trust_remote_code=True,
            )

            results = []
            for item in dataset:
                results.append({
                    "audio_path": item.get("path", ""),
                    "text": item.get("sentence", ""),
                    "lang": language,
                    "source": "common_voice",
                    "duration": item.get("duration", 0),
                })

            logger.info(f"Chargé {len(results)} exemples Common Voice {language}")
            return results

        except Exception as e:
            logger.error(f"Erreur Common Voice {language}: {e}")
            return []

    def load_opus100(self, source: str = "en", target: str = "fr") -> List[Dict]:
        """
        Charge数据集 OPUS-100 pour traduction FR
        """
        try:
            logger.info(f"Chargement OPUS-100 {source}-{target}")
            dataset = load_dataset(
                "opus100",
                f"{source}-{target}",
                split="train",
            )

            results = []
            for item in dataset:
                results.append({
                    "text": item.get("translation", {}).get(target, ""),
                    "source_text": item.get("translation", {}).get(source, ""),
                    "lang": target,
                    "source": f"opus100/{source}-{target}",
                })

            logger.info(f"Chargé {len(results)} exemples OPUS-100")
            return results

        except Exception as e:
            logger.error(f"Erreur OPUS-100: {e}")
            return []

    def search_huggingface(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Recherche datasets sur HuggingFace

        Args:
            query: Terme de recherche
            max_results: Nombre max de résultats

        Returns:
            Liste de datasets disponibles
        """
        try:
            # Via l'API HF
            api_url = f"https://huggingface.co/api/datasets?search={query}"
            response = requests.get(api_url, timeout=30)
            data = response.json()

            results = []
            for item in data[:max_results]:
                results.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "downloads": item.get("downloads", 0),
                    "tags": item.get("tags", [])[:5],
                })

            return results

        except Exception as e:
            logger.error(f"Erreur recherche HF: {e}")
            return []

    def explore_available(self) -> Dict[str, List[Dict]]:
        """Explore les datasets africains disponibles"""
        logger.info("Exploration datasets africains...")

        results = {
            "ewe": self.load_masakhane("ewe"),
            "common_voice_ewe": self.load_common_voice("ewe"),
        }

        return results


def explore_datasets():
    """Explore et affiche les datasets disponibles"""
    loader = DatasetLoader()

    # Recherche pour langues africaines
    logger.info("Recherche datasets 'african language'...")
    results = loader.search_huggingface("african language text")

    print("\n=== Datasets Africains Disponibles ===")
    for r in results[:15]:
        print(f"- {r['name']}: {r['downloads']:,} téléchargements")

    return results


def load_all_sources() -> List[Dict]:
    """Charge toutes les sources disponibles"""
    loader = DatasetLoader()
    all_data = []

    # Masakhane Ewe (langue sœur du Mina)
    ewe_data = loader.load_masakhane("ewe")
    if ewe_data:
        all_data.extend(ewe_data)
        logger.info(f"Ewe: {len(ewe_data)} exemples")

    return all_data


if __name__ == "__main__":
    explore_datasets()