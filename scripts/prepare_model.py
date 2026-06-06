"""
Script de préparation du modèle pour l'inférence.
Télécharge le modèle de base Qwen et fusionne avec les adaptateurs LoRA.

Usage: python scripts/prepare_model.py
"""
import os
import sys
from pathlib import Path

# Ajouter le parent au path pour pouvoir importer
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mina-translator"
FINAL_DIR = MODEL_DIR / "final"
MERGED_DIR = MODEL_DIR / "merged"
BASE_MODEL = "Qwen/Qwen1.5-0.5B"  # Modèle de base utilisé pour le fine-tuning

logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")


def download_and_merge():
    """Télécharge le modèle de base et fusionne avec les adaptateurs LoRA."""

    logger.info("=" * 60)
    logger.info("PRÉPARATION DU MODÈLE POUR L'INFÉRENCE")
    logger.info("=" * 60)

    # 1. Créer le répertoire de sortie
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Répertoire: {MERGED_DIR}")

    # 2. Vérifier si les adaptateurs LoRA existent
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required_files if not (FINAL_DIR / f).exists()]
    if missing:
        logger.error(f"Fichiers manquants dans {FINAL_DIR}: {missing}")
        logger.error("Lancez d'abord l'entraînement pour générer les adaptateurs LoRA.")
        return False

    # 3. Télécharger le modèle de base
    logger.info(f"\n[1/2] Téléchargement du modèle de base: {BASE_MODEL}")
    logger.info("Cela peut prendre 5-10 minutes (environ 1 Go à télécharger)...")

    try:
        from transformers import AutoTokenizer, AutoConfig
        import torch

        # Télécharger le tokenizer et la config
        logger.info("Téléchargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            cache_dir=str(PROJECT_ROOT / ".cache" / "hub")
        )
        tokenizer.save_pretrained(MERGED_DIR)
        logger.info("Tokenizer téléchargé ✓")

        # Télécharger la config
        logger.info("Téléchargement de la configuration...")
        config = AutoConfig.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            cache_dir=str(PROJECT_ROOT / ".cache" / "hub")
        )
        config.save_pretrained(MERGED_DIR)
        logger.info("Configuration téléchargée ✓")

        # Copier les fichiers de configuration supplémentaires
        import shutil
        for fname in ["added_tokens.json", "special_tokens_map.json",
                      "tokenizer_config.json", "tokenizer.json",
                      "merges.txt", "vocab.json"]:
            src = FINAL_DIR / fname
            if src.exists():
                shutil.copy(src, MERGED_DIR / fname)
                logger.info(f"Copié: {fname}")

        # Télécharger le modèle de base (non量化)
        logger.info("\nTéléchargement du modèle de base (pytorch_model.bin)...")

        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            cache_dir=str(PROJECT_ROOT / ".cache" / "hub")
        )
        base_model.save_pretrained(MERGED_DIR)
        logger.info("Modèle de base téléchargé ✓")

        # 4. Charger et fusionner les adaptateurs LoRA
        logger.info(f"\n[2/2] Fusion avec les adaptateurs LoRA...")

        from peft import PeftModel

        # Charger le modèle avec les adaptateurs
        model = PeftModel.from_pretrained(
            base_model,
            str(FINAL_DIR),
            device_map="cpu"  # Fusion sur CPU pour éviter les problèmes CUDA
        )

        # Fusionner les poids (merge + unload)
        logger.info("Fusion des poids LoRA avec le modèle de base...")
        merged_model = model.merge_and_unload()

        # Sauvegarder le modèle fusionné
        logger.info("Sauvegarde du modèle fusionné...")
        merged_model.save_pretrained(MERGED_DIR)

        logger.info("\n" + "=" * 60)
        logger.info("✅ MODÈLE PRÊT POUR L'INFÉRENCE!")
        logger.info("=" * 60)
        logger.info(f"\nRépertoire: {MERGED_DIR}")
        logger.info("\nLancez maintenant:")
        logger.info("  python scripts/inference_test.py")

        return True

    except ImportError as e:
        logger.error(f"Module manquant: {e}")
        logger.error("Assurez-vous d'activer l'environnement: venv_train\\Scripts\\activate")
        return False
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_merged_model():
    """Vérifie si le modèle fusionné existe."""
    required = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    return all((MERGED_DIR / f).exists() for f in required)


if __name__ == "__main__":
    if check_merged_model():
        logger.info("✅ Modèle fusionné déjà présent!")
        logger.info(f"Répertoire: {MERGED_DIR}")
    else:
        success = download_and_merge()
        sys.exit(0 if success else 1)