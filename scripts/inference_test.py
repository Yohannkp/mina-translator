"""
scripts/inference_test.py - Test d'inférence du modèle Mina-Translator
======================================================================

Script de test pour vérifier les performances d'inférence du modèle fine-tuné.
Charge le modèle depuis models/mina-translator/ et teste la traduction
de phrases inédites en français vers le Mina.

Optimisé pour RTX 4060 (8 Go VRAM) avec température basse pour éviter les hallucinations.

Usage:
    python scripts/inference_test.py [--model-path PATH] [--verbose]

Exemples:
    # Test par défaut (5 phrases de base)
    python scripts/inference_test.py

    # Test avec phrases personnalisées
    python scripts/inference_test.py --phrases "Bonjour comment allez-vous" "Je veux manger" "Où est la gare"

    # Mode verbose (afficher plus de détails)
    python scripts/inference_test.py --verbose

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# PyTorch
import torch

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Logging
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mina-translator"
OUTPUT_RESULTS = PROJECT_ROOT / "logs" / "inference_results.json"

# Paramètres d'inférence optimisés
INFERENCE_CONFIG = {
    "temperature": 0.3,          # Un peu plus haut pour diversité
    "top_p": 0.95,               # Nucleus sampling
    "top_k": 50,                 # Top-k sampling
    "max_new_tokens": 128,       # Longueur max de la réponse (Mina phrases courtes)
    "repetition_penalty": 1.2,   # Éviter les répétitions
    "do_sample": True,           # Sampling au lieu de greedy
    "use_cache": True,           # Utiliser le cache KV
}

# Phrases de test par défaut (français -> Mina)
DEFAULT_TEST_PHRASES = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Je voudrais un rendez-vous à l'hôpital demain matin.",
    "Où se trouve la pharmacie la plus proche ?",
    "Je dois payer ma facture d'électricité.",
    "Mon enfant est malade, j'ai besoin d'un médecin.",
]


# =============================================================================
# LOGGING
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
# CHARGEMENT DU MODÈLE
# =============================================================================

def load_model(model_path: Path, device: str = "cuda"):
    """
    Charge le modèle fine-tuné avec optimisation pour l'inférence

    Args:
        model_path: Chemin vers le dossier du modèle
        device: Dispositif (cuda/cpu)

    Returns:
        Tuple (model, tokenizer, processor)
    """
    logger.info("=" * 60)
    logger.info("CHARGEMENT DU MODÈLE")
    logger.info("=" * 60)
    logger.info(f"Chemin: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Le modèle n'existe pas: {model_path}")

    # Vérifier CUDA
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA non disponible, utilisation du CPU")
        device = "cpu"

    logger.info(f"Dispositif: {device}")

    # =============================================================================
    # DÉTECTION DU FORMAT DU MODÈLE
    # =============================================================================
    # Le modèle peut être:
    # 1. Format merged: config.json + pytorch_model.bin dans model_path/
    # 2. Format LoRA: adapter_config.json + adapter_model.safetensors dans model_path/
    #    (nécessite le modèle de base Qwen/Qwen1.5-0.5B)

    has_merged_model = (model_path / "config.json").exists() and (
        (model_path / "pytorch_model.bin").exists() or
        (model_path / "model.safetensors").exists()
    )
    has_lora_adapters = (model_path / "adapter_config.json").exists()

    base_model_id = "Qwen/Qwen2-0.5B-Instruct"

    if has_merged_model:
        logger.info("Format détecté: Modèle fusionné (merged)")
        model_dir = model_path
    elif has_lora_adapters:
        logger.info("Format détecté: Adaptateurs LoRA")
        logger.info(f"Modèle de base requis: {base_model_id}")
        logger.info("Téléchargement automatique du modèle de base...")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        # Télécharger et charger le modèle de base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            device_map="cuda",  # Force GPU
        )
        logger.info(f"Modèle de base chargé: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params")

        # Charger les adaptateurs LoRA
        logger.info("Chargement des adaptateurs LoRA...")
        model = PeftModel.from_pretrained(
            base_model,
            str(model_path),
            device_map="auto",
        )
        model.eval()
        logger.info("Adaptateurs LoRA fusionnés ✓")

        # Charger le tokenizer (du dossier LoRA)
        logger.info("Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("=" * 60)
        logger.info("MODÈLE PRÊT (base + LoRA)")
        logger.info("=" * 60)

        return model, tokenizer, None
    else:
        # Essayer de télécharger le modèle de base
        logger.warning("Format de modèle non reconnu.")
        logger.warning(f"Téléchargement automatique de {base_model_id}...")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                device_map="auto",
            )

            # Chercher les adaptateurs dans les sous-dossiers
            for subdir in model_path.iterdir():
                if subdir.is_dir() and (subdir / "adapter_config.json").exists():
                    logger.info(f"Adaptateurs trouvés: {subdir.name}")
                    model = PeftModel.from_pretrained(base_model, str(subdir))
                    break
            else:
                raise FileNotFoundError("Aucun adaptateur LoRA trouvé")

            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("=" * 60)
            logger.info("MODÈLE PRÊT")
            logger.info("=" * 60)

            return model, tokenizer, None

        except Exception as e:
            logger.error(f"Impossible de charger le modèle: {e}")
            logger.error("\nOptions:")
            logger.error("1. Lancez: python scripts/prepare_model.py")
            logger.error("2. Ou: python scripts/train_mini_llm.py --steps 10")
            raise

    # Charger le tokenizer
    logger.info("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Configure padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Charger le modèle
    logger.info("Chargement du modèle...")

    # Configuration pour 8 Go VRAM
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Si GPU disponible, utiliser quantization 4-bit ou FP16
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"VRAM disponible: {gpu_memory:.1f} Go")

        if gpu_memory >= 6:
            # Utiliser FP16 pour de meilleures performances
            load_kwargs["torch_dtype"] = torch.float16
            logger.info("Mode: FP16 (meilleure qualité)")
        else:
            # Fallback vers FP32
            load_kwargs["torch_dtype"] = torch.float32
            logger.warning("Mode: FP32 (qualité réduite)")

    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )

    model.eval()  # Mode évaluation

    logger.info(f"Modèle chargé: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M paramètres")

    # Charger le processor (si disponible)
    processor = None
    processor_path = model_path / "processor"
    if processor_path.exists():
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            logger.info("Processor chargé")
        except Exception as e:
            logger.warning(f"Processor non chargé: {e}")

    logger.info("=" * 60)
    logger.info("MODÈLE PRÊT")
    logger.info("=" * 60)

    return model, tokenizer, processor


# =============================================================================
# FONCTIONS DE TRADUCTION
# =============================================================================

def create_translation_prompt(text: str, source_lang: str = "français", target_lang: str = "mina") -> str:
    """
    Crée le prompt pour la traduction (format identique à l'entraînement)

    Args:
        text: Texte à traduire
        source_lang: Langue source
        target_lang: Langue cible

    Returns:
        Prompt formaté
    """
    if target_lang == "mina" or target_lang == "ewe":
        # Français -> Mina
        prompt = f"<|im_start|>user\nTraduis en Ewe: {text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Mina -> Français
        prompt = f"<|im_start|>user\nTraduis en Français: {text}<|im_end|>\n<|im_start|>assistant\n"

    return prompt


def translate(
    model,
    tokenizer,
    text: str,
    config: dict = None,
) -> tuple[str, float]:
    """
    Traduit une phrase du français vers le Mina

    Args:
        model: Modèle loaded
        tokenizer: Tokenizer
        text: Texte à traduire
        config: Configuration d'inférence

    Returns:
        Tuple (traduction, temps_en_ms)
    """
    if config is None:
        config = INFERENCE_CONFIG

    # Créer le prompt
    prompt = create_translation_prompt(text)

    # Tokeniser
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Déplacer vers le GPU si disponible
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Mesurer le temps d'inférence
    start_time = time.time()

    # Générer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            do_sample=config["do_sample"],
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Mesurer le temps total
    elapsed_ms = (time.time() - start_time) * 1000

    # Décoder la réponse complète
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extraire uniquement la traduction (après <|im_start|>assistant\n)
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in response:
        translation = response.split(assistant_marker)[-1].strip()
    else:
        # Fallback: utiliser le prompt original
        translation = response[len(prompt):].strip()

    # Nettoyer la réponse - supprimer les balises spéciales
    translation = translation.replace("<|im_end|>", "").strip()
    translation = translation.split("\n")[0].strip()

    return translation, elapsed_ms


def translate_batch(
    model,
    tokenizer,
    texts: List[str],
    config: dict = None,
) -> List[tuple[str, float]]:
    """
    Traduit plusieurs phrases en batch

    Args:
        model: Modèle
        tokenizer: Tokenizer
        texts: Liste de textes
        config: Configuration

    Returns:
        Liste de (traduction, temps) tuples
    """
    results = []

    for text in texts:
        translation, elapsed = translate(model, tokenizer, text, config)
        results.append((translation, elapsed))

    return results


# =============================================================================
# AFFICHAGE DES RÉSULTATS
# =============================================================================

def display_results(results: List[dict], verbose: bool = False):
    """
    Affiche les résultats de traduction de manière claire

    Args:
        results: Liste de résultats
        verbose: Mode détaillé
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("RÉSULTATS DE TRADUCTION")
    logger.info("=" * 70)

    total_time = 0

    for i, result in enumerate(results, 1):
        logger.info("")
        logger.info(f"--- Phrase {i} ---")
        logger.info(f"Français: {result['source']}")
        logger.info(f"Mina:     {result['translation']}")
        logger.info(f"Temps:    {result['time_ms']:.1f} ms")
        total_time += result['time_ms']

        if verbose:
            logger.debug(f"Tokens générés: ~{len(result['translation']) // 4}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("STATISTIQUES GLOBALES")
    logger.info("=" * 70)
    logger.info(f"Nombre de phrases: {len(results)}")
    logger.info(f"Temps total:       {total_time:.1f} ms")
    logger.info(f"Temps moyen:      {total_time / len(results):.1f} ms")
    logger.info(f"Temps min:         {min(r['time_ms'] for r in results):.1f} ms")
    logger.info(f"Temps max:         {max(r['time_ms'] for r in results):.1f} ms")

    # Estimer les performances
    avg_time = total_time / len(results)
    if avg_time < 200:
        perf_level = "EXCELLENT ⚡⚡⚡"
    elif avg_time < 500:
        perf_level = "BON ⚡⚡"
    elif avg_time < 1000:
        perf_level = "ACCEPTABLE ⚡"
    else:
        perf_level = "LENT - Optimisation nécessaire"

    logger.info("")
    logger.info(f"Performance API: {perf_level}")
    logger.info("=" * 70)


def save_results(results: List[dict], output_path: Path):
    """
    Sauvegarde les résultats en JSON

    Args:
        results: Liste de résultats
        output_path: Chemin du fichier de sortie
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "config": INFERENCE_CONFIG,
        "results": results,
        "stats": {
            "total_phrases": len(results),
            "total_time_ms": sum(r['time_ms'] for r in results),
            "avg_time_ms": sum(r['time_ms'] for r in results) / len(results),
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Résultats sauvegardés: {output_path}")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def run_inference_test(
    model_path: Path = DEFAULT_MODEL_PATH,
    phrases: Optional[List[str]] = None,
    verbose: bool = False,
    save_output: bool = True,
):
    """
    Exécute le test d'inférence complet

    Args:
        model_path: Chemin vers le modèle
        phrases: Liste de phrases à tester (ou None pour défaut)
        verbose: Mode détaillé
        save_output: Sauvegarder les résultats
    """
    setup_logging(verbose)

    logger.info("=" * 60)
    logger.info("TEST D'INFÉRENCE MINA-TRANSLATOR")
    logger.info("=" * 60)
    logger.info(f"Modèle: {model_path}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Phrases à tester
    if phrases is None:
        phrases = DEFAULT_TEST_PHRASES

    logger.info(f"Phrases à tester: {len(phrases)}")

    # Charger le modèle
    try:
        model, tokenizer, processor = load_model(model_path)
    except FileNotFoundError:
        logger.error(f"Modèle non trouvé: {model_path}")
        logger.info("Conseil: Lancez d'abord l'entraînement ou spécifiez le chemin correct.")
        logger.info(f"Répertoire des modèles: {PROJECT_ROOT / 'models'}")
        return None

    # Exécuter les traductions
    results = []

    for i, phrase in enumerate(phrases, 1):
        logger.info("")
        logger.info(f"Test {i}/{len(phrases)}: \"{phrase[:50]}...\"")

        try:
            translation, elapsed_ms = translate(model, tokenizer, phrase)

            results.append({
                "source": phrase,
                "translation": translation,
                "time_ms": elapsed_ms,
            })

            logger.info(f"  → \"{translation[:50]}...\" ({elapsed_ms:.0f} ms)")

        except Exception as e:
            logger.error(f"Erreur de traduction: {e}")
            results.append({
                "source": phrase,
                "translation": f"[ERREUR: {e}]",
                "time_ms": 0,
            })

    # Afficher les résultats
    display_results(results, verbose)

    # Sauvegarder
    if save_output:
        save_results(results, OUTPUT_RESULTS)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test d'inférence Mina-Translator")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Chemin vers le modèle fine-tuné",
    )
    parser.add_argument(
        "--phrases",
        type=str,
        nargs="+",
        help="Phrases à traduire (séparées par des espaces)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode détaillé",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder les résultats",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Température de génération (défaut: 0.1)",
    )

    args = parser.parse_args()

    # Mettre à jour la config
    INFERENCE_CONFIG["temperature"] = args.temperature

    # Lancer le test
    run_inference_test(
        model_path=Path(args.model_path),
        phrases=args.phrases,
        verbose=args.verbose,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()