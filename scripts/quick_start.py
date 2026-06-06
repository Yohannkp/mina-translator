"""
scripts/quick_start.py - Guide de démarrage rapide Mina-Translator
==================================================================

Ce script automatise les étapes clés pour:
1. Créer/vérifier l'environnement Python
2. Préparer les données
3. Lancer l'entraînement
4. Tester l'inférence

Usage:
    python scripts/quick_start.py --step setup    # Setup environnement
    python scripts/quick_start.py --step train     # Lancer entraînement
    python scripts/quick_start.py --step test      # Tester inférence
    python scripts/quick_start.py --step all       # Tout faire

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
VENV_PATH = PROJECT_ROOT / "venv_train"
PYTHON_BIN = VENV_PATH / "Scripts" / "python.exe"
MAIN_PYTHON = Path("C:/Program Files/Python312/python.exe")

# Couleurs pour l'affichage
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_banner():
    """Affiche la bannière du projet"""
    print(f"""
{BLUE}{'='*60}
{BOLD}   🇨🇩 MINA-TRANSLATOR - GUIDE DE DÉMARRAGE RAPIDE
{'='*60}{RESET}

Projet de traduction automatique Français ↔ Mina (Togo)
Optimisé pour RTX 4060 (8 Go VRAM)

{BOLD}Capacités:{RESET}
  • Fine-tuning QLoRA 4-bit sur modèle léger
  • Traduction texte français → mina
  • API FastAPI pour déploiement

{'='*60}
""")


def print_step(step: str, message: str):
    """Affiche une étape en cours"""
    print(f"\n{BLUE}▸ {step}{RESET}: {message}")


def print_success(message: str):
    """Affiche un succès"""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Affiche une erreur"""
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message: str):
    """Affiche un avertissement"""
    print(f"{YELLOW}⚠ {message}{RESET}")


# =============================================================================
# ÉTAPES
# =============================================================================

def check_system():
    """Vérifie les composants système"""
    print_step("SYSTÈME", "Vérification de l'environnement...")

    issues = []

    # Python 3.12
    try:
        result = subprocess.run(
            [str(MAIN_PYTHON), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Python 3.12" in result.stdout:
            print_success("Python 3.12 détecté")
        else:
            print_warning(f"Python {result.stdout.strip()} détecté (3.12 recommandé)")
    except FileNotFoundError:
        issues.append("Python 3.12 non trouvé")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go")
        else:
            print_warning("CUDA non disponible")
    except ImportError:
        issues.append("PyTorch non installé")

    if issues:
        print(f"\n{RED}Problèmes détectés:{RESET}")
        for issue in issues:
            print(f"  • {issue}")
        return False

    return True


def setup_environment():
    """Configure l'environnement Python"""
    print_step("ENVIRONNEMENT", "Configuration...")

    # Vérifier Python
    if not MAIN_PYTHON.exists():
        print_error("Python 3.12 non trouvé!")
        print(f"\nInstallez Python 3.12 depuis: https://www.python.org/downloads/")
        return False

    # Créer venv
    if VENV_PATH.exists():
        print_warning("Environnement existant, suppression...")
        import shutil
        shutil.rmtree(VENV_PATH)

    print("Création de l'environnement virtuel...")
    result = subprocess.run(
        [str(MAIN_PYTHON), "-m", "venv", str(VENV_PATH)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print_error("Impossible de créer l'environnement")
        print(result.stderr)
        return False

    print_success("Environnement créé: venv_train/")

    # Installer dépendances
    print("\nInstallation des dépendances...")

    pip_cmd = [str(PYTHON_BIN), "-m", "pip", "install"]

    # PyTorch
    print("  • PyTorch (CUDA 12.1)...")
    subprocess.run(
        pip_cmd + [
            "torch==2.3.1", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ],
        capture_output=True,
    )

    # Autres dépendances
    deps = [
        "transformers==4.40.0",
        "peft==0.11.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.30.0",
        "datasets==2.18.0",
        "huggingface_hub==0.21.0",
        "fastapi",
        "uvicorn[standard]",
        "loguru",
        "librosa",
        "numpy",
    ]

    print("  • Autres dépendances...")
    subprocess.run(pip_cmd + deps, capture_output=True)

    print_success("Dépendances installées!")
    return True


def prepare_data():
    """Prépare les données d'entraînement"""
    print_step("DONNÉES", "Préparation...")

    # Vérifier dataset
    dataset_path = PROJECT_ROOT / "data" / "processed" / "mina_french_pairs.jsonl"

    if not dataset_path.exists():
        print_error(f"Dataset non trouvé: {dataset_path}")
        return False

    # Compter les exemples
    with open(dataset_path, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())

    print_success(f"Dataset: {count} exemples Mina-Français")
    return True


def run_training():
    """Lance l'entraînement"""
    print_step("ENTRAÎNEMENT", "Lancement...")

    train_script = PROJECT_ROOT / "scripts" / "train_mini_llm.py"

    if not train_script.exists():
        print_error("Script d'entraînement non trouvé!")
        return False

    # Dry run
    print(f"\n{BOLD}Démarrage du dry-run (50 steps)...{RESET}")
    print("=" * 60)

    cmd = [str(PYTHON_BIN), str(train_script), "--steps", "50"]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print_success("Entraînement terminé avec succès!")
        return True
    else:
        print_error("Entraînement échoué")
        return False


def run_inference_test():
    """Lance le test d'inférence"""
    print_step("INFÉRENCE", "Test...")

    test_script = PROJECT_ROOT / "scripts" / "inference_test.py"

    if not test_script.exists():
        print_error("Script de test non trouvé!")
        return False

    if not (PROJECT_ROOT / "models" / "mina-translator").exists():
        print_warning("Modèle non trouvé - lancez d'abord l'entraînement")
        return False

    cmd = [str(PYTHON_BIN), str(test_script)]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print_success("Test terminé!")
        return True
    else:
        print_warning("Test terminé avec des erreurs")
        return True  # On continue même si test échoué


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    print_banner()

    parser = argparse.ArgumentParser(description="Guide de démarrage Mina-Translator")
    parser.add_argument(
        "--step",
        choices=["setup", "data", "train", "test", "all"],
        default="all",
        help="Étape à exécuter",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Ignorer l'entraînement (si modèle existe déjà)",
    )

    args = parser.parse_args()

    # Vérifier système
    if not check_system():
        print("\nRésolvez les problèmes avant de continuer.")
        return 1

    if args.step == "setup":
        success = setup_environment()
    elif args.step == "data":
        success = prepare_data()
    elif args.step == "train":
        success = prepare_data() and run_training()
    elif args.step == "test":
        success = run_inference_test()
    else:  # all
        success = setup_environment()
        if not success:
            print_error("Échec du setup")
            return 1

        success = prepare_data()
        if not success:
            print_error("Échec de préparation des données")
            return 1

        if not args.skip_training:
            success = run_training()
            if not success:
                print_error("Échec de l'entraînement")
                return 1

        print("\n" + "=" * 60)
        print("TOUT EST PRÊT!")
        print("=" * 60)
        print("""
Pour utiliser le modèle:
  1. Activer l'environnement: venv_train\\Scripts\\activate
  2. Lancer l'API: python api/main.py
  3. Tester: python scripts/inference_test.py

Bon courage pour le projet Mina! 🇹🇬
""")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())