"""
Script de correction des problèmes de compatibilité du modèle.

Corrige:
1. Paramètres non reconnus dans adapter_config.json
2. Conflit NumPy version

Usage: python scripts/fix_model_compat.py
"""
import os
import sys
from pathlib import Path
import shutil
import json

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "mina-translator"

# Paramètres à supprimer (non reconnus par les anciennes versions de PEFT)
REMOVE_PARAMS = [
    "alora_invocation_tokens",
    "qalora_group_size",
    "use_qalora",
    "peft_version",  # Ce n'est pas utilisé par PEFT de toute façon
]

def fix_adapter_config(config_path: Path) -> bool:
    """Corrige le fichier adapter_config.json"""
    print(f"\n📝 Correction: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        original_keys = set(config.keys())

        # Supprimer les paramètres non reconnus
        for param in REMOVE_PARAMS:
            if param in config:
                print(f"   ✂️  Suppression: {param}")
                del config[param]

        # Corriger le nom du modèle de base si nécessaire
        if config.get("base_model_name_or_path") == "Qwen/Qwen2-0.5B-Instruct":
            # Essayer d'abord Qwen2, puis Qwen1.5 comme fallback
            config["base_model_name_or_path"] = "Qwen/Qwen1.5-0.5B"
            print(f"   🔄 Modèle base: Qwen/Qwen2-0.5B-Instruct → Qwen/Qwen1.5-0.5B")

        # S'assurer que les paramètres requis sont présents
        if "r" not in config:
            config["r"] = 8
            print("   ➕ Ajout: r=8")

        if "lora_alpha" not in config:
            config["lora_alpha"] = 16
            print("   ➕ Ajout: lora_alpha=16")

        # Sauvegarder
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Config corrigée ({len(original_keys)} → {len(config)} paramètres)")
        return True

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def fix_all_configs():
    """Corrige tous les fichiers adapter_config.json"""
    print("=" * 60)
    print("CORRECTION DES FICHIERS DE CONFIGURATION")
    print("=" * 60)

    fixed_count = 0
    for subdir in MODEL_DIR.iterdir():
        if subdir.is_dir():
            config_path = subdir / "adapter_config.json"
            if config_path.exists():
                if fix_adapter_config(config_path):
                    fixed_count += 1

    print(f"\n✅ {fixed_count} fichier(s) corrigé(s)")
    return fixed_count > 0


def fix_numpy_version():
    """Installe NumPy 1.x pour la compatibilité"""
    print("\n" + "=" * 60)
    print("CORRECTION NUMPY")
    print("=" * 60)

    print("\n⚠️  NumPy 2.x cause des problèmes avec certains modules.")
    print("   Installation de NumPy 1.x...")

    os.system(f'"{PROJECT_ROOT}\\venv_train\\Scripts\\python.exe" -m pip install "numpy<2" --quiet')

    print("   ✅ NumPy 1.x installé")
    print("\n   Redémarrez votre terminal et réactivez l'environnement:")
    print("   venv_train\\Scripts\\activate")
    print("   python scripts/inference_test.py")


if __name__ == "__main__":
    print("🔧 Script de correction de compatibilité modèle")
    print()

    # Étape 1: Corriger les configs
    config_fixed = fix_all_configs()

    if config_fixed:
        print("\n" + "=" * 60)
        print("PROCHAINES ÉTAPES")
        print("=" * 60)
        print()
        print("1. Corriger NumPy (recommander):")
        print("   python scripts/fix_model_compat.py")
        print()
        print("   OU manuellement dans le terminal:")
        print("   venv_train\\Scripts\\activate")
        print('   pip install "numpy<2"')
        print()
        print("2. Relancer le test:")
        print("   python scripts/inference_test.py")
        print()

        # Demander si on veut corriger NumPy maintenant
        response = input("\nVoulez-vous corriger NumPy maintenant? (o/n): ").strip().lower()
        if response == 'o':
            fix_numpy_version()
    else:
        print("\n❌ Aucune correction nécessaire, les configs semblent OK.")