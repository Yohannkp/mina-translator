"""
scripts/train_mina_translator.py - Fine-tuning Qwen2 pour FR↔Mina
================================================================

Fine-tune Qwen2-0.5B-Instruct avec LoRA sur le corpus Mina-Français.
Utilise les 360 paires FR→Mina fournies pour créer un modèle spécialisé.

Architecture:
- Modèle: Qwen/Qwen2-0.5B-Instruct
- Technique: QLoRA (4-bit, ultra-efficient)
- Entraînement: ~4-6 GB VRAM (RTX 4060 compatible)

Usage:
    python scripts/train_mina_translator.py [--epochs 3] [--batch 4]

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Modèle de base - petit et efficient pour 8GB VRAM
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Chemins
PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CORPUS_FILE = PROJECT_DIR / "data/corpus/dataset_mina.jsonl"
OUTPUT_DIR = PROJECT_DIR / "models" / "mina-translator"
MODELS_DIR = PROJECT_DIR / "models"

# Hyperparamètres
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Échantillon pour test (0 = tous)
TEST_SAMPLE = 0  # Mettez 10 pour tester rapidement

# =============================================================================
# VÉRIFICATIONS
# =============================================================================

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"   {title}")
    print(f"{'='*70}\n")

def check_gpu():
    """Vérifie le GPU et la VRAM disponible."""
    print_section("VÉRIFICATION GPU")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"✅ GPU: {gpu_name}")
            print(f"   VRAM Totale: {vram_total:.1f} GB")
            print(f"   VRAM Libre: {vram_total - vram_free:.1f} GB")
            return True
        else:
            print("❌ CUDA non disponible")
            return False
    except ImportError:
        print("❌ PyTorch non installé")
        return False

def check_dependencies():
    """Vérifie les dépendances."""
    print_section("VÉRIFICATION DES DÉPENDANCES")

    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("accelerate", "Accelerate"),
        ("bitsandbytes", "BitsAndBytes"),
        ("datasets", "Datasets"),
    ]

    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - installer avec: pip install {module}")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Dépendances manquantes: {', '.join(missing)}")
        print("   Installez avec:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("   pip install transformers peft accelerate bitsandbytes datasets")
        return False

    return True

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_corpus(file_path: Path) -> List[Dict]:
    """Charge le corpus FR→Mina."""
    print_section("CHARGEMENT DU CORPUS")

    if not file_path.exists():
        print(f"❌ Fichier non trouvé: {file_path}")
        return []

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError:
                    continue

    print(f"   📁 Fichier: {file_path.name}")
    print(f"   📊 Paires FR→Mina: {len(data)}")

    # Afficher quelques exemples
    print(f"\n   📝 Exemples (3 premiers):")
    for i, entry in enumerate(data[:3]):
        fr = entry.get('input', entry.get('fr', ''))
        mina = entry.get('output', entry.get('mina', ''))
        print(f"   [{i+1}] FR: {fr[:50]}...")
        print(f"       MINA: {mina[:50]}...")
        print()

    return data


def format_instruction(entry: Dict) -> str:
    """Formate une entrée pour l'entraînement."""
    instruction = entry.get('instruction', 'Traduis cette phrase française en mina (ewé du Togo).')
    input_text = entry.get('input', entry.get('fr', ''))
    output_text = entry.get('output', entry.get('mina', ''))

    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""


# =============================================================================
# ENTRENNEMENT
# =============================================================================

def train_model():
    """Fine-tune le modèle avec QLoRA."""
    print_section("DÉMARRAGE DE L'ENTRAÎNEMENT")

    # Charger les données
    corpus = load_corpus(CORPUS_FILE)
    if not corpus:
        print("❌ Impossible de continuer sans données")
        return False

    # Échantillon de test ?
    if TEST_SAMPLE > 0:
        corpus = corpus[:TEST_SAMPLE]
        print(f"⚠️  Mode test: {len(corpus)} paires seulement")

    print(f"\n📚 Données d'entraînement: {len(corpus)} paires")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"   Max length: {MAX_LENGTH}")
    print(f"   LoRA: r={LORA_R}, alpha={LORA_ALPHA}")

    # Import des bibliothèques
    print("\n⏳ Chargement des bibliothèques...")
    start_time = time.time()

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            set_seed
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        from bitsandbytes.optimization import Adam8bit
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("   Assurez-vous d'avoir installé toutes les dépendances")
        return False

    set_seed(42)

    # Charger le tokenizer
    print("\n🔄 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )

    # Ajouter les tokens spéciaux
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Préparer les données
    print("\n📝 Préparation des données...")
    formatted_data = [format_instruction(entry) for entry in corpus]

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": formatted_data})
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Charger le modèle avec QLoRA
    print("\n🔄 Chargement du modèle avec QLoRA 4-bit...")
    print("   (Cela peut prendre quelques minutes la première fois)")

    # Configuration BitsAndBytes pour 4-bit
    bnb_config = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        **bnb_config
    )

    # Configuration LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Entraîner
    print(f"\n🚀 Début de l'entraînement...")
    print(f"   ⏱️  Temps estimé: {len(corpus) * EPOCHS * 2 // 60} minutes")
    print()

    trainer.train()

    # Sauvegarder le modèle
    print(f"\n💾 Sauvegarde du modèle...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    elapsed = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {elapsed // 60:.0f} minutes")

    return True


# =============================================================================
# TEST DU MODÈLE
# =============================================================================

def test_model():
    """Test le modèle fine-tuné."""
    print_section("TEST DU MODÈLE FINE-TUNÉ")

    model_path = OUTPUT_DIR / "final"

    if not model_path.exists():
        print(f"⚠️  Modèle non trouvé: {model_path}")
        print("   Entraînez d'abord le modèle avec --train")
        return False

    print(f"   📁 Modèle: {model_path}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        print("❌ Modules requis non installés")
        return False

    # Charger le modèle et tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Créer un pipeline
    translator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
    )

    # Test phrases
    test_phrases = [
        "La santé est importante.",
        "Je suis malade.",
        "Le médecin est venu.",
        "Où est la pharmacie?",
        "Combien ça coûte?",
    ]

    print("\n🧪 Tests de traduction FR→Mina:")
    print("-" * 50)

    for phrase in test_phrases:
        prompt = f"""Traduis cette phrase française en mina (ewé du Togo).

Français: {phrase}

Mina:"""

        result = translator(prompt, max_new_tokens=100)[0]["generated_text"]
        # Extraire la réponse
        if "Mina:" in result:
            response = result.split("Mina:")[-1].strip()
        else:
            response = result.strip()

        print(f"\nFR: {phrase}")
        print(f"MINA: {response}")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning Mina-Translator")
    parser.add_argument("--train", "-t", action="store_true", help="Lancer l'entraînement")
    parser.add_argument("--test", action="store_true", help="Tester le modèle")
    parser.add_argument("--all", "-a", action="store_true", help="Entraîner ET tester")
    parser.add_argument("--epochs", "-e", type=int, default=EPOCHS, help="Nombre d'epochs")
    parser.add_argument("--batch", "-b", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--sample", "-s", type=int, default=TEST_SAMPLE, help="Échantillon (0=tous)")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("   🔧 MINA-TRANSLATOR: FINE-TUNING QWEN2-0.5B")
    print("   📚 Corpus: dataset_mina.jsonl (360 paires FR→Mina)")
    print("=" * 70)

    global EPOCHS, BATCH_SIZE, TEST_SAMPLE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    TEST_SAMPLE = args.sample

    # Vérifications
    if not check_dependencies():
        sys.exit(1)

    if not check_gpu():
        print("\n⚠️  GPU requis pour l'entraînement")
        sys.exit(1)

    # Mode
    if args.all:
        if train_model():
            test_model()
    elif args.train:
        train_model()
    elif args.test:
        test_model()
    else:
        print("\n📋 OPTIONS:")
        print("   --train       Lancer l'entraînement")
        print("   --test        Tester le modèle existant")
        print("   --all         Entraîner ET tester")
        print("   --epochs 5    Nombre d'epochs")
        print("   --batch 4     Batch size")
        print("   --sample 10   Échantillon de test (0=tous)")
        print("\n📖 EXEMPLES:")
        print("   python scripts/train_mina_translator.py --train")
        print("   python scripts/train_mina_translator.py --all --epochs 5")
        print("   python scripts/train_mina_translator.py --sample 10 --all  # Test rapide")


if __name__ == "__main__":
    main()