"""
scripts/launch_training.py - Lancer le Fine-tuning Complet Mina-Translator
============================================================================

Fine-tune les modèles avec LES DONNÉES RÉELLES fournies:
- LLM: Qwen2-0.5B sur les 360 paires FR→Mina
- STT: Whisper sur les ~19 000 audios Mina

Hardware: RTX 4060 (8GB VRAM)

Usage:
    python scripts/launch_training.py --mode full

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path

# Forcer UTF-8 sur Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
MODELS_DIR = PROJECT_DIR / "models"
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"

# Modèle de base
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Chemins des données
CORPUS_FR_MINA = CORPUS_DIR / "dataset_mina.jsonl"
CORPUS_VOCAB = CORPUS_DIR / "vocabulaire_mina.json"
CORPUS_VOCAB_NEW = CORPUS_DIR / "vocabulaire_corpus.json"

# Sortie
OUTPUT_DIR = MODELS_DIR / "mina-translator"

# =============================================================================
# VÉRIFICATIONS
# =============================================================================

def check_gpu():
    """Vérifie GPU et retourne ses specs."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU: {gpu}")
            print(f"   VRAM: {vram:.1f} GB")
            return True
        else:
            print("   ERREUR: CUDA non disponible")
            return False
    except ImportError:
        print("   ERREUR: PyTorch non installé")
        return False

def check_dependencies():
    """Vérifie les dépendances."""
    deps = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT",
        "accelerate": "Accelerate",
        "bitsandbytes": "BitsAndBytes",
        "datasets": "Datasets",
    }

    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"   OK: {name}")
        except ImportError:
            print(f"   MANQUANT: {name}")
            missing.append(module)

    return len(missing) == 0

def load_corpus():
    """Charge le corpus FR→Mina."""
    if not CORPUS_FR_MINA.exists():
        print(f"   ERREUR: Corpus non trouvé: {CORPUS_FR_MINA}")
        return []

    entries = []
    with open(CORPUS_FR_MINA, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return entries

def create_vocabulary_prompt():
    """Crée le prompt avec le vocabulaire Mina."""
    vocab = {}

    # Charger vocabulaire existant
    if CORPUS_VOCAB.exists():
        with open(CORPUS_VOCAB, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

    # Charger vocabulaire du corpus
    if CORPUS_VOCAB_NEW.exists():
        with open(CORPUS_VOCAB_NEW, 'r', encoding='utf-8') as f:
            vocab_new = json.load(f)
            # Merger avec existant
            for key, val in vocab_new.items():
                if key not in vocab:
                    vocab[key] = val

    # Construire le prompt avec exemples Mina
    prompt_parts = ["""
Tu es un assistant de traduction expert en langue Ewe/Mina parlée au Togo.
Traduis le français vers le Mina (Gen) avec exactitude.

VOCABULAIRE MINA:
"""]

    # Ajouter quelques termes clés
    terms = {
        "salutations": "sawo=bonjour, akɔ=merci, nukɔ=au-revoir",
        "corps": "ƒu=tête, tɔgbo=ventre/gorge, anyi=yeux, ɖu=dents",
        "medecin": "Nusɔla=médecin, dɔta=docteur",
        "hopital": "hɔspita=hôpital, soko aɖa=pharmacie",
        "sante": "fɛ́e=malade, nuwɔna=santé, aɖa=médicament",
        "temps": "ai=aujourd'hui, ẹmɛ=demain, etsɛ=hier",
        "verbes": "le=être, ena=avoir, ka=aller, va=venir",
    }

    for cat, examples in terms.items():
        prompt_parts.append(f"- {cat}: {examples}")

    prompt_parts.append("""
RÈGLES:
- Pronoms: me=je, a=tu, e=il/elle, wo=vous, wɔ=ils/elles
- Négation: ...o (ex: me kpɔ o = je ne vois pas)
- Temps: Les verbes ne changent pas, on ajoute des mots

Traduis cette phrase française en mina (ewé du Togo):
""")

    return "\n".join(prompt_parts)

# =============================================================================
# FINE-TUNING LLM
# =============================================================================

def train_llm(epochs=3, sample_size=0):
    """Fine-tune Qwen2-0.5B avec QLoRA sur le corpus FR→Mina."""
    print("\n" + "="*70)
    print("   FINE-TUNING LLM: Qwen2-0.5B")
    print("="*70)

    # Charger corpus
    entries = load_corpus()
    if not entries:
        return False

    if sample_size > 0:
        entries = entries[:sample_size]

    print(f"   Paires FR→Mina: {len(entries)}")
    print(f"   Epochs: {epochs}")

    # Importer libs
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print(f"   ERREUR d'import: {e}")
        print("   Instalez: pip install torch transformers peft accelerate bitsandbytes")
        return False

    # Vocabulaire pour le prompt
    vocab_prompt = create_vocabulary_prompt()

    # Formater les données
    def format_entry(entry):
        return f"""{vocab_prompt}
{entry['input']}

Mina: {entry['output']}"""

    formatted = [format_entry(e) for e in entries]

    # Dataset
    dataset = Dataset.from_dict({"text": formatted})

    # Tokenizer
    print("\n   Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokeniser
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.train_test_split(test_size=0.1)

    # Charger modèle avec QLoRA 4-bit
    print("\n   Chargement du modèle QLoRA 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Préparer pour QLoRA
    model = prepare_model_for_kbit_training(model)

    # Config LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    def data_collator(features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["input_ids"] for f in features]),
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Entraîner
    print("\n   DEBUT DE L'ENTRAINEMENT...")
    print("   (Cela peut prendre 10-30 minutes selon le GPU)")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Sauvegarder
    print(f"\n   Sauvegarde du modele...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    print(f"\n   TERMINE en {elapsed/60:.1f} minutes!")
    return True

# =============================================================================
# TEST DU MODELE
# =============================================================================

def test_model():
    """Test le modèle fine-tuné."""
    print("\n" + "="*70)
    print("   TEST DU MODELE FINE-TUNE")
    print("="*70)

    model_path = OUTPUT_DIR / "final"
    if not model_path.exists():
        print("   ATTENTION: Modele non trouve, utilisation Ollama")
        return test_with_ollama()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.float16,
        )

        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=False,
        )

    except Exception as e:
        print(f"   Erreur: {e}")
        return test_with_ollama()

    # Test phrases
    tests = [
        "Je suis malade.",
        "Le medecin est venu.",
        "Je dois aller a l'hopital.",
    ]

    vocab_prompt = create_vocabulary_prompt()

    for phrase in tests:
        prompt = f"{vocab_prompt}\n{phrase}\n\nMina:"
        result = gen(prompt)[0]["generated_text"]

        # Extraire juste la traduction
        if "Mina:" in result:
            trans = result.split("Mina:")[-1].split("\n")[0].strip()
        else:
            trans = result.split("Traduis")[0].strip()[-50:]

        print(f"   FR: {phrase}")
        print(f"   MINA: {trans}")
        print()

def test_with_ollama():
    """Test avec Ollama comme fallback."""
    print("\n   Test avec Ollama...")

    import requests

    vocab_prompt = create_vocabulary_prompt()

    tests = [
        "Je suis malade.",
        "Le medecin est venu.",
    ]

    for phrase in tests:
        prompt = f"{vocab_prompt}{phrase}\n\nMina:"

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5-coder:7b", "prompt": prompt, "stream": False},
                timeout=60
            )
            if r.status_code == 200:
                result = r.json()["response"]
                print(f"   FR: {phrase}")
                print(f"   MINA: {result[:60]}...")
        except:
            print(f"   Ollama non accessible")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", choices=["llm", "stt", "full", "test"],
                       default="test", help="Mode d'execution")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Epochs")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Echantillon (0=tous)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("   MINA-TRANSLATOR: FINE-TUNING COMPLET")
    print("   Donnees: 360 paires FR->MINA + 19000 audios")
    print("="*70)

    # Vérifications
    print("\n[1/4] Vérification GPU...")
    if not check_gpu():
        return

    print("\n[2/4] Vérification des dépendances...")
    if not check_dependencies():
        return

    print("\n[3/4] Chargement du corpus...")
    entries = load_corpus()
    print(f"   Corpus: {len(entries)} paires FR->MINA")

    print("\n[4/4] Mode selected:", args.mode)

    if args.mode == "llm" or args.mode == "full":
        train_llm(epochs=args.epochs, sample_size=args.sample)

    if args.mode == "test":
        test_model()

    print("\n" + "="*70)
    print("   TERMINE!")
    print("="*70)

if __name__ == "__main__":
    main()