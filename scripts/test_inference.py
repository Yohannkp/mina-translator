"""
scripts/test_inference.py - Test d'inférence pour Mina-Translator
============================================================

Teste le modèle fine-tuné avec des phrases françaises inédites.
Mesure le temps d'inférence en millisecondes.

Usage:
    python scripts/test_inference.py

Auteur: Claude Opus 4.8
Date: 2026-06-04
"""

import sys
import io
import time

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = "c:/Ce PC/Projet_python/IA traduction Français Mina"
MODEL_PATH = f"{PROJECT_ROOT}/models/mina-translator-trained"
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================

def load_model():
    """Charge le modèle fine-tuné."""
    print("=" * 60)
    print("CHARGEMENT DU MODÈLE MINA-TRANSLATOR")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {MODEL_PATH}")
    print()

    # Configuration 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Charger modèle de base
    print("Chargement du modèle de base (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # Charger LoRA adapter
    print("Chargement de l'adaptateur LoRA...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    print(f"   GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # Charger tokenizer
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print()
    print("=" * 60)
    print("MODÈLE CHARGÉ AVEC SUCCÈS")
    print("=" * 60)
    print()

    return model, tokenizer


def translate(model, tokenizer, text: str, temperature: float = 0.2) -> tuple[str, float]:
    """Traduit une phrase française en Mina."""

    prompt = f"<|im_start|>user\nTraduis en Ewe: {text}<|im_end|>\n<|im_start|>assistant\n"

    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or 151643,
        )

    elapsed_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract translation
    if "<|im_start|>assistant\n" in response:
        translation = response.split("<|im_start|>assistant\n")[-1].strip()
    else:
        translation = response.split("assistant\n")[-1].strip() if "assistant\n" in response else response

    # Clean up
    translation = translation.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    translation = translation.split("\n")[0].strip()

    return translation, elapsed_ms


# =============================================================================
# TESTS
# =============================================================================

# Phrases de test - INÉDITES (pas dans le dataset)
TEST_PHRASES = [
    "Le president est arrive a Lome pour la ceremony.",
    "Ou puis-je trouver un taxi pour aller a Kpalime?",
    "J'ai besoin d'un medicament pour la fiere.",
    "Mon enfant doit passer son examen de mathematiques.",
    "La pluie a cause des inundations dans le quartier.",
]

def main():
    print()
    print("=" * 60)
    print("MINA-TRANSLATOR - TEST D'INFÉRENCE")
    print("=" * 60)
    print()

    # Charger le modèle
    model, tokenizer = load_model()

    print()
    print("=" * 60)
    print("RÉSULTATS DE TRADUCTION")
    print("=" * 60)
    print()

    times = []

    for i, phrase in enumerate(TEST_PHRASES, 1):
        translation, elapsed_ms = translate(model, tokenizer, phrase)
        times.append(elapsed_ms)

        print(f"[{i}/5] Français: {phrase}")
        print(f"       Mina:     {translation}")
        print(f"       Temps:    {elapsed_ms:.1f} ms")
        print()

    # Statistiques
    avg_time = sum(times) / len(times)
    print("=" * 60)
    print(f"Temps moyen d'inférence: {avg_time:.1f} ms")
    print(f"Temps min: {min(times):.1f} ms")
    print(f"Temps max: {max(times):.1f} ms")
    print("=" * 60)
    print()

    # Benchmark de performance
    print("Benchmark de performance:")
    print(f"  - Temps moyen: {avg_time:.0f} ms par requête")
    print(f"  - Débit estimé: {60000/avg_time:.1f} requêtes/minute")
    print(f"  - Latence P50: {sorted(times)[len(times)//2]:.0f} ms")
    print()


if __name__ == "__main__":
    main()