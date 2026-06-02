#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test d'inférence pour Mina-Translator
Charge le modèle fine-tuné et teste avec 5 phrases françaises inédites.
Mesure le temps d'inférence en millisecondes.
"""

import sys
import io
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def load_model():
    """Charger le modèle fine-tuné depuis models/mina-translator/"""
    model_path = "c:/Ce PC/Projet_python/IA traduction Français Mina/models/mina-translator/final"

    print("=" * 65)
    print("MINA TRANSLATOR - TEST D'INFERENCE")
    print("=" * 65)
    print(f"\nChargement du modele depuis: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # VRAM info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {allocated:.1f} GB / {total:.1f} GB")

    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    print(f"Model loaded: OK")

    return model, tokenizer


def translate(model, tokenizer, text, temperature=0.3, max_new_tokens=60):
    """
    Générer une traduction Mina avec paramètres optimisés.
    Température basse = moins d'hallucinations, plus cohérent.
    """
    prompt = f"Français: {text}\nMina:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    end = time.perf_counter()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    mina = response.split("Mina:")[-1].strip()
    # Clean up - take only first line
    mina = mina.split("\n")[0].strip()
    # Remove the prompt if it leaked through
    if "Français:" in mina:
        mina = mina.split("Français:")[0].strip()

    inference_ms = (end - start) * 1000
    return mina, inference_ms


def main():
    model, tokenizer = load_model()

    # 5 phrases françaises TOTALEMENT INÉDITES (pas dans le corpus)
    # Ces phrases测试ent la capacité du modèle à généraliser
    test_phrases = [
        "Le president a arrive a Lome.",
        "Je voudrais ouvrir un compte bancaire.",
        "Il y a une reunion importante demain matin.",
        "Ma fille a reussi son examen avec mention tres bien.",
        "Le taxi s'est arrete devant la mosquee.",
    ]

    print("\n" + "=" * 65)
    print("TEST DE TRADUCTION - 5 PHRASES INÉDITES")
    print("Parametres: temperature=0.3, top_p=0.85, top_k=50")
    print("=" * 65)

    results = []

    for i, fr in enumerate(test_phrases, 1):
        mina, ms = translate(model, tokenizer, fr, temperature=0.3)
        results.append((fr, mina, ms))

        print(f"\n[{i}/5] Francais: {fr}")
        print(f"    Mina:      {mina}")
        print(f"    Temps:     {ms:.1f} ms")

    # Stats
    avg_ms = sum(r[2] for r in results) / len(results)
    min_ms = min(r[2] for r in results)
    max_ms = max(r[2] for r in results)

    print("\n" + "=" * 65)
    print("STATISTIQUES D'INFERENCE")
    print("=" * 65)
    print(f"  Temps moyen:  {avg_ms:.1f} ms")
    print(f"  Temps min:    {min_ms:.1f} ms")
    print(f"  Temps max:    {max_ms:.1f} ms")
    print(f"  Throughput:   {1000/avg_ms:.1f} traductions/seconde")

    print("\n" + "=" * 65)
    print("ANALYSE DE LA QUALITE")
    print("=" * 65)

    # Analyse qualitative basique
    correct_keywords = {
        "Lome": "Lome",
        "bancaire": "bank",
        "reunion": "meeting",
        "examen": "exam",
        "taxi": "taxi",
    }

    for fr, mina, ms in results:
        # Check if key words from French appear in the Mina output
        found = []
        fr_lower = fr.lower()
        for key, label in correct_keywords.items():
            if key.lower() in fr_lower and key.lower() in mina.lower():
                found.append(f"{key}")
            elif key.lower() in fr_lower:
                found.append(f"?{key}")

        if found:
            print(f"  - '{fr[:30]}...' -> mots cles detects: {', '.join(found)}")

    print("\n" + "=" * 65)
    print("TEST TERMINE AVEC SUCCES")
    print("=" * 65)


if __name__ == "__main__":
    main()