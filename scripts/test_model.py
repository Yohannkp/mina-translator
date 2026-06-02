#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for Mina Translator model"""

import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def safe_print(text):
    """Print safely handling encoding issues"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

def main():
    model_path = 'c:/Ce PC/Projet_python/IA traduction Français Mina/models/mina-translator/final'

    print("Chargement du modele Mina-Translator...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()

    print("=" * 60)
    print("TEST DE TRADUCTION MINA")
    print("=" * 60)

    test_phrases = [
        'La sante est importante.',
        'Je suis malade.',
        'Il a mal a la tete.',
        'Ou est hopital?',
        "J'ai besoin d'un medecin.",
        'Le marche est ouvert.',
        'Je veux acheter du riz.',
        'Combien ca coute?',
    ]

    for fr in test_phrases:
        prompt = f'Français: {fr}\nMina:'
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if 'Mina:' in response:
            mina = response.split('Mina:')[-1].strip()
        else:
            mina = response.replace(prompt, '').strip()

        # Clean up response
        mina = mina.split('\n')[0].strip()
        mina = mina.split('Français:')[0].strip()

        safe_print(f"FR: {fr}")
        safe_print(f"MINA: {mina}")
        print("-" * 40)

    print("\nTest termine!")

if __name__ == "__main__":
    main()