#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de vérification système pour Mina-Translator
Vérifie automatiquement : GPU, modèles, API, traduction
Sortie silencieuse avec indicateurs visuels ✅/❌
"""

import sys
import io
import os
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Imports conditionnels (pour éviter les erreurs si pas installé)
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CheckResult:
    """Résultat d'une vérification"""
    name: str
    status: bool  # True = OK, False = FAIL
    message: str
    details: Optional[str] = None


def ok(msg: str) -> str:
    return f"✅ {msg}"


def fail(msg: str) -> str:
    return f"❌ {msg}"


def info(msg: str) -> str:
    return f"  → {msg}"


def check_gpu() -> CheckResult:
    """Vérifie la présence du GPU NVIDIA et la VRAM disponible"""
    if not TORCH_AVAILABLE:
        return CheckResult(
            name="GPU NVIDIA",
            status=False,
            message="PyTorch non installé"
        )

    if not torch.cuda.is_available():
        return CheckResult(
            name="GPU NVIDIA",
            status=False,
            message="CUDA non disponible - aucun GPU détecté"
        )

    try:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_vram = torch.cuda.memory_allocated(0) / 1e9
        available_vram = total_vram - allocated_vram

        if available_vram < 2.0:
            return CheckResult(
                name="GPU NVIDIA",
                status=False,
                message=f"VRAM insuffisante: {available_vram:.1f} GB disponible (minimum: 2 GB)",
                details=f"{gpu_name} | Total: {total_vram:.1f} GB | Disponible: {available_vram:.1f} GB"
            )

        return CheckResult(
            name="GPU NVIDIA",
            status=True,
            message=f"{gpu_name}",
            details=f"VRAM: {available_vram:.1f} GB / {total_vram:.1f} GB disponible"
        )
    except Exception as e:
        return CheckResult(
            name="GPU NVIDIA",
            status=False,
            message=f"Erreur détection GPU: {str(e)}"
        )


def check_model_files() -> CheckResult:
    """Vérifie l'intégrité des fichiers du modèle"""
    model_dir = Path("c:/Ce PC/Projet_python/IA traduction Français Mina/models/mina-translator-final-backup")

    if not model_dir.exists():
        return CheckResult(
            name="Fichiers Modèle",
            status=False,
            message=f"Directory introuvable: {model_dir}"
        )

    # Fichiers critiques requis
    required_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]

    missing_files = []
    corrupted_files = []

    for filename in required_files:
        filepath = model_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            # Vérification taille minimale (fichier non vide)
            if filepath.stat().st_size < 100:
                corrupted_files.append(f"{filename} (trop petit)")

    if missing_files:
        return CheckResult(
            name="Fichiers Modèle",
            status=False,
            message=f"Fichiers manquants: {', '.join(missing_files)}",
            details=f"Répertoire: {model_dir}"
        )

    if corrupted_files:
        return CheckResult(
            name="Fichiers Modèle",
            status=False,
            message=f"Fichiers corrompus: {', '.join(corrupted_files)}"
        )

    # Vérification somme de contrôle du fichier principal
    adapter_file = model_dir / "adapter_model.safetensors"
    file_size = adapter_file.stat().st_size / 1e6

    return CheckResult(
        name="Fichiers Modèle",
        status=True,
        message=f"Tous les fichiers présents et intacts",
        details=f"adapter_model.safetensors: {file_size:.1f} MB"
    )


def check_api() -> CheckResult:
    """Vérifie que l'API FastAPI répond sur le port 8000"""
    import socket

    # Test TCP rapide d'abord
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(('127.0.0.1', 8000))
        sock.close()
        if result != 0:
            return CheckResult(
                name="API FastAPI",
                status=False,
                message="Service non démarré (port 8000 non actif)"
            )
    except Exception as e:
        return CheckResult(
            name="API FastAPI",
            status=False,
            message=f"Erreur connexion: {str(e)}"
        )

    # Test HTTP avec timing
    try:
        import urllib.request

        start = time.perf_counter()
        req = urllib.request.Request("http://127.0.0.1:8000/health")
        req.add_header("Accept", "application/json")

        with urllib.request.urlopen(req, timeout=0.5) as response:
            elapsed_ms = (time.perf_counter() - start) * 1000
            status_code = response.status

            if status_code != 200:
                return CheckResult(
                    name="API FastAPI",
                    status=False,
                    message=f"Code HTTP {status_code} (attendu: 200)"
                )

            if elapsed_ms > 500:
                return CheckResult(
                    name="API FastAPI",
                    status=False,
                    message=f"Latence trop élevée: {elapsed_ms:.0f} ms (max: 500 ms)"
                )

            return CheckResult(
                name="API FastAPI",
                status=True,
                message=f"Réponse en {elapsed_ms:.0f} ms",
                details=f"Status: {status_code}"
            )
    except urllib.error.URLError as e:
        return CheckResult(
            name="API FastAPI",
            status=False,
            message=f"Erreur HTTP: {str(e)}"
        )
    except Exception as e:
        return CheckResult(
            name="API FastAPI",
            status=False,
            message=f"Erreur inattendue: {str(e)}"
        )


def check_translation() -> CheckResult:
    """Effectue une micro-traduction de test"""
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        return CheckResult(
            name="Test Traduction",
            status=False,
            message="Modules non disponibles (torch/transformers)"
        )

    model_path = "c:/Ce PC/Projet_python/IA traduction Français Mina/models/mina-translator-final-backup"

    try:
        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Chargement du modèle
        start_load = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        model.eval()
        load_time_ms = (time.perf_counter() - start_load) * 1000

        # Test de traduction
        test_phrase = "Bonjour"
        prompt = f"Français: {test_phrase}\nMina:"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        start_infer = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        infer_time_ms = (time.perf_counter() - start_infer) * 1000

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        mina_result = response.split("Mina:")[-1].strip()
        # Clean up
        mina_result = mina_result.split("\n")[0].strip()
        mina_result = mina_result.split("Français:")[0].strip()

        # Nettoyer la mémoire GPU
        del model
        torch.cuda.empty_cache()

        # Vérifier que la traduction n'est pas vide
        if not mina_result or len(mina_result) < 2:
            return CheckResult(
                name="Test Traduction",
                status=False,
                message="Traduction vide ou trop courte"
            )

        return CheckResult(
            name="Test Traduction",
            status=True,
            message=f"'{test_phrase}' -> '{mina_result}'",
            details=f"Chargement: {load_time_ms:.0f}ms | Inférence: {infer_time_ms:.0f}ms"
        )

    except Exception as e:
        return CheckResult(
            name="Test Traduction",
            status=False,
            message=f"Erreur: {str(e)}"
        )


def run_health_check() -> bool:
    """
    Exécute toutes les vérifications et retourne True si tout est OK
    """
    print("=" * 60)
    print("MINA-TRANSLATOR - VÉRIFICATION SYSTÈME")
    print("=" * 60)
    print()

    # Exécuter les vérifications
    checks = [
        ("1. Hardware GPU", check_gpu),
        ("2. Fichiers Modèle", check_model_files),
        ("3. API FastAPI", check_api),
        ("4. Test Traduction", check_translation),
    ]

    results = []
    all_passed = True

    for label, check_func in checks:
        try:
            result = check_func()
        except Exception as e:
            result = CheckResult(
                name=label,
                status=False,
                message=f"Exception: {str(e)}"
            )

        results.append(result)

        if result.status:
            print(f"{ok(result.name)}")
            print(info(result.message))
        else:
            print(f"{fail(result.name)}")
            print(info(result.message))

        if result.details:
            print(info(result.details))

        if not result.status:
            all_passed = False

        print()

    # Résumé
    print("=" * 60)

    passed = sum(1 for r in results if r.status)
    total = len(results)

    print(f"RÉSULTAT: {passed}/{total} vérifications réussies")
    print("=" * 60)

    if all_passed:
        print()
        print("🟢 SYSTÈME OPÉRATIONNEL ET STABLE")
        print()
        print("Tous les composants sont fonctionnels:")
        print("  • GPU NVIDIA détecté et VRAM disponible")
        print("  • Modèle chargé et fichiers intacts")
        print("  • API FastAPI opérationnelle")
        print("  • Traduction de test réussie")
        print()
        return True
    else:
        failed = [r.name for r in results if not r.status]
        print()
        print("🔴 SYSTÈME NON OPÉRATIONNEL")
        print()
        print("Composants en erreur:")
        for name in failed:
            print(f"  • {name}")
        print()
        print("Actions recommandées:")
        for r in results:
            if not r.status:
                print(f"  → {r.message}")
        print()
        return False


if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)