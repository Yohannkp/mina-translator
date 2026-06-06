"""
scripts/train_whisper_native.py - Fine-tuning Whisper avec Transformers
=========================================================================

Fine-tune Whisper sur les données audio Mina pour créer un modèle STT
capable de transcrire le Mina parlé au Togo.

Utilise la bibliothèque Transformers de HuggingFace pour un fine-tuning
correct avec CTC Loss et le tokenizer Whisper.

Optimisé pour RTX 4060 Laptop (8GB VRAM).

Usage:
    # Préparer les données d'abord
    python scripts/prepare_whisper_data.py

    # Lancer le training
    python scripts/train_whisper_native.py --epochs 3

    # Mode test (quelques steps)
    python scripts/train_whisper_native.py --dry-run

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
import gc
from pathlib import Path
from dataclasses import dataclass

# UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Modèle Whisper (tiny=39M, base=74M)
    model_name: str = "openai/whisper-base"

    # Entraînement
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 1e-4

    # Audio
    sample_rate: int = 16000

    # Données
    data_dir: Path = Path("c:/Ce PC/Projet_python/IA traduction Français Mina/data/whisper_data")
    output_dir: Path = Path("c:/Ce PC/Projet_python/IA traduction Français Mina/models/whisper-mina")


def load_data(config: Config):
    """Charge les manifestes de données."""
    train_data = []
    val_data = []

    # Charger train
    train_path = config.data_dir / "train.jsonl"
    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line))

    # Charger validation
    val_path = config.data_dir / "validation.jsonl"
    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    val_data.append(json.loads(line))

    return train_data, val_data


def train_with_whisper(config: Config, train_data, val_data):
    """Fine-tune Whisper avec la méthode de transcription."""
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperFeatureExtractor,
        WhisperTokenizer
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n   Device: {DEVICE}")

    print(f"\n   Chargement du modèle Whisper {config.model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    processor = WhisperProcessor.from_pretrained(config.model_name)

    # Forcer la langue française
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.lang_to_id = {"french": 1}  # Forcer le français comme langue source

    model = model.to(DEVICE)

    # Optimiseur
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Entraînement
    print(f"\n   Début de l'entraînement...")
    print(f"   Données train: {len(train_data)}")
    print(f"   Données validation: {len(val_data)}")

    global_step = 0
    max_steps = min(len(train_data), 500)  # Limiter pour le test

    model.train()

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"   EPOCH {epoch+1}/{config.epochs}")
        print(f"{'='*60}")

        # Mélanger les données
        import random
        random.shuffle(train_data)

        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(range(0, max_steps), desc=f"Epoch {epoch+1}")

        for step in progress_bar:
            idx = step % len(train_data)
            entry = train_data[idx]

            try:
                # Charger l'audio
                import whisper as whisper_audio
                audio = whisper_audio.load_audio(entry['audio_filepath'])
                audio = whisper_audio.pad_or_trim(audio)

                # Convertir en input features
                input_features = processor(
                    audio,
                    sampling_rate=config.sample_rate,
                    return_tensors="pt"
                ).input_features.to(DEVICE)

                # Encoder le texte (labels)
                labels = processor.tokenizer(
                    entry['text'],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=448
                ).input_ids.to(DEVICE)

                # Forward
                outputs = model(
                    input_features=input_features,
                    labels=labels
                )

                loss = outputs.loss / config.grad_accum
                epoch_loss += loss.item() * config.grad_accum
                epoch_steps += 1

                # Backward
                loss.backward()

                # Gradient accumulation
                if (step + 1) % config.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Afficher
                if epoch_steps % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': f'{epoch_loss/epoch_steps:.4f}',
                        'step': global_step
                    })

                # Afficher VRAM
                if DEVICE == "cuda" and epoch_steps % 20 == 0:
                    mem = torch.cuda.memory_allocated() / 1e9
                    progress_bar.set_postfix({
                        'loss': f'{epoch_loss/epoch_steps:.4f}',
                        'VRAM': f'{mem:.1f}GB'
                    })

                # Limiter pour le dry-run
                if config.epochs == 1 and global_step >= 10:
                    print("\n   === DRY-RUN TERMINÉ ===")
                    break

            except Exception as e:
                print(f"\n   Erreur: {e}")
                continue

        print(f"\n   Epoch {epoch+1} - Loss moyenne: {epoch_loss/max(epoch_steps, 1):.4f}")

        # Sauvegarder checkpoint
        checkpoint_path = config.output_dir / f"checkpoint-epoch-{epoch+1}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        print(f"   Checkpoint sauvegardé: {checkpoint_path}")

        # Réinitialiser max_steps pour les epochs suivants
        max_steps = len(train_data)

    # Sauvegarder le modèle final
    final_path = config.output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n   Modele final sauvegardé: {final_path}")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune Whisper for Mina')
    parser.add_argument('--model', type=str, default='base',
                        help='Model size: tiny, base')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick test mode')

    args = parser.parse_args()

    # Mapping model name
    model_map = {
        'tiny': 'openai/whisper-tiny',
        'base': 'openai/whisper-base',
        'small': 'openai/whisper-small',
        'medium': 'openai/whisper-medium',
    }
    model_name = model_map.get(args.model, f'openai/whisper-{args.model}')

    config = Config(
        model_name=model_name,
        epochs=1 if args.dry_run else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("\n" + "="*70)
    print("   FINE-TUNING WHISPER POUR LE MINA")
    print("="*70)
    print(f"\n   Configuration:")
    print(f"      Modele: {config.model_name}")
    print(f"      Epochs: {config.epochs}")
    print(f"      Batch size: {config.batch_size}")
    print(f"      Learning rate: {config.learning_rate}")
    print(f"      Output: {config.output_dir}")

    if args.dry_run:
        print("\n   *** MODE DRY-RUN (10 steps) ***")

    # Charger les données
    print("\n   Chargement des données...")
    train_data, val_data = load_data(config)

    if not train_data:
        print("\n   ERREUR: Aucune donnée train trouvee!")
        print("   Lancez d'abord: python scripts/prepare_whisper_data.py")
        return

    print(f"   Train: {len(train_data)} entrées")
    print(f"   Val: {len(val_data)} entrées")

    # Lancer l'entraînement
    model = train_with_whisper(config, train_data, val_data)

    print("\n" + "="*70)
    print("   FINE-TUNING TERMINÉ!")
    print("="*70)
    print(f"\n   Modele sauvegarde: {config.output_dir}")
    print("\n   Prochaine etape: Tester avec")
    print(f"   python scripts/test_whisper_pipeline.py")


if __name__ == "__main__":
    main()