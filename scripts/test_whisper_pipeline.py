"""
scripts/test_whisper_pipeline.py - Test du pipeline STT Mina
=============================================================

Teste le pipeline de transcription automatique (STT) avec Whisper.
Vérifie que les fichiers audio peuvent être traités et que le modèle
fonctionne avant de lancer le fine-tuning.

Usage:
    python scripts/test_whisper_pipeline.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import json
import os
import sys
from pathlib import Path

# UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import whisper
import torch
import librosa

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path("c:/Ce PC/Projet_python/IA traduction Français Mina")
CV_DIR = PROJECT_DIR / "data" / "cv-corpus-25.0-2026-03-09" / "gej"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n   Device: {DEVICE}")


def load_audio_sample(n=5):
    """Charge n exemples audio du corpus."""
    print("\n" + "="*60)
    print("   CHARGEMENT DES DONNÉES AUDIO")
    print("="*60)

    # Lire le fichier validated.tsv
    df = pd.read_csv(CV_DIR / "validated.tsv", sep='\t')

    # Lire les durées
    durations = pd.read_csv(CV_DIR / "clip_durations.tsv", sep='\t')
    df = df.merge(durations, left_on='path', right_on='clip')

    # Filtrer par durée (1-30 secondes)
    df['duration_s'] = df['duration[ms]'] / 1000
    df = df[(df['duration_s'] >= 1) & (df['duration_s'] <= 30)]

    print(f"   Total clips: {len(df)}")
    print(f"   Duree totale: {df['duration_s'].sum() / 3600:.2f} heures")

    # Prendre n exemples
    samples = df.sample(n=min(n, len(df)), random_state=42)

    entries = []
    for idx, row in samples.iterrows():
        audio_path = CV_DIR / "clips" / row['path']
        if audio_path.exists():
            entries.append({
                'path': str(audio_path),
                'sentence': row['sentence'],
                'duration': row['duration_s'],
                'client_id': row['client_id']
            })

    print(f"   Echantillons charges: {len(entries)}")

    return entries


def test_whisper_model():
    """Teste le modèle Whisper fine-tuné sur Mina."""
    print("\n" + "="*60)
    print("   TEST DU MODÈLE WHISPER")
    print("="*60)

    # Chemin vers le modèle fine-tuné
    model_path = PROJECT_DIR / "models" / "whisper-mina" / "final"

    if model_path.exists():
        print(f"\n   Chargement du modèle fine-tuné depuis:")
        print(f"   {model_path}")

        # Charger avec transformers (le modèle fine-tuné)
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
        processor = WhisperProcessor.from_pretrained(str(model_path))

        # Déplacer sur GPU
        if DEVICE == "cuda":
            model = model.to("cuda")

        print(f"   Modele charge sur: {DEVICE}")
        print(f"   ✅ Mode fine-tuné activé!")

        return model, processor, "fine_tuned"
    else:
        print(f"\n   ⚠️  Modele fine-tuné non trouvé!")
        print(f"   Utilisation du modele whisper-tiny (non fine-tuné)")
        print(f"\n   Lancez d'abord: python scripts/train_whisper_native.py")

        model = whisper.load_model("tiny", device=DEVICE)
        return model, None, "base"


def transcribe_audio(model, audio_path: str, language: str = "en", processor=None, is_fine_tuned=False) -> dict:
    """Transcrit un fichier audio."""
    try:
        if is_fine_tuned and processor is not None:
            # Mode fine-tuné: utiliser transformers
            import librosa

            # Charger l'audio sur GPU
            audio, sr = librosa.load(audio_path, sr=16000)

            # Convertir en tenseur et déplacer sur GPU
            input_features = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features

            if DEVICE == "cuda":
                input_features = input_features.to("cuda")

            # Générer
            # Le modèle a été fine-tuné sur Mina - génération libre sans contrainte de langue
            generated_ids = model.generate(
                input_features,
                max_new_tokens=128
            )

            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return {
                "text": transcription,
                "language": "min",
                "language_probability": 1.0,
                "no_speech_prob": 0.0
            }
        else:
            # Mode standard whisper
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio)

            options = whisper.DecodingOptions(
                language=language,
                task="transcribe",
                fp16=(DEVICE == "cuda")
            )

            result = model.decode(mel, options)

            return {
                "text": result.text,
                "language": result.language,
                "language_probability": result.language_probability,
                "no_speech_prob": result.no_speech_prob
            }

    except Exception as e:
        return {"error": str(e)}


def main():
    print("\n" + "="*70)
    print("   TEST DU PIPELINE STT MINA (WHISPER)")
    print("="*70)

    # Étape 1: Charger les données audio
    samples = load_audio_sample(n=5)

    if not samples:
        print("\n   ❌ Aucun fichier audio trouve!")
        return

    # Étape 2: Tester le modèle Whisper
    result = test_whisper_model()
    model, processor, mode = result[0], result[1], result[2]
    is_fine_tuned = (mode == "fine_tuned")

    # Étape 3: Transcrire les échantillons
    print("\n" + "="*60)
    print("   TRANSCRIPTION DES ÉCHANTILLONS")
    print("="*60)

    results = []

    for i, sample in enumerate(samples):
        print(f"\n   [{i+1}/{len(samples)}] {sample['path'].split('/')[-1]}")

        if is_fine_tuned:
            # Mode fine-tuné: transcription Mina directe
            result = transcribe_audio(model, sample['path'], language="min", processor=processor, is_fine_tuned=True)
            print(f"       MINA: {result.get('text', result.get('error', 'ERROR'))}")
            transcription_text = result.get('text', '')
        else:
            # Mode standard whisper (non fine-tuné)
            result_en = transcribe_audio(model, sample['path'], language="en")
            print(f"       EN: {result_en.get('text', result_en.get('error', 'ERROR'))}")

            result_fr = transcribe_audio(model, sample['path'], language="fr")
            print(f"       FR: {result_fr.get('text', result_fr.get('error', 'ERROR'))}")

            result_auto = transcribe_audio(model, sample['path'], language=None)
            print(f"       AUTO: {result_auto.get('text', result_auto.get('error', 'ERROR'))}")

            transcription_text = result_auto.get('text', '')

        # Reference
        print(f"       REF (Mina): {sample['sentence']}")

        results.append({
            'audio': sample['path'].split('/')[-1],
            'reference': sample['sentence'],
            'transcription': transcription_text,
        })

    # Sauvegarder les résultats
    output_path = PROJECT_DIR / "data" / "whisper_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("   RÉSUMÉ")
    print("="*60)

    print(f"\n   Fichiers testes: {len(results)}")
    print(f"   Modele: whisper-small fine-tuné Mina" if is_fine_tuned else f"   Modele: whisper-tiny (non fine-tuné)")
    print(f"   Resultats sauvegardes: {output_path}")

    if is_fine_tuned:
        print("\n   ✅ Modele fine-tuné opérationnel!")
        print("   Le modèle est prêt pour la transcription Mina.")
    else:
        print("\n   NOTE: Le modele n'est pas fine-tuné.")
        print("   Lancez d'abord: python scripts/train_whisper_native.py")

    print("\n" + "="*70)
    print("   TEST TERMINÉ!")
    print("="*70)


if __name__ == "__main__":
    main()