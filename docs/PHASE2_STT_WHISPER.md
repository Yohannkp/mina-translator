# Phase 2: Fine-tuning Whisper (STT) pour le Mina

## 📋 Vue d'ensemble

Cette phase permet de fine-tuner Whisper pour créer un modèle de **reconnaissance vocale automatique (ASR/STT)** capable de transcrire le Mina parlé au Togo.

---

## 🎯 Objectifs

| Objectif | Détails |
|----------|---------|
| **Données audio** | ~11.4 heures de audio Mina (Common Voice) |
| **Clips validés** | 16,417 clips avec transcription |
| **Phrases uniques** | 3,188 phrases Mina |
| **Orateurs uniques** | 19 orateurs |

---

## 📊 Données Disponibles

### Common Voice Mina (Ewe) - Togo

```
data/cv-corpus-25.0-2026-03-09/gej/
├── clips/                    # 16,773 fichiers MP3
│   ├── common_voice_gej_43111985.mp3
│   ├── common_voice_gej_43111986.mp3
│   └── ...
├── validated.tsv            # Métadonnées validées
├── dev.tsv                  # Split développement
├── test.tsv                 # Split test
└── train.tsv                # Split train
```

### Statistiques

| Métrique | Valeur |
|----------|--------|
| Durée totale | 11.39 heures |
| Durée moyenne | 2.44 secondes |
| Clips | 16,773 |
| Phrases uniques | 3,188 |
| Orateurs | 19 |

---

## 🚀 Guide de Démarrage

### Étape 1: Préparer les données

```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
.\venv_train\Scripts\python.exe scripts/prepare_whisper_data.py
```

**Ce script:**
1. Charge les métadonnées Common Voice
2. Filtre les clips par durée (1-30 secondes)
3. Convertit les MP3 en WAV 16kHz mono
4. Crée les fichiers JSONL (train/val/test)
5. Génère un rapport de statistiques

**Sortie:**
```
data/whisper_data/
├── train.jsonl       # 85% des données
├── validation.jsonl  # 10% des données
├── test.jsonl        # 5% des données
└── prepare_report.json
```

### Étape 2: Tester le pipeline (Dry-run)

```powershell
# Tester avec quelques échantillons
.\venv_train\Scripts\python.exe scripts/test_whisper_pipeline.py
```

**Résultat attendu:**
```
=== TEST DU PIPELINE STT MINA (WHISPER) ===
   Device: cuda
   Total clips: 16,417
   Echantillons charges: 5
   Modele: whisper-tiny (non fine-tuné)
   Transcription: Mu, akutuna.
```

### Étape 3: Lancer le Fine-tuning

```powershell
# Mode complet (3 epochs)
.\venv_train\Scripts\python.exe scripts/train_whisper_native.py --epochs 3

# Mode test (1 epoch, 10 steps)
.\venv_train\Scripts\python.exe scripts/train_whisper_native.py --dry-run

# Avec modèle small (meilleur mais plus VRAM)
.\venv_train\Scripts\python.exe scripts/train_whisper_native.py --model small --epochs 3
```

---

## 📁 Scripts Créés

| Script | Description |
|--------|-------------|
| `prepare_whisper_data.py` | Prépare les données audio pour Whisper |
| `test_whisper_pipeline.py` | Test le pipeline STT avant fine-tuning |
| `train_whisper_native.py` | Script principal de fine-tuning Whisper |

---

## ⚙️ Configuration Recommandée

Pour RTX 4060 Laptop (8 GB VRAM):

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Modèle** | `tiny` ou `base` | tiny=39M params, base=74M params |
| **Batch size** | 8 | Clips par batch |
| **Gradient accumulation** | 4 | Batch effectif = 32 |
| **Learning rate** | 1e-4 | Taux d'apprentissage |
| **Epochs** | 3-5 | Nombre d'epochs |
| **VRAM utilisée** | ~6-7 GB | Avec fp16 |

---

## 🧠 Technique de Fine-tuning

### Architecture Whisper

```
Audio Input (MP3/WAV)
    │
    ▼
┌─────────────────────┐
│  Mel Spectrogram    │  80 mel bins, 3000 frames (30s)
│  Feature Extractor  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Encoder (Conformer)│  Transforme mel → audio features
│  3000 → 1500        │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Decoder (Transformer)│  Génère tokens de transcription
│  cross-attention    │
└─────────────────────┘
    │
    ▼
Text Output (Transcription)
```

### Stratégie de Fine-tuning

1. **Geler l'encodeur** → Only fine-tune le décodeur
2. **LoRA** → Optional (non implémenté dans ce script)
3. **Freeze embeddings** → Garde les embeddings originaux

---

## 📈 Résultats Attendus

| Métrique | Avant Fine-tuning | Après Fine-tuning |
|----------|-------------------|-------------------|
| WER (Word Error Rate) | ~70% | ~30-40% |
| Précision Mina | Faible | Bonne |
| Temps d'inférence | Variable | ~500ms/clips |

---

## 🔧 Dépannage

### Erreur: "No module named 'whisper'"

```powershell
.\venv_train\Scripts\pip.exe install openai-whisper
```

### Erreur: "CUDA out of memory"

Réduire le batch size:
```powershell
.\venv_train\Scripts\python.exe scripts/train_whisper_native.py --batch-size 4
```

### Erreur: "File not found"

Vérifier que les fichiers audio existent:
```powershell
ls "data/cv-corpus-25.0-2026-03-09/gej/clips/" | head -5
```

---

## 📝 Prochaines Étapes

### Phase 3: Intégration LLM + TTS

Après Whisper fine-tuné:

```
Audio Input (Mina)
    │
    ▼
Whisper (fine-tuned) ────▶ Texte (Mina)
    │
    ▼
Qwen2/Phi3 (fine-tuned) ────▶ Traduction (Français)
    │
    ▼
TTS (Coqui/Fish Speech) ────▶ Audio Output (Français)
```

### Tâches

1. **API complète** avec pipeline STT → LLM → TTS
2. **Fine-tuning TTS** avec voix togolaises
3. **Déploiement** sur serveur avec GPU

---

## 📚 Ressources

- [Whisper GitHub](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/whisper)
- [Common Voice Dataset](https://commonvoice.mozilla.org/datasets)

---

*Document généré automatiquement - Yohann - 2026*