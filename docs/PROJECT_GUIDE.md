# Mina-Translator - Système de Traduction/Transcription Mina-Français

## 📋 Table des Matières

- [Installation rapide](#-installation-rapide)
- [Structure du projet](#-structure-du-projet)
- [Scripts disponibles](#-scripts-disponibles)
- [API FastAPI](#-api-fastapi)
- [Fine-tuning des modèles](#-fine-tuning-des-modèles)
- [Corpus de données](#-corpus-de-données)
- [FAQ](#-faq)

---

## 🚀 Installation rapide

```bash
# Cloner le projet
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"

# Créer l'environnement virtuel
py -m venv venv_train
venv_train\Scripts\activate

# Installer les dépendances
pip install torch transformers accelerate peft bitsandbytes
pip install datasets huggingface_hub
pip install fastapi uvicorn loguru
pip install openai-whisper soundfile librosa scipy

# Vérifier CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 Structure du projet

```
IA traduction Français Mina/
├── api/
│   ├── main.py              # API principale
│   ├── mina_api.py          # API v2 avec pipeline complet
│   ├── auth.py              # Authentification
│   ├── database.py          # Base de données SQLite
│   ├── security.py          # Sécurité
│   └── logger.py            # Logging
├── scripts/
│   ├── build_parallel_corpus.py  # Génère traductions FR
│   ├── train_mina_llm.py    # Fine-tune Qwen2
│   ├── fine_tune_whisper.py # Fine-tune Whisper STT
│   ├── speech_to_text_pipeline.py # Pipeline audio->FR
│   └── generate_translations.py   # Alternative Ollama
├── data/
│   ├── corpus/
│   │   ├── mina_ewe_corpus.json     # 16,417 phrases Mina (Common Voice)
│   │   ├── parallel_corpus.jsonl     # Paires Mina->Français (en construction)
│   │   └── training_dataset.jsonl    # Dataset pour training
│   └── cv-corpus-25.0-2026-03-09/
│       └── gej/              # Common Voice Ewe (16k fichiers audio)
├── models/
│   ├── mina-translator/     # Modèle LoRA actuel (Qwen2-0.5B)
│   └── whisper-mina-ewe/     # Whisper fine-tuné (à créer)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 🔧 Scripts disponibles

### 1. Générer le corpus parallèle (Mina → Français)

```bash
# Test rapide (20 phrases)
py scripts/build_parallel_corpus.py --test

# 500 phrases (~2h)
py scripts/build_parallel_corpus.py --sample 500

# Corpus complet (~12h)
py scripts/build_parallel_corpus.py --full
```

### 2. Fine-tuner Qwen2 sur le corpus Mina

```bash
# Dry run (50 steps)
py scripts/train_mina_llm.py --dry-run

# Entraînement complet (3 epochs)
py scripts/train_mina_llm.py --epochs 3

# Test avec peu de données
py scripts/train_mina_llm.py --sample 500 --epochs 1
```

### 3. Fine-tuner Whisper sur l'Ewe (STT)

```bash
# Préparer le dataset
py scripts/fine_tune_whisper.py --prepare

# Lancer l'entraînement
py scripts/fine_tune_whisper.py --train --epochs 3

# Test rapide
py scripts/fine_tune_whisper.py --dry-run
```

### 4. Tester le pipeline audio complet

```bash
# Test avec fichiers Common Voice
py scripts/speech_to_text_pipeline.py --test-common-voice --num-samples 10
```

---

## 🌐 API FastAPI

### Lancer l'API

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py -m uvicorn api.mina_api:app --reload --port 8000
```

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| GET | `/health` | Health check |
| POST | `/translate` | Traduire du texte |
| POST | `/transcribe` | Transcrire audio → Mina |
| POST | `/pipeline` | Audio → Mina → Français |

### Exemples

```bash
# Traduire du Mina vers le Français
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Kopuwo", "direction": "mina_to_french"}'

# Transcrire un fichier audio
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3"

# Pipeline complet (audio → français)
curl -X POST http://localhost:8000/pipeline \
  -F "file=@audio.mp3"
```

---

## 🎯 Fine-tuning des modèles

### Qwen2-0.5B-Instruct (Traducteur)

**Configuration optimisée pour RTX 4060 (8GB VRAM):**

| Paramètre | Valeur |
|-----------|--------|
| Technique | QLoRA 4-bit |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Batch size | 2 (effective: 8 avec grad accum) |
| Learning rate | 2e-4 |
| Max length | 256 tokens |

### Whisper (STT - Speech-to-Text)

**Pour la reconnaissance vocale de l'Ewe:**

| Paramètre | Valeur |
|-----------|--------|
| Modèle | whisper-base ou whisper-small |
| Dataset | Common Voice Ewe (16,417 fichiers) |
| Epochs | 3-5 |
| Batch size | 8 |
| Fine-tuning | LoRA sur les couches d'attention |

---

## 📊 Corpus de données

### Sources

1. **Common Voice Ewe** (Mozilla)
   - 16,417 phrases Mina avec audio
   - Source: `data/cv-corpus-25.0-2026-03-09/gej/`
   - Transcriptions validées dans `validated.tsv`

2. **Corpus parallèle Mina→Français**
   - Généré avec Ollama (Llama 3.1)
   - Stocké dans: `data/corpus/parallel_corpus.jsonl`
   - Nécessite validation humaine

### Format des données

```json
{
  "mina": "Kopuwo",
  "french": "Bonjour (salutation Ewe)",
  "audio": "chemin/vers/fichier.mp3",
  "timestamp": "2026-06-03T10:30:00"
}
```

---

## ❓ FAQ

### Pourquoi Whisper ne reconnaît pas l'Ewe ?

Whisper a été entraîné principalement sur l'anglais et les langues majoritaires. L'Ewe/Mina est une langue à faibles ressources. Il faut **fine-tuner** Whisper sur les données Common Voice Ewe pour qu'il apprenne à reconnaître cette langue.

### Comment améliorer la traduction ?

1. **Générer plus de traductions** avec Ollama
2. **Valider les traductions** manuellement (certaines sont des hallucinations)
3. **Fine-tuner Qwen2** sur le corpus validé
4. **Ajouter des traductions humaines** pour les phrases importantes

### Le modèle est-il prêt pour la production ?

**Non** - Le modèle actuel nécessite:
1. Fine-tuning sur un corpus validé
2. Tests approfondis
3. Validation par des locuteurs Ewe natifs

### Comment contribuer ?

1. Valider les traductions dans `parallel_corpus.jsonl`
2. Corriger les erreurs dans les fichiers JSONL
3. Ajouter des traductions de haute qualité

---

## 📞 Contact

Projet développé par Yohann avec Claude Opus 4.8
Pour le Togo - Inclusion numérique