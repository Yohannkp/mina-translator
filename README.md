# Mina-Translator: Système de Traduction Mina-Français

API pour la traduction et transcription vocale entre le Mina (langue du Togo) et le Français.

## 🗂️ Structure du Projet

```
IA traduction Français Mina/
├── api/
│   └── main.py              # API FastAPI
├── config/
│   └── settings.py          # Configuration
├── data/
│   └── corpus/
│       └── dataset_mina.jsonl  # Dataset Mina (500 phrases)
├── modules/
│   ├── stt.py              # Speech-to-Text (Whisper)
│   └── tts.py             # Text-to-Speech
├── scripts/
│   ├── generate_dataset.py  # Génère le dataset
│   └── train_translation.py # Fine-tuning QLoRA
├── tests/
│   └── test_api.py         # Tests API
├── requirements.txt        # Dépendances
└── README.md              # Ce fichier
```

## 🚀 Installation

```bash
# Cloner ou accéder au projet
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"

# Créer un environnement virtuel (recommandé)
python -m venv venv
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📦 Dépendances Principales

- **Transformers** - Modèles de langue (HuggingFace)
- **Peft** - Fine-tuning LoRA/QLoRA
- **BitsAndBytes** - Quantification 4-bit
- **Whisper** - Transcription audio
- **FastAPI** - API web
- **Edge-TTS** - Synthèse vocale (légère)

## 🎯 Utilisation

### 1. Générer le Dataset

```bash
python scripts/generate_dataset.py
```

Génère 500 paires de traduction FR → Mina dans `data/corpus/dataset_mina.jsonl`.

### 2. Fine-tuning (Optionnel)

```bash
python scripts/train_translation.py
```

Fine-tune un modèle (Phi-3-mini ou Llama) avec QLoRA sur GPU (8 Go VRAM requis).

**Temps estimé:** 2-4 heures sur RTX 4060

### 3. Lancer l'API

```bash
# Méthode 1: Direct
python -m uvicorn api.main:app --reload --port 8000

# Méthode 2: Script
python api/main.py
```

L'API sera disponible sur `http://localhost:8000`

### 4. Tester l'API

```bash
# Santé
curl http://localhost:8000/health

# Traduction FR → Mina
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour", "source_lang": "fr", "target_lang": "mina"}'

# Documentation Swagger
# http://localhost:8000/docs
```

## 📡 Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | État de l'API et GPU |
| `/translate` | POST | Traduit du texte |
| `/transcribe` | POST | Transcrit de l'audio |
| `/speak` | POST | Synthétise la parole |

## 🎤 Exemples de Traduction

| Français | Mina |
|----------|------|
| Bonjour, comment allez-vous? | Akpe ɖe o, efy nye? |
| Je vais bien merci. | Mele go ye. |
| Combien ça coûte? | Elɛ womɛna? |
| J'ai mal à la tête. | Mekɔ ta. |

## 💻 Configuration Matérielle Requise

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| GPU | 6 Go VRAM | 8+ Go VRAM |
| RAM | 16 Go | 32 Go |
| CPU | 4 cœurs | 8+ cœurs |
| Stockage | 20 Go | 50 Go SSD |

## 🔧 Configuration

Variables d'environnement optionnelles:

```bash
# API Anthropic (génération dataset)
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_AUTH_TOKEN=votre_token

# Dossier de sortie (par défaut: ./data)
OUTPUT_DIR=./data

# Device (cuda ou cpu)
DEVICE=cuda
```

## 📚 Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Utilisateur   │────▶│   API FastAPI   │
└─────────────────┘     └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Traduction  │       │  Transcription│       │  Synthèse     │
│   (LLM+QLoRA) │       │   (Whisper)   │       │   (Edge-TTS)  │
└───────────────┘       └───────────────┘       └───────────────┘
```

## 🔄 Pipeline STT → Traduction → TTS

1. **STT (Speech-to-Text)**: Whisper transcrit l'audio Mina/Français
2. **LLM**: Fine-tuned Mina-Translator traduit FR ↔ Mina
3. **TTS**: Edge-TTS synthétise la réponse en audio

## 🐛 Dépannage

### Erreur CUDA Out of Memory
```bash
# Réduire la taille du batch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Whisper slow
```bash
# Utiliser un modèle plus petit
python -m modules.stt --model tiny
```

### edge-tts non trouvé
```bash
pip install edge-tts
```

## 📄 Licence

Projet développé pour l'inclusion numérique au Togo.

## 👤 Auteur

YO-HAR