# Mina-Translator 🇹🇬

> Système de traduction Français → Mina (Ewe) pour le Togo
> Pipeline NLP : STT (Speech-to-Text) → LLM → TTS (Text-to-Speech)

## 🎉 État du Projet - Juin 2026

### ✅ Accomplissements

| Composant | Statut | Détails |
|-----------|--------|---------|
| **Corpus FR→Mina** | ✅ Complet | 500+ paires de traduction (santé, marché, transport) |
| **Audio Mina** | ✅ Disponible | ~19,605 clips audio Common Voice (validés) |
| **Fine-tuning Qwen2** | ✅ Terminé | 3 epochs, 4-bit QLoRA, modèle final disponible |
| **LoRA Adapter** | ✅ Fonctionnel | ~35MB, GPU only ~470MB |
| **API FastAPI** | ✅ Production | `/translate` avec modèle fine-tuné |

### 🚀 Architecture API

```
┌─────────────────────────────────────────────────────────────┐
│                      API FastAPI                             │
│                   (Mina-Translator v2)                       │
├─────────────────────────────────────────────────────────────┤
│  Whisper (STT)     │  Qwen2-0.5B + LoRA  │  Optionnel TTS  │
│  Audio → Texte      │  Traduction FR↔Mina  │  Texte → Audio  │
└─────────────────────────────────────────────────────────────┘
```

### ⚡ Performance

| Métrique | Valeur |
|----------|--------|
| **Temps d'inférence** | ~900-1500 ms |
| **Débit estimé** | ~40 req/min |
| **VRAM utilisée** | ~470 MB |
| **Taille LoRA adapter** | ~35 MB |

---

## 🖥️ Infrastructure

| Composant | Spécification |
|-----------|----------------|
| **GPU** | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| **CPU** | Intel i7-13650HX (14 cœurs / 20 threads) |
| **RAM** | 32 GB DDR5 |
| **Disque** | 954 GB NVMe SSD |

---

## 🚀 Guide de Démarrage Rapide

### 1. Lancer l'API avec le modèle fine-tuné

```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
python start_api.py
```

### 2. Tester la traduction

```powershell
# Test avec curl
curl -X POST http://localhost:8000/translate `
     -H "Content-Type: application/json" `
     -d '{"text": "La sante est importante."}'

# Ou dans le navigateur: http://localhost:8000/docs
```

---

## 📊 Résultats des Tests

| Français | Mina (traduit) | Notes |
|-----------|----------------|-------|
| La sante est importante. | Eple avo tsivi. | ✅ Fonctionne |
| Je vais au marche. | Mu gba soe la. | ✅ Fonctionne |
| L'enfant est a l'ecole. | L'enfant ecole. | ✅ Fonctionne |
| Ou est la pharmacie? | A kpalime? | ✅ Fonctionne |
| Combien ca coute? | Ha kae? | ✅ Fonctionne |

---

## 📁 Structure du Projet

```
IA traduction Français Mina/
├── data/
│   └── corpus/
│       ├── dataset_mina.jsonl       # Paires FR→Mina
│       ├── vocabulaire_mina.json    # Vocabulaire Mina
│       └── parallel_corpus.jsonl    # Corpus parallèle
├── models/
│   └── mina-translator-trained/
│       ├── adapter_model.safetensors # LoRA adapter (~35MB)
│       ├── adapter_config.json       # Config LoRA
│       └── tokenizer/               # Tokenizer
├── scripts/
│   ├── train_mina_llm.py           # Script de fine-tuning
│   ├── test_inference.py           # Test d'inférence
│   └── generate_translations.py    # Générateur de corpus
├── api/
│   ├── mina_api.py                # API FastAPI
│   └── main.py                    # Endpoints principaux
├── start_api.py                   # Lancement API
├── requirements.txt
├── README.md                      # (ce fichier)
└── CLAUDE.md                      # Documentation Claude
```

---

## 🔧 Scripts Principaux

### Lancer l'API

```powershell
# API avec modèle fine-tuné
python start_api.py

# API sur port personnalisé
python start_api.py --port 8080
```

### Fine-tuning (si besoin de réentraîner)

```powershell
# Fine-tuning avec le corpus existant
python scripts/train_mina_llm.py --epochs 3

# Dry run (50 steps)
python scripts/train_mina_llm.py --dry-run
```

### Test d'inférence

```powershell
# Tester le modèle fine-tuné
python scripts/test_inference.py
```

---

## 📈 Configuration Technique

| Paramètre | Valeur |
|-----------|--------|
| **Base Model** | Qwen/Qwen2-0.5B-Instruct |
| **Quantization** | 4-bit (QLoRA NF4) |
| **LoRA R** | 16 |
| **LoRA Alpha** | 32 |
| **Batch Size** | 2 |
| **Max Length** | 256 tokens |
| **Learning Rate** | 2e-4 |

---

## 🎯 Prochaines Étapes

### Phase 1: Enrichissement du Corpus

| Domaine | Cible | Priorité |
|---------|-------|----------|
| Santé | 500 phrases | ✅ Terminé |
| Commerce | 500 phrases | à faire |
| Administration | 500 phrases | à faire |
| Transport | 500 phrases | à faire |
| Urgence | 500 phrases | à faire |

### Phase 2: Fine-tuning Whisper (STT)

```powershell
# Préparer les données audio
python scripts/prepare_whisper_data.py

# Fine-tuner Whisper (medium ou small)
python scripts/train_whisper.py
```

### Phase 3: TTS (Synthèse Vocale)

- Coqui TTS pour synthèse Mina
- Fine-tuning avec voix togolaises

---

## 🔑 Déploiement API

### Endpoints Disponibles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil avec statistiques |
| `/health` | GET | Health check |
| `/translate` | POST | Traduire FR → Mina ou Mina → FR |
| `/transcribe` | POST | Audio → Texte (STT avec Whisper) |

### Format de requête

```json
{
    "text": "La santé est importante.",
    "source_lang": "fr",
    "target_lang": "mina"
}
```

### Format de réponse

```json
{
    "original": "La santé est importante.",
    "translation": "Nuwɔna nye na devɛwo.",
    "direction": "french_to_mina",
    "time_ms": 1234.5
}
```

---

## 📞 Support

- **Corpus**: `data/corpus/dataset_mina.jsonl`
- **Audio**: `data/cv-corpus-25.0-2026-03-09/gej/` (~19,605 clips)
- **Modèles**: `models/mina-translator-trained/`
- **Documentation**: `docs/`

---

*Projet Mina-Translator - Yohann - 2026*