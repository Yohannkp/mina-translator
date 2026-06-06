# Mina-Translator - État du Projet

**Date**: 2026-06-05  
**Version**: 1.0.0

---

## ✅ Progrès Accomplis

### 1. Modèle de Traduction (LLM)

| Composant | Détails |
|-----------|---------|
| **Modèle de base** | Qwen/Qwen2-0.5B-Instruct |
| **Technique** | QLoRA 4-bit |
| **Corpus** | 360 paires FR-Mina |
| **Epochs** | 3 |
| **VRAM utilisée** | ~4-5 Go |
| **Emplacement** | `models/mina-translator-trained/` |
| **Statut** | ✅ Entraîné |

### 2. Modèle STT (Speech-to-Text)

| Composant | Détails |
|-----------|---------|
| **Modèle de base** | openai/whisper-small |
| **Dataset** | 19,605 clips audio Common Voice Mina |
| **Training steps** | 500 |
| **Loss final** | 0.007755 |
| **WER estimé** | ~15-20% |
| **Emplacement** | `models/mina-whisper-v1/` |
| **Statut** | ✅ Entraîné |

### 3. API FastAPI

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/health` | GET | Health check |
| `/translate` | POST | Traduire FR → Mina |
| `/translate_mina` | POST | Traduire Mina → FR |
| `/translate/batch` | POST | Traduction par lot (max 10) |
| `/languages` | GET | Langues supportées |

### 4. Scripts Disponibles

| Script | Usage |
|--------|-------|
| `scripts/train_mini_llm.py` | Fine-tune Qwen2 sur corpus Mina |
| `scripts/train_whisper_simple.py` | Fine-tune Whisper sur audio Mina |
| `scripts/test_api.py` | Test l'API |
| `scripts/inference_test.py` | Test d'inférence direct |

---

## 🚀 Lancement Rapide

### 1. Lancer l'API

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py -m uvicorn api.main:app --reload --port 8000
```

### 2. Tester l'API

```bash
# Dans un autre terminal
py scripts/test_api.py
```

### 3. Documentation Swagger

Ouvrir: http://localhost:8000/docs

---

## 📁 Structure des Fichiers

```
IA traduction Français Mina/
├── api/
│   └── main.py              # API FastAPI production-ready
├── models/
│   ├── mina-translator-trained/   # Modèle LLM fine-tuné
│   └── mina-whisper-v1/           # Modèle STT fine-tuné
├── scripts/
│   ├── train_mini_llm.py          # Entraînement LLM
│   ├── train_whisper_simple.py    # Entraînement STT
│   ├── test_api.py                # Tests API
│   └── inference_test.py           # Tests inférence
├── data/
│   ├── corpus/
│   │   └── parallel_corpus.jsonl  # Corpus FR-Mina
│   └── whisper_data/
│       └── audio/                  # Audio Common Voice Mina
├── docs/
│   └── API_B2B_DOCUMENTATION.md    # Documentation B2B
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 🔄 Prochaines Étapes

1. **Validation humaine des traductions** - Vérifier la qualité des traductions générées
2. **Enrichissement du corpus** - Passer de 360 à 1000+ paires FR-Mina
3. **Intégration STT + Traduction** - Pipeline audio → transcription → traduction
4. **Déploiement production** - Mettre en ligne l'API
5. **Tests de charge** - Valider les performances avec charge réelle

---

## 📊 Performance Actuelle

| Métrique | Valeur |
|----------|--------|
| Temps d'inférence moyen | ~1000-2000 ms |
| Taille modèle LLM | ~1 Go (compressé) |
| Taille modèle STT | ~300 Mo |
| Précision traduction | À valider manuellement |
| WER STT | ~15-20% (estimé) |

---

## 🔧 Configuration Matériel

- **GPU**: NVIDIA RTX 4060 Laptop (8 Go VRAM)
- **CPU**: Intel i7-13650HX (14 cœurs)
- **RAM**: 32 Go DDR5
- **Disque**: 954 Go NVMe SSD