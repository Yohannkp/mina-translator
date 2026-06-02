# Mina-Translator 🇹🇬

> Système de traduction Français → Mina (Ewe) pour le Togo
> Pipeline NLP : STT (Speech-to-Text) → LLM → TTS (Text-to-Speech)

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Infrastructure](#-infrastructure)
- [Installation](#-installation)
- [Structure du Projet](#-structure-du-projet)
- [Guide de Démarrage Rapide](#-guide-de-démarrage-rapide)
- [Entraînement du Modèle](#-entraînement-du-modèle)
- [API FastAPI](#-api-fastapi)
- [Test d'Inférence](#-test-dinf%C3%A9rence)
- [Corpus Mina](#-corpus-mina)
- [Prochaines Étapes](#-prochaines-étapes)
- [FAQ](#-faq)

---

## 🎯 Aperçu

Ce projet vise à créer une API de traduction/transcription pour favoriser l'inclusion numérique au Togo, permettant de traduire et transcrire des échanges vocaux entre le Français et le Mina.

### Architecture Cible

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   Audio     │───▶│  STT (Whisper)│───▶│  LLM (Qwen) │───▶│ TTS (Coqui) │
│   Input     │    │              │    │  Traduction │    │   Output    │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                            │
                                    ┌───────┴───────┐
                                    │  Mina ↔ Français │
                                    └───────────────┘
```

---

## 🖥️ Infrastructure

| Composant | Spécification |
|-----------|----------------|
| **GPU** | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| **CPU** | Intel i7-13650HX (14 cœurs / 20 threads) |
| **RAM** | 32 GB DDR5 |
| **Disque** | 954 GB NVMe SSD |
| **OS** | Windows 11 |

### Optimisations pour 8 GB VRAM

- **Quantification** : QLoRA 4-bit (obligatoire)
- **Batch size** : 2-4 (effective via gradient accumulation)
- **Max sequence length** : 128-256 tokens
- **Modèle推荐**: Qwen2-0.5B-Instruct ou Phi-3-mini-4k-instruct

---

## 📦 Installation

### 1. Prérequis

```bash
# Python 3.10+
python --version  # >= 3.10

# CUDA 11.8+ pour PyTorch
nvidia-smi  # Vérifier CUDA
```

### 2. Cloner/Copier le projet

```bash
cd "c:/Ce PC/Projet_python/"
# Le projet est déjà présent dans ce dossier
```

### 3. Créer l'environnement Virtuel

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 4. Installer les Dépendances

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
pip install datasets scipy
pip install fastapi uvicorn loguru
pip install huggingface_hub
```

Ou utiliser `requirements.txt` existant :

```bash
pip install -r requirements.txt
```

---

## 📁 Structure du Projet

```
IA traduction Français Mina/
├── data/
│   └── corpus/
│       ├── corpus_mina_5000.jsonl      # Corpus original
│       └── final_clean_dataset.jsonl   # Corpus nettoyé
├── models/
│   ├── mina-translator/               # Modèle entraîné (LoRA)
│   │   ├── checkpoint-epoch-1/
│   │   ├── checkpoint-epoch-2/
│   │   ├── checkpoint-epoch-3/
│   │   └── final/                     # ← Version finale
│   └── mina-translator-final-backup/  # ← BACKUP DE SÉCURITÉ
├── scripts/
│   ├── train_minimal.py               # Script d'entraînement
│   ├── inference_test.py             # Test d'inférence
│   └── generate_corpus.py             # Générateur de phrases
├── api/
│   └── main.py                        # API FastAPI
├── docs/
│   ├── PLAN_ENRICHISSEMENT.md
│   └── README.md                      # (ce fichier)
├── requirements.txt
├── README.md                           # ← README principal
└── CLAUDE.md                           # Documentation pour Claude
```

---

## 🚀 Guide de Démarrage Rapide

### Option 1 : Tester le Modèle Entraîné

```bash
# Lancer le test d'inférence
python scripts/inference_test.py
```

**Résultat attendu** :
```
=================================================================
MINA TRANSLATOR - TEST D'INFERENCE
=================================================================
[1/5] Francais: Le president a arrive a Lome.
    Mina:      Nusɔla va le ta.
    Temps:     1188.9 ms
...
Temps moyen:  1579.2 ms
=================================================================
```

### Option 2 : Lancer l'API FastAPI

```bash
# Dans un terminal
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
python -m uvicorn api.main:app --reload --port 8000
```

Puis ouvrir : http://localhost:8000/docs

### Option 3 : Relancer l'Entraînement

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
python scripts/train_minimal.py
```

**Durée estimée** : ~6 minutes pour 3 epochs

---

## 🏋️ Entraînement du Modèle

### Configuration Actuelle

| Paramètre | Valeur |
|-----------|--------|
| **Modèle de base** | Qwen/Qwen2-0.5B-Instruct |
| **Technique** | LoRA (r=8, alpha=16) |
| **Corpus** | 360 phrases Mina-Français |
| **Epochs** | 3 |
| **Batch size** | 2 (effective: 8 avec grad accum) |
| **Learning rate** | 2e-4 |
| **Max length** | 128 tokens |
| **VRAM utilisée** | ~4-5 GB |

### Modifier les Paramètres

Éditer `scripts/train_minimal.py` :

```python
# Configuration
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"  # ou "microsoft/Phi-3-mini-4k-instruct"
EPOCHS = 5                                  # Augmenter pour meilleur résultats
BATCH_SIZE = 2
GRAD_ACCUM = 4                              # Effective batch = 8
MAX_LEN = 256                               # Phrases plus longues
LEARNING_RATE = 2e-4
LORA_R = 16                                 # Plus de paramètres LoRA
```

### Reprendre depuis un Checkpoint

```python
# Dans scripts/train_minimal.py, modifier :
model_path = "models/mina-translator/checkpoint-epoch-1"
# OU
model_path = "models/mina-translator/final"
```

### Avec un Nouveau Corpus

1. Préparer le fichier JSONL avec le format :
```json
{"fr": "Phrase en français", "mina": "Phrase en mina", "domain": "sante"}
```

2. Modifier le chemin dans `train_minimal.py` :
```python
corpus_file = Path("data/corpus/nouveau_corpus.jsonl")
```

---

## 🌐 API FastAPI

### Points de Terminaison

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `GET /` | GET | Page d'accueil avec stats |
| `POST /translate` | POST | Traduire texte FR → Mina |
| `POST /translate_mina` | POST | Traduire Mina → Français |
| `GET /health` | GET | Health check |

### Exemple d'Utilisation

```python
import requests

# Traduction FR → Mina
response = requests.post(
    "http://localhost:8000/translate",
    json={"text": "La santé est importante.", "temperature": 0.3}
)
print(response.json())
# {"original": "La santé est importante.", "translation": "Nuwɔna nye na devɛwo."}
```

### Lancer en Production

```bash
# Avec Gunicorn (Linux)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app

# Avec uvicorn (，开发)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## ⚡ Test d'Inférence

Le script `scripts/inference_test.py` permet de :

1. Charger le modèle fine-tuné depuis `models/mina-translator/final`
2. Tester avec 5 phrases françaises inédites
3. Mesurer le temps d'inférence en millisecondes
4. Analyser la qualité des traductions

```bash
python scripts/inference_test.py
```

**Paramètres recommandés pour l'inférence** :

```python
{
    "max_new_tokens": 60,
    "temperature": 0.3,      # Basse = moins d'hallucinations
    "do_sample": True,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 1.2
}
```

---

## 📚 Corpus Mina

### Format JSONL

```json
{"fr": "Phrase française", "mina": "Phrase mina", "domain": "domaine"}
```

### Domaines Couverts

| Domaine | Phrases | Priorité |
|---------|---------|----------|
| Santé | 52 | 🔴 Haute |
| Famille | 52 | 🔴 Haute |
| Administration | 52 | 🔴 Haute |
| Marché | 51 | 🟡 Moyenne |
| Transport | 51 | 🟡 Moyenne |
| Éducation | 51 | 🟡 Moyenne |
| Justice | 50 | 🟡 Moyenne |
| Religion | 1 | ⚠️ À enrichir |

### Enrichir le Corpus

1. **Génération automatique** (voir `scripts/generer_corpus.py`)
2. **Collecte manuelle** via l'outil d'annotation (`mina_annotator.py`)
3. **Web scraping** de sources togolaises

---

## 🔮 Prochaines Étapes

### Phase 1 : Amélioration du Modèle (1-2 semaines)

- [ ] Enrichir le corpus à 2000+ phrases
- [ ] Ajouter des domaines manquants (religion, agriculture, technologie)
- [ ] Entraîner 5-10 epochs supplémentaires
- [ ] Tester avec Qwen2-1.5B ou Phi-3-mini

### Phase 2 : Intégration STT/TTS (2-4 semaines)

- [ ] Fine-tuner Whisper pour le Mina
- [ ] Intégrer Coqui TTS pour la synthèse vocale
- [ ] Pipeline audio complet

### Phase 3 : Déploiement (4-8 semaines)

- [ ] Optimiser l'inférence (ONNX, TensorRT)
- [ ] Déployer sur serveur/cloud
- [ ] Tests A/B et monitoring

---

## ❓ FAQ

### Q: Le modèle ne traduit pas bien les phrases complexes

**R**: Le corpus actuel (360 phrases) est limité. Enrichissez-le avec des exemples dans le domaine concerné. Un corpus de 2000+ phrases est recommandé pour une qualité acceptable.

### Q: L'entraînement échoue avec "CUDA out of memory"

**R**: Réduisez le batch size et max_length, ou utilisez la quantification QLoRA 4-bit. Voir `scripts/train_minimal.py` pour les paramètres optimisés pour 8 GB VRAM.

### Q: Comment ajouter une nouvelle langue ?

**R**: Dupliquez le corpus avec la nouvelle langue source/cible, puis relancez l'entraînement avec un nouveau modèle de base multilingue.

### Q: Le modèle hallucine / génère du texte incohérent

**R**: Baissez la température (0.2-0.3) et augmentez le repetition_penalty. Voir `scripts/inference_test.py`.

### Q: Comment faire une sauvegarde complète ?

**R**:
```bash
# Sauvegarder modèle + corpus + configs
tar -czf mina-translator-backup.tar.gz \
    models/mina-translator-final-backup/ \
    data/corpus/ \
    scripts/ \
    api/
```

---

## 📞 Support

Pour toute question sur le projet, vérifier :
1. [docs/PLAN_ENRICHISSEMENT.md](docs/PLAN_ENRICHISSEMENT.md) - Plan de développement
2. Les logs dans `models/mina-translator/training.log`
3. Les checkpoints dans `models/mina-translator/checkpoint-epoch-*/`

---

## 📄 Licence

Projet développé dans le cadre de l'inclusion numérique au Togo.

**Dernière mise à jour** : 2 juin 2026  
**Version** : 0.1.0 (prototype)