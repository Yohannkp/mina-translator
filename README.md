# Mina-Translator

Traduction **français ↔ mina** par LLM affiné, avec transcription de la parole en amont. Le mina (aussi appelé gen, code ISO `gej`) est parlé dans le sud du Togo et au Bénin ; c'est une langue peu dotée : quasiment aucun corpus parallèle public, aucun modèle de traduction disponible.

Ce projet part de là. Il construit le corpus qui manque, affine un petit modèle dessus, et sert le tout derrière une API.

---

## Le problème

Affiner un modèle de traduction est la partie facile. Sur une langue peu dotée, **la difficulté est la donnée** :

- aucun corpus parallèle français–mina n'existe publiquement ;
- l'orthographe n'est pas standardisée, les mêmes mots s'écrivent de plusieurs façons selon le locuteur ;
- les rares ressources audio (Common Voice `gej`) sont transcrites en mina, mais **sans traduction française**.

Le projet répond aux trois points : un corpus parallèle construit à la main, une application de collecte participative pour l'étendre, et un pipeline d'entraînement qui tient sur un GPU grand public.

---

## Architecture

```
  Audio (mina)                      Texte (français)
       │                                   │
       ▼                                   │
┌──────────────┐                           │
│   Whisper    │  transcription             │
│   (STT)      │                            │
└──────┬───────┘                            │
       │ texte mina                         │
       ▼                                    ▼
┌────────────────────────────────────────────────┐
│   Qwen2-0.5B-Instruct + adaptateur LoRA         │
│   traduction bidirectionnelle FR ↔ Mina         │
└────────────────────┬───────────────────────────┘
                     ▼
              ┌──────────────┐
              │  API FastAPI  │  /translate  /transcribe  /pipeline
              └──────────────┘
```

Le modèle de base fait 0,5 milliard de paramètres et l'adaptateur LoRA pèse ~35 Mo : l'ensemble tient dans **environ 470 Mo de VRAM**, ce qui permet de le faire tourner sur une machine ordinaire plutôt que sur un serveur dédié. C'était une contrainte de conception, pas un accident : le public visé est au Togo.

---

## Données

| Ressource | Volume | Origine |
|---|---|---|
| Corpus parallèle FR ↔ Mina | **500 paires** (`parallel_corpus.jsonl`) | construit à la main — santé, marché, transport |
| Corpus fusionné et nettoyé | **494 paires** (`corpus_merged.jsonl`) | dédoublonnage + validation du précédent |
| Transcriptions mina | **19 605 lignes** (`mina_full_dataset.jsonl`) | Common Voice 25.0, sous-ensemble `gej` |
| Clips audio | **16 773 fichiers** | idem, utilisés pour l'axe reconnaissance vocale |

Les données Common Voice sont sous licence CC0. Le corpus parallèle est le travail original du projet.

**Pourquoi le dépôt est volumineux (~220 Mo)** : les clips audio sont versionnés pour que le pipeline soit reproductible sans re-télécharger l'archive Common Voice complète. Si seul le code vous intéresse, un clone partiel suffit :

```bash
git clone --filter=blob:none --sparse https://github.com/Yohannkp/mina-translator.git
cd mina-translator && git sparse-checkout set --no-cone '/*' '!/data'
```

---

## Installation

Python 3.10+ et, pour l'entraînement, un GPU NVIDIA avec 6 Go de VRAM au minimum.

```bash
git clone https://github.com/Yohannkp/mina-translator.git
cd mina-translator
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Pour le GPU, installer PyTorch compilé pour votre version de CUDA **avant** le reste :

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Configuration : copier `.env.example` en `.env` et l'adapter. Aucune clé n'est nécessaire pour un usage local.

---

## Utilisation

### Servir l'API

```bash
python start_api.py            # http://localhost:8000
python start_api.py --port 8080
```

Documentation interactive sur `http://localhost:8000/docs`.

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | état du service et du modèle chargé |
| `/translate` | POST | français → mina |
| `/translate_mina` | POST | mina → français |
| `/translate/batch` | POST | traduction par lot |
| `/transcribe` | POST | audio → texte (Whisper) |
| `/pipeline` | POST | audio mina → texte français, chaîne complète |
| `/languages` | GET | langues et directions disponibles |

```bash
curl -X POST http://localhost:8000/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "La santé est importante.", "source_lang": "fr", "target_lang": "mina"}'
```

### Réentraîner

```bash
python scripts/train_mina_llm.py --epochs 3
python scripts/train_mina_llm.py --dry-run     # 50 pas, pour valider la configuration
python scripts/test_inference.py
```

### Collecte participative

Une application Streamlit présente un clip audio mina et demande sa traduction française, pour étendre le corpus parallèle avec l'aide de locuteurs natifs :

```bash
streamlit run crowdsource_app.py
```

Les contributions vont dans SQLite, avec une synchronisation optionnelle vers Google Sheets (`scripts/sync_google_sheets.py`, mise en place détaillée dans `docs/SETUP_GOOGLE_SHEETS.md`).

---

## Configuration d'entraînement

| Paramètre | Valeur |
|---|---|
| Modèle de base | `Qwen/Qwen2-0.5B-Instruct` |
| Quantification | 4 bits, QLoRA NF4 |
| LoRA `r` / `alpha` | 16 / 32 |
| Longueur maximale | 256 tokens |
| Taille de lot | 2 |
| Taux d'apprentissage | 2e-4 |
| Époques | 3 |

Inférence mesurée entre **900 et 1500 ms** par requête sur GPU portable (RTX 4060, 8 Go), soit un débit d'environ 40 requêtes par minute — largement suffisant pour un usage interactif, insuffisant pour du trafic soutenu sans mise en lot.

---

## Où en est le modèle — honnêtement

Le modèle produit des traductions **approximatives et pas encore exploitables en production**. Sur les phrases courtes du domaine couvert par le corpus, il retrouve souvent le sens ; dès qu'il sort de ce domaine, il tronque ou invente.

| Français | Sortie du modèle | Lecture |
|---|---|---|
| Combien ça coûte ? | `Ha kae?` | correct |
| Je vais au marché. | `Mu gba soe la.` | sens transmis, formulation approximative |
| L'enfant est à l'école. | `L'enfant ecole.` | **échec** — le modèle recopie le français |

C'est le résultat attendu avec 500 paires d'entraînement. La limite n'est ni le modèle ni la méthode : **c'est la taille du corpus**, et c'est pour cela que la collecte participative existe. Publier ce constat plutôt qu'un tableau de succès me paraît plus utile à quiconque voudrait reprendre le travail.

---

## Structure

```
mina-translator/
├── api/                  API FastAPI — routes, authentification, quotas, journalisation
├── ml/
│   ├── translation/      pipeline de traduction (chargement LoRA, inférence)
│   └── stt/              pipeline de reconnaissance vocale
├── modules/              briques STT et TTS
├── data/
│   ├── corpus/           corpus parallèles et vocabulaire
│   ├── collectors/       chargement de jeux de données, scraping
│   ├── annotation/       outils d'annotation
│   └── cv-corpus-…/gej/  sous-ensemble Common Voice (audio + transcriptions)
├── scripts/              entraînement, évaluation, fusion de corpus, synchronisation
├── webapp/               interface web de démonstration
├── crowdsource_app.py    application Streamlit de collecte participative
├── start_api.py          point d'entrée de l'API
└── docs/                 notes techniques, guides de mise en place
```

---

## Suite

1. **Étendre le corpus** — commerce, administration, urgences. C'est le seul levier qui change vraiment la qualité.
2. **Affiner Whisper** sur le sous-ensemble `gej` : le Whisper générique ne connaît pas le mina, la transcription est le point faible de la chaîne audio.
3. **Brancher la synthèse vocale** : le module existe (`modules/tts.py`, Coqui XTTS v2) mais n'est pas câblé dans la chaîne principale de l'API — c'est ce qui manque pour fermer la boucle parole → parole.
4. **Évaluation chiffrée** — BLEU / chrF sur un jeu de test tenu à l'écart, ce qui manque aujourd'hui pour mesurer les progrès autrement qu'à l'œil.

---

## Licence

Code : à définir. Données Common Voice : CC0. Corpus parallèle original : libre de réutilisation avec attribution.
