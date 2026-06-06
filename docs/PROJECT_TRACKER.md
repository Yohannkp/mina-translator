# 📋 MINA-TRANSLATOR - SUIVI DE PROJET COMPLET

**Projet**: Système de traduction/transcription Français ↔ Mina (Ewe du Togo)  
**Machine**: RTX 4060 Laptop (8GB VRAM), i7-13650HX, 32GB RAM, 954GB SSD  
**Date de création**: 2026-06-06  
**Dernière mise à jour**: 2026-06-06

---

## 🎯 OBJECTIFS DU PROJET

### Objectif Principal
Créer une API de traduction/transcription pour favoriser l'inclusion numérique au Togo, permettant de traduire et transcrire des échanges vocaux entre le Français et le Mina (Ewe).

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

## 📊 ÉTAT ACTUEL DU PROJET

| Composant | Statut | Emplacement | Notes |
|-----------|--------|-------------|-------|
| **Corpus Mina** | ✅ Complété | `data/corpus/` | ~19,605 phrases Mina, 360 paires FR-Mina |
| **Audio Common Voice** | ✅ Téléchargé | `data/cv-corpus-25.0-2026-03-09/` | 19,605 clips audio Mina |
| **LLM Qwen2 Fine-tuné** | ✅ Entraîné | `models/mina-translator-trained/` | 3 epochs, QLoRA 4-bit |
| **Whisper STT Fine-tuné** | ✅ Entraîné | `models/mina-whisper-v1/` | 500 steps, loss 0.007755 |
| **API FastAPI** | ⚠️ Partiellement fonctionnel | `api/main.py` | Modèle non chargé au démarrage |
| **WebApp** | ✅ Créé | `webapp/` | Interface vocale complète |
| **Crowdsource App** | ✅ Créé | `crowdsource_app.py` | Application Streamlit pour collecte de traductions |
| **Documentation** | ✅ Créée | `docs/` | Multiple docs créées |

---

## 📝 HISTORIQUE DES ÉTAPES COMPLÉTÉES

### Phase 1: Préparation des Données ✅

| Date | Étape | Détails | Résultat |
|------|-------|---------|----------|
| 2026-06-02 | Collecte corpus Mina | 19,605 phrases Mina extraites | ✅ Succès |
| 2026-06-02 | Création vocabulaire | `vocabulaire_mina.json` créé | ✅ Succès |
| 2026-06-02 | Parallel corpus | 360 paires FR-Mina créées | ✅ Succès |
| 2026-06-03 | Corpus enrichi | `corpus_enriched.jsonl` généré | ✅ Succès |
| 2026-06-03 | Corpus fusionné | `corpus_merged.jsonl` (425 paires) | ✅ Succès |
| 2026-06-04 | Génération traductions | 500 phrases avec templates | ✅ Succès |

### Phase 2: Entraînement LLM ✅

| Date | Étape | Détails | Résultat |
|------|-------|---------|----------|
| 2026-06-02 | Script train_mini_llm.py | QLoRA fine-tuning créé | ✅ Succès |
| 2026-06-02 | Entraînement Qwen2-0.5B | 3 epochs, batch 2, lr 2e-4 | ✅ Succès |
| 2026-06-03 | Validation inférence | Test avec 5 phrases FR | ✅ Fonctionne |

### Phase 3: Entraînement STT ✅

| Date | Étape | Détails | Résultat |
|------|-------|---------|----------|
| 2026-06-02 | Script train_whisper.py | Whisper fine-tuning créé | ✅ Succès |
| 2026-06-02 | Préparation données audio | 19,605 clips Common Voice Mina | ✅ Succès |
| 2026-06-03 | Entraînement Whisper | 500 steps, loss finale 0.007755 | ✅ Succès |

### Phase 4: API FastAPI ✅

| Date | Étape | Détails | Résultat |
|------|-------|---------|----------|
| 2026-06-02 | Création api/main.py | API FastAPI basique | ✅ Succès |
| 2026-06-02 | Authentification | API keys + Master Key | ✅ Complété |
| 2026-06-02 | Audit logging | `api/audit.py`, `api/logger.py` | ✅ Complété |
| 2026-06-03 | Documentation B2B | `API_PARTENAIRES.md` | ✅ Complété |
| 2026-06-05 | Intégration Whisper | Endpoint `/transcribe` ajouté | ✅ Complété |
| 2026-06-05 | Problème modèle | VRAM saturée au démarrage | ⚠️ En cours |

### Phase 5: WebApp ✅

| Date | Étape | Détails | Résultat |
|------|-------|---------|----------|
| 2026-06-05 | Création interface | `webapp/index.html` | ✅ Complété |
| 2026-06-05 | Styles CSS | `webapp/styles.css` | ✅ Complété |
| 2026-06-05 | JavaScript | `webapp/app.js` | ✅ Complété |
| 2026-06-05 | Test | Connexion API | ⚠️ API déconnectée |

---

## 🐛 ERREURS ET SOLUTIONS

### Erreur 1: VRAM saturée au démarrage de l'API

| Champ | Détail |
|-------|--------|
| **Date** | 2026-06-05 |
| **Symptôme** | Le modèle ne charge pas au démarrage de l'API (`model_loaded: false`), VRAM à 8.59/8.59 Go |
| **Cause probable** | La VRAM est presque entièrement utilisée par un autre processus ou le modèle précédent n'a pas été libéré |
| **Solution** | Redémarrer le processus ou libérer la VRAM avant de lancer l'API |
| **Status** | 🔄 En cours de résolution |

**Commandes de diagnostic:**
```powershell
# Vérifier les processus GPU
nvidia-smi

# Redémarrer l'API après avoir libéré la mémoire
# (Ctrl+C dans le terminal API, puis:)
py -m uvicorn api.main:app --reload --port 8000
```

### Erreur 2: Serveur WebApp affichant les dossiers du projet

| Champ | Détail |
|-------|--------|
| **Date** | 2026-06-05 |
| **Symptôme** | `http://localhost:3000` affiche la liste des fichiers au lieu de l'application web |
| **Cause** | Le serveur HTTP était lancé depuis le répertoire racine au lieu de `webapp/` |
| **Solution** | Lancer le serveur depuis le dossier `webapp`: `cd webapp && py -m http.server 3000` |
| **Status** | ✅ Résolu |

### Erreur 3: Fine-tuning Whisper échoué (plusieurs tentatives)

| Champ | Détail |
|-------|--------|
| **Date** | 2026-06-03 au 2026-06-04 |
| **Symptôme** | Multiple erreurs de segmentation (exit code 139) et échecs avec exit code 1 |
| **Cause** | Problèmes de compatibilité CUDA, VRAM insuffisante pour certains paramètres |
| **Solution** | Réduction du batch size, utilisation de `train_whisper_simple.py` avec paramètres optimisés |
| **Status** | ✅ Résolu - Entraînement réussi avec 500 steps |

### Erreur 4: Fine-tuning Qwen2 échoué (exit code 139)

| Champ | Détail |
|-------|--------|
| **Date** | 2026-06-03 |
| **Symptôme** | Segmentation fault lors du chargement du modèle Qwen2-0.5B |
| **Cause** | Problèmes avec la quantification bitsandbytes sur Windows |
| **Solution** | Utilisation de `train_mini_llm.py` avec une approche plus simple, sans quantization excessive |
| **Status** | ✅ Résolu - Entraînement réussi |

---

## 🎙️ Application Crowdsourcing (2026-06-06)

### Concept
Application Streamlit pour collecter des traductions Mina → Français par crowdsourcing.
Les utilisateurs écoutent des phrases en Mina et proposent la traduction en français.

### Fonctionnement
1. L'utilisateur arrive sur la page Streamlit
2. Il écoute l'audio Mina (Common Voice)
3. Il tape la traduction française
4. Il clique "Valider"
5. L'audio suivant apparaît (jamais le même deux fois)

### Fichiers créés

| Fichier | Description | Statut |
|---------|-------------|--------|
| `crowdsource_app.py` | Application Streamlit principale | ✅ Créé |
| `scripts/init_crowdsource_db.py` | Initialisation base SQLite | ✅ Créé |
| `scripts/sync_db_github.py` | Synchronisation GitHub | ✅ Créé |
| `docs/CROWDSOURCE_DEPLOYMENT.md` | Guide de déploiement | ✅ Créé |
| `launch_crowdsource.bat` | Script de lancement Windows | ✅ Créé |

### Base de données SQLite

**Table: translations**
- `audio_id`: ID de l'audio Common Voice
- `mina_text`: Texte Mina original
- `french_translation`: Traduction proposée
- `translation_hash`: Hash pour déduplication
- `validation_count`: Nombre de fois validé

**Table: seen_audios**
- `audio_id`: ID de l'audio
- `session_id`: Session de l'utilisateur
- Empêche de montrer le même audio deux fois

### Déploiement

**Streamlit Cloud (gratuit)**:
```bash
streamlit run crowdsource_app.py
```

**Stockage SQLite sur GitHub**:
- L'utilisateur a trouvé un moyen de stocker des fichiers > 5GB sur GitHub
- La base SQLite sera stockée sur GitHub pour synchronisation

### Commandes

```powershell
# Initialiser la base
py scripts/init_crowdsource_db.py

# Lancer l'application
streamlit run crowdsource_app.py

# Synchroniser avec GitHub
py scripts/sync_db_github.py --push
py scripts/sync_db_github.py --pull
```

---

## 📁 FICHIERS CRÉÉS/PERSONNALISÉS

### Scripts d'entraînement

| Fichier | Usage | Statut |
|---------|-------|--------|
| `scripts/train_mini_llm.py` | Fine-tune Qwen2 avec QLoRA | ✅ Fonctionnel |
| `scripts/train_whisper_simple.py` | Fine-tune Whisper (simplifié) | ✅ Fonctionnel |
| `scripts/generate_translations.py` | Génère phrases FR-Mina | ✅ Fonctionnel |
| `scripts/merge_corpus.py` | Fusionne les corpus | ✅ Fonctionnel |
| `scripts/build_parallel_corpus.py` | Construit corpus parallèle | ✅ Fonctionnel |

### API et Backend

| Fichier | Usage | Statut |
|---------|-------|--------|
| `api/main.py` | API FastAPI principale | ⚠️ Modèle non chargé |
| `api/auth.py` | Authentification API keys | ✅ Fonctionnel |
| `api/audit.py` | Logging des requêtes | ✅ Fonctionnel |
| `api/database.py` | Base de données SQLite | ✅ Fonctionnel |
| `api/security.py` | Sécurité et hashing | ✅ Fonctionnel |

### Frontend WebApp

| Fichier | Usage | Statut |
|---------|-------|--------|
| `webapp/index.html` | Interface principale | ✅ Fonctionnel |
| `webapp/styles.css` | Styles CSS | ✅ Fonctionnel |
| `webapp/app.js` | Logique JavaScript | ✅ Fonctionnel |
| `webapp/launch_webapp.bat` | Script de lancement | ✅ Fonctionnel |

### Documentation

| Fichier | Description | Statut |
|---------|-------------|--------|
| `docs/PROJECT_STATUS.md` | État du projet | ✅ À jour |
| `docs/API_B2B_DOCUMENTATION.md` | Documentation B2B | ✅ Complète |
| `docs/PHASE2_STT_WHISPER.md` | Guide STT | ✅ Complète |
| `docs/PLAN_ENRICHISSEMENT.md` | Plan d'enrichissement corpus | ✅ Complète |
| `README.md` | README principal | ✅ À jour |

---

## 📊 MÉTRIQUES DE PERFORMANCE

### LLM (Qwen2-0.5B)

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Temps d'inférence moyen | ~1000-2000 ms | Sur RTX 4060 |
| Taille modèle | ~1 Go | Compressé |
| Batch size utilisé | 2 | Limité par VRAM |
| Learning rate | 2e-4 | |
| Epochs entraînement | 3 | |
| Loss finale | Non évalué | |

### STT (Whisper-small)

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Training steps | 500 | |
| Loss finale | 0.007755 | Bon |
| WER estimé | ~15-20% | Non validé sur données réelles |
| Dataset | 19,605 clips | Common Voice Mina |

### API

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Temps de réponse | ~1000-2000 ms | Dépend du texte |
| Concurrence | Non testé | |
| Disponibilité | ⚠️ Partiel | Modèle non chargé |

---

## 🔧 CONFIGURATION MATÉRIEL

```
╔════════════════════════════════════════════════════════╗
║                    CONFIGURATION                        ║
╠════════════════════════════════════════════════════════╣
║ GPU:    NVIDIA RTX 4060 Laptop (8 GB VRAM)           ║
║ CPU:    Intel i7-13650HX (14 cœurs / 20 threads)      ║
║ RAM:    32 GB DDR5                                    ║
║ Disque: 954 GB NVMe SSD                               ║
║ OS:     Windows 11 Home                              ║
╚════════════════════════════════════════════════════════╝
```

### Optimisations appliquées

- **Quantification**: QLoRA 4-bit pour LLM (obligatoire pour 8GB VRAM)
- **Batch size**: 2-4 (effective via gradient accumulation)
- **Max sequence length**: 128-256 tokens
- **Modèle recommandé**: Qwen2-0.5B-Instruct (3.8B params, ~4.5 Go VRAM)

---

## 📋 PROCHAINES ÉTAPES

### Priorité 1: Résoudre le problème de chargement du modèle API

- [ ] Vérifier les processus GPU avec `nvidia-smi`
- [ ] Redémarrer l'API proprement
- [ ] Implémenter le lazy loading si le problème persiste

### Priorité 2: Validation des traductions

- [ ] Tester manuellement 50 phrases FR → Mina
- [ ] Évaluer la qualité des traductions
- [ ] Corriger les erreurs fréquentes

### Priorité 3: Enrichissement du corpus

- [ ] Passer de 425 à 1000+ paires FR-Mina
- [ ] Valider les nouvelles traductions avec des locuteurs natifs
- [ ] Documenter les expressions idiomatiques

### Priorité 4: Intégration pipeline complet

- [ ] STT → Traduction → TTS
- [ ] Tests end-to-end
- [ ] Optimisation des performances

### Priorité 5: Déploiement production

- [ ] Configuration serveur
- [ ] Tests de charge
- [ ] Monitoring et alerting

---

## 📝 NOTES ET OBSERVATIONS

### Observations importantes

1. **Problème VRAM récurrent**: La RTX 4060 Laptop a seulement 8GB de VRAM, ce qui est limitant pour les modèles de langue. Il faut toujours utiliser la quantification QLoRA 4-bit.

2. **Corpus Mina limité**: Avec seulement ~425 paires FR-Mina, le modèle a des connaissances limitées. L'enrichissement du corpus est crucial.

3. **Audio Common Voice**: 19,605 clips audio Mina sont disponibles et ont été préparés pour l'entraînement Whisper.

4. **Scripts multiples**: Il existe beaucoup de scripts similaires/doublons dans le dossier `scripts/`. Une nettoyage serait bénéfique.

### Recommandations

1. Pour le prochain entraînement LLM, utiliser 1000+ paires FR-Mina
2. Implémenter un système de validation humaine pour les traductions
3. Préparer le modèle pour un déploiement cloud (si VRAM insuffisante)

---

## 📞 COMMANDES DE LANCEMENT

### Lancer l'API
```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py -m uvicorn api.main:app --reload --port 8000
```

### Lancer le WebApp
```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina/webapp"
py -m http.server 3000
```

### Lancer l'entraînement LLM
```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py scripts/train_mini_llm.py --epochs 3
```

### Lancer l'entraînement Whisper
```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py scripts/train_whisper_simple.py --steps 500
```

### Tester l'API
```powershell
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py scripts/test_api.py
```

---

## 🔄 JOURNAL DE MISE À JOUR

| Date | Modifications | Par |
|------|---------------|-----|
| 2026-06-06 | Création du document PROJECT_TRACKER.md | Claude |
| 2026-06-05 | Création PROJECT_STATUS.md | Claude |
| 2026-06-02 | Initialisation du projet | Claude |

---

*Document à mettre à jour à chaque étape du projet.*
*Pour toute question, consulter le README.md ou la documentation B2B.*