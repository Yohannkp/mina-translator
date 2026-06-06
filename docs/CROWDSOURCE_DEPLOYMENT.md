# 🎙️ Mina Crowdsource - Guide de Déploiement

Application Streamlit pour collecter des traductions Mina → Français par crowdsourcing.

---

## 📋 Prérequis

- Python 3.10+
- Streamlit
- Compte Streamlit Cloud (gratuit)
- GitHub (pour stocker la base SQLite)
- Dataset Common Voice Mina

---

## 🚀 Installation Locale

### 1. Cloner le projet

```bash
git clone https://github.com/VOTRE_USER/VOTRE_REPO.git
cd VOTRE_REPO
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances

```bash
pip install streamlit pandas
```

### 4. Télécharger le dataset Common Voice Mina

```bash
# Le dataset doit être dans:
# data/cv-corpus-25.0-2026-03-09/gej/

# Vérifier la structure:
ls data/cv-corpus-25.0-2026-03-09/gej/
# Doit contenir: clips/, validated.tsv, etc.
```

### 5. Initialiser la base de données

```bash
python scripts/init_crowdsource_db.py
```

### 6. Lancer l'application

```bash
streamlit run crowdsource_app.py
```

L'application sera accessible sur: http://localhost:8501

---

## ☁️ Déploiement sur Streamlit Cloud

### 1. Préparer le repository GitHub

```
VOTRE_REPO/
├── crowdsource_app.py          # Application Streamlit
├── scripts/
│   ├── init_crowdsource_db.py  # Initialisation BDD
│   └── sync_db_github.py       # Sync GitHub
├── data/
│   ├── mina_crowdsource.db     # Base SQLite (git LFS recommandé)
│   └── cv-corpus-25.0-2026-03-09/  # Dataset audio
│       └── gej/
│           ├── clips/         # ~16,000 fichiers audio MP3
│           └── validated.tsv
├── requirements.txt
└── README.md
```

### 2. Créer requirements.txt

```txt
streamlit>=1.28.0
pandas>=2.0.0
```

### 3. Pousser sur GitHub

```bash
git init
git add .
git commit -m "Initial commit - Mina Crowdsource"
git branch -M main
git remote add origin https://github.com/VOTRE_USER/VOTRE_REPO.git
git push -u origin main
```

### 4. Déployer sur Streamlit Cloud

1. Aller sur: https://streamlit.io/cloud

2. Cliquer sur **"New app"**

3. Configurer:
   - **Repository**: VOTRE_USER/VOTRE_REPO
   - **Branch**: main
   - **Main file path**: `crowdsource_app.py`
   - **Python version**: 3.10

4. Cliquer sur **"Deploy!"**

### 5. Configuration avancée (optionnel)

Dans Streamlit Cloud, vous pouvez configurer:
- **Secrets**: Pour les variables sensibles
- **App URL**: URL publique de l'app
- **Custom domain**: Domaine personnalisé

---

## 🔄 Synchronisation de la Base SQLite

### Option 1: Git LFS (Recommandé)

Git LFS permet de stocker des fichiers volumineux sur GitHub.

```bash
# Installer Git LFS
git lfs install

# Tracker les fichiers SQLite
git lfs track "*.db"

# Commit et push
git add .gitattributes
git add data/mina_crowdsource.db
git commit -m "Add crowdsource database"
git push
```

### Option 2: Script de Sync

```bash
# Upload vers GitHub
python scripts/sync_db_github.py --push

# Download depuis GitHub
python scripts/sync_db_github.py --pull

# Voir les stats
python scripts/sync_db_github.py --stats
```

### Option 3: GitHub Actions (Automatique)

Créer `.github/workflows/sync-db.yml`:

```yaml
name: Sync Database

on:
  schedule:
    - cron: '0 */6 * * *'  # Toutes les 6 heures
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Sync DB
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          pip install requests
          python scripts/sync_db_github.py --push
```

---

## 📱 Interface Utilisateur

### Écran principal

```
┌─────────────────────────────────────────────┐
│ 🇹🇬 Mina Crowdsource                        │
│                                             │
│ Aidez-nous à traduire le Mina en Français!  │
│                                             │
│ 🎵 Écoutez la phrase en Mina                │
│ [▶️ Bouton Play Audio]                      │
│                                             │
│ ✍️ Votre traduction en français:            │
│ [_________________________________]         │
│ [_________________________________]         │
│                                             │
│ [✅ Valider]  [⏭️ Passer]                   │
│                                             │
│ 📚 16,000  🌍 1,234  👤 45  ⏳ 15,955       │
└─────────────────────────────────────────────┘
```

### Flux utilisateur

1. L'utilisateur arrive sur la page
2. L'audio Mina se joue automatiquement (ou au clic)
3. L'utilisateur tape la traduction française
4. L'utilisateur clique "Valider"
5. L'audio suivant apparaît (jamais le même deux fois)
6. Répéter...

---

## 📊 Monitoring et Stats

### Stats globales (dashboard)

- Total traductions collectées
- Audios uniques traduits
- Traductions validées (consensus)
- Nombre de contributeurs

### Exporter les données

```bash
# Exporter en CSV
python -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/mina_crowdsource.db')
df = pd.read_sql('SELECT * FROM translations', conn)
df.to_csv('translations_export.csv', index=False)
print(f'Exporté {len(df)} traductions')
"
```

---

## 🛠️ Maintenance

### Sauvegarder la base

```bash
# Local
cp data/mina_crowdsource.db data/mina_crowdsource_backup.db

# GitHub
python scripts/sync_db_github.py --push
```

### Réinitialiser la base

```bash
python scripts/init_crowdsource_db.py --reset
```

### Ajouter des audios

1. Télécharger le nouveau dataset Common Voice
2. Mettre à jour `data/cv-corpus-25.0-2026-03-09/gej/`
3. Réinitialiser la queue:
```bash
python scripts/init_crowdsource_db.py --reset
```

---

## ❓ FAQ

### Q: Comment ajouter plus d'audios ?
R: Téléchargez une version plus récente de Common Voice Mina et remplacez le dossier `gej/`.

### Q: Comment changer le nombre de validations nécessaires ?
R: Modifiez `translations_needed` dans `audio_queue` ou dans `init_crowdsource_db.py`.

### Q: Comment voir les traductions validées ?
R: Exécutez une requête SQL:
```sql
SELECT * FROM validated_translations;
```

### Q: Comment supprimer les doublons ?
R: Le système utilise un hash pour dédupliquer automatiquement.

---

## 🤝 Contribuer

1. Fork le projet
2. Créez une branche (`git checkout -b feature/xxx`)
3. Commit (`git commit -am 'Add xxx'`)
4. Push (`git push origin feature/xxx`)
5. Ouvrez une Pull Request

---

## 📄 Licence

MIT License - Libre d'utilisation et de modification.

---

**💡 Astuce**: Pour un meilleur engagement, partagez l'application sur les réseaux sociaux togolais et dans les communautés Mina en ligne !