# 🚀 Guide d'Installation Google Sheets pour Mina Crowdsource

Ce guide explique comment configurer Google Sheets pour que les données de traduction soient stockées en sécurité dans le cloud.

---

## 📋 Sommaire

1. [Créer un projet Google Cloud](#1-créer-un-projet-google-cloud)
2. [Activer l'API Google Sheets](#2-activer-lapi-google-sheets)
3. [Créer un compte de service](#3-créer-un-compte-de-service)
4. [Télécharger les credentials](#4-télécharger-les-credentials)
5. [Créer le Google Sheet](#5-créer-le-google-sheet)
6. [Partager le Sheet avec le service account](#6-partager-le-sheet-avec-le-service-account)
7. [Configurer Streamlit Cloud](#7-configurer-streamlit-cloud)
8. [Tester localement](#8-tester-localement)

---

## 1. Créer un projet Google Cloud

1. Allez sur [console.cloud.google.com](https://console.cloud.google.com)
2. Connectez-vous avec votre compte Google
3. Cliquez sur **"Select a project"** → **"New Project"**
4. Nom du projet : `mina-translator`
5. Cliquez **Create**

---

## 2. Activer l'API Google Sheets

1. Dans le menu gauche : **APIs & Services** → **Library**
2. Recherchez : **Google Sheets API**
3. Cliquez dessus → **Enable**

---

## 3. Créer un compte de service

1. Allez dans **APIs & Services** → **Credentials**
2. Cliquez **+ CREATE CREDENTIALS** → **Service Account**
3. Nom : `mina-crowdsource`
4. Cliquez **Create** → **Done**

---

## 4. Télécharger les credentials

1. Allez dans **APIs & Services** → **Credentials**
2. Cliquez sur le service account que vous venez de créer
3. Onglet **Keys** → **Add Key** → **JSON**
4. Le fichier JSON sera téléchargé
5. Renommez-le en `credentials.json`
6. Placez-le dans le dossier du projet : `credentials.json`

---

## 5. Créer le Google Sheet

1. Allez sur [sheets.google.com](https://sheets.google.com)
2. Créez un **Nouveau tableau**
3. Nommez-le : `Mina Crowdsource - Traductions`
4. Copiez l'ID du URL :
   ```
   https://docs.google.com/spreadsheets/d/[ICI EST L'ID]/edit
   ```
   L'ID est la longue chaîne de caractères entre `/d/` et `/edit`

---

## 6. Partager le Sheet avec le service account

1. Ouvrez le Google Sheet
2. Cliquez **Partager** (bouton vert)
3. Ajoutez l'email du service account :
   ```
   mina-crowdsource@mina-translator.iam.gserviceaccount.com
   ```
4. Rôle : **Éditeur**
5. Cliquez **Envoyer**

---

## 7. Configurer Streamlit Cloud

Dans Streamlit Cloud, allez dans **Settings** → **Secrets** et ajoutez :

```toml
USE_GOOGLE_SHEETS = true
GOOGLE_SPREADSHEET_ID = "votre_id_du_sheet"
ADMIN_CODE = "votre_code_secret"
```

---

## 8. Tester localement

### Installer les dépendances :

```powershell
pip install gspread google-auth google-auth-oauthlib google-auth-httplib2
```

### Variables d'environnement :

Créez un fichier `.env` (NE JAMAIS COMMITER!) :

```bash
USE_GOOGLE_SHEETS=true
GOOGLE_SPREADSHEET_ID=votre_id_du_sheet
ADMIN_CODE=mina2026
```

### Lancer l'app :

```powershell
streamlit run crowdsource_app.py
```

---

## 🔍 Vérification

Quand vous lancez l'app, vous devriez voir en haut :

```
☁️ Données sauvegardées dans Google Sheets ✅
```

Et quand vous validez une traduction, elle apparaît dans votre Google Sheet !

---

## 📊 Format des données dans Google Sheets

| audio_id | audio_path | mina_text | french_text | translation_type | created_at | session_id |
|----------|------------|-----------|-------------|------------------|------------|-------------|
| abc123 | /path/to/file.mp3 | Mina texto | French translation | text | 2026-06-06T... | abc12345 |

---

## ⚠️ Important

- **NE JAMAIS** commiter `credentials.json` sur GitHub !
- Ajoutez `credentials.json` à votre `.gitignore`
- Le fichier `.gitignore` est déjà configuré pour ça

---

## 🎯 Prochaines étapes

1. Configurez Google Sheets ✅
2. Relancez le crowdsourcing avec vos amis
3. Quand vous voulez exporter les données pour l'entraînement :
   - Téléchargez le Google Sheet en CSV
   - Utilisez `scripts/export_google_sheets.py` pour convertir en JSONL

---

## 💬 Besoin d'aide ?

Si vous avez des questions, vérifiez :
1. L'API Google Sheets est bien activée
2. Le service account a accès au Sheet
3. L'ID du Sheet est correct dans les secrets
4. Le fichier `credentials.json` est présent et valide