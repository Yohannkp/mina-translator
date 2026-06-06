# Mina-Voice WebApp 🇹🇬

> Application web pour tester le système de traduction vocale Français ↔ Mina

## 🎯 Fonctionnalités

- 🎤 **Enregistrement vocal** - Utilisez votre microphone pour parler en Mina ou Français
- 🎤 **Transcription Whisper** - L'audio est transcrit automatiquement en texte
- 🔄 **Traduction automatique** - Traduction FR ↔ Mina en temps réel
- ⌨️ **Mode texte** - Entrez du texte directement pour la traduction
- 🔊 **Synthèse vocale** - Écoutez la traduction

## 🚀 Démarrage Rapide

### 1. Lancer l'API (Terminal 1)

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"
py -m uvicorn api.main:app --reload --port 8000
```

### 2. Lancer le WebApp (Terminal 2)

```bash
cd "c:/Ce PC/Projet_python/IA traduction Français Mina/webapp"
py -m http.server 3000
```

Ou double-cliquez sur `launch.bat` (lance les 2 en même temps).

### 3. Ouvrir dans le navigateur

```
http://localhost:3000
```

## 📋 Utilisation

### Mode Vocal

1. Sélectionnez la direction de traduction (**Mina → Français** ou **Français → Mina**)
2. Cliquez sur **🎤 Enregistrer**
3. Parlez clairement dans votre microphone
4. Cliquez sur **Arrêter**
5. Cliquez sur **🎯 Transcrire et Traduire**
6. Consultez les résultats

### Mode Texte

1. Sélectionnez **⌨️ Texte** comme mode d'entrée
2. Choisissez la direction de traduction
3. Tapez votre texte (ou utilisez une phrase rapide)
4. Cliquez sur **Traduire**

## ⌨️ Raccourcis Clavier

| Raccourci | Action |
|-----------|--------|
| `Espace` | Enregistrer / Arrêter (mode vocal) |
| `Ctrl+Entrée` | Traduire (mode texte) |

## 🔌 Points de Terminaison API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/transcribe` | Transcription audio → texte (Whisper) |
| POST | `/translate` | Traduction FR → Mina |
| POST | `/translate_mina` | Traduction Mina → FR |
| GET | `/health` | État de l'API |

## 🛠️ Dépannage

### Le microphone ne fonctionne pas ?
- Vérifiez les permissions du navigateur
- Chrome: `Paramètres → Confidentialité → Microphone`
- Firefox: `Paramètres → Permissions → Microphone`

### L'API ne répond pas ?
- Vérifiez que l'API est bien lancée sur le port 8000
- Vérifiez le statut dans l'interface (point vert/rouge)

### Whisper ne transcrit pas ?
- Whisper utilise le modèle `whisper-small` par défaut
- Le fine-tuning Mina est utilisé s'il est disponible dans `models/mina-whisper-v1`

## 📁 Structure des Fichiers

```
webapp/
├── index.html      # Interface utilisateur
├── styles.css      # Styles CSS
├── app.js          # Logique JavaScript
├── launch.bat      # Script de lancement rapide
└── README.md       # Ce fichier
```

## 🎨 Design

- Interface sombre moderne
- Support mobile (responsive)
- Visualisation de l'audio en temps réel
- Indicateurs de confiance et temps de traitement

## 🔒 Notes de Sécurité

- API accessible depuis `localhost` uniquement en développement
- CORS activé pour permettre les requêtes cross-origin
- Pour la production, ajoutez une authentification API Key

## 📞 Support

Pour toute question, contactez l'équipe de développement Mina-Translator.