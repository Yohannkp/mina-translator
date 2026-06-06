# API Mina-Translator - Guide des Partenaires B2B
> **Version 2.0.0** | Documentation pour les entreprises togolaises
> *Dernière mise à jour: 2026-06-02*

---

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Obtention d'une Clé API](#obtention-dune-clé-api)
3. [Authentification](#authentification)
4. [Endpoints Disponibles](#endpoints-disponibles)
5. [Codes d'Erreur](#codes-derreur)
6. [Quotas et Limites](#quotas-et-limites)
7. [Tarification](#tarification)
8. [Exemples de Code](#exemples-de-code)

---

## 🎯 Introduction

L'API Mina-Translator permet aux entreprises togolaises d'intégrer la traduction automatique Français ↔ Mina dans leurs applications.

### Cas d'usage

| Secteur | Application |
|---------|-------------|
| **Santé** | Traduction de prescriptions médicales |
| **Justice** | Assistance juridique pour les citoyens |
| **Commerce** | Transactions sur les marchés |
| **Administration** | Formulaires et documents officiels |
| **Éducation** | Ressources pédagogiques |
| **Urgence** | Services d'urgence et sécurité |

---

## 🔑 Obtention d'une Clé API

### Étape 1: Contacter le support

Envoyez un email à **api@mina-translator.tg** avec:
- Nom de l'entreprise
- Numéro RCC
- Volume de requêtes estimé
- Plan souhaité (Basic/Pro/Enterprise)

### Étape 2: Réception de votre clé

Vous recevrez une clé au format:
```
mtk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

⚠️ **IMPORTANT**: Cette clé ne sera affichée qu'une seule fois. Conservez-la en lieu sûr.

### Étape 3: Configuration

Ajoutez votre clé dans les headers de vos requêtes:
```
X-API-Key: mtk_live_votre_cle_ici
```

---

## 🔐 Authentification

Toutes les requêtes (sauf `/translate/open`) nécessitent une authentification.

### Header Requis

```
X-API-Key: mtk_live_votre_cle_api
```

### Codes de Réponse

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 401 | Clé API manquante ou invalide |
| 429 | Quota ou rate limit atteint |
| 500 | Erreur interne |

---

## 🌐 Endpoints Disponibles

### 1. Traduction Authentifiée (Recommandé)

**POST** `/translate`

```http
POST /translate HTTP/1.1
Host: api.mina-translator.tg
Content-Type: application/json
X-API-Key: mtk_live_votre_cle_ici

{
  "text": "Bonjour, comment allez-vous?",
  "source_lang": "fr",
  "target_lang": "mina"
}
```

**Réponse:**
```json
{
  "original": "Bonjour, comment allez-vous?",
  "translated": "Woezor, ayi?",
  "source_lang": "fr",
  "target_lang": "mina",
  "confidence": 0.85,
  "inference_ms": 145,
  "remaining_quota": {
    "daily_remaining": 945,
    "monthly_remaining": 8950
  }
}
```

---

### 2. Traduction Démonstration (Sans Clé)

**POST** `/translate/open`

⚠️ Usage limité à des fins de test. Textes max 200 caractères.

```http
POST /translate/open HTTP/1.1
Host: api.mina-translator.tg
Content-Type: application/json

{
  "text": "Bonjour",
  "source_lang": "fr",
  "target_lang": "mina"
}
```

---

### 3. Vérification de Santé

**GET** `/health`

```http
GET /health HTTP/1.1
```

**Réponse:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "gpu_memory_used": 4.2,
  "gpu_memory_total": 8.0,
  "api_version": "2.0.0"
}
```

---

## ❌ Codes d'Erreur

### Erreurs 4xx - Problèmes Client

| Code HTTP | Error | Description | Solution |
|-----------|-------|-------------|----------|
| 400 | `TEXT_TOO_LONG` | Texte dépasse 1000 caractères | Réduisez la taille du texte |
| 400 | `INVALID_LANGUAGE` | Langue non supportée | Utilisez `fr` ou `mina` |
| 401 | `MISSING_API_KEY` | Header X-API-Key absent | Ajoutez le header |
| 401 | `INVALID_API_KEY` | Clé invalide ou expirée | Vérifiez votre clé |
| 429 | `RATE_LIMIT_EXCEEDED` | Plus de 60 req/min | Attendez et réessayez |
| 429 | `QUOTA_LIMIT` | Quota épuisé | Contactez le support |

### Erreurs 5xx - Problèmes Serveur

| Code HTTP | Error | Description |
|-----------|-------|-------------|
| 500 | `INTERNAL_ERROR` | Erreur interne |
| 503 | `MODEL_NOT_LOADED` | Modèle temporairement indisponible |

### Format de Réponse d'Erreur

```json
{
  "error": "QUOTA_EXCEEDED",
  "message": "Quota quotidien épuisé",
  "code": "QUOTA_LIMIT",
  "details": {
    "daily_limit": 1000,
    "daily_used": 1000
  },
  "hint": "Contactez le support pour augmenter votre quota"
}
```

---

## 📊 Quotas et Limites

### Par Plan

| Plan | Quotidien | Mensuel | Rate Limit |
|------|-----------|---------|------------|
| **Basic** | 500 | 5,000 | 60 req/min |
| **Pro** | 2,000 | 20,000 | 120 req/min |
| **Enterprise** | 10,000 | 100,000 | 300 req/min |

### Mode Démonstration (`/translate/open`)

- 100 caractères max par requête
- 10 requêtes par heure
- **Pas de facturation**

---

## 💰 Tarification

### Plans Mensuels

| Plan | Prix | Inclus |
|------|------|--------|
| Basic | 25,000 XOF/mois | 5,000 traductions |
| Pro | 75,000 XOF/mois | 20,000 traductions |
| Enterprise | 200,000 XOF/mois | 100,000 traductions |

### Dépassement

| Volume | Prix |
|--------|------|
| 1-100 traductions | 5 XOF/traduction |
| 100-1000 traductions | 4 XOF/traduction |
| 1000+ traductions | 3 XOF/traduction |

### Facturation

- Facture mensuelle envoyée par email
- Paiement par Mobile Money (T-Money, Flooz)
- Délai de paiement: 30 jours

---

## 💻 Exemples de Code

### Python

```python
import requests

API_KEY = "mtk_live_votre_cle_ici"
API_URL = "https://api.mina-translator.tg"

def translate(text, source_lang="fr", target_lang="mina"):
    response = requests.post(
        f"{API_URL}/translate",
        headers={"X-API-Key": API_KEY},
        json={
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
    )

    if response.status_code == 200:
        return response.json()["translated"]
    else:
        error = response.json()
        raise Exception(f"{error['code']}: {error['message']}")

# Utilisation
result = translate("Bonjour, comment allez-vous?")
print(result)  # "Woezor, ayi?"
```

### JavaScript (Node.js)

```javascript
const API_KEY = 'mtk_live_votre_cle_ici';
const API_URL = 'https://api.mina-translator.tg';

async function translate(text, sourceLang = 'fr', targetLang = 'mina') {
  const response = await fetch(`${API_URL}/translate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify({
      text,
      source_lang: sourceLang,
      target_lang: targetLang
    })
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(`${data.code}: ${data.message}`);
  }

  return data.translated;
}

// Utilisation
const result = await translate("Bonjour, comment allez-vous?");
console.log(result); // "Woezor, ayi?"
```

### cURL

```bash
# Traduction FR → MINA
curl -X POST "https://api.mina-translator.tg/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mtk_live_votre_cle_ici" \
  -d '{
    "text": "Bonjour, comment allez-vous?",
    "source_lang": "fr",
    "target_lang": "mina"
  }'

# Traduction MINA → FR
curl -X POST "https://api.mina-translator.tg/translate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mtk_live_votre_cle_ici" \
  -d '{
    "text": "Woezor, ayi?",
    "source_lang": "mina",
    "target_lang": "fr"
  }'
```

### PHP

```php
<?php

$api_key = 'mtk_live_votre_cle_ici';
$api_url = 'https://api.mina-translator.tg/translate';

$data = [
    'text' => 'Bonjour, comment allez-vous?',
    'source_lang' => 'fr',
    'target_lang' => 'mina'
];

$ch = curl_init($api_url);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'X-API-Key: ' . $api_key
]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);
echo $result['translated']; // "Woezor, ayi?"
```

---

## 🔧 Intégration SDK

### Installation Python

```bash
pip install mina-translator-sdk
```

### Utilisation SDK

```python
from mina_translator import MinaTranslator

client = MinaTranslator(api_key="mtk_live_votre_cle_ici")

# Traduction simple
result = client.translate("Bonjour", "fr", "mina")

# Traduction par lot
texts = ["Bonjour", "Merci", "Au revoir"]
results = client.translate_batch(texts, "fr", "mina")

# Vérifier les quotas
quotas = client.get_quotas()
print(f"Requêtes restantes aujourd'hui: {quotas['daily_remaining']}")
```

---

## 📞 Support

| Canal | Contact |
|-------|---------|
| Email | api@mina-translator.tg |
| Téléphone | +228 22 XX XX XX |
| WhatsApp | +228 XX XX XX XX |
| Heures | Lun-Ven 8h-18h TGT |

---

## 📜 RGPD & Confidentialité

- Les données de traduction sont supprimées après traitement
- Aucune donnée n'est partagée avec des tiers
- Journalisation uniquement à des fins de facturation
- Droit de suppression sur demande

---

*Document généré automatiquement - Version 2.0.0*
*© 2026 Mina-Translator - Tous droits réservés*