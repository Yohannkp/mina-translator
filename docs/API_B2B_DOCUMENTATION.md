# API B2B - Mina-Translator
## Documentation Technique de Production

---

## 1. Architecture de Sécurité

### 1.1 Système d'Authentification

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUX D'AUTHENTIFICATION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Client                                                          │
│      │                                                            │
│      ▼ X-API-Key: mtk_abc123...                                    │
│   ┌─────────────┐                                                 │
│   │  Validation │                                                 │
│   │  Format     │                                                 │
│   └──────┬──────┘                                                 │
│          │ OK                                                     │
│          ▼                                                        │
│   ┌─────────────┐     Master Key?                                  │
│   │  Vérification│ ──────────────────────────▶ ACCÈS ILLIMITÉ     │
│   │  Master Key │                          (pas de quota, pas log)  │
│   └──────┬──────┘                                                 │
│          │ NON                                                    │
│          ▼                                                        │
│   ┌─────────────┐     Valide?                                      │
│   │  Vérification│ ──────────────────────────▶ 401 UNAUTHORIZED   │
│   │  DB SQLite  │     Non trouvé                                   │
│   └──────┬──────┘                                                 │
│          │ OUI                                                    │
│          ▼                                                        │
│   ┌─────────────┐     Quota ok?                                   │
│   │  Vérification│ ──────────────────────────▶ 429 TOO_MANY_REQ   │
│   │  Quota      │     Épuisé                                       │
│   └──────┬──────┘                                                 │
│          │ OUI                                                    │
│          ▼                                                        │
│   ┌─────────────┐                                                 │
│   │  Traitement │                                                  │
│   │  + Log      │                                                  │
│   └─────────────┘                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Hiérarchie des Clés

| Type | Préfixe | Usage | Quota | Logging |
|------|---------|-------|-------|---------|
| **Master Key** | None | Administration, debug | Illimité | Limité |
| **Client API Key** | `mtk_` | Production | Selon plan | Complet |

---

## 2. Guide de Déploiement

### 2.1 Installation Initiale

```bash
# 1. Cloner le projet
cd "c:/Ce PC/Projet_python/IA traduction Français Mina"

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Initialiser la base de données et générer la Master Key
python scripts/setup_db.py
```

### 2.2 Lancement de l'API

```bash
# Mode développement (avec hot-reload)
uvicorn api.main:app --reload --port 8000

# Mode production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2.3 Vérification

```bash
# Santé de l'API
curl http://localhost:8000/health

# Test avec Master Key (admin)
curl -H "X-API-Key: VOTRE_MASTER_KEY" http://localhost:8000/admin/clients

# Test avec clé client
curl -X POST -H "X-API-Key: mtk_..." -H "Content-Type: application/json" \
  -d '{"text": "Bonjour", "source_lang": "fr", "target_lang": "mina"}' \
  http://localhost:8000/translate
```

---

## 3. Endpoints API

### 3.1 Endpoints Publics

| Endpoint | Méthode | Description | Limite |
|----------|---------|-------------|--------|
| `/` | GET | Page d'accueil | - |
| `/health` | GET | État de l'API | - |
| `/translate/open` | POST | Traduction demo | 200 caractères |

### 3.2 Endpoints Authentifiés

| Endpoint | Méthode | Description | Auth |
|----------|---------|-------------|------|
| `/translate` | POST | Traduction | X-API-Key |
| `/transcribe` | POST | Transcription audio | X-API-Key |

### 3.3 Endpoints Admin (Master Key)

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/admin/clients` | GET | Liste des clients |
| `/admin/api-keys` | POST | Créer clé API |
| `/admin/stats` | GET | Statistiques globales |
| `/admin/billing/{client_id}` | GET | Facturation client |

---

## 4. Codes d'Erreur HTTP

| Code | Erreur | Cause | Solution |
|------|--------|-------|----------|
| **400** | `TEXT_TOO_LONG` | Texte > 500 caractères | Réduire la taille |
| **400** | `TRANSLATION_NOT_FOUND` | Pas de correspondance | Entraîner le modèle |
| **401** | `UNAUTHORIZED` | Clé invalide | Vérifier la clé API |
| **401** | `ADMIN_ACCESS_REQUIRED` | Pas de Master Key | Utiliser Master Key |
| **429** | `QUOTA_EXCEEDED` | Quota épuisé | Attendre reset ou upgrader |
| **500** | `INTERNAL_ERROR` | Erreur serveur | Contacter le support |

---

## 5. Formats de Réponse

### 5.1 Traduction Réussie

```json
{
  "original": "Bonjour, comment allez-vous?",
  "translated": "Bonjou, nongoa wu?",
  "source_lang": "fr",
  "target_lang": "mina",
  "confidence": 0.87,
  "inference_ms": 145,
  "remaining_quota": {
    "daily_remaining": 950,
    "monthly_remaining": 4500
  }
}
```

### 5.2 Erreur Standardisée

```json
{
  "error": "QUOTA_EXCEEDED",
  "message": "Quota limite atteint",
  "code": "TOO_MANY_REQUESTS",
  "details": {
    "daily_remaining": 0,
    "monthly_remaining": 0
  },
  "hint": "Contactez admin@mina-translator.com pour augmenter le quota"
}
```

---

## 6. Audit et Logging

### 6.1 Fichier d'Audit (logs/audit.log)

```json
{"timestamp": "2026-06-02T15:30:00", "event_type": "request",
 "client_id": 1, "client_name": "Test-Client", "endpoint": "/translate",
 "inference_ms": 145, "status_code": 200, "is_master": false}
```

### 6.2 Rotation des Logs

- Fichier: `logs/audit.log`
- Rotation: 10 MB par fichier
- Rétention: 7 jours

---

## 7. Sécurité - Meilleures Pratiques

### 7.1 Stockage des Clés

```bash
# .env (NE JAMAIS COMMITER!)
MINA_MASTER_KEY=votre_cle_super_secrete_ici
```

### 7.2 Rotation des Clés

```python
# Générer une nouvelle Master Key
from api.security import generate_master_key
new_key = generate_master_key(48)
# Mettre à jour .env
```

### 7.3 Validation des Entrées

- Longueur max: 500 caractères
- Caractères suspects: bloqués (`<script`, `{{`, etc.)
- Unicode normalisé

---

## 8. Schéma de la Base de Données

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    clients      │────<│    api_keys     │     │     quotas      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │     │ id (PK)         │
│ name            │     │ key_id (UNIQUE) │     │ client_id (FK)  │
│ email           │     │ key_hash        │────<│ daily_limit     │
│ company         │     │ client_id (FK)  │     │ daily_used      │
│ plan            │     │ active          │     │ monthly_limit   │
│ active          │     │ expires_at      │     │ monthly_used    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 9. Monitoring et Alertes

### 9.1 Santé de l'API

```bash
# Statut rapide
curl http://localhost:8000/health | jq

# Réponse
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "api_version": "3.0.0"
}
```

### 9.2 Statistiques en Temps Réel

```bash
# Avec Master Key
curl -H "X-API-Key: MASTER_KEY" http://localhost:8000/admin/stats
```

---

## 10. Support

- **Email**: admin@mina-translator.com
- **Documentation**: http://localhost:8000/docs
- **Logs**: `logs/audit.log`

---

*Document généré automatiquement - Mina-Translator v3.0.0*
*Dernière mise à jour: 2026-06-02*