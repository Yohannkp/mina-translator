# 🚀 Guide de Démarrage Rapide - Mina-Translator

## 🎯 Objectif
Traducteur automatique Français ↔ Mina (langue du Togo) avec fine-tuning QLoRA optimisé pour RTX 4060.

---

## ⚡ Démarrage Ultra-Rapide

### Option 1: Script Automatique (Recommandé)

```bash
# 1. Lancer le script de démarrage
python scripts/quick_start.py --step all

# 2. Accepter les installations
# 3. Attendre le dry-run (5-10 minutes)
```

### Option 2: Manuel Étape par Étape

```bash
# 1. Setup environnement
scripts\setup_env.bat

# 2. Activer l'environnement
venv_train\Scripts\activate

# 3. Dry-run (50 steps)
python scripts/train_mini_llm.py --steps 50

# 4. Test d'inférence
python scripts/inference_test.py
```

---

## 📁 Structure des Scripts

| Script | Description | Utilisation |
|--------|-------------|-------------|
| `quick_start.py` | Guide interactif | `python scripts/quick_start.py --step all` |
| `setup_env.bat` | Setup Windows | Double-cliquer ou `scripts\setup_env.bat` |
| `train_mini_llm.py` | Fine-tuning QLoRA | `python scripts/train_mini_llm.py --steps 50` |
| `inference_test.py` | Test d'inférence | `python scripts/inference_test.py` |

---

## 🔧 Commandes d'Entraînement

### Dry-Run (50 steps) - ~10 minutes
```bash
python scripts/train_mini_llm.py --steps 50
```

### Entraînement Complet (1000 steps) - ~2-3 heures
```bash
python scripts/train_mini_llm.py --full
```

### Avec Modèle Personnalisé
```bash
# Phi-3-mini (recommandé pour 8GB VRAM)
python scripts/train_mini_llm.py --model microsoft/Phi-3-mini-4k-instruct

# Qwen2-0.5B (plus rapide, moins performant)
python scripts/train_mini_llm.py --model Qwen/Qwen2-0.5B

# Llama-3.2-1B (option alternative)
python scripts/train_mini_llm.py --model meta-llama/Llama-3.2-1B
```

---

## 🧪 Test d'Inférence

### Test Simple
```bash
python scripts/inference_test.py
```

### Avec Phrases Personnalisées
```bash
python scripts/inference_test.py --phrases "Bonjour" "Merci beaucoup" "Je vais bien"
```

### Mode Verbeux
```bash
python scripts/inference_test.py --verbose
```

---

## 🐛 Dépannage

### Erreur: Segmentation Fault (exit code 139)

**Cause**: Incompatibilité de versions entre PyTorch/CUDA.

**Solution**:
```bash
# 1. Vérifier les versions
python -c "import torch; print(torch.__version__)"

# 2. Réinstaller PyTorch propre
pip uninstall torch -y
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Erreur: CUDA Out of Memory

**Cause**: Batch size trop grand.

**Solution**:
```bash
python scripts/train_mini_llm.py --batch 1 --steps 50
```

### Erreur: Model not found

**Cause**: Modèle non téléchargé.

**Solution**:
```bash
# Télécharger manuellement
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"
```

---

## 📊 Configuration Optimale (RTX 4060 8GB)

```python
{
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_steps": 1000,
    "learning_rate": 2e-4,
    "quantization": "4-bit NF4",
    "fp16": True,
    "optim": "paged_adamw_8bit",
}
```

**Temps estimé**:
- Dry-run (50 steps): ~10 minutes
- Entraînement complet (1000 steps): ~2-3 heures
- Inférence: ~100-200ms par phrase

---

## 📁 Structure du Projet

```
IA traduction Français Mina/
├── scripts/
│   ├── quick_start.py        # Guide interactif
│   ├── setup_env.bat         # Setup Windows
│   ├── train_mini_llm.py     # Fine-tuning QLoRA
│   ├── inference_test.py     # Test d'inférence
│   └── ...
├── data/
│   └── processed/
│       └── mina_french_pairs.jsonl  # Dataset Mina
├── models/
│   └── mina-translator/      # Modèle fine-tuné
├── api/
│   └── main.py               # API FastAPI
└── docs/
    └── ...
```

---

## 🎓 Prochaines Étapes

1. **✅ Dry-run** - Valider que l'entraînement fonctionne
2. **✅ Entraînement complet** - Fine-tuner le modèle
3. **✅ Test d'inférence** - Vérifier les traductions
4. **⬜ Enrichir le dataset** - Ajouter plus de phrases Mina
5. **⬜ Déployer l'API** - `python api/main.py`

---

## 📞 Besoin d'Aide?

Si vous avez des erreurs:
1. Vérifiez la section **Dépannage** ci-dessus
2. Consultez les logs dans `logs/training_*.log`
3. Contactez-moi avec le message d'erreur exact

**Bon courage pour le projet Mina! 🇹🇬**