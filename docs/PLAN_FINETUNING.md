# Plan de Fine-tuning Mina-Translator

## Configuration Machine

| Composant | Spécification |
|-----------|---------------|
| GPU | RTX 4060 Laptop |
| VRAM | 8 Go |
| RAM | 32 Go DDR5 |
| CPU | i7-13650HX (14 cœurs) |
| Stockage | 1 To SSD NVMe |

## Modèles Recommandés pour 8 Go VRAM

### 1. Phi-3-mini (RECOMMANDÉ)
- **Paramètres**: 3.8B
- **VRAM avec QLoRA 4-bit**: ~4.5 Go
- **Contexte**: 4096 tokens
- **Avantages**: Optimisé pour tâches de dialogue, excellent ratio qualité/vRAM

### 2. Llama-3.2-3B-Instruct
- **Paramètres**: 3B
- **VRAM avec QLoRA 4-bit**: ~3.5 Go
- **Contexte**: 128K tokens
- **Avantages**: Contexte large, bon pour génération longue

### 3. Llama-3.2-1B-Instruct (Plus léger)
- **Paramètres**: 1.3B
- **VRAM avec QLoRA 4-bit**: ~2.5 Go
- **Contexte**: 128K tokens
- **Avantages**: Très peu VRAM, entraînement rapide

### 4. Mistral-7B-Instruct (Déconseillé)
- **Paramètres**: 7B
- **VRAM avec QLoRA 4-bit**: ~6.5 Go
- **Risque**: Peut saturer la VRAM sur certains batches

---

## Étapes du Fine-tuning

### 1. Préparation du Dataset

Format LoRA (Instruction Tuning):
```json
{
  "instruction": "Traduis en mina cette phrase en français",
  "input": "Bonjour, comment allez-vous?",
  "output": "Akpe ɖe o, efy nye?",
  "theme": "quotidien",
  "validated": true
}
```

Format d'entrée pour le modèle:
```
<|user|>
Traduis en mina cette phrase en français: Bonjour, comment allez-vous?
<|assistant|>
Akpe ɖe o, efy nye?<|end|>
```

### 2. Configuration QLoRA (4-bit)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Compression supplémentaire
    bnb_4bit_quant_type="nf4",        # Normal Float 4
)
```

**NF4 vs FP4**:
- NF4: Meilleure précision pour poids de réseau neuronal
- FP4: Plus rapide mais moins précis

### 3. Paramètres LoRA Optimisés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `r` | 16 | Rang de la décomposition (impact mémoire = r²) |
| `lora_alpha` | 32 | Scaling factor (2x le rang habituel) |
| `lora_dropout` | 0.05 | Régularisation |
| `target_modules` | Tous les modules | Q, K, V, O, Gate, Up, Down |
| `bias` | none | Plus efficace que "all" |

### 4. Paramètres d'Entraînement pour RTX 4060

```python
training_args = {
    # Batch & Accumulation
    "per_device_train_batch_size": 4,      # Ajustable selon VRAM
    "gradient_accumulation_steps": 4,        # Effective batch = 16
    
    # Mémoire
    "gradient_checkpointing": True,          # Réduit VRAM de ~30%
    "fp16": True,                            # Mixed precision
    
    # Optimiseur
    "optim": "paged_adamw_32bit",           # Mémoire optimisé parbitsandbytes
    
    # Learning rate
    "learning_rate": 2e-4,                  # Standard pour LoRA
    "weight_decay": 0.001,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Sauvegarde
    "save_strategy": "steps",
    "save_steps": 100,
    
    # Stabilité
    "max_grad_norm": 0.3,                   # Gradient clipping
}
```

**Effective Batch Size**: 4 × 4 = 16 (équivalent à batch 16 sur GPU normale)

### 5. Durée Estimée par Epoch

| Modèle | 1,000 samples | 5,000 samples | 10,000 samples |
|--------|---------------|---------------|-----------------|
| Phi-3-mini | 12-18 min | 60-90 min | 120-180 min |
| Llama-3.2-3B | 15-22 min | 75-110 min | 150-220 min |
| Llama-3.2-1B | 8-12 min | 40-60 min | 80-120 min |
| Mistral-7B | 25-40 min | 125-200 min | 250-400 min |

**Dataset actuel**: ~2,500 exemples → ~30-45 min/epoch avec Phi-3-mini

### 6. Critères d'Arrêt (Early Stopping)

```python
early_stopping_config = {
    "patience": 3,           # Arrêter après 3 epochs sans amélioration
    "threshold": 0.01,       # Amélioration minimale requise (1%)
}
```

**Conditions d'arrêt recommandées**:
- Loss de validation n'améliore plus pendant 3 epochs
- Overfitting détecté (train loss ↓, val loss ↑)
- Temps maximal dépassé (ex: 4 heures)

### 7. Évaluation Post-Fine-tuning

Métriques recommandées:
- **BLEU Score**: Similarité avec traductions de référence
- **CHRIF**: Score de caractère (Togo-specific)
- **Exact Match**: Pour phrases courtes
- **Test manuel**: 10-20 phrases représentatives

---

## Script de Fine-tuning Complet

### Utilisation

```bash
# Mode dry-run (afficher config sans exécuter)
python scripts/finetune_mina.py --dry-run

# Entraînement standard (Phi-3-mini, 3 epochs)
python scripts/finetune_mina.py --model phi3-mini --epochs 3

# Entraînement avec early stopping
python scripts/finetune_mina.py --model phi3-mini --epochs 5 --early-stop

# Personnaliser les paramètres
python scripts/finetune_mina.py \
    --model llama3.2-3b \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --lora-r 8

# Évaluer un modèle existant
python scripts/finetune_mina.py --evaluate-only --output models/mina-translator-phi3-mini
```

### Arguments Disponibles

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--model` | str | phi3-mini | Modèle de base |
| `--epochs` | int | 3 | Nombre d'epochs |
| `--batch-size` | int | 4 | Batch size |
| `--learning-rate` | float | 2e-4 | Taux d'apprentissage |
| `--max-length` | int | 256 | Longueur max |
| `--lora-r` | int | 16 | Rang LoRA |
| `--early-stop` | flag | False | Activer early stopping |
| `--output` | str | auto | Dossier de sortie |
| `--evaluate-only` | flag | False | Évaluer sans entraîner |
| `--dry-run` | flag | False | Afficher config |

---

## Optimisations VRAM

### Si VRAM insuffisante (< 7 Go utilisés)

1. **Réduire batch size**: `--batch-size 2`
2. **Réduire LoRA r**: `--lora-r 8`
3. **Réduire max_length**: `--max-length 128`
4. **Activer gradient checkpointing**: automatique dans le script

### Pour RTX 4060 Laptop (8 Go)

Configuration conservative (si problèmes):
```bash
python scripts/finetune_mina.py \
    --model llama3.2-1b \
    --epochs 3 \
    --batch-size 2 \
    --max-length 128
```

---

## Dépannage

| Problème | Solution |
|----------|----------|
| OOM (Out of Memory) | Réduire batch-size ou max-length |
| Training très lent | Vérifier que CUDA est actif (`torch.cuda.is_available()`) |
| Loss NaN | Réduire learning rate (1e-4) ou augmenter warmup |
| Modèle ne génère pas | Vérifier le format des prompts |
| Ergebnisse mauvais | Plus de données, plus d'epochs, learning rate différent |

---

## Installation des Dépendances

```bash
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install bitsandbytes>=0.43.0
pip install accelerate>=0.30.0
pip install datasets>=2.18.0

# Pour GPU CUDA 12.1 (recommandé pour RTX 40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```