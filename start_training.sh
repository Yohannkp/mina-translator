#!/bin/bash
# start_training.sh - Lance l'entraînement Whisper et le monitoring simultanément
# =============================================================================
#
# Usage:
#   bash start_training.sh
#   ./start_training.sh
#
# Auteur: Claude Opus 4.8
# Date: 2026-06-02

# Répertoire du projet
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Configuration
LOG_FILE="$PROJECT_DIR/logs/training.log"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train_whisper.py"
MONITOR_SCRIPT="$PROJECT_DIR/scripts/monitor_training.py"
PYTHON_CMD="python"

# Créer le répertoire de logs
mkdir -p "$PROJECT_DIR/logs"

# Couleurs pour l'affichage
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}║           MINA-WHISPER TRAINING LAUNCHER - RTX 4060                      ║${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo

# Vérifier que le script d'entraînement existe
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "${RED}✗ ERREUR: Script d'entraînement non trouvé: $TRAIN_SCRIPT${NC}"
    exit 1
fi

# Vérifier que le moniteur existe
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo -e "${RED}✗ ERREUR: Script de monitoring non trouvé: $MONITOR_SCRIPT${NC}"
    exit 1
fi

# Vérifier GPU
echo -e "${YELLOW}▸ Vérification du GPU...${NC}"
"$PYTHON_CMD" -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go')" 2>/dev/null || {
    echo -e "${RED}✗ ERREUR: GPU non détecté ou PyTorch non compatible CUDA${NC}"
    echo "  Assurez-vous d'avoir installé PyTorch avec CUDA:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
}

echo -e "${GREEN}✓ GPU détecté${NC}"
echo

# Démarrer l'entraînement en arrière-plan avec logs
echo -e "${YELLOW}▸ Démarrage de l'entraînement en arrière-plan...${NC}"
echo -e "${CYAN}  Log file: $LOG_FILE${NC}"
echo

# Lancer l'entraînement avec redirection des logs
nohup "$PYTHON_CMD" "$TRAIN_SCRIPT" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo -e "${GREEN}✓ Entraînement démarré (PID: $TRAIN_PID)${NC}"
echo

# Attendre 3 secondes que les premiers logs apparaissent
echo -e "${YELLOW}▸ Initialisation du moniteur...${NC}"
sleep 3

# Lancer le monitoring (blocant)
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}║                         TABLEAU DE BORD MONITORING                      ║${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo

# Lancer le moniteur
"$PYTHON_CMD" "$MONITOR_SCRIPT" --log-file "$LOG_FILE"
MONITOR_EXIT=$?

# Quand le moniteur se termine, afficher le résumé
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}║                            RÉSUMÉ FINAL                                ║${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
echo

if [ -f "$PROJECT_DIR/models/mina-whisper-v1/trainer_state.json" ]; then
    echo -e "${CYAN}▸ Métriques finales:${NC}"

    "$PYTHON_CMD" -c "
import json
from pathlib import Path

state_file = Path('$PROJECT_DIR/models/mina-whisper-v1/trainer_state.json')
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)

    print(f'  Steps: {state.get(\"step\", \"N/A\")}')
    print(f'  Epoch: {state.get(\"epoch\", \"N/A\")}')

    if 'log_history' in state and state['log_history']:
        last = state['log_history'][-1]
        if 'loss' in last:
            print(f'  Dernière Loss: {last[\"loss\"]:.6f}')
        if 'best_metric' in state:
            print(f'  Best Metric: {state[\"best_metric\"]:.6f}')
"
fi

echo
echo -e "${GREEN}✓ Entraînement terminé${NC}"
echo -e "${CYAN}  Modèle sauvegardé dans: models/mina-whisper-v1/${NC}"
echo
echo -e "Pour voir les logs: tail -f $LOG_FILE"
echo -e "Pour tester le modèle: python scripts/test_whisper.py"
echo