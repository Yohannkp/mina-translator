@echo off
REM start_training.bat - Lance l'entraînement Whisper et le monitoring (Windows)
REM =============================================================================
REM
REM Usage:
REM   start_training.bat
REM   double-clic sur le fichier
REM
REM Auteur: Claude Opus 4.8
REM Date: 2026-06-02

setlocal enabledelayedexpansion

REM Configuration
set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%"

set "LOG_FILE=%PROJECT_DIR%\logs\training.log"
set "TRAIN_SCRIPT=%PROJECT_DIR%\scripts\train_whisper.py"
set "MONITOR_SCRIPT=%PROJECT_DIR%\scripts\monitor_training.py"

REM Couleurs (ANSI sur Windows 10+)
color 0B

echo ================================================================================
echo.
echo              MINA-WHISPER TRAINING LAUNCHER - RTX 4060
echo.
echo ================================================================================
echo.

REM Créer le répertoire de logs
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM Vérifier que le script d'entraînement existe
if not exist "%TRAIN_SCRIPT%" (
    echo [ERREUR] Script d'entraînement non trouvé: %TRAIN_SCRIPT%
    pause
    exit /b 1
)

REM Vérifier que le moniteur existe
if not exist "%MONITOR_SCRIPT%" (
    echo [ERREUR] Script de monitoring non trouvé: %MONITOR_SCRIPT%
    pause
    exit /b 1
)

REM Vérifier GPU
echo [INFO] Vérification du GPU...
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go')" 2>nul
if errorlevel 1 (
    echo [ERREUR] GPU non détecté ou PyTorch non compatible CUDA
    echo.
    echo Assurez-vous d'avoir installé PyTorch avec CUDA:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

echo [OK] GPU détecté
echo.

REM Démarrer l'entraînement en arrière-plan avec logs
echo [INFO] Démarrage de l'entraînement en arrière-plan...
echo [INFO] Log file: %LOG_FILE%
echo.

REM Lancer l'entraînement avec redirection des logs
start /min "Whisper Training" cmd /c "python \"%TRAIN_SCRIPT%\" > \"%LOG_FILE%\" 2>&1"
set TRAIN_PID=!errorlevel!

timeout /t 3 /nobreak >nul

echo ================================================================================
echo.
echo              TABLEAU DE BORD MONITORING
echo.
echo ================================================================================
echo.

REM Lancer le monitoring (blocant)
python "%MONITOR_SCRIPT%" --log-file "%LOG_FILE%"
set MONITOR_EXIT=!errorlevel!

echo.
echo ================================================================================
echo.
echo              RÉSUMÉ FINAL
echo.
echo ================================================================================
echo.

REM Afficher les métriques finales
if exist "%PROJECT_DIR%\models\mina-whisper-v1\trainer_state.json" (
    echo [INFO] Métriques finales:
    python -c "
import json
from pathlib import Path

state_file = Path(r'%PROJECT_DIR%\models\mina-whisper-v1\trainer_state.json')
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)

    print(f'  Steps: {state.get(\"step\", \"N/A\")}')
    print(f'  Epoch: {state.get(\"epoch\", \"N/A\")}')

    if 'log_history' in state and state['log_history']:
        last = state['log_history'][-1]
        if 'loss' in last:
            print(f'  Derniere Loss: {last[\"loss\"]:.6f}')
        if 'best_metric' in state:
            print(f'  Best Metric: {state[\"best_metric\"]:.6f}')
"
)

echo.
echo [OK] Entraînement terminé
echo [INFO] Modèle sauvegardé dans: models\mina-whisper-v1\
echo.
echo Pour voir les logs en temps réel: tail -f logs\training.log
echo Pour tester le modèle: python scripts\test_whisper.py
echo.

pause