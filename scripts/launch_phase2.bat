@echo off
REM Phase 2: Fine-tuning Whisper (STT) pour le Mina
REM =================================================

echo.
echo ================================================
echo    PHASE 2: PREPARATION STT MINA
echo ================================================
echo.

cd /d "c:\Ce PC\Projet_python\IA traduction Français Mina"

echo [1/3] Preparation des donnees audio...
echo.
call .\venv_train\Scripts\python.exe scripts\prepare_whisper_data.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERREUR] La preparation a echoue!
    pause
    exit /b 1
)

echo.
echo [2/3] Test du pipeline Whisper...
echo.
call .\venv_train\Scripts\python.exe scripts\test_whisper_pipeline.py

echo.
echo [3/3] Lancement du fine-tuning (optionnel)...
echo.
echo Tapez 'O' pour lancer le fine-tuning complet
echo Tapez 'N' pour quitter
echo.

choice /C ON /M "Lancer le fine-tuning Whisper (3 epochs, ~30 min)?"

if %ERRORLEVEL% EQU 1 (
    echo.
    echo Lancement du fine-tuning...
    echo.
    call .\venv_train\Scripts\python.exe scripts\train_whisper_native.py --epochs 3
) else (
    echo.
    echo Fine-tuning annule.
    echo Pour le lancer plus tard:
    echo   .\venv_train\Scripts\python.exe scripts\train_whisper_native.py --epochs 3
)

echo.
echo ================================================
echo    PHASE 2 TERMINEE
echo ================================================
echo.
echo Prochaine etape: Phase 3 - Integration LLM + TTS
echo Documentation: docs\PHASE2_STT_WHISPER.md
echo.

pause