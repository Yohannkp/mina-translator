@echo off
REM ============================================================
REM Mina-Translator - Lanceur automatique
REM ============================================================

echo.
echo ============================================================
echo MINA-TRANSLATOR - SYSTEME DE TRADUCTION MINA-FRANCAIS
echo ============================================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

REM Activer l'environnement virtuel
if exist venv_train\Scripts\activate.bat (
    echo Activation de l'environnement virtuel...
    call venv_train\Scripts\activate.bat
) else (
    echo Creation de l'environnement virtuel...
    python -m venv venv_train
    call venv_train\Scripts\activate.bat
    echo Installation des dependances...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate peft bitsandbytes
    pip install datasets huggingface_hub
    pip install fastapi uvicorn loguru pydantic-settings
    pip install openai-whisper soundfile librosa scipy
)

echo.
echo ============================================================
echo MENU PRINCIPAL
echo ============================================================
echo.
echo 1. Lancer l'API (http://localhost:8000)
echo 2. Generer le corpus de traductions (500 phrases)
echo 3. Fine-tuner Qwen2 sur le corpus Mina
echo 4. Fine-tuner Whisper sur l'Ewe
echo 5. Tester le pipeline audio
echo 6. Quitter
echo.
set /p choice="Votre choix: "

if "%choice%"=="1" goto api
if "%choice%"=="2" goto corpus
if "%choice%"=="3" goto train_llm
if "%choice%"=="4" goto train_whisper
if "%choice%"=="5" goto test_pipeline
if "%choice%"=="6" goto end

:api
echo.
echo Lancement de l'API...
python -m uvicorn api.mina_api:app --reload --port 8000
goto end

:corpus
echo.
echo Generation du corpus de traductions...
echo Cette operation prend environ 2 heures pour 500 phrases.
echo.
python scripts/build_parallel_corpus.py --sample 500
goto end

:train_llm
echo.
echo Fine-tuning Qwen2 sur le corpus Mina...
python scripts/train_mina_llm.py --epochs 3
goto end

:train_whisper
echo.
echo Fine-tuning Whisper sur l'Ewe (STT)...
python scripts/fine_tune_whisper.py --prepare
python scripts/fine_tune_whisper.py --train --epochs 3
goto end

:test_pipeline
echo.
echo Test du pipeline audio complet...
python scripts/speech_to_text_pipeline.py --test-common-voice --num-samples 5
goto end

:end
pause