@echo off
REM scripts/setup_env.bat - Configure l'environnement Mina-Translator
REM =================================================================

echo.
echo ============================================================
echo  MINA-TRANSLATOR - Configuration de l'Environnement
echo ============================================================
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe.
    echo Telechargez Python sur: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Aller dans le repertoire du projet
cd /d "%~dp0.."

echo [1/5] Creation de l'environnement virtuel...
if not exist "venv_train" (
    python -m venv venv_train
    echo    OK - venv_train cree
) else (
    echo    OK - venv_train existe deja
)

echo.
echo [2/5] Activation de l'environnement...
call venv_train\Scripts\activate.bat

echo.
echo [3/5] Installation des dependances...
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes
pip install datasets scipy
pip install fastapi uvicorn loguru
pip install huggingface_hub
pip install whisper soundfile librosa

echo.
echo [4/5] Telechargement du modele Qwen2-0.5B...
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct')"

echo.
echo [5/5] Verification de CUDA...
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go') if torch.cuda.is_available() else None"

echo.
echo ============================================================
echo  CONFIGURATION TERMINEE
echo ============================================================
echo.
echo Prochaine etape: Entrainer le modele
echo   python scripts/train_mina_llm.py
echo.
echo Ou lancer l'API directement:
echo   python start_api.py
echo.
pause