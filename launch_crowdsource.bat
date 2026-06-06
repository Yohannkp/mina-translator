@echo off
title Mina Crowdsource - Traduction App
color 0A

echo.
echo ========================================
echo   MINA CROWDSOURCE - APPLICATION STREAMLIT
echo ========================================
echo.

:: Vérifier si le module init existe
if not exist "scripts\init_crowdsource_db.py" (
    echo ❌ Erreur: Script init_crowdsource_db.py non trouvé
    pause
    exit /b 1
)

:: Vérifier si le dataset Common Voice existe
if not exist "data\cv-corpus-25.0-2026-03-09\gej\validated.tsv" (
    echo ⚠️ Dataset Common Voice Mina non trouvé !
    echo    Les audios ne seront pas disponibles.
    echo.
    echo    Téléchargez le dataset depuis:
    echo    https://commonvoice.mozilla.org/ge/datasets
    echo.
    choice /c:OC /n "Voulez-vous continuer quand même? (O/N): "
    if errorlevel 2 exit /b 1
)

:: Vérifier si la base de données existe
if not exist "data\mina_crowdsource.db" (
    echo ℹ️ Base de données non trouvée.
    echo    Initialisation en cours...
    echo.

    py scripts\init_crowdsource_db.py

    if errorlevel 1 (
        echo ❌ Erreur lors de l'initialisation
        pause
        exit /b 1
    )
)

echo ✅ Prêt !
echo.
echo Lancement de l'application Streamlit...
echo.
echo ========================================
echo.

:: Lancer Streamlit
streamlit run crowdsource_app.py --server.port 8501 --server.headless false

pause