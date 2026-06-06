@echo off
title Mina-Voice

REM ========================================
REM Lancement API + WebApp
REM ========================================

echo.
echo ========================================
echo    MINA-VOICE - SYSTEME DE TRADUCTION
echo ========================================
echo.
echo  Lancement des services...
echo.

REM Option: Lancer l'API en arrière-plan
REM start cmd /k "py -m uvicorn api.main:app --reload --port 8000"

echo.
echo  1. Lancer l'API (nouveau terminal):
echo     py -m uvicorn api.main:app --reload --port 8000
echo.
echo  2. Lancer le WebApp (nouveau terminal):
echo     cd webapp ^&^& py -m http.server 3000
echo.
echo ========================================
echo.

REM Lancer le WebApp depuis le dossier webapp
cd /d "%~dp0webapp"
python -m http.server 3000