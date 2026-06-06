@echo off
title Mina-Voice WebApp
color 0A
cd /d "%~dp0webapp"

echo.
echo ========================================
echo   MINA-VOICE WEBAPP
echo ========================================
echo.
echo   -> API:  http://localhost:8000/docs
echo   -> WebApp: http://localhost:3000
echo.
echo   Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.

python -m http.server 3000
pause