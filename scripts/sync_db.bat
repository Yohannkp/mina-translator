@echo off
REM scripts/sync_db.bat - Synchronise la base de données depuis GitHub
REM ============================================================================

echo.
echo ============================================================
echo 🔄 SYNCHRONISATION DE LA BASE DE DONNEES
echo ============================================================
echo.

cd /d "%~dp0\.."

echo 📥 git pull origin main...
git pull origin main

echo.
echo ============================================================
echo ✅ Termine!
echo ============================================================
echo.

pause