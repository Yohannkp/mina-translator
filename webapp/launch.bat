@echo off
title Mina-Voice WebApp
cd /d "%~dp0"
start http://localhost:8000/docs
cd /d "%~dp0webapp"
python -m http.server 3000