@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
python -c "import fastapi, uvicorn, numpy, scipy, soundfile" 2>nul
if errorlevel 1 python -m pip install -r requirements.txt
python app\server.py
pause
