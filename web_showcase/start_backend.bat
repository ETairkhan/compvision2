@echo off
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 (echo Install Python 3.11 first. & pause & exit /b 1)
)
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. & pause & exit /b 1)
call .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
