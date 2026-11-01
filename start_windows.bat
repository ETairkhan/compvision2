@echo off
cd /d "%~dp0"
start "Object Atlas API" cmd /k "start_backend.bat"
start "Object Atlas Frontend" cmd /k "start_frontend.bat"
echo Starting two terminals. Open http://127.0.0.1:5173 once Vite is ready.
pause
