@echo off
cd /d "%~dp0frontend"
where npm >nul 2>&1
if errorlevel 1 (echo Install Node.js LTS first. & pause & exit /b 1)
if not exist "node_modules" call npm install
if errorlevel 1 (echo npm install failed. & pause & exit /b 1)
call npm run dev
pause
