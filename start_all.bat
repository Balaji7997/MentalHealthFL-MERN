@echo off
title MentalHealthFL Launcher
echo Starting MentalHealthFL Full-Stack Platform...

start "1 - FastAPI AI Service (8000)" cmd /k "cd /d C:\Users\ASUS\MentalHealthFL-MERN\ai-service && call venv\Scripts\activate && uvicorn main:app --host 127.0.0.1 --port 8000"

timeout /t 4 /nobreak >nul

start "2 - Express Backend (5000)" cmd /k "cd /d C:\Users\ASUS\MentalHealthFL-MERN\backend && node server.js"

timeout /t 2 /nobreak >nul

start "3 - React Dashboard (5173)" cmd /k "cd /d C:\Users\ASUS\MentalHealthFL-MERN && npm run dev"

timeout /t 4 /nobreak >nul
start http://localhost:5173