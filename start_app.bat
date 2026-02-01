@echo off
echo Starting Backend Server...
start "Backend Server" cmd /k "backend\venv\Scripts\python backend\main.py"

echo Starting Frontend...
cd frontend
start "Frontend" cmd /k "npm run dev"
echo App started! Access it at http://localhost:5173
