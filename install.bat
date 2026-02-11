@echo off
echo ============================================
echo   TheFlowBoard - Installation
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo   Installation complete!
echo.
echo   To run TheFlowBoard:
echo     1. activate venv:  venv\Scripts\activate.bat
echo     2. run:            streamlit run app.py
echo ============================================
pause
