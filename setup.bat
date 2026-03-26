@echo off
cd /d "%~dp0"

echo ============================================
echo  SquatLock Setup
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

:: Activate venv and install packages
echo Installing dependencies...
call .venv\Scripts\activate
pip install -r requirements.txt
echo.
echo [OK] Dependencies installed.
echo.

:: Download MediaPipe Pose model if not present
if not exist "pose_landmarker_lite.task" (
    echo Downloading MediaPipe Pose model...
    curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
    echo [OK] Model downloaded.
) else (
    echo [OK] Pose model already present.
)

echo.
echo ============================================
echo  Setup complete!
echo  Run SquatLock.bat to start the app.
echo ============================================
pause
