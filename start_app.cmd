@echo off
setlocal

cd /d "%~dp0"

set "PY_LAUNCHER="
set "PY_VERSION_TAG="

where py >nul 2>nul
if %errorlevel% neq 0 (
    echo Python launcher not found. Install Python 3 and make sure "py" is available.
    pause
    exit /b 1
)

py -3.13 -c "import sys" >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_LAUNCHER=py -3.13"
    set "PY_VERSION_TAG=313"
)

if not defined PY_LAUNCHER (
    py -3.12 -c "import sys" >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_LAUNCHER=py -3.12"
        set "PY_VERSION_TAG=312"
    )
)

if not defined PY_LAUNCHER (
    py -3.11 -c "import sys" >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_LAUNCHER=py -3.11"
        set "PY_VERSION_TAG=311"
    )
)

if not defined PY_LAUNCHER (
    echo No compatible Python version found.
    echo Install Python 3.11, 3.12, or 3.13, then run this launcher again.
    pause
    exit /b 1
)

set "VENV_DIR=.venv-py%PY_VERSION_TAG%"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment with %PY_LAUNCHER%...
    %PY_LAUNCHER% -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

if exist ".venv\pyvenv.cfg" (
    echo Legacy .venv detected. This launcher will use %VENV_DIR% instead.
)

echo Installing or updating dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

if not exist "models\face_landmarker.task" (
    echo Downloading MediaPipe models...
    "%PYTHON_EXE%" scripts\download_models.py
    if %errorlevel% neq 0 (
        echo Failed to download model files.
        pause
        exit /b 1
    )
)

echo Starting AR Filter app...
echo Open http://127.0.0.1:8000 in your browser.
echo Press Ctrl+C in this window to stop the server.
echo.

"%PYTHON_EXE%" app.py

echo.
echo App stopped.
pause
