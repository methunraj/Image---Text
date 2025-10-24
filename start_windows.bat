@echo off
REM ======================================================================
REM Windows startup script
REM - Creates .venv with `python -m venv` if missing
REM - Activates it via .venv\Scripts\activate.bat
REM - Installs dependencies from requirements.txt (if present)
REM - Ensures Streamlit is available
REM - Launches Streamlit app with clear console feedback
REM - Works regardless of the current directory by cd'ing to script folder
REM - Handles missing Python and common error cases gracefully
REM ======================================================================

setlocal EnableExtensions EnableDelayedExpansion

REM Change to the directory of this script
cd /d "%~dp0"

echo [INFO] Starting setup...

REM --------------------------------------------------
REM Locate a Python 3 interpreter
REM --------------------------------------------------
set "PYTHON_EXE="

REM Prefer `python` if it is Python 3
python -c "import sys; raise SystemExit(0 if sys.version_info[0]==3 else 1)" 1>nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_EXE=python"
) else (
  REM Try the Windows launcher for Python 3
  py -3 --version 1>nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_EXE=py -3"
  )
)

if not defined PYTHON_EXE (
  echo [ERROR] Python 3 is not installed or not on PATH. Please install Python 3 and retry.
  goto :hold
)

for /f "delims=" %%V in ('%PYTHON_EXE% --version 2^>^&1') do set "PY_VER=%%V"
echo [INFO] Using !PY_VER!

REM --------------------------------------------------
REM Create virtual environment if needed
REM --------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment in .venv\
  %PYTHON_EXE% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    goto :hold
  )
) else (
  echo [INFO] Virtual environment already exists (.venv).
)

REM --------------------------------------------------
REM Activate the virtual environment
REM --------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
  echo [ERROR] Could not find venv activation script at .venv\Scripts\activate.bat
  goto :hold
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate the virtual environment.
  goto :hold
)
echo [INFO] Virtual environment activated.

REM Ensure pip is available from this venv
python -m pip --version 1>nul 2>nul
if errorlevel 1 (
  echo [ERROR] pip not found in the virtual environment.
  goto :hold
)

REM --------------------------------------------------
REM Install dependencies
REM --------------------------------------------------
if exist requirements.txt (
  echo [INFO] Installing dependencies from requirements.txt
  python -m pip install --upgrade pip 1>nul 2>nul
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from requirements.txt
    goto :hold
  )
) else (
  echo [WARN] requirements.txt not found — skipping dependency installation.
)

REM Ensure Streamlit is available (in case requirements.txt was missing or incomplete)
python -c "import streamlit" 1>nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing Streamlit
  python -m pip install streamlit
  if errorlevel 1 (
    echo [ERROR] Failed to install Streamlit.
    goto :hold
  )
)

REM --------------------------------------------------
REM Determine app entry point
REM --------------------------------------------------
set "APP_FILE="

REM Allow override via STREAMLIT_APP environment variable if it points to a file
if defined STREAMLIT_APP (
  if exist "%STREAMLIT_APP%" (
    set "APP_FILE=%STREAMLIT_APP%"
  )
)

if not defined APP_FILE (
  if exist "app.py" (
    set "APP_FILE=app.py"
  ) else if exist "app\main.py" (
    set "APP_FILE=app\main.py"
  ) else if exist "main.py" (
    set "APP_FILE=main.py"
  ) else if exist "streamlit_app.py" (
    set "APP_FILE=streamlit_app.py"
  )
)

if not defined APP_FILE (
  echo [ERROR] Could not find an app entry point (missing app.py or app\main.py).
  echo         Set STREAMLIT_APP to your app path or create app.py.
  goto :hold
)

echo [INFO] Launching Streamlit app: %APP_FILE%
echo.

REM Run Streamlit in the foreground so logs are visible
python -m streamlit run "%APP_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Streamlit exited with code %EXIT_CODE%
) else (
  echo [INFO] Streamlit exited normally.
)

:hold
echo.
pause
endlocal & exit /b %EXIT_CODE%

