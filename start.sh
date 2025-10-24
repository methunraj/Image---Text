#!/usr/bin/env bash
#
# Cross-platform startup script for macOS/Linux
# - Creates (if missing) and activates a Python venv in .venv/
# - Installs dependencies from requirements.txt (if present)
# - Ensures Streamlit is available
# - Launches the Streamlit app
# - Works when run from any directory by cd'ing to the script's location
# - Provides clear logging and graceful error handling

set -uo pipefail

# ---------------
# Helper functions
# ---------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*" 1>&2; }

# Prompt the user before closing if running in a terminal (TTY)
prompt_to_close() {
  if [[ -t 0 && -t 1 ]]; then
    echo
    read -r -p "Press Enter to close this window..." _
  fi
}
trap prompt_to_close EXIT

# Resolve the directory of this script, even if sourced via symlink
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
cd "$SCRIPT_DIR" || { error "Failed to cd to script directory: $SCRIPT_DIR"; exit 1; }

# Make the script executable for future one-click runs (harmless if already executable)
chmod +x "$0" >/dev/null 2>&1 || true

# Pick the best available Python 3 executable
find_python() {
  local py
  if command -v python3 >/dev/null 2>&1; then
    py=python3
  elif command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(0 if sys.version_info[0]==3 else 1)'; then
    py=python
  else
    py=""
  fi
  printf '%s' "$py"
}

PYTHON="$(find_python)"
if [[ -z "$PYTHON" ]]; then
  error "Python 3 is not installed or not on PATH. Please install Python 3 and retry."
  exit 1
fi

info "Using Python: $($PYTHON --version 2>&1)"

# --------------------
# Create virtual env
# --------------------
if [[ ! -x .venv/bin/python ]]; then
  info "Creating virtual environment in .venv/"
  if ! "$PYTHON" -m venv .venv; then
    error "Failed to create virtual environment."
    exit 1
  fi
else
  info "Virtual environment already exists (.venv)."
fi

# --------------------
# Activate virtual env
# --------------------
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  info "Virtual environment activated."
else
  error "Could not find venv activation script at .venv/bin/activate"
  exit 1
fi

# Ensure pip is usable within the venv
if ! python -m pip --version >/dev/null 2>&1; then
  error "pip not found in the virtual environment."
  exit 1
fi

# -----------------------------
# Install project dependencies
# -----------------------------
if [[ -f requirements.txt ]]; then
  info "Installing dependencies from requirements.txt"
  if ! python -m pip install --upgrade pip >/dev/null 2>&1; then
    warn "Unable to upgrade pip (continuing)."
  fi
  if ! python -m pip install -r requirements.txt; then
    error "Failed to install dependencies from requirements.txt"
    exit 1
  fi
else
  warn "requirements.txt not found — skipping dependency installation."
fi

# Ensure Streamlit is available (in case requirements.txt is missing or incomplete)
if ! python -c "import streamlit" >/dev/null 2>&1; then
  info "Installing Streamlit"
  if ! python -m pip install streamlit; then
    error "Failed to install Streamlit."
    exit 1
  fi
fi

# -----------------------------
# Determine app entry point
# -----------------------------
# Priority: env var STREAMLIT_APP -> app.py -> app/main.py -> main.py -> streamlit_app.py
APP_FILE=""
if [[ -n "${STREAMLIT_APP:-}" && -f "$STREAMLIT_APP" ]]; then
  APP_FILE="$STREAMLIT_APP"
elif [[ -f app.py ]]; then
  APP_FILE="app.py"
elif [[ -f app/main.py ]]; then
  APP_FILE="app/main.py"
elif [[ -f main.py ]]; then
  APP_FILE="main.py"
elif [[ -f streamlit_app.py ]]; then
  APP_FILE="streamlit_app.py"
fi

if [[ -z "$APP_FILE" ]]; then
  error "Could not find an app entry point (missing app.py or app/main.py)."
  error "Set STREAMLIT_APP to your app path or create app.py."
  exit 1
fi

info "Launching Streamlit app: $APP_FILE"
echo

# Run Streamlit in the foreground so logs are visible in this window
python -m streamlit run "$APP_FILE"
exit_code=$?

if [[ $exit_code -ne 0 ]]; then
  error "Streamlit exited with code $exit_code"
else
  info "Streamlit exited normally."
fi

exit "$exit_code"

