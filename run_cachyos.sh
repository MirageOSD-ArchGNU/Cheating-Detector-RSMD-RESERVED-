#!/usr/bin/env bash
set -e

echo "=================================================="
echo "   MirageIME WEBCHEAT - CachyOS / Arch Launcher   "
echo "=================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Installing via pacman..."
    sudo pacman -S --needed --noconfirm python python-pip
fi

# Ensure system dependencies for OpenCV & Wayland/X11 GUI
echo "[*] Checking CachyOS system libraries (opencv, v4l-utils, tk/qt)..."
if command -v pacman &> /dev/null; then
    sudo pacman -S --needed --noconfirm python-virtualenv v4l-utils qt6-base libxcb xcb-util-cursor || true
fi

# Create isolated virtual environment (avoids Arch PEP 668 externally-managed-environment error)
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[*] Upgrading pip and installing AI/CV dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "[*] Launching MirageIME WEBCHEAT..."
# Force XCB if running under Wayland without native OpenCV Wayland support, fallback gracefully
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}
python3 mirage_webcheat.py "$@"
