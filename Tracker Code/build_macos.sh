#!/bin/bash
# Build script for macOS Screenshot Tracker
# Creates a virtual environment, installs dependencies, and builds the .app bundle

set -e  # Exit on any error

echo "=========================================="
echo "macOS Screenshot Tracker Build Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}ERROR: This script must be run on macOS${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}ERROR: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Python version: $(python3 --version)${NC}"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf __pycache__/
rm -rf *.spec.bak
echo -e "${GREEN}Clean complete${NC}"
echo ""

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
echo -e "${GREEN}Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: requirements.txt not found${NC}"
    exit 1
fi

pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed${NC}"
echo ""

# Install PyInstaller
echo "Installing PyInstaller..."
pip install pyinstaller>=5.13.0
echo -e "${GREEN}PyInstaller installed${NC}"
echo ""

# Build with PyInstaller
echo "Building macOS application with PyInstaller..."
echo "This may take a few minutes..."
echo ""

pyinstaller app_macos.spec

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Build successful!"
    echo "==========================================${NC}"
    echo ""
    echo "Application bundle created at:"
    echo -e "${GREEN}  dist/ScreenshotTracker.app${NC}"
    echo ""
    echo "To run the app:"
    echo "  open dist/ScreenshotTracker.app"
    echo ""
    echo "Note: On first run, macOS Gatekeeper may block the app."
    echo "See README.md for instructions on handling Gatekeeper."
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "Build failed!"
    echo "==========================================${NC}"
    echo ""
    exit 1
fi

# Deactivate virtual environment
deactivate

echo -e "${GREEN}Build script completed successfully${NC}"
