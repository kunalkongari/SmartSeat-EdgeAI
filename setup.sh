#!/bin/bash
# setup.sh — Run once to install dependencies and download model
# Usage: bash setup.sh

set -e

echo "================================================"
echo "  Smart Seat Occupancy — YOLOv8 Setup"
echo "================================================"

# 1. Create venv if not present
if [ ! -d "venv" ]; then
  echo "[1/4] Creating virtual environment..."
  python3 -m venv venv
else
  echo "[1/4] Virtual environment already exists."
fi

# 2. Activate
source venv/bin/activate

# 3. Install dependencies
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 4. Download YOLOv8n model
echo "[3/4] Downloading YOLOv8n model..."
python3 download_model.py

echo "[4/4] Setup complete!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  python3 app.py"
echo ""
echo "Then open: http://localhost:5000"
