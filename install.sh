#!/bin/bash
echo "============================================"
echo "  TheFlowBoard - Installation"
echo "============================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.9+ first."
    exit 1
fi

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  To run TheFlowBoard:"
echo "    1. activate venv:  source venv/bin/activate"
echo "    2. run:            streamlit run app.py"
echo "============================================"
