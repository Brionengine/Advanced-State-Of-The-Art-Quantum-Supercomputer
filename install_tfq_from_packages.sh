#!/bin/bash

# TensorFlow Quantum Installation Script (from local packages)
# This script installs TensorFlow Quantum from locally downloaded wheel files

echo "🚀 Installing TensorFlow Quantum from local packages..."
echo "======================================================="

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 not found. Installing..."
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
fi

# Check if packages directory exists
if [ ! -d "packages/wheels" ]; then
    echo "❌ Package directory not found!"
    echo "Please run the download script first or ensure packages are in 'packages/wheels/'."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "tfq_env" ]; then
    echo "📦 Creating Python 3.10 virtual environment..."
    python3.10 -m venv tfq_env
fi

# Activate environment
echo "🔧 Activating virtual environment..."
source tfq_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Count wheel files
wheel_count=$(ls packages/wheels/*.whl 2>/dev/null | wc -l)
echo "📊 Found $wheel_count wheel files to install"

if [ $wheel_count -eq 0 ]; then
    echo "❌ No wheel files found in packages/wheels/"
    echo "Please run the download script first."
    exit 1
fi

# Install from local wheels
echo "📦 Installing packages from local wheels..."
pip install --no-index --find-links packages/wheels tensorflow==2.15.0 tensorflow-quantum cirq sympy

# Verify installation
echo "✅ Verifying installation..."
python -c "
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
print(f'TensorFlow: {tf.__version__}')
print(f'TensorFlow Quantum: {tfq.__version__}')
print(f'Cirq: {cirq.__version__}')
print('✅ All packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 TensorFlow Quantum installation completed successfully!"
    echo ""
    echo "To activate the environment in the future:"
    echo "  source tfq_env/bin/activate"
    echo ""
    echo "To test the installation:"
    echo "  python simple_tfq_test.py"
    echo ""
    echo "To deactivate when done:"
    echo "  deactivate"
else
    echo "❌ Installation verification failed!"
    exit 1
fi