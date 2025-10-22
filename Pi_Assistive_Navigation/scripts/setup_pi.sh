#!/bin/bash
# Automated setup script for Raspberry Pi 4
# Run with: bash scripts/setup_pi.sh

set -e  # Exit on error

echo "=========================================="
echo "Assistive Navigation System - Setup"
echo "=========================================="
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "WARNING: This doesn't appear to be a Raspberry Pi!"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    git \
    libatlas-base-dev \
    libhdf5-dev \
    espeak \
    portaudio19-dev \
    libsdl2-dev \
    libsdl2-mixer-dev \
    pulseaudio \
    pulseaudio-module-bluetooth

echo ""
echo "Step 3: Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo ""
echo "Step 4: Installing TFLite Runtime (optimized for Pi)..."
pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime || {
    echo "WARNING: Could not install tflite_runtime, falling back to tensorflow.lite"
}

echo ""
echo "Step 5: Setting up GPIO permissions..."
sudo usermod -a -G gpio $USER
sudo usermod -a -G video $USER
sudo usermod -a -G audio $USER

echo ""
echo "Step 6: Creating directory structure..."
mkdir -p soundscapes/{forest,park,beach,city,residential,indoor,museum,plaza,parking}
mkdir -p models
mkdir -p logs

echo ""
echo "Step 7: Setting file permissions..."
chmod +x main.py
chmod +x scripts/*.py
chmod +x scripts/*.sh

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your TFLite models to models/"
echo "2. Add audio files to soundscapes/*/"
echo "3. Wire up HC-SR04 ultrasonic sensor"
echo "4. Connect USB camera"
echo "5. Test with: python3 main.py"
echo ""
echo "Useful commands:"
echo "  Test ultrasonic: python3 scripts/test_ultrasonic.py"
echo "  Test camera:     python3 scripts/test_camera.py"
echo "  Benchmark:       python3 scripts/benchmark_models.py"
echo ""
echo "You may need to reboot for group permissions to take effect."
read -p "Reboot now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi

