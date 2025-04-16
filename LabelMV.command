#!/bin/bash

# Move to the directory of this script
cd "$(dirname "$0")"

# Ensure script exits on error
set -e

echo "Checking if OpenCV, Ultralytics, and PyQt5 are installed..."

# Check for opencv-python
if pip3 show opencv-python > /dev/null 2>&1; then
    echo "OpenCV is already installed."
else
    echo "OpenCV is not installed. Installing now..."
    pip3 install opencv-python
fi

# Check for ultralytics
if pip3 show ultralytics > /dev/null 2>&1; then
    echo "Ultralytics is already installed."
else
    echo "Ultralytics is not installed. Installing now..."
    pip3 install ultralytics
fi

# Check for PyQt5
if pip3 show PyQt5 > /dev/null 2>&1; then
    echo "PyQt5 is already installed."
else
    echo "PyQt5 is not installed. Installing now..."
    pip3 install PyQt5
fi

# Run the Python script
python3 launcher.py