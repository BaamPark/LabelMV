#!/bin/bash

# Function to check and install a package
check_and_install() {
    package_name=$1
    echo "Checking for $package_name..."
    pip list | grep "$package_name" > /dev/null

    if [ $? -eq 0 ]; then
        echo "$package_name is already installed."
    else
        echo "$package_name is not installed. Installing now..."
        pip install "$package_name"
    fi
}

# Check if Python and pip are available
echo "Checking for Python and pip..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Exiting..."
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Exiting..."
    exit 1
fi

# Check and install opencv-python and ultralytics
check_and_install "opencv-python"
check_and_install "ultralytics"

# Run launcher.py
echo "Running launcher.py..."
python3 launcher.py
