#!/bin/bash
set -e

git submodule update --init --recursive

# Create venv only if it doesn't exist
if [ ! -d "venv" ]; then
    uv venv --python 3.11 venv
fi
source venv/bin/activate

uv pip install -e ./deps/robosuite
uv pip install -e ./deps/robocasa

# Setup macros if not already done
if [ ! -f "deps/robocasa/robocasa/macros_private.py" ]; then
    cd deps/robocasa && python -m robocasa.scripts.setup_macros && cd ../..
fi
if [ ! -f "deps/robosuite/robosuite/macros_private.py" ]; then
    cd deps/robosuite && python -m robosuite.scripts.setup_macros && cd ../..
fi

# Download assets only if not already present
if [ ! -d "deps/robocasa/robocasa/models/assets/textures" ]; then
    cd deps/robocasa && python -m robocasa.scripts.download_kitchen_assets && cd ../..
else
    echo "Assets already downloaded, skipping."
fi
