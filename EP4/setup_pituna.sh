#!/bin/bash
# Prepares a Linux machine without conda (pituna): Miniforge in the home folder, the aga511 environment, then the test.
# Run it inside tmux:   tmux new -s setup 'bash setup_pituna.sh; bash'
set -e
cd "$(dirname "$0")"
FORGE=${FORGE:-$HOME/miniforge3}
INSTALLER=$HOME/Miniforge3-Linux-x86_64.sh

if ! command -v conda >/dev/null && [ ! -x "$FORGE/bin/conda" ]; then
    echo "=== installing Miniforge into $FORGE"
    curl -L -o "$INSTALLER" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$INSTALLER" -b -p "$FORGE"
fi
source "$FORGE/bin/activate"
if ! conda env list | grep -q '^aga511 '; then
    echo "=== creating the aga511 environment"
    conda env create -f environment.yml
fi
conda activate aga511
echo "=== $(python --version), $(nvcc --version | tail -1)"
python main.py test --yes
