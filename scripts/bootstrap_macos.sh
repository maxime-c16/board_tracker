#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -e "${ROOT_DIR}"

echo "macOS environment ready at ${VENV_DIR}"
echo "Recommended next step for Apple Silicon: python -m pip install torch torchvision"
echo "Activate with: source ${VENV_DIR}/bin/activate"
