#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "  P-H3-Image WebUI Launcher (Linux / macOS)"
echo "  A1111-Style Easy Interface for PrunaAI / MiniMax-H3"
echo "============================================================"
echo ""

# 1. Check Git
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is not installed. Please install git first."
    exit 1
fi

# 2. Clone ComfyUI engine if missing
if [ ! -f "ComfyUI/main.py" ]; then
    echo "[1/4] Cloning headless ComfyUI inference engine..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
else
    echo "[1/4] ComfyUI inference engine found."
fi

# 3. Create Python virtual environment if missing
OS_NAME="$(uname -s)"
if [ ! -x ".venv/bin/python" ]; then
    echo "[2/4] Setting up Python virtual environment (.venv)..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv --python 3.12
        if [ "$OS_NAME" = "Darwin" ]; then
            uv pip install --python .venv/bin/python torch torchvision torchaudio
        else
            uv pip install --python .venv/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        fi
        uv pip install --python .venv/bin/python -r ComfyUI/requirements.txt gradio huggingface_hub hf_transfer websocket-client requests
    else
        PYTHON_BIN="${PYTHON_BIN:-python3}"
        "$PYTHON_BIN" -m venv .venv
        .venv/bin/python -m pip install --upgrade pip
        if [ "$OS_NAME" = "Darwin" ]; then
            .venv/bin/python -m pip install torch torchvision torchaudio
        else
            .venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        fi
        .venv/bin/python -m pip install -r ComfyUI/requirements.txt gradio huggingface_hub hf_transfer websocket-client requests
    fi
else
    echo "[2/4] Python virtual environment (.venv) found."
fi

# 4. Check & Download P-H3-Image weights
PROFILE="${PH3_PROFILE:-24gb}"
echo "[3/4] Checking P-H3-Image model weights (Profile: ${PROFILE})..."
.venv/bin/python download_models.py --profile "${PROFILE}"

# 5. Launch A1111-style WebUI
echo "[4/4] Starting P-H3-Image WebUI..."
exec .venv/bin/python webui.py "$@"
