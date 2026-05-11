#!/bin/bash
#
# setup_venv.sh — Create venvs and install dependencies for vLLM and SGLang.
#
# ⚠️  vLLM and SGLang have conflicting llguidance dependencies and MUST be
#     installed in separate virtual environments.
#
#     venv_vllm/   → vLLM + LiteLLM proxy
#     venv_sglang/ → SGLang (EAGLE speculative decoding)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

setup_one() {
    VENV_NAME="$1"
    REQ_FILE="$2"
    VENV_DIR="$SCRIPT_DIR/$VENV_NAME"

    if [ -d "$VENV_DIR" ]; then
        echo "✅ $VENV_NAME already exists, skipping create"
    else
        echo "📦 Creating $VENV_NAME virtual environment..."
        python3 -m venv "$VENV_DIR"
        echo "✅ $VENV_NAME created"
    fi

    echo "📥 Installing dependencies into $VENV_NAME..."
    "$VENV_DIR/bin/pip" install -U pip
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/$REQ_FILE"
    echo "✅ $VENV_NAME dependencies installed"
    echo ""
}

echo "============================================"
echo "Setting up vLLM stack (venv_vllm/)..."
echo "============================================"
setup_one "venv_vllm" "requirements_vllm.txt"

echo "============================================"
echo "Setting up SGLang stack (venv_sglang/)..."
echo "============================================"
setup_one "venv_sglang" "requirements_sglang.txt"

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "Activate vLLM stack:"
echo "   source venv_vllm/bin/activate"
echo ""
echo "Activate SGLang stack:"
echo "   source venv_sglang/bin/activate"
echo ""
echo "Then start the server:"
echo "   ./start.sh both         # vLLM backend"
echo "   ./start_sglang.sh both # SGLang backend"
