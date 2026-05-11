#!/bin/bash
#
# start_sglang.sh — Zyphra/ZAYA1-8B with SGLang + EAGLE speculative decoding.
#
# Uses SGLang's built-in EAGLE algorithm: 3 speculative steps × 4 draft
# tokens per cycle.
#
# ⚠️  Requirements:
#   1. SGLang special build (see below)
#   2. Zyphra/ZAYA1-8B model downloaded (~24 GB BF16)
#
# Install SGLang:
#   source venv_sglang/bin/activate
#   pip install -r requirements_sglang.txt
#
# Model is already in HuggingFace cache:
#   Zyphra/ZAYA1-8B  (~24 GB BF16)
#
# Usage:
#   ./start_sglang.sh           # SGLang backend + token stats + LiteLLM proxy
#   ./start_sglang.sh backend   # SGLang only
#   ./start_sglang.sh proxy     # LiteLLM proxy only
#
# Tuning knobs (env):
#   SGLANG_MODEL               Model path (default Zyphra/ZAYA1-8B)
#   SGLANG_SPEC_NUM_STEPS      EAGLE steps (default 3)
#   SGLANG_SPEC_EAGLE_TOPK     Top-k per step (default 1)
#   SGLANG_SPEC_DRAFT_TOKENS   Draft tokens per cycle (default 4)
#   SGLANG_MEM_FRACTION        Static memory fraction (default 0.45)
#   SGLANG_MAX_MODEL_LEN       Context window (default 131072)
#   SGLANG_TP_SIZE             Tensor parallel size (default 1)
#
# Logs:  logs/sglang_server.log, logs/litellm_proxy.log
# Ports: 11111 LiteLLM proxy · 11112 SGLang backend · 11113 token stats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect venv (prefer venv_sglang for SGLang stack)
if [ -d "$SCRIPT_DIR/venv_sglang" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv_sglang/bin/python3"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

mkdir -p "$SCRIPT_DIR/logs"

export SGLANG_MODEL="${SGLANG_MODEL:-Zyphra/ZAYA1-8B}"
export SGLANG_SPEC_NUM_STEPS="${SGLANG_SPEC_NUM_STEPS:-3}"
export SGLANG_SPEC_EAGLE_TOPK="${SGLANG_SPEC_EAGLE_TOPK:-1}"
export SGLANG_SPEC_DRAFT_TOKENS="${SGLANG_SPEC_DRAFT_TOKENS:-4}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.45}"
export SGLANG_MAX_MODEL_LEN="${SGLANG_MAX_MODEL_LEN:-131072}"
export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"

echo "🦅 SGLang EAGLE config:"
echo "   model=${SGLANG_MODEL}"
echo "   steps=${SGLANG_SPEC_NUM_STEPS}  topk=${SGLANG_SPEC_EAGLE_TOPK}  draft_tokens=${SGLANG_SPEC_DRAFT_TOKENS}"
echo "   mem=${SGLANG_MEM_FRACTION}  ctx=${SGLANG_MAX_MODEL_LEN}  tp=${SGLANG_TP_SIZE}"
echo ""

start_backend() {
    echo "🚀 Starting SGLang backend (port 11112)..."
    $VENV_PYTHON "$SCRIPT_DIR/zaya_sglang_server.py" \
        >> "$SCRIPT_DIR/logs/sglang_server.log" 2>&1 &
    echo "Backend PID: $!"
}

start_stats() {
    echo "📊 Starting token stats server (port 11113)..."
    $VENV_PYTHON "$SCRIPT_DIR/zaya_token_tracker.py" &
    echo "Stats PID: $!"
}

start_proxy() {
    echo "🔀 Starting LiteLLM proxy (port 11111)..."
    $VENV_PYTHON "$SCRIPT_DIR/server_compress.py" &
    echo "Proxy PID: $!"
}

case "${1:-both}" in
    both)
        start_backend
        sleep 5
        start_stats
        sleep 1
        start_proxy
        echo ""
        echo "✅ All services started (SGLang EAGLE):"
        echo "   SGLang backend: http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Token stats:    http://0.0.0.0:11113"
        echo ""
        echo "Tail logs with:  tail -f logs/sglang_server.log"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy|stats]"
        exit 1
        ;;
esac