#!/bin/bash
#
# Start Zyphra/ZAYA1-8B serving stack (vLLM backend + LiteLLM proxy)
#
# Usage:
#   ./start.sh          # start both vLLM backend and LiteLLM proxy
#   ./start.sh backend  # start only vLLM backend (11112)
#   ./start.sh proxy    # start only LiteLLM proxy (11111)
#   ./start_no_thinking.sh  # same stack, tool calling on, reasoning parser off
#
# Architecture:
#   Copilot -> LiteLLM (11111) -> vLLM (11112)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect venv (prefer venv_vllm for vLLM stack)
if [ -d "$SCRIPT_DIR/venv_vllm" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv_vllm/bin/python3"
elif [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
elif [ -d "../lunch-model/venv" ]; then
    VENV_PYTHON="../lunch-model/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

mkdir -p logs
STREAM_LOGS="${ZAYA_STREAM_LOGS:-1}"
BACKEND_HEALTH_URL="${ZAYA_BACKEND_HEALTH_URL:-http://localhost:11112/health}"
BACKEND_WAIT_TIMEOUT="${ZAYA_BACKEND_WAIT_TIMEOUT:-900}"
BACKEND_WAIT_INTERVAL="${ZAYA_BACKEND_WAIT_INTERVAL:-5}"
ZAYA_VLLM_ATTENTION_BACKEND="${ZAYA_VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export ZAYA_VLLM_ATTENTION_BACKEND
ZAYA_BLOCK_FLASH_ATTN="${ZAYA_BLOCK_FLASH_ATTN:-1}"
export ZAYA_BLOCK_FLASH_ATTN
ZAYA_ENABLE_AUTO_TOOL_CHOICE="${ZAYA_ENABLE_AUTO_TOOL_CHOICE:-1}"
export ZAYA_ENABLE_AUTO_TOOL_CHOICE
ZAYA_TOOL_CALL_PARSER="${ZAYA_TOOL_CALL_PARSER:-zaya_xml}"
export ZAYA_TOOL_CALL_PARSER
ZAYA_ENABLE_REASONING="${ZAYA_ENABLE_REASONING:-1}"
export ZAYA_ENABLE_REASONING
ZAYA_REASONING_PARSER="${ZAYA_REASONING_PARSER:-qwen3}"
export ZAYA_REASONING_PARSER
ZAYA_THINKING_BUDGET="${ZAYA_THINKING_BUDGET:-8192}"
export ZAYA_THINKING_BUDGET
ZAYA_CHAT_TEMPLATE="${ZAYA_CHAT_TEMPLATE:-}"
export ZAYA_CHAT_TEMPLATE
ZAYA_BEHAVIOR_GUARD="${ZAYA_BEHAVIOR_GUARD:-1}"
export ZAYA_BEHAVIOR_GUARD
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0f}"
export TORCH_CUDA_ARCH_LIST
if [ -x "/usr/local/cuda/bin/ptxas" ]; then
    TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
    export TRITON_PTXAS_PATH
fi
SERVICE_PID=""

start_service() {
    local label="$1"
    local logfile="$2"
    shift 2

    : > "$logfile"
    if [ "$STREAM_LOGS" = "1" ]; then
        echo "   Streaming logs here and saving to: $logfile"
        "$@" > >(tee -a "$logfile") 2>&1 &
    else
        echo "   Logs: $logfile"
        "$@" > "$logfile" 2>&1 &
    fi

    SERVICE_PID=$!
    echo "$label PID: $SERVICE_PID"
}

start_backend() {
    echo "🚀 Starting vLLM backend (port 11112)..."
    echo "   Attention backend: $ZAYA_VLLM_ATTENTION_BACKEND"
    echo "   Block flash-attn import: $ZAYA_BLOCK_FLASH_ATTN"
    echo "   Auto tool choice: $ZAYA_ENABLE_AUTO_TOOL_CHOICE"
    echo "   Tool call parser: ${ZAYA_TOOL_CALL_PARSER:-disabled}"
    echo "   Reasoning parser: $([ "$ZAYA_ENABLE_REASONING" = "1" ] && echo "${ZAYA_REASONING_PARSER:-disabled}" || echo "disabled")"
    echo "   Thinking budget:  $([ "$ZAYA_ENABLE_REASONING" = "1" ] && echo "${ZAYA_THINKING_BUDGET} tokens (0=unlimited)" || echo "n/a")"
    echo "   Chat template: ${ZAYA_CHAT_TEMPLATE:-model default}"
    echo "   Behavior guard: $([ "$ZAYA_BEHAVIOR_GUARD" = "1" ] && echo "enabled" || echo "disabled")"
    echo "   TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    echo "   TRITON_PTXAS_PATH: ${TRITON_PTXAS_PATH:-not set}"
    print_cuda_stack
    start_service "Backend" "$SCRIPT_DIR/logs/vllm_backend.log" "$VENV_PYTHON" zaya_server.py
    BACKEND_PID="$SERVICE_PID"
}

print_cuda_stack() {
    echo "   CUDA/Python packages:"
    "$VENV_PYTHON" -m pip list 2>/dev/null \
        | awk 'BEGIN{IGNORECASE=1} /flash|flashinfer|vllm|torch|triton|cuda/ {print "     " $0}' \
        || true
}

new_token_session() {
    NEW_SID=$("$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import zaya_token_tracker
sid = zaya_token_tracker.new_session()
print(sid, end='')
" 2>/dev/null)
    echo "📊 New token session: ${NEW_SID:-unknown}"
}

start_stats() {
    echo "🚀 Starting token stats server (port 11113)..."
    start_service "Stats" "$SCRIPT_DIR/logs/token_stats.log" "$VENV_PYTHON" "$SCRIPT_DIR/zaya_token_tracker.py"
}

start_proxy() {
    echo "🚀 Starting LiteLLM proxy (port 11111)..."
    start_service "Proxy" "$SCRIPT_DIR/logs/lite_llm.log" "$VENV_PYTHON" server_compress.py
}

wait_for_backend() {
    local elapsed=0

    echo "⏳ Waiting for vLLM backend health: $BACKEND_HEALTH_URL"
    echo "   Timeout: ${BACKEND_WAIT_TIMEOUT}s (ZAYA_BACKEND_WAIT_TIMEOUT to override)"

    while [ "$elapsed" -lt "$BACKEND_WAIT_TIMEOUT" ]; do
        if curl -fsS --max-time 2 "$BACKEND_HEALTH_URL" >/dev/null 2>&1; then
            echo "✅ vLLM backend is healthy after ${elapsed}s"
            return 0
        fi

        if [ -n "${BACKEND_PID:-}" ] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo "❌ vLLM backend process exited before becoming healthy."
            echo "   Check: $SCRIPT_DIR/logs/vllm_backend.log"
            return 1
        fi

        sleep "$BACKEND_WAIT_INTERVAL"
        elapsed=$((elapsed + BACKEND_WAIT_INTERVAL))
        echo "   Still loading... ${elapsed}s elapsed"
    done

    echo "❌ Timed out waiting for vLLM backend after ${BACKEND_WAIT_TIMEOUT}s."
    echo "   Keep watching this terminal or check: $SCRIPT_DIR/logs/vllm_backend.log"
    echo "   Increase timeout with: ZAYA_BACKEND_WAIT_TIMEOUT=1800 ./start.sh both"
    return 1
}

case "${1:-both}" in
    both)
        start_backend
        new_token_session
        start_stats
        wait_for_backend
        start_proxy
        echo ""
        echo "✅ All services started:"
        echo "   vLLM backend:   http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Token stats:    http://0.0.0.0:11113"
        echo "   Terminal logs:  $([ "$STREAM_LOGS" = "1" ] && echo "enabled" || echo "disabled")"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:11112/health"
        echo "  curl http://localhost:11111/health"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        new_token_session
        wait_for_backend
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    diagnose)
        echo "Python: $VENV_PYTHON"
        echo "Attention backend: $ZAYA_VLLM_ATTENTION_BACKEND"
        echo "Block flash-attn import: $ZAYA_BLOCK_FLASH_ATTN"
        echo "Auto tool choice: $ZAYA_ENABLE_AUTO_TOOL_CHOICE"
        echo "Tool call parser: ${ZAYA_TOOL_CALL_PARSER:-disabled}"
        echo "Reasoning parser: $([ "$ZAYA_ENABLE_REASONING" = "1" ] && echo "${ZAYA_REASONING_PARSER:-disabled}" || echo "disabled")"
        echo "Thinking budget:  $([ "$ZAYA_ENABLE_REASONING" = "1" ] && echo "${ZAYA_THINKING_BUDGET} tokens (0=unlimited)" || echo "n/a")"
        echo "Chat template: ${ZAYA_CHAT_TEMPLATE:-model default}"
        echo "Behavior guard: $([ "$ZAYA_BEHAVIOR_GUARD" = "1" ] && echo "enabled" || echo "disabled")"
        echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
        echo "TRITON_PTXAS_PATH: ${TRITON_PTXAS_PATH:-not set}"
        print_cuda_stack
        "$VENV_PYTHON" - <<'PY'
import importlib.util
mods = [
    "flash_attn", "flash_attn_2_cuda", "flash_attn_3_cuda",
    "flash_attn_interface", "flash_attn_cuda",
    "flashinfer", "vllm", "torch", "triton",
]
for name in mods:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else 'not found'}")
PY
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy|stats|diagnose]"
        exit 1
        ;;
esac
