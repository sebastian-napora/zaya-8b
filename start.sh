#!/bin/bash
#
# Start Zyphra/ZAYA1-8B serving stack (vLLM backend + LiteLLM proxy)
#
# Usage:
#   ./start.sh          # start both vLLM backend and LiteLLM proxy
#   ./start.sh backend  # start only vLLM backend (11112)
#   ./start.sh proxy    # start only LiteLLM proxy (11111)
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
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
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
    start_service "Backend" "$SCRIPT_DIR/logs/vllm_backend.log" "$VENV_PYTHON" zaya_server.py
    BACKEND_PID="$SERVICE_PID"
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
    *)
        echo "Usage: $0 [both|backend|proxy|stats]"
        exit 1
        ;;
esac
