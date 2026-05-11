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

    echo "$label PID: $!"
}

start_backend() {
    echo "🚀 Starting vLLM backend (port 11112)..."
    start_service "Backend" "$SCRIPT_DIR/logs/vllm_backend.log" "$VENV_PYTHON" zaya_server.py
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

case "${1:-both}" in
    both)
        start_backend
        new_token_session
        echo "Waiting 5s for backend to initialize..."
        sleep 5
        start_stats
        sleep 1
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
