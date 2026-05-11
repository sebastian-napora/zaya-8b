#!/bin/bash
#
# Start ZAYA in no-thinking mode while keeping OpenAI tool calling enabled.
#
# Usage:
#   ./start_no_thinking.sh          # backend + token stats + LiteLLM proxy
#   ./start_no_thinking.sh backend  # backend only
#   ./start_no_thinking.sh proxy    # proxy only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ZAYA_ENABLE_REASONING=0
export ZAYA_REASONING_PARSER=""
export ZAYA_ENABLE_AUTO_TOOL_CHOICE="${ZAYA_ENABLE_AUTO_TOOL_CHOICE:-1}"
export ZAYA_TOOL_CALL_PARSER="${ZAYA_TOOL_CALL_PARSER:-zaya_xml}"

echo "Starting ZAYA no-thinking mode:"
echo "   Reasoning parser: disabled"
echo "   Auto tool choice: $ZAYA_ENABLE_AUTO_TOOL_CHOICE"
echo "   Tool call parser: ${ZAYA_TOOL_CALL_PARSER:-disabled}"
echo ""

exec "$SCRIPT_DIR/start.sh" "$@"
