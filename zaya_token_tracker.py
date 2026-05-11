#!/usr/bin/env python3
"""
Token usage tracker for ZAYA1-8B serving stack.

Records per-request token usage (prompt/completion/total tokens) and
cumulative costs to logs/token_stats.json.
"""

import os
import json
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
STATS_FILE = LOG_DIR / "token_stats.json"

_session_id: str = ""
_lock = threading.Lock()
_stats: dict = defaultdict(float)
_request_counts: dict = defaultdict(int)
_model_stats: dict = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0})

_current_session_file = LOG_DIR / ".current_session"


def new_session() -> str:
    """Create a fresh session ID."""
    global _session_id
    with _lock:
        _session_id = f"zaya-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        _stats.clear()
        _request_counts.clear()
        _model_stats.clear()
        _current_session_file.write_text(_session_id)
        return _session_id


def get_session() -> str:
    """Return current session ID, loading from disk if needed."""
    global _session_id
    if not _session_id and _current_session_file.exists():
        _session_id = _current_session_file.read_text().strip()
    return _session_id or "unknown"


class TokenTracker:
    """Custom LiteLLM callback to track token usage per model."""

    def __init__(self):
        self._request_start: dict = {}

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        model = kwargs.get("model", "unknown")
        try:
            prompt_tokens = getattr(response_obj, "usage", None) and getattr(response_obj.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response_obj, "usage", None) and getattr(response_obj.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response_obj, "usage", None) and getattr(response_obj.usage, "total_tokens", 0) or 0
        except Exception:
            prompt_tokens = completion_tokens = total_tokens = 0

        with _lock:
            ms = (end_time - start_time) * 1000
            _model_stats[model]["prompt_tokens"] += prompt_tokens
            _model_stats[model]["completion_tokens"] += completion_tokens
            _model_stats[model]["total_tokens"] += total_tokens
            _model_stats[model]["requests"] += 1
            _stats["total_tokens"] += total_tokens
            _stats["prompt_tokens"] += prompt_tokens
            _stats["completion_tokens"] += completion_tokens
            _request_counts[model] += 1

        entry = {
            "session": get_session(),
            "ts": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": round(ms, 1),
        }
        stats_log = LOG_DIR / "token_requests.jsonl"
        with open(stats_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass


_tracker = TokenTracker()


def register():
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _Callback(CustomLogger):
        async def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _tracker.log_success_event(kwargs, response_obj, start_time, end_time)
        async def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            _tracker.log_failure_event(kwargs, response_obj, start_time, end_time)

    litellm.callbacks.append(_Callback())


# ── Token stats HTTP server ────────────────────────────────────────────────────
def start_stats_server(port: int = 11113):
    """Minimal HTTP server exposing token stats as JSON."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json as _json

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/stats" or self.path == "/":
                with _lock:
                    data = {
                        "session": get_session(),
                        "cumulative": dict(_stats),
                        "by_model": dict(_model_stats),
                        "request_counts": dict(_request_counts),
                    }
                body = _json.dumps(data, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_):
            pass

    HTTPServer(("0.0.0.0", port), _Handler).serve_forever()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11113
    print(f"📊 Token stats server starting on port {port}…")
    start_stats_server(port)
