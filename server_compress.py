#!/usr/bin/env python3
"""
LiteLLM proxy for Zyphra/ZAYA1-8B.

Architecture:
    Copilot -> LiteLLM (11111) -> vLLM (11112)
    or
    Copilot -> LiteLLM (11111) -> SGLang (11112)

Usage:
    # Terminal 1: start vLLM backend on 11112
    python3 zaya_server.py

    # Terminal 2: start LiteLLM proxy on 11111
    python3 server_compress.py
"""

import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from zaya_logging import configure_logging

logger = configure_logging("zaya_litellm", "litellm_detailed.log")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_LEVEL = os.environ.get("ZAYA_LOG_LEVEL", "DEBUG").upper()
LOG_LEVEL_VALUE = getattr(logging, LOG_LEVEL, logging.DEBUG)
os.environ.setdefault("LITELLM_LOG", LOG_LEVEL)
os.environ["LITELLM_REQUEST_LOGGING"] = "true"

logger.info("Importing LiteLLM")
import litellm
logger.info("LiteLLM imported")

logger.info("Registering token tracker callback")
import zaya_token_tracker  # noqa: F401 — records per-request token usage
zaya_token_tracker.register()
logger.info("Token tracker callback registered")

# Create dedicated logger for image requests
litellm_logger = logging.getLogger("zaya.litellm.requests")
litellm_logger.setLevel(logging.DEBUG)
fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "litellm_requests.log"),
    maxBytes=int(os.environ.get("ZAYA_LOG_MAX_BYTES", "52428800")),
    backupCount=int(os.environ.get("ZAYA_LOG_BACKUPS", "5")),
)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(process)d %(name)-28s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(fh)

litellm_logger.info("=" * 80)
litellm_logger.info("LiteLLM ZAYA1-8B Proxy Started")

# Keep LiteLLM console logging at the configured level.
litellm_main_logger = logging.getLogger("litellm")
litellm_main_logger.setLevel(LOG_LEVEL_VALUE)

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "11111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(__file__).parent / "lite_llm_config.yaml"

logger.info("Starting LiteLLM proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)

os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)

logger.info("=" * 60)
logger.info("LiteLLM proxy starting in-process")
logger.info("=" * 60)

# Verify callback is registered before starting
from litellm.integrations.custom_logger import CustomLogger
registered_callbacks = [cb for cb in litellm.callbacks if isinstance(cb, CustomLogger)]
logger.info("Registered custom callbacks: %d", len(registered_callbacks))
for cb in registered_callbacks:
    logger.info("  - %s", type(cb).__name__)

def main():
    import uvicorn
    logger.info("Starting uvicorn for LiteLLM proxy")
    try:
        uvicorn.run(
            "litellm.proxy.proxy_server:app",
            host=LITELLM_HOST,
            port=int(LITELLM_PORT),
            reload=False,
            log_level=LOG_LEVEL.lower(),
        )
    except Exception:
        logger.exception("LiteLLM proxy exited with a fatal error")
        raise


if __name__ == "__main__":
    main()
