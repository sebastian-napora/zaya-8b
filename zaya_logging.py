#!/usr/bin/env python3
"""Shared logging setup for the ZAYA serving scripts."""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import socket
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FORMAT = "%(asctime)s %(process)d %(name)-28s %(levelname)-8s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_FAULT_FILES = []


def _level_from_env(default: str = "DEBUG") -> int:
    level_name = os.environ.get("ZAYA_LOG_LEVEL", default).upper()
    return getattr(logging, level_name, logging.DEBUG)


def _make_file_handler(path: Path, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=int(os.environ.get("ZAYA_LOG_MAX_BYTES", "52428800")),
        backupCount=int(os.environ.get("ZAYA_LOG_BACKUPS", "5")),
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    return handler


def configure_logging(service_name: str, log_file: str, *, default_level: str = "DEBUG") -> logging.Logger:
    """Configure root logging and useful third-party loggers.

    The launcher scripts already redirect stdout/stderr into service logs. This
    function makes Python logging, warnings, and uncaught exceptions land there
    too, while also keeping structured per-service files.
    """

    LOG_DIR.mkdir(exist_ok=True)
    level = _level_from_env(default_level)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    os.environ.setdefault("HF_HUB_VERBOSITY", "info")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(_make_file_handler(LOG_DIR / log_file, level))

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    fault_path = LOG_DIR / f"{service_name}_faulthandler.log"
    fault_file = open(fault_path, "a", buffering=1)
    _FAULT_FILES.append(fault_file)
    faulthandler.enable(file=fault_file, all_threads=True)

    logger = logging.getLogger(service_name)
    logger.info("=" * 80)
    logger.info("%s logging initialized", service_name)
    logger.info("log_dir=%s main_log=%s fault_log=%s", LOG_DIR, LOG_DIR / log_file, fault_path)
    logger.info("python=%s executable=%s", sys.version.replace("\n", " "), sys.executable)
    logger.info("platform=%s host=%s pid=%s cwd=%s", platform.platform(), socket.gethostname(), os.getpid(), os.getcwd())
    logger.info(
        "env ZAYA_LOG_LEVEL=%s TRANSFORMERS_VERBOSITY=%s HF_HUB_VERBOSITY=%s NCCL_DEBUG=%s",
        logging.getLevelName(level),
        os.environ.get("TRANSFORMERS_VERBOSITY"),
        os.environ.get("HF_HUB_VERBOSITY"),
        os.environ.get("NCCL_DEBUG"),
    )

    for logger_name in (
        "vllm",
        "vllm.engine",
        "vllm.model_executor",
        "transformers",
        "huggingface_hub",
        "torch",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "litellm",
        "LiteLLM",
        "LiteLLM Proxy",
    ):
        logging.getLogger(logger_name).setLevel(level)

    old_excepthook = sys.excepthook

    def log_uncaught(exc_type, exc, tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        old_excepthook(exc_type, exc, tb)

    sys.excepthook = log_uncaught
    return logger


def log_torch_cuda(logger: logging.Logger) -> None:
    """Log CUDA visibility if torch is importable."""

    try:
        import torch
    except Exception:
        logger.exception("Could not import torch while collecting CUDA diagnostics")
        return

    try:
        logger.info("torch=%s cuda_available=%s cuda_version=%s", torch.__version__, torch.cuda.is_available(), torch.version.cuda)
        if torch.cuda.is_available():
            logger.info("cuda_device_count=%s current_device=%s", torch.cuda.device_count(), torch.cuda.current_device())
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_gib = props.total_memory / (1024**3)
                logger.info(
                    "cuda_device[%d] name=%s capability=%s.%s total_memory=%.2f GiB",
                    index,
                    props.name,
                    props.major,
                    props.minor,
                    total_gib,
                )
    except Exception:
        logger.exception("Could not collect CUDA diagnostics")
