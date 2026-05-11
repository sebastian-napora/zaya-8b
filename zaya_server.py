#!/usr/bin/env python3
"""
Local vLLM API Server for Zyphra/ZAYA1-8B on NVIDIA GB10
with a plain OpenAI-compatible chat endpoint.

Architecture:
  Copilot -> LiteLLM (11111) -> vLLM (11112)

Model: Zyphra/ZAYA1-8B (local HF cache)
Context: 131,072 tokens max
Requires: vLLM 0.14.0+, transformers 5.0.0+
"""

import sys
import os
import logging
import traceback
import inspect
import json
import re
import uuid
from logging.handlers import RotatingFileHandler

from zaya_logging import configure_logging, log_torch_cuda

# Allow long max_model_len
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
# DGX Spark / GB10 Blackwell can accidentally pick a flash-attn wheel compiled
# for the wrong GPU architecture. Triton is slower but avoids that native crash.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", os.environ.get("ZAYA_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"))

# Setup logging
logger = configure_logging("zaya_vllm", "vllm_detailed.log")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
vllm_logger = logging.getLogger("zaya.vllm.requests")
vllm_logger.setLevel(logging.DEBUG)
if not any(getattr(h, "baseFilename", "") == os.path.join(LOG_DIR, "vllm_requests.log") for h in vllm_logger.handlers):
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "vllm_requests.log"),
        maxBytes=int(os.environ.get("ZAYA_LOG_MAX_BYTES", "52428800")),
        backupCount=int(os.environ.get("ZAYA_LOG_BACKUPS", "5")),
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(process)d %(name)-28s %(levelname)-8s %(message)s"))
    vllm_logger.addHandler(fh)
vllm_logger.info("=" * 80)
vllm_logger.info("vLLM ZAYA1-8B request logger started")

# Ensure venv packages take priority
_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if os.path.exists(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if os.path.exists(_venv_lib) and _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)

def _get_tool_parser_manager():
    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    return ToolParserManager


def _get_tool_parser_class(parser_name):
    try:
        manager = _get_tool_parser_manager()
        return manager.get_tool_parser(parser_name)
    except Exception as exc:
        logger.warning("vLLM tool parser %r is not available: %s", parser_name, exc)
        return None


ZYPHRA_TOOL_CALL_RE = re.compile(
    r"<zyphra_tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</zyphra_tool_call>",
    re.DOTALL,
)
ZYPHRA_TOOL_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)</parameter>",
    re.DOTALL,
)


def _parse_tool_scalar(value):
    text = value.strip()
    if not text:
        return ""

    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() == "null":
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", text):
            return float(text)
    except Exception:
        pass

    return text


def _extract_zyphra_tool_calls(text):
    if not isinstance(text, str) or "<zyphra_tool_call>" not in text:
        return [], text

    tool_calls = []
    for match in ZYPHRA_TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        arguments = {
            param.group(1).strip(): _parse_tool_scalar(param.group(2))
            for param in ZYPHRA_TOOL_PARAM_RE.finditer(body)
        }
        if name:
            tool_calls.append({"name": name, "arguments": arguments})

    if not tool_calls:
        return [], text

    return tool_calls, None


def _openai_tool_call_dict(tool_call):
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False),
        },
    }


def _make_vllm_tool_call(tool_call):
    payload = _openai_tool_call_dict(tool_call)
    try:
        from vllm.entrypoints.openai.protocol import FunctionCall, ToolCall

        function = FunctionCall(
            name=payload["function"]["name"],
            arguments=payload["function"]["arguments"],
        )
        try:
            return ToolCall(
                id=payload["id"],
                type="function",
                function=function,
            )
        except TypeError:
            return ToolCall(
                type="function",
                function=function,
            )
    except Exception:
        try:
            from vllm.entrypoints.openai.protocol import ChatCompletionMessageToolCall, FunctionCall

            return ChatCompletionMessageToolCall(
                id=payload["id"],
                type="function",
                function=FunctionCall(
                    name=payload["function"]["name"],
                    arguments=payload["function"]["arguments"],
                ),
            )
        except Exception:
            return payload


def _get_obj_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set_obj_field(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
        return True
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        try:
            object.__setattr__(obj, name, value)
            return True
        except Exception:
            logger.exception("Unable to set response field %s on %s", name, type(obj).__name__)
            return False


def _recover_leaked_tool_calls_in_response(response):
    choices = _get_obj_field(response, "choices", None)
    if not choices:
        return False

    recovered = 0
    for choice in choices:
        message = _get_obj_field(choice, "message", None)
        if message is None:
            continue
        content = _get_obj_field(message, "content", None)
        tool_calls, _ = _extract_zyphra_tool_calls(content)
        if not tool_calls:
            continue

        _set_obj_field(message, "content", None)
        _set_obj_field(message, "tool_calls", [_make_vllm_tool_call(call) for call in tool_calls])
        _set_obj_field(choice, "finish_reason", "tool_calls")
        recovered += len(tool_calls)

    if recovered:
        logger.warning("Recovered %d leaked ZAYA XML tool call(s) from final response", recovered)
        return True
    return False


def _patch_zyphra_tool_extraction(parser_cls):
    if getattr(parser_cls, "_zaya_xml_recovery_patch", False):
        return
    if not hasattr(parser_cls, "extract_tool_calls"):
        logger.warning("zaya_xml parser has no extract_tool_calls method; XML recovery patch skipped")
        return

    original_extract = parser_cls.extract_tool_calls

    def patched_extract_tool_calls(self, model_output, request):
        tool_calls, content = _extract_zyphra_tool_calls(model_output)
        if tool_calls:
            try:
                from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation

                logger.warning(
                    "Recovered %d leaked ZAYA XML tool call(s) from model output",
                    len(tool_calls),
                )
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[_make_vllm_tool_call(call) for call in tool_calls],
                    content=content,
                )
            except Exception:
                logger.exception("Failed to build vLLM tool-call response from leaked ZAYA XML")

        return original_extract(self, model_output, request)

    parser_cls.extract_tool_calls = patched_extract_tool_calls
    parser_cls._zaya_xml_recovery_patch = True
    logger.warning("Patched zaya_xml parser to recover leaked <zyphra_tool_call> blocks")


# ─── Main ─────────────────────────────────────────────────────────────────────
def _patch_zaya_xml_tool_parser():
    """Adapt vLLM's ZAYA tool parser to newer parser constructor calls."""
    try:
        parser_cls = _get_tool_parser_class("zaya_xml")
        if parser_cls is None:
            return False
        _patch_zyphra_tool_extraction(parser_cls)
        init_sig = inspect.signature(parser_cls.__init__)
        accepts_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in init_sig.parameters.values()
        )
        if accepts_varargs or len(init_sig.parameters) >= 3:
            logger.info("zaya_xml tool parser already accepts tokenizer and tools")
            return True

        original_init = parser_cls.__init__

        def patched_init(self, tokenizer, tools=None):
            original_init(self, tokenizer)
            if tools is None:
                return
            self.tools = tools
            parser = getattr(self, "parser", None)
            if hasattr(parser, "set_tools"):
                parser.set_tools(tools)
            if hasattr(self, "set_tools"):
                self.set_tools(tools)

        parser_cls.__init__ = patched_init
        parser_cls._zaya_constructor_patch = True
        logger.warning(
            "Patched zaya_xml tool parser constructor to accept tokenizer, tools. "
            "This keeps native <zyphra_tool_call> markup from leaking into content."
        )
        return True
    except Exception:
        logger.exception("Unable to patch zaya_xml tool parser; falling back to another parser may be required")
        return False


def _resolve_tool_call_parser(requested_parser, zaya_xml_parser_available):
    if not requested_parser:
        return ""

    fallback_parsers = ["zaya_xml", "qwen3_xml", "hermes"]
    candidates = [requested_parser]
    if requested_parser == "zaya_xml" and not zaya_xml_parser_available:
        candidates = []
    candidates.extend(fallback_parsers)

    seen = set()
    for parser_name in candidates:
        if not parser_name or parser_name in seen:
            continue
        seen.add(parser_name)
        if parser_name == "zaya_xml" and not zaya_xml_parser_available:
            continue
        if _get_tool_parser_class(parser_name) is not None:
            if parser_name != requested_parser:
                logger.warning(
                    "Using vLLM tool parser %r instead of requested parser %r",
                    parser_name,
                    requested_parser,
                )
            return parser_name

    raise RuntimeError(
        "Tool calling is enabled, but no supported vLLM tool parser was found. "
        f"Tried: {', '.join(seen) or requested_parser}. "
        "Set ZAYA_TOOL_CALL_PARSER to an installed parser, or set "
        "ZAYA_ENABLE_AUTO_TOOL_CHOICE=0 for plain chat."
    )


BEHAVIOR_GUARD_MARKER = "[ZAYA behavior guard]"
DEFAULT_BEHAVIOR_GUARD = f"""{BEHAVIOR_GUARD_MARKER}
You are a focused coding assistant operating inside one local repository.
Interpret the user's latest request narrowly and stay within the requested scope.
Do not turn a small request into a broad repository improvement plan unless the
user explicitly asks for broad planning, review, or architecture work.
For coding tasks, inspect the relevant local files, make the smallest coherent
change, and verify it. If the request is ambiguous, choose the conservative
local fix or ask one short clarifying question.
Use private reasoning to choose the right next action, but keep that reasoning
scoped to the latest request and the files directly involved.
Use tools when they help, but do not narrate a long tool-use plan before acting.
When using a tool, emit the tool call only; do not include explanatory prose
before or after the tool call.
In user-facing replies, provide concise rationale, concrete changes, and test
results. Do not expose private chain-of-thought or lengthy hidden reasoning."""


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_behavior_guard_text():
    if not _env_flag("ZAYA_BEHAVIOR_GUARD", "1"):
        return ""

    guard_file = os.environ.get("ZAYA_BEHAVIOR_GUARD_FILE", "").strip()
    if guard_file:
        try:
            with open(guard_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            return text or DEFAULT_BEHAVIOR_GUARD
        except Exception:
            logger.exception("Failed to read ZAYA_BEHAVIOR_GUARD_FILE=%s; using default guard", guard_file)

    return os.environ.get("ZAYA_BEHAVIOR_GUARD_TEXT", DEFAULT_BEHAVIOR_GUARD).strip()


def _message_role(msg):
    return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)


def _message_content(msg):
    return msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)


def _set_message_content(msg, content):
    if isinstance(msg, dict):
        msg["content"] = content
        return True
    try:
        setattr(msg, "content", content)
        return True
    except Exception:
        return False


def _content_contains_guard(content):
    if isinstance(content, str):
        return BEHAVIOR_GUARD_MARKER in content
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and BEHAVIOR_GUARD_MARKER in str(part.get("text", ""))
            for part in content
        )
    return False


def _merge_guard_content(guard_text, content):
    if content is None:
        return guard_text
    if isinstance(content, str):
        return f"{guard_text}\n\n{content}"
    if isinstance(content, list):
        return [{"type": "text", "text": guard_text}, *content]
    return f"{guard_text}\n\n{content}"


def _apply_behavior_guard(request):
    guard_text = _get_behavior_guard_text()
    if not guard_text:
        return False

    messages = getattr(request, "messages", None)
    if not messages:
        return False

    for msg in messages:
        if _message_role(msg) != "system":
            continue
        content = _message_content(msg)
        if _content_contains_guard(content):
            return False
        if _set_message_content(msg, _merge_guard_content(guard_text, content)):
            return True
        break

    messages.insert(0, {"role": "system", "content": guard_text})
    return True


async def main():
    logger.info("Importing vLLM OpenAI server entrypoints")
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client,
        build_app,
        init_app_state,
        setup_server,
        serve_http,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    logger.info("vLLM imports completed")
    log_torch_cuda(logger)
    zaya_xml_parser_available = _patch_zaya_xml_tool_parser()

    # ── Build vLLM args ─────────────────────────────────────────────────────────
    logger.info("Building vLLM serve argument parser")
    parser = FlexibleArgumentParser(prog="zaya1-8b-server")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve")
    serve_parser = make_arg_parser(serve_parser)
    serve_option_strings = {
        option
        for action in serve_parser._actions
        for option in getattr(action, "option_strings", [])
    }

    # ── Env-configurable knobs ──────────────────────────────────────────────
    MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "131072")
    GPU_MEM_UTIL  = os.environ.get("VLLM_GPU_MEM_UTIL",  "0.30")
    OPT_LEVEL     = os.environ.get("VLLM_OPT_LEVEL",      "1")
    MODEL_NAME    = os.environ.get("ZAYA_MODEL",          "Zyphra/ZAYA1-8B")
    ATTENTION_BACKEND = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
    ENABLE_AUTO_TOOL_CHOICE = os.environ.get("ZAYA_ENABLE_AUTO_TOOL_CHOICE", "1") == "1"
    TOOL_CALL_PARSER = os.environ.get("ZAYA_TOOL_CALL_PARSER", "zaya_xml").strip()
    REASONING_PARSER = os.environ.get("ZAYA_REASONING_PARSER", "qwen3").strip()
    ENABLE_REASONING = os.environ.get("ZAYA_ENABLE_REASONING", "1") == "1"
    CHAT_TEMPLATE = os.environ.get("ZAYA_CHAT_TEMPLATE", "").strip()
    BEHAVIOR_GUARD_ENABLED = _env_flag("ZAYA_BEHAVIOR_GUARD", "1")

    if ENABLE_AUTO_TOOL_CHOICE:
        TOOL_CALL_PARSER = _resolve_tool_call_parser(TOOL_CALL_PARSER, zaya_xml_parser_available)

    logger.info(
        (
            "Resolved backend config model=%s max_model_len=%s gpu_memory_utilization=%s "
            "optimization_level=%s attention_backend=%s auto_tools=%s tool_parser=%s "
            "reasoning=%s reasoning_parser=%s chat_template=%s"
        ),
        MODEL_NAME,
        MAX_MODEL_LEN,
        GPU_MEM_UTIL,
        OPT_LEVEL,
        ATTENTION_BACKEND,
        ENABLE_AUTO_TOOL_CHOICE,
        TOOL_CALL_PARSER or "disabled",
        ENABLE_REASONING,
        REASONING_PARSER or "disabled",
        CHAT_TEMPLATE or "model default",
    )
    logger.info(
        "Behavior guard %s; override with ZAYA_BEHAVIOR_GUARD_TEXT or ZAYA_BEHAVIOR_GUARD_FILE",
        "enabled" if BEHAVIOR_GUARD_ENABLED else "disabled",
    )

    argv = [
        MODEL_NAME,
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--mamba-cache-dtype", "float32",
        "--load-format", "safetensors",
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-num-batched-tokens", "65536",
        "--optimization-level", OPT_LEVEL,
        "--port", "11112",
        "--host", "0.0.0.0",
    ]

    if ENABLE_AUTO_TOOL_CHOICE:
        argv += ["--enable-auto-tool-choice"]
        if TOOL_CALL_PARSER:
            argv += ["--tool-call-parser", TOOL_CALL_PARSER]
        else:
            logger.warning("Auto tool choice enabled but ZAYA_TOOL_CALL_PARSER is empty")
    else:
        logger.info("Auto tool choice disabled by ZAYA_ENABLE_AUTO_TOOL_CHOICE=0")

    if ENABLE_REASONING:
        if "--enable-reasoning" in serve_option_strings:
            argv += ["--enable-reasoning"]
        if REASONING_PARSER:
            argv += ["--reasoning-parser", REASONING_PARSER]
        else:
            logger.warning("Reasoning enabled but ZAYA_REASONING_PARSER is empty")
    else:
        logger.info("Reasoning parser disabled by ZAYA_ENABLE_REASONING=0")

    if CHAT_TEMPLATE and "--chat-template" in serve_option_strings:
        argv += ["--chat-template", CHAT_TEMPLATE]
    elif CHAT_TEMPLATE:
        logger.warning("Ignoring ZAYA_CHAT_TEMPLATE=%s because this vLLM parser has no --chat-template flag", CHAT_TEMPLATE)

    if "--attention-backend" in serve_option_strings:
        argv += ["--attention-backend", ATTENTION_BACKEND]
        logger.info("Passing vLLM CLI attention backend: %s", ATTENTION_BACKEND)
    else:
        logger.info("vLLM parser has no --attention-backend flag; using VLLM_ATTENTION_BACKEND env only")

    # ── Speculative decoding (N-gram / Prompt Lookup) ───────────────────────
    # Drafts tokens by matching n-grams from the input prompt.  No separate
    # model required — zero extra memory.
    import json as _json
    SPEC_NGRAM     = os.environ.get("VLLM_SPEC_NGRAM",     "0") == "1"
    SPEC_NGRAM_K   = os.environ.get("VLLM_SPEC_NGRAM_K",   "5")
    SPEC_NGRAM_MIN = os.environ.get("VLLM_SPEC_NGRAM_MIN", "3")
    SPEC_NGRAM_MAX = os.environ.get("VLLM_SPEC_NGRAM_MAX", "5")

    if SPEC_NGRAM:
        spec_cfg = _json.dumps({
            "method": "ngram",
            "num_speculative_tokens": int(SPEC_NGRAM_K),
            "prompt_lookup_min": int(SPEC_NGRAM_MIN),
            "prompt_lookup_max": int(SPEC_NGRAM_MAX),
        })
        argv += ["--speculative-config", spec_cfg]
        print(f"🔍 N-gram speculative decoding ENABLED  (k={SPEC_NGRAM_K})")
        logger.info("N-gram speculative decoding enabled config=%s", spec_cfg)
    else:
        print("ℹ️  Speculative decoding disabled.")
        logger.info("Speculative decoding disabled")

    logger.debug("vLLM serve argv=%s", argv)
    args = serve_parser.parse_args(argv)
    args.command = "serve"
    args.model_tag = argv[0]
    args.model = args.model_tag
    validate_parsed_serve_args(args)
    logger.info("vLLM serve args parsed and validated")

    # ── Start engine (blocking until model is loaded) ──────────────────────────
    print("⏳ Loading Zyphra/ZAYA1-8B… this may take a few minutes.")
    logger.info("Starting vLLM engine client; model weights load begins now")
    try:
        async with build_async_engine_client(args) as engine_client:
            logger.info("vLLM engine client started; model weights loaded")
            supported_tasks = await engine_client.get_supported_tasks()
            model_config = engine_client.model_config
            logger.info("Supported tasks: %s", supported_tasks)
            logger.info("Model config: %r", model_config)

            # ── Build FastAPI app ───────────────────────────────────────────────────
            logger.info("Building FastAPI app and initializing app state")
            app = build_app(args, supported_tasks, model_config)
            await init_app_state(engine_client, app.state, args, supported_tasks)
            logger.info("FastAPI app state initialized")

            from fastapi import Request
            from vllm.entrypoints.openai.chat_completion.serving import (
                OpenAIServingChat,
            )
            from vllm.entrypoints.openai.chat_completion.protocol import (
                ChatCompletionRequest,
            )

            serving_chat: OpenAIServingChat = app.state.openai_serving_chat

            # Wrap the original method to log and normalize all requests.
            original_create = serving_chat.create_chat_completion

            async def logged_create_chat_completion(request: ChatCompletionRequest, raw_request: Request = None, **kwargs):
                guard_applied = _apply_behavior_guard(request)
                vllm_logger.info("=" * 80)
                vllm_logger.info("/v1/chat/completions request received")
                vllm_logger.info("Model: %s", request.model)
                vllm_logger.info("Stream: %s", request.stream)
                vllm_logger.info("Behavior guard: %s", "applied" if guard_applied else ("enabled" if BEHAVIOR_GUARD_ENABLED else "disabled"))
                vllm_logger.info("Message count: %d", len(request.messages))

                for i, msg in enumerate(request.messages):
                    if isinstance(msg, dict):
                        msg_role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                    else:
                        msg_role = getattr(msg, "role", "unknown")
                        content = getattr(msg, "content", "")

                    if isinstance(content, list):
                        image_types = [c for c in content if c.get("type") == "image_url"]
                        text_parts  = [c for c in content if c.get("type") == "text"]
                        vllm_logger.info(
                            "  msg[%d] role=%s: %d image_url items, %d text items",
                            i, msg_role, len(image_types), len(text_parts)
                        )
                        for j, part in enumerate(content):
                            if part.get("type") == "image_url":
                                img_url = part.get("image_url", {})
                                if isinstance(img_url, dict):
                                    url = img_url.get("url", "")[:100]
                                    detail = img_url.get("detail", "not_set")
                                else:
                                    url = str(img_url)[:100]
                                    detail = "not_set"
                                vllm_logger.info(
                                    "    image_url[%d]: url_len=%d, detail=%s, url=%s...",
                                    j, len(img_url.get("url", "")) if isinstance(img_url, dict) else len(str(img_url)), detail, url
                                )
                            elif part.get("type") == "text":
                                vllm_logger.info("    text[%d]: %s", j, part.get("text", "")[:300])
                    elif isinstance(content, str):
                        vllm_logger.info("  msg[%d] role=%s: %s", i, msg_role, content[:300])

                if hasattr(request, "extra_body") and request.extra_body:
                    vllm_logger.info("Extra body: %s", request.extra_body)

                vllm_logger.info("=" * 80)

                try:
                    result = await original_create(request, raw_request, **kwargs)
                    recovered = _recover_leaked_tool_calls_in_response(result)
                    if request.stream and not recovered:
                        vllm_logger.debug(
                            "Streaming response returned; leaked XML tool-call recovery is handled by parser hook only"
                        )
                    vllm_logger.info("Request completed successfully")
                    return result
                except Exception as e:
                    vllm_logger.error("Request failed: %s", str(e))
                    vllm_logger.error(traceback.format_exc())
                    raise

            serving_chat.create_chat_completion = logged_create_chat_completion

            # ─── Health check ──────────────────────────────────────────────────────
            @app.get("/health")
            async def health():
                return {"status": "ok", "model": "Zyphra/ZAYA1-8B"}

            # ── Serve ───────────────────────────────────────────────────────────────
            listen_address, sock = setup_server(args)
            logger.info("Listening on %s", listen_address)
            print(f"\n🚀 ZAYA1-8B Server @ {MAX_MODEL_LEN} Context")
            print(f"📡 Chat API:    http://0.0.0.0:11112/v1/chat/completions")
            print(f"❤️  Health:     http://0.0.0.0:11112/health")
            print()
            await serve_http(
                app,
                sock=sock,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=30,
            )
    except Exception:
        logger.exception("vLLM backend failed during import, model load, app startup, or serving")
        raise


def _run_main():
    import uvloop
    try:
        uvloop.run(main())
    except Exception:
        logger.exception("zaya_server.py exited with a fatal error")
        raise


if __name__ == "__main__":
    _run_main()


def run():
    """Synchronous entry point for console_scripts."""
    _run_main()
