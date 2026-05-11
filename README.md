# Zyphra/ZAYA1-8B Serving Setup

Local vLLM + LiteLLM proxy stack for `Zyphra/ZAYA1-8B`.

## Architecture

```
GitHub Copilot (VS Code) -> LiteLLM (11111) -> vLLM (11112)
                                            (or SGLang 11112)
```

- **vLLM/SGLang backend** (port 11112): loads model and handles inference
- **LiteLLM proxy** (port 11111): OpenAI-compatible HTTP API, handles auth/routing/callbacks
- **Token stats server** (port 11113): per-request token usage tracking

## Files

| File | Purpose |
|------|---------|
| `zaya_server.py` | vLLM backend server |
| `zaya_sglang_server.py` | SGLang backend server (EAGLE speculative decoding) |
| `server_compress.py` | LiteLLM proxy entrypoint |
| `zaya_token_tracker.py` | Per-request token usage tracker |
| `lite_llm_config.yaml` | LiteLLM model routing config |
| `start.sh` | Start vLLM + LiteLLM stack |
| `start_sglang.sh` | Start SGLang + LiteLLM stack |
| `kill.sh` | Kill all serving processes |

## Quick Start

### vLLM backend
```bash
./start.sh both
```
Service logs are written under `logs/`, especially `logs/vllm_backend.log`
for model-load failures. By default, `./start.sh both` also streams service
logs into the same terminal.

The vLLM backend can take several minutes to load weights and finish warmup.
`start.sh both` waits for `http://localhost:11112/health` before starting
LiteLLM, so the proxy does not accept requests while the backend is still
loading. To wait longer than the default 900 seconds:

```bash
ZAYA_BACKEND_WAIT_TIMEOUT=1800 ./start.sh both
```

To save logs without streaming them, run:

```bash
ZAYA_STREAM_LOGS=0 ./start.sh both
```

Useful log files:

| File | Contents |
|---|---|
| `logs/vllm_backend.log` | stdout/stderr from the vLLM backend process |
| `logs/vllm_detailed.log` | structured Python/vLLM/transformers/HF/uvicorn logs |
| `logs/vllm_requests.log` | backend chat request summaries and request errors |
| `logs/lite_llm.log` | stdout/stderr from the LiteLLM proxy process |
| `logs/litellm_detailed.log` | structured LiteLLM/proxy/uvicorn logs |
| `logs/litellm_requests.log` | proxy request logger |
| `logs/token_stats.json` | current token usage totals |
| `logs/token_requests.jsonl` | per-request token usage and failures |
| `logs/*_faulthandler.log` | Python crash dumps for hard failures |

Set `ZAYA_LOG_LEVEL=INFO` for quieter logs, or leave the default `DEBUG`
while diagnosing startup and model-load issues.

If startup prints `FATAL: FlashAttention requires building with sm version...`,
the installed `flash-attn` wheel was built for the wrong GPU architecture.
The vLLM backend defaults to `ZAYA_VLLM_ATTENTION_BACKEND=TRITON_ATTN` and
`ZAYA_BLOCK_FLASH_ATTN=1` on DGX Spark / GB10 to avoid that package. The
launcher exports this repo on `PYTHONPATH` so `sitecustomize.py` hides
`flash_attn` from worker processes too. It also defaults
`TORCH_CUDA_ARCH_LIST=12.0f` and uses `/usr/local/cuda/bin/ptxas` for Triton
when available.

Inspect the CUDA/vLLM package stack with:

```bash
./start.sh diagnose
```

Try another backend without editing code:

```bash
ZAYA_VLLM_ATTENTION_BACKEND=FLASHINFER ZAYA_BLOCK_FLASH_ATTN=0 ./start.sh both
```

### Reasoning and tool calling

This setup serves ZAYA with vLLM reasoning output parsing enabled by default:
`ZAYA_ENABLE_REASONING=1` and `ZAYA_REASONING_PARSER=qwen3`. That means
responses can expose a `reasoning_content` field when the model/template emits
thinking-style output.

Automatic tool choice is also enabled by default for Copilot and other
OpenAI-compatible clients:

| Env var | Default | Notes |
|---|---|---|
| `ZAYA_ENABLE_AUTO_TOOL_CHOICE` | `1` | Enables `tool_choice: auto` support |
| `ZAYA_TOOL_CALL_PARSER` | `qwen3_xml` | Avoids the broken `zaya_xml` parser on current vLLM builds |
| `ZAYA_ENABLE_REASONING` | `1` | Enables reasoning parser flags |
| `ZAYA_REASONING_PARSER` | `qwen3` | Parser used for `reasoning_content` |
| `ZAYA_CHAT_TEMPLATE` | unset | Optional custom/tool-use chat template |

If Copilot returns a parser error, try a different parser without editing code:

```bash
ZAYA_TOOL_CALL_PARSER=hermes ./start.sh both
```

For plain chat testing with no auto tool parser:

```bash
ZAYA_ENABLE_AUTO_TOOL_CHOICE=0 ./start.sh both
```

### SGLang backend (EAGLE speculative decoding)
```bash
./start_sglang.sh both
```

Typical startup time: **2-3 min**.
Peak RAM during load: **~50-60 GB** (BF16, 24 GB model on GB10).

### Memory / context tuning (env vars)

| Env var | Default | Notes |
|---|---|---|
| `VLLM_MAX_MODEL_LEN` | `131072` | Max context tokens |
| `VLLM_GPU_MEM_UTIL` | `0.50` | Fraction of 128 GB reserved for vLLM pool |
| `VLLM_OPT_LEVEL` | `1` | `1`=CUDA graphs only (recommended). `0`=eager |

### SGLang tuning (env vars)

| Env var | Default | Notes |
|---|---|---|
| `SGLANG_MAX_MODEL_LEN` | `131072` | Context window |
| `SGLANG_MEM_FRACTION` | `0.45` | Static memory fraction |
| `SGLANG_SPEC_NUM_STEPS` | `3` | EAGLE steps |
| `SGLANG_SPEC_DRAFT_TOKENS` | `4` | Draft tokens per cycle |
| `SGLANG_DISABLE_CUDA_GRAPH` | `1` | Disable CUDA graphs on Blackwell |

## Manual start (two terminals)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python zaya_server.py        # terminal 1
python server_compress.py   # terminal 2
```

The detached helpers can also be run directly:
```bash
./start_vllm.py
./start_litellm.py
```

## API Endpoints

| Endpoint | Port | Purpose |
|----------|------|---------|
| `POST /v1/chat/completions` | 11111/11112 | Chat completion |
| `GET /health` | 11111/11112 | Health check |
| `GET /stats` | 11113 | Token usage stats |

## Copilot Integration

In VS Code with GitHub Copilot, configure:
```
http://localhost:11111/v1/chat/completions
```

## Model Details

- **Model**: `Zyphra/ZAYA1-8B`
- **Type**: 16-expert MoE (80 layers, hidden_size=2048)
- **Quantization**: BF16 (no quantization in this setup)
- **Context**: 131,072 tokens max
- **Size**: ~24 GB BF16
- **KV heads**: 2 (GQA)
- **Activation**: SwiGLU with EDA/MOD extensions
