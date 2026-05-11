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
The vLLM backend defaults to `ZAYA_VLLM_ATTENTION_BACKEND=FLASHINFER` to avoid
that package. Try another backend without editing code:

```bash
ZAYA_VLLM_ATTENTION_BACKEND=TRITON_ATTN ./start.sh both
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
