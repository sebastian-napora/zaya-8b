#!/usr/bin/env python3
"""
SGLang API Server for Zyphra/ZAYA1-8B — EAGLE Speculative Decoding.

Uses SGLang's built-in EAGLE speculative algorithm which predicts multiple
draft tokens using the model's own hidden states.

Architecture:
  Copilot -> LiteLLM (11111) -> SGLang (11112)

Model: Zyphra/ZAYA1-8B (local HF cache, ~24 GB BF16)
       16-expert MoE, 80 layers, hidden_size=2048
       131,072 tokens max context

EAGLE speculative decoding config:
  3 EAGLE steps × 4 draft tokens per cycle.

Known limitations on Blackwell (GB10, CC 12.1) with SGLang v0.5.11
────────────────────────────────────────────────────────────────────
  1. CUDA graphs may need to be disabled if issues arise.
  2. Triton attention backend recommended on Blackwell.

Requirements
────────────
  SGLang:
    pip install "sglang[all]==0.5.11" \\
        --extra-index-url https://flashinfer.ai/whl/cu130/torch2.9/
    pip install git+https://github.com/huggingface/transformers.git

  Model is already in HuggingFace cache:
    Zyphra/ZAYA1-8B  (~24 GB BF16, in ~/.cache/huggingface/hub/)

Env knobs
─────────
  SGLANG_MODEL               Model path (default Zyphra/ZAYA1-8B)
  SGLANG_PORT                Server port (default 11112)
  SGLANG_HOST                Bind address (default 0.0.0.0)
  SGLANG_MEM_FRACTION        Static memory fraction (default 0.45)
  SGLANG_SPEC_NUM_STEPS      EAGLE steps (default 3)
  SGLANG_SPEC_EAGLE_TOPK     Top-k per step (default 1)
  SGLANG_SPEC_DRAFT_TOKENS   Draft tokens per cycle (default 4)
  SGLANG_TP_SIZE             Tensor parallel size (default 1 for single GB10)
  SGLANG_MAX_MODEL_LEN       Context window (default 131072)
  SGLANG_DISABLE_CUDA_GRAPH  Disable CUDA graphs (default 1 on Blackwell)
  SGLANG_OVERRIDE_ARGS       Extra JSON passed to --json-model-override-args

Logs:  logs/sglang_server.log
Ports: 11111 LiteLLM proxy · 11112 SGLang backend
"""

import sys
import os
import json
import glob
import logging

_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if os.path.exists(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if os.path.exists(_venv_lib) and _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)


# ── Patch: register ZAYA model type before SGLang loads ──────────────────────
def _register_zaya_model_type():
    """Register ZayaConfig for model_type='zaya' so SGLang can load ZAYA1-8B."""
    try:
        from transformers import AutoConfig, PretrainedConfig
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

        # Load raw config to get fields
        cfg_files = glob.glob(
            os.path.expanduser("~/.cache/huggingface/hub/models--Zyphra--ZAYA1-8B/snapshots/*/config.json")
        )
        if not cfg_files:
            logging.warning("ZAYA1-8B config.json not found — skipping ZayaConfig patch")
            return

        with open(cfg_files[0]) as f:
            raw = json.load(f)

        # Build a ZayaConfig with the right model_type
        # Inherit from PretrainedConfig so all HF/SGLang internals work
        class ZayaConfig(PretrainedConfig):
            model_type = "zaya"
            is_composition = False

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Map ZAYA-specific fields into standard PretrainedConfig fields
                for k, v in raw.items():
                    if not hasattr(self, k) and k not in (
                        "model_type",
                        "architectures",
                        "transformers_version",
                    ):
                        setattr(self, k, v)

        # Register with both AutoConfig and SGLang's registry
        AutoConfig.register("zaya", ZayaConfig)
        _CONFIG_REGISTRY["zaya"] = ZayaConfig

        logging.info("Registered ZayaConfig for model_type='zaya'")

    except Exception as e:
        logging.warning(f"ZayaConfig patch failed (non-fatal): {e}")
        import traceback
        logging.warning(traceback.format_exc())


_register_zaya_model_type()

# ── Check SGLang is installed ────────────────────────────────────────────────
try:
    import sglang  # noqa: F401
except ImportError:
    print("❌  SGLang is not installed.")
    print()
    print("   Install SGLang v0.5.11:")
    print()
    print('     pip install "sglang[all]==0.5.11" \\')
    print("         --extra-index-url https://flashinfer.ai/whl/cu130/torch2.9/")
    print("     pip install git+https://github.com/huggingface/transformers.git")
    print()
    print("   Then re-run:  ./start_sglang.sh")
    sys.exit(1)


def main():
    MODEL              = os.environ.get("SGLANG_MODEL",             "Zyphra/ZAYA1-8B")
    PORT               = os.environ.get("SGLANG_PORT",              "11112")
    HOST               = os.environ.get("SGLANG_HOST",              "0.0.0.0")
    MEM_FRACTION       = os.environ.get("SGLANG_MEM_FRACTION",      "0.45")
    SPEC_NUM_STEPS     = os.environ.get("SGLANG_SPEC_NUM_STEPS",    "3")
    SPEC_EAGLE_TOPK    = os.environ.get("SGLANG_SPEC_EAGLE_TOPK",   "1")
    SPEC_DRAFT_TOKENS  = os.environ.get("SGLANG_SPEC_DRAFT_TOKENS", "4")
    TP_SIZE            = os.environ.get("SGLANG_TP_SIZE",           "1")
    MAX_MODEL_LEN      = os.environ.get("SGLANG_MAX_MODEL_LEN",    "131072")
    DISABLE_CUDA_GRAPH = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "1")
    SPEC_ADAPTIVE      = os.environ.get("SGLANG_SPEC_ADAPTIVE",    "0")
    OVERRIDE_ARGS      = os.environ.get("SGLANG_OVERRIDE_ARGS",     "")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",                   MODEL,
        "--tp-size",                      TP_SIZE,
        "--speculative-algorithm",        "EAGLE",
        "--speculative-num-steps",        SPEC_NUM_STEPS,
        "--speculative-eagle-topk",       SPEC_EAGLE_TOPK,
        "--speculative-num-draft-tokens", SPEC_DRAFT_TOKENS,
        "--mem-fraction-static",         MEM_FRACTION,
        "--context-length",              MAX_MODEL_LEN,
        "--served-model-name",            "zaya1-8b",
        "--host",                         HOST,
        "--port",                         PORT,
        "--attention-backend",           "triton",
        "--speculative-draft-attention-backend", "triton",
        "--disable-piecewise-cuda-graph",
    ]

    if DISABLE_CUDA_GRAPH == "1":
        cmd.append("--disable-cuda-graph")

    if SPEC_ADAPTIVE == "1":
        cmd.append("--speculative-adaptive")

    if OVERRIDE_ARGS:
        cmd += ["--json-model-override-args", OVERRIDE_ARGS]

    print(f"🚀 SGLang EAGLE speculative decoding")
    print(f"   Model:  {MODEL}  (BF16, ~24 GB)")
    print(f"   EAGLE:  steps={SPEC_NUM_STEPS}  topk={SPEC_EAGLE_TOPK}  draft_tokens={SPEC_DRAFT_TOKENS}  adaptive={'yes' if SPEC_ADAPTIVE=='1' else 'no'}")
    print(f"   TP:     {TP_SIZE}   ctx={MAX_MODEL_LEN}   mem={MEM_FRACTION}")
    if OVERRIDE_ARGS:
        print(f"   Override: {OVERRIDE_ARGS}")
    print(f"   URL:    http://{HOST}:{PORT}")
    print()
    print("⏳ Loading model — ~24 GB BF16, allow 1–2 min on GB10…")
    print()

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
