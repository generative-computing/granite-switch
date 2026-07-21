# SPDX-License-Identifier: Apache-2.0
#
# Serve Granite Switch with vLLM's OpenAI-compatible API server.
#
# Built on the official vLLM image, which already ships vLLM 0.19.1, PyTorch,
# and CUDA 12.x — matching this repo's default (vLLM 0.20+ needs CUDA 13+).
# We only add the granite_switch package on top; vLLM auto-discovers the model
# via the `vllm.general_plugins` entry point (see pyproject.toml), so no manual
# registration is needed at runtime.
#
# Build:
#   docker build -t granite-switch-vllm .
#
# Run (needs an NVIDIA GPU + the NVIDIA container toolkit):
#   docker run --gpus all -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     granite-switch-vllm
#
# Serve a different model / checkpoint:
#   docker run --gpus all -p 8000:8000 \
#     -e MODEL=ibm-granite/granite-switch-4.1-8b-preview \
#     granite-switch-vllm
#
# Pass extra vLLM args (appended after the entrypoint):
#   docker run --gpus all -p 8000:8000 granite-switch-vllm \
#     --tensor-parallel-size 2 --max-model-len 8192
#
# Gated/private models: pass a token with `-e HF_TOKEN=...`.

# Pin to CUDA 12.x / vLLM 0.19.1. The plain tag is the x86_64 CUDA 12 build;
# the `-cu130` variants target CUDA 13. Bump in lockstep with pyproject's
# vllm pin.
FROM vllm/vllm-openai:v0.19.1

# Serve the 3b preview by default; override with `-e MODEL=...`.
ENV MODEL=ibm-granite/granite-switch-4.1-3b-preview \
    HOST=0.0.0.0 \
    PORT=8000 \
    # Faster HF downloads on first run (base image includes hf_transfer).
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# Install granite_switch itself. vLLM/torch/CUDA already live in the base
# image, so we deliberately install WITHOUT the [vllm] extra to avoid pulling
# a second, possibly conflicting vLLM wheel. Copy only what's needed to build
# the wheel so edits to tests/docs don't bust this layer's cache.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps . \
    && python -c "import granite_switch.vllm  # sanity: package imports"

EXPOSE 8000

# The base image's entrypoint is the vLLM API server. We can't reference $MODEL
# from an exec-form ENTRYPOINT, so wrap the launch in a tiny shell that expands
# the env vars and forwards any extra CLI args ("$@") to vLLM.
ENTRYPOINT ["/bin/bash", "-c", \
    "exec python3 -m vllm.entrypoints.openai.api_server --model \"$MODEL\" --host \"$HOST\" --port \"$PORT\" \"$@\"", \
    "--"]
