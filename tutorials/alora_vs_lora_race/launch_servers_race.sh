#!/usr/bin/env bash
# Launch two vLLM servers side-by-side for the ALORA vs LoRA race.
#
# ALORA server: pre-built Granite Switch checkpoint (port 8111)
# LoRA server:  compose your own with --technology-filter lora (port 8112)
#
# To build the LoRA-only model:
#   python -m granite_switch.composer.compose_granite_switch \
#     --base-model ibm-granite/granite-4.1-3b \
#     --adapters ibm-granite/granitelib-rag-r1.0 \
#                ibm-granite/granitelib-core-r1.0 \
#                ibm-granite/granitelib-guardian-r1.0 \
#     --technology-filter lora \
#     --output ./granite-switch-lora-only

LORA_MODEL="${1:-./granite-switch-lora-only}"

CUDA_VISIBLE_DEVICES=0 vllm serve ibm-granite/granite-switch-4.1-3b-preview --port 8111 &> vllm_alora.log &
CUDA_VISIBLE_DEVICES=1 vllm serve "$LORA_MODEL" --port 8112 &> vllm_lora.log &
