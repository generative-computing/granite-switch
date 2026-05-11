#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hello Adapter — Granite Switch demo with HuggingFace (guardian-core adapter).

Demonstrates how to activate adapters in a Granite Switch model using
the chat template's ``adapter_name`` parameter. It shows:

1. Loading a composed model
2. Activating an adapter via ``apply_chat_template(adapter_name=...)``
3. Using the correct Guardian prompt protocol

This example uses the **guardian-core** adapter, which evaluates text
against a safety criterion and returns a structured Yes/No score.

Requirements:
    pip install "granite-switch[hf,compose]"

    # CUDA GPU required

Usage:
    # Option 1: Use a pre-composed model from HuggingFace
    MODEL_PATH=ibm-granite/granite-switch-4.1-3b-preview python hello_adapter.py

    # Option 2: Compose your own (takes a few minutes)
    python hello_adapter.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# ── Guardian prompt protocol ──────────────────────────────────────────────

GUARDIAN_JUDGE_THINK = (
    "<think>As a judge agent, carefully analyze whether the provided text meets the "
    "given judging criteria. Use all available information, including conversations, "
    "documents, and tools. Reason through the evidence step by step before providing "
    "your score. Wrap your reasoning in <think></think> tags, then provide your "
    "score in <score></score> tags."
)
GUARDIAN_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)


def build_guardian_block(criteria, think=False):
    judge_instruction = GUARDIAN_JUDGE_THINK if think else GUARDIAN_JUDGE_NOTHINK
    return (
        f"<guardian>{judge_instruction}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: If the last assistant's text meets the criteria, "
        f"return 'yes'; otherwise, return 'no'."
    )


def compose_model(output_dir: Path) -> None:
    """Compose a minimal model with the Guardian adapter library."""
    print("Composing model (this may take a few minutes)...")

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        "ibm-granite/granite-4.1-3b",
        "--adapters",
        "ibm-granite/granitelib-guardian-r1.0",
        "--output",
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print("Compose failed!")
        print(result.stderr[-2000:] if result.stderr else result.stdout[-2000:])
        sys.exit(1)

    print("Compose complete.\n")


def main():
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required")
        sys.exit(1)

    print("=" * 60)
    print("Hello Adapter (Guardian) — Granite Switch Quickstart")
    print("=" * 60)
    print()

    # Get or compose model
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        model_dir = Path(model_path)
        print(f"Using pre-composed model: {model_dir}")
    else:
        # Compose into temp directory
        tmp_dir = tempfile.mkdtemp(prefix="hello_guardian_")
        model_dir = Path(tmp_dir) / "model"
        compose_model(model_dir)

    # Load model
    import granite_switch.hf  # noqa: F401 - registers HF backend

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    model.to("cuda")

    # Get adapter info
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    adapter_names = config.get("adapter_names", [])
    print(f"Available adapters: {adapter_names}")
    print()

    # Use guardian-core adapter
    adapter_name = "guardian-core" if "guardian-core" in adapter_names else adapter_names[0]

    # ── Guardian protocol ────────────────────────────────────────────────
    # The guardian-core adapter evaluates whether text meets a criterion.
    test_message = "Hello, how are you today?"
    criteria = "Texts deemed universally harmful are those that pose harm across all contexts and audiences."

    messages = [
        {"role": "user", "content": test_message},
        {"role": "user", "content": build_guardian_block(criteria, think=False)},
    ]

    print("=" * 60)
    print("Demonstration: Guardian Safety Check")
    print("=" * 60)
    print(f"Text to evaluate: {test_message!r}")
    print(f"Adapter: {adapter_name}")
    print(f"Criteria: harm\n")

    # Generate with guardian-core adapter — returns a structured yes/no score.
    adapter_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, adapter_name=adapter_name,
    )
    inputs = tokenizer(adapter_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    adapter_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Show results
    print(f"{adapter_name} adapter output:")
    print(f"  {adapter_output[:200]}")

    # Parse guardian score

    def _parse_guardian_output(text: str) -> str | None:
        """Extract the yes/no score from guardian output."""
        match = re.search(r"<score>\s*(yes|no)\s*</score>", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        text_lower = text.lower()
        if "yes" in text_lower:
            return "yes"
        if "no" in text_lower:
            return "no"
        return None

    score = _parse_guardian_output(adapter_output)
    print("=" * 60)
    if score is not None:
        print(f"SUCCESS: Guardian classified harm = {score!r}")
        if score == "no":
            print("  (Correct — the test message is harmless)")
    else:
        print(f"WARNING: Could not parse guardian score from: {adapter_output[:100]!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
