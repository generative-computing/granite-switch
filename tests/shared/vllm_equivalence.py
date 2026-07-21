# SPDX-License-Identifier: Apache-2.0
"""Shared vLLM infrastructure for Granite 4 upstream equivalence tests.

Provides helpers for:
- Saving HF models (upstream + switch) to disk with random weights
- Running vLLM inference via vllm.LLM and extracting prompt logprobs
- Full integration equivalence pipeline

All vLLM-specific imports are lazy (inside functions) so this module can
be imported even without vLLM installed.

Used by:
- tests/vllm/test_upstream_equivalence.py (combinatorial unit-scaled tests)
- tests/vllm/test_granite4_mini.py (miniaturized real-scaled tests)
- tests/vllm/test_granite4_fullsize.py (full-dimension tests)
"""

import gc
import json
import os
import tempfile

import torch

# ── Model creation (kept for other tests) ─────────────────────────


def make_vllm_config(config, architectures, max_tokens=None):
    """Create VllmConfig with ModelConfig from a transformers config.

    Writes config.json to a temp directory, then builds ModelConfig from it.
    """
    from vllm.config import ModelConfig, VllmConfig

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    tmpdir = tempfile.mkdtemp(prefix="granite_switch_equiv_")
    config_dict = config.to_dict()
    config_dict["architectures"] = architectures
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    if max_tokens is None:
        max_tokens = 256

    model_config = ModelConfig(
        model=tmpdir,
        task="generate",
        dtype="bfloat16",
        max_model_len=min(config.max_position_embeddings, max_tokens),
    )
    return VllmConfig(model_config=model_config)


def init_model_weights(model, seed=0):
    """Initialize model weights with small random values.

    vLLM uses torch.empty for parameter allocation, so we need explicit init.
    Skips LoRA parameters and layer norms (which use default init from vLLM).
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if "lora_A" in name or "lora_B" in name:
                continue
            if "layernorm" in name or "norm" in name:
                continue
            param.data.normal_(0, 0.02)


# ── HF model saving ──────────────────────────────────────────────


def save_upstream_model(cfg_dict, seed, tmpdir):
    """Create HF GraniteMoeHybridForCausalLM with random weights, save to disk.

    Returns (save_dir, state_dict).
    State dict is returned for weight transfer to switch model without
    re-loading from disk.
    """
    from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
        GraniteMoeHybridConfig,
    )
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridForCausalLM,
    )

    torch.manual_seed(seed)
    config = GraniteMoeHybridConfig(**cfg_dict)
    model = GraniteMoeHybridForCausalLM(config).eval()

    save_dir = os.path.join(str(tmpdir), "upstream")
    model.save_pretrained(save_dir)

    sd = model.state_dict()
    del model
    gc.collect()

    return save_dir, sd


def save_switch_model(upstream_sd, cfg_dict, tmpdir):
    """Create HF GraniteSwitch (num_adapters=0), transfer upstream weights, save.

    Args:
        upstream_sd: state dict from upstream HF model
        cfg_dict: model config dict (same as upstream)
        tmpdir: parent directory

    Returns save_dir path.
    """
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM as HFSwitch
    from tests.shared.granite4_equivalence import transfer_weights_strict

    switch_cfg = GraniteSwitchConfig(**cfg_dict, num_adapters=0)
    switch = HFSwitch(switch_cfg).eval()

    transfer_weights_strict(upstream_sd, switch.state_dict())

    save_dir = os.path.join(str(tmpdir), "switch")
    switch.save_pretrained(save_dir)
    del switch
    gc.collect()

    return save_dir


def save_zero_adapter_model(upstream_sd, cfg_dict, tmpdir, num_adapters=2):
    """Create GraniteSwitch with adapters, transfer base weights, zero LoRA, save.

    Args:
        upstream_sd: state dict from upstream HF model
        cfg_dict: base model config dict (same as upstream, without adapter fields)
        tmpdir: parent directory
        num_adapters: number of adapters to create (default 2)

    Returns:
        save_dir path.
    """
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM as HFSwitch
    from tests.shared.granite4_equivalence import (
        augment_cfg_with_adapters,
        transfer_weights,
        zero_lora_weights,
    )

    switch_cfg_dict = augment_cfg_with_adapters(cfg_dict, num_adapters=num_adapters)
    switch_cfg = GraniteSwitchConfig(**switch_cfg_dict)
    switch = HFSwitch(switch_cfg).eval()

    # Transfer base weights (non-strict: LoRA/switch params left unloaded)
    transfer_weights(upstream_sd, switch.state_dict())

    # Zero all LoRA weights defensively
    zero_lora_weights(switch)

    save_dir = os.path.join(str(tmpdir), "switch-zero")
    switch.save_pretrained(save_dir)
    del switch
    gc.collect()

    return save_dir


# ── vLLM logprob extraction ──────────────────────────────────────


def extract_logprobs_tensor(outputs, vocab_size):
    """Convert vLLM prompt_logprobs to dense [seq_len-1, vocab_size] tensor.

    outputs[0].prompt_logprobs is List[Optional[Dict[int, Logprob]]].
    Position 0 is always None (no previous token to condition on).
    """
    prompt_logprobs = outputs[0].prompt_logprobs
    seq_len = len(prompt_logprobs) - 1
    tensor = torch.full((seq_len, vocab_size), float("-inf"))

    for i, pos_logprobs in enumerate(prompt_logprobs[1:]):
        if pos_logprobs is not None:
            for token_id, logprob_obj in pos_logprobs.items():
                tensor[i, token_id] = logprob_obj.logprob

    return tensor


def dump_model_params(model_dir, dump_path):
    """Load model via vLLM and dump parameter checksums to a file."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM

    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
    )

    # Access model params via internal API
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    checksums = {}
    for name, param in model.named_parameters():
        checksums[name] = (
            param.shape,
            param.sum().item(),
            param.abs().max().item(),
            param.float().norm().item(),
        )

    torch.save(checksums, dump_path)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def run_vllm_logprobs(model_dir, input_ids_list, vocab_size, **llm_kwargs):
    """Load model via vllm.LLM, extract prompt logprobs as dense tensor.

    Args:
        model_dir: path to saved HF model (config.json + model.safetensors)
        input_ids_list: list of int token IDs
        vocab_size: vocab size for dense tensor allocation
        **llm_kwargs: extra kwargs for LLM() constructor (e.g. max_model_len,
            gpu_memory_utilization)

    Returns:
        [seq_len-1, vocab_size] float tensor of log-probabilities
    """
    # vLLM v1 forks a subprocess for EngineCore; if CUDA was initialized
    # in the parent (e.g. by torch.cuda.is_available() probe), fork fails.
    # Spawn avoids inheriting the parent's CUDA state.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Allow requesting logprobs for all vocab tokens
    llm_kwargs.setdefault("max_logprobs", vocab_size)

    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        **llm_kwargs,
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=vocab_size,
    )

    prompt = TokensPrompt(prompt_token_ids=input_ids_list)
    outputs = llm.generate(prompt, sampling_params=sampling_params)

    logprobs_tensor = extract_logprobs_tensor(outputs, vocab_size)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return logprobs_tensor


# ── Full integration pipeline ─────────────────────────────────────


def run_equivalence_integration(cfg_dict, *, seq_len=16, seed=0, tmpdir, **llm_kwargs):
    """Full integration equivalence pipeline via vllm.LLM.

    1. Create HF upstream model (random weights) -> save_pretrained
    2. Transfer weights to HF switch model (num_adapters=0) -> save_pretrained
    3. Delete HF models (CPU only, free memory)
    4. Load upstream via LLM -> extract prompt_logprobs -> delete LLM
    5. Load switch via LLM -> extract prompt_logprobs -> delete LLM
    6. Return (upstream_logprobs, switch_logprobs)

    Args:
        cfg_dict: model config dict
        seq_len: input sequence length
        seed: random seed for weight initialization
        tmpdir: directory for saving model checkpoints
        **llm_kwargs: extra kwargs for LLM() (e.g. max_model_len,
            gpu_memory_utilization)

    Returns:
        (upstream_logprobs, switch_logprobs) — each [seq_len-1, vocab_size]
    """
    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    # Phase 1: save HF models to disk
    upstream_dir, upstream_sd = save_upstream_model(cfg_dict, seed, tmpdir)
    switch_dir = save_switch_model(upstream_sd, cfg_dict, tmpdir)
    del upstream_sd
    gc.collect()

    # Generate deterministic input
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg_dict["vocab_size"], (seq_len,)).tolist()
    vocab_size = cfg_dict["vocab_size"]

    # Phase 2: run both models through vLLM's actual serving path
    upstream_logprobs = run_vllm_logprobs(
        upstream_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )

    return upstream_logprobs, switch_logprobs


def run_zero_adapter_no_hiding_equivalence(
    cfg_dict, *, use_control_tokens=False, seq_len=16, seed=0, tmpdir, **llm_kwargs
):
    """Integration pipeline for zero-adapter switch.

    Creates a switch model with adapter infrastructure (LoRA wrappers, switch
    layer). Zero LoRA weights means zero delta. When not using control tokens,
    output should be bit-exact with upstream.

    Args:
        cfg_dict: base model config dict (without adapter fields)
        use_control_tokens: if True, input includes control tokens that
            trigger adapter switching; if False, plain random tokens
        seq_len: input sequence length
        seed: random seed for weight initialization
        tmpdir: directory for saving model checkpoints
        **llm_kwargs: extra kwargs for LLM()

    Returns:
        (upstream_logprobs, switch_logprobs) -- each [seq_len-1, vocab_size]
    """
    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    # Phase 1: save HF models to disk
    upstream_dir, upstream_sd = save_upstream_model(cfg_dict, seed, tmpdir)
    switch_dir = save_zero_adapter_model(upstream_sd, cfg_dict, tmpdir)
    del upstream_sd
    gc.collect()

    # Generate input
    if use_control_tokens:
        from tests.shared.granite4_equivalence import make_active_adapter_input

        input_ids = make_active_adapter_input(1, seq_len, seed=42)
        input_ids = input_ids[0].tolist()
    else:
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (seq_len,)).tolist()
    vocab_size = cfg_dict["vocab_size"]

    # Phase 2: run both models through vLLM's actual serving path
    upstream_logprobs = run_vllm_logprobs(
        upstream_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )

    return upstream_logprobs, switch_logprobs


def run_zero_adapter_equivalence(cfg_dict, *, seq_len=16, seed=0, tmpdir, **llm_kwargs):
    """Integration pipeline for zero-adapter switch with hiding enabled.

    Creates a switch model WITH adapter infrastructure and zero LoRA weights.
    Input includes control tokens that trigger adapter switching. Hidden
    positions are intentionally different; compare only visible positions.

    Args:
        cfg_dict: base model config dict (without adapter fields)
        seq_len: input sequence length
        seed: random seed for weight initialization
        tmpdir: directory for saving model checkpoints
        **llm_kwargs: extra kwargs for LLM()

    Returns:
        (upstream_logprobs, switch_logprobs) — each [seq_len-1, vocab_size]
    """
    from granite_switch.vllm import register as register_granite_switch
    from tests.shared.granite4_equivalence import make_active_adapter_input

    register_granite_switch()

    # Phase 1: save HF models to disk
    upstream_dir, upstream_sd = save_upstream_model(cfg_dict, seed, tmpdir)
    switch_dir = save_zero_adapter_model(upstream_sd, cfg_dict, tmpdir)
    del upstream_sd
    gc.collect()

    # Generate input with control tokens that trigger adapter switching
    input_ids = make_active_adapter_input(1, seq_len, seed=42)
    input_ids = input_ids[0].tolist()  # vLLM expects flat list
    vocab_size = cfg_dict["vocab_size"]

    # Phase 2: run both models through vLLM's actual serving path
    upstream_logprobs = run_vllm_logprobs(
        upstream_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        **llm_kwargs,
    )

    return upstream_logprobs, switch_logprobs


# ── Gap equivalence pipeline ──────────────────────────────────────


def run_gap_equivalence(cfg_dict, *, seq_len, ctrl_pos, seed=0, tmpdir, **llm_kwargs):
    """Integration pipeline for KV hiding gap equivalence via vLLM.

    Creates upstream + 1-adapter switch (zero LoRA), inserts a hidden control
    token at ctrl_pos, and runs both through vLLM's serving path.

    Args:
        cfg_dict: base model config dict (without adapter fields)
        seq_len: upstream sequence length (switch gets seq_len + 1)
        ctrl_pos: position to insert hidden control token
        seed: random seed for weight initialization
        tmpdir: directory for saving model checkpoints
        **llm_kwargs: extra kwargs for LLM()

    Returns:
        (upstream_logprobs, switch_logprobs) — each dense logprob tensors.
        upstream: [seq_len-1, vocab_size], switch: [seq_len, vocab_size].
    """
    from granite_switch.vllm import register as register_granite_switch
    from tests.shared.gap_equivalence import make_gapped_inputs

    register_granite_switch()

    # Phase 1: save HF models to disk
    upstream_dir, upstream_sd = save_upstream_model(cfg_dict, seed, tmpdir)
    switch_dir = save_zero_adapter_model(upstream_sd, cfg_dict, tmpdir, num_adapters=1)
    del upstream_sd
    gc.collect()

    # Generate gapped inputs
    upstream_ids, switch_ids = make_gapped_inputs(seq_len, ctrl_pos, seed=42)
    vocab_size = cfg_dict["vocab_size"]

    # Phase 2: run both models through vLLM's actual serving path
    upstream_logprobs = run_vllm_logprobs(
        upstream_dir,
        upstream_ids[0].tolist(),
        vocab_size,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        switch_ids[0].tolist(),
        vocab_size,
        **llm_kwargs,
    )

    return upstream_logprobs, switch_logprobs
