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
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

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


#: The GPU worker that actually stands up each engine (see module docstring).
#: Runs in its OWN process so this -- the pytest --
#: process never creates a CUDA context.
_LOGPROBS_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_vllm_logprobs_worker.py"
)
#: Per engine: HF checkpoint load + vLLM init + a single teacher-forced generate.
#: Generous; the real gate is correctness, and a hung child should fail loudly.
_WORKER_TIMEOUT = float(os.environ.get("GS_LOGPROBS_WORKER_TIMEOUT", "1200"))
#: Fresh-worker attempts on a transient device-unavailable init failure (see
#: _spawn_logprobs_worker). Each attempt is a brand-new process, which is what
#: lets it ride through the host's sticky-but-transient Exclusive_Process window.
_WORKER_INIT_RETRIES = int(os.environ.get("GS_LOGPROBS_WORKER_RETRIES", "4"))
#: Seconds to let the device settle between those attempts.
_WORKER_RETRY_DELAY = float(os.environ.get("GS_LOGPROBS_WORKER_RETRY_DELAY", "15"))
#: Monotonic counter so each engine run gets its own handoff files inside tmpdir.
#: Also drives the per-engine round-robin device pin (see _visible_device_count /
#: GS_LOGPROBS_PIN_DEVICE in _vllm_logprobs_worker.py): consecutive engines land on
#: alternating GPUs, so a worker whose EngineCore child is still releasing a device
#: does not collide with the next worker trying to acquire the SAME device under
#: Exclusive_Process. (The primary cluster-2 fix is keeping the worker PARENT
#: CUDA-free -- see vllm_gpu_mem.gpu_mem_util; round-robin just spreads the two
#: workers of a pair across the two cards so their teardown/acquire windows never
#: overlap on one device.)
_logprobs_run_counter = itertools.count()
#: Set GS_LOGPROBS_NO_ROUND_ROBIN=1 to disable device round-robin (e.g. to A/B it or
#: on a single-GPU host where it is a no-op anyway).
_ROUND_ROBIN_DEVICES = not os.environ.get("GS_LOGPROBS_NO_ROUND_ROBIN")


def _visible_device_count():
    """How many GPUs are visible to the workers, WITHOUT creating a CUDA context.

    The pytest process must stay CUDA-free, so
    this never calls ``torch.cuda``. It reads ``CUDA_VISIBLE_DEVICES`` when set
    (the authoritative allowlist the workers also parse), else asks ``nvidia-smi``.
    Returns ``1`` on any ambiguity so the round-robin degrades to "always device 0"
    -- i.e. today's single-device pin -- rather than pointing off the end.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [v.strip() for v in visible.split(",") if v.strip() != ""]
        return max(1, len(entries))
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        n = len([line for line in out.splitlines() if line.strip()])
        return max(1, n)
    except Exception:
        return 1


def _is_transient_device_error(text: str) -> bool:
    """True if worker output shows the Exclusive_Process device-availability race.

    The signature is ``cudaErrorDevicesUnavailable`` ("CUDA-capable device(s)
    is/are busy or unavailable"), raised at engine ``init_device`` -- a transient
    the host throws under NVIDIA Exclusive_Process compute mode, independent of the
    model or weights. Matching the worker's
    FULL stdout+stderr (not a wrapped parent exception string) is why this can key
    on the real error text, which the old in-process retry could not see.
    """
    low = text.lower()
    return (
        "cudaerrordevicesunavailable" in low
        or "device(s) is/are busy or unavailable" in low
    )


def run_vllm_logprobs(model_dir, input_ids_list, vocab_size, *, tmpdir, **llm_kwargs):
    """Run one vLLM engine in a subprocess, return its dense prompt logprobs.

    The engine is NOT built in this process. Under the GPU host's
    Exclusive_Process compute mode, if this (the pytest) process ever created a
    CUDA context -- which ``gpu_mem_util``/``shutdown_llm`` would, via
    ``torch.cuda.mem_get_info``/``empty_cache`` -- it would own the device and
    every ``EngineCore`` child spawned afterward would die with
    ``cudaErrorDevicesUnavailable``. So the engine, and every ``torch.cuda`` call,
    live entirely in ``_vllm_logprobs_worker.py``, one fresh process per engine.
    This process only writes a JSON spec, shells out, and loads back a CPU tensor.

    Args:
        model_dir: path to saved HF model (config.json + model.safetensors)
        input_ids_list: list of int token IDs
        vocab_size: vocab size for dense tensor allocation
        tmpdir: directory for the subprocess handoff files (the caller's tmp_path)
        **llm_kwargs: extra kwargs for LLM() constructor (e.g. max_model_len,
            enforce_eager), forwarded verbatim to the worker

    Returns:
        [seq_len-1, vocab_size] float tensor of log-probabilities
    """
    # enable_prefix_caching: a SYMMETRIC-config default, applied here (no CUDA) so
    # both engines of a pair get it. upstream's GraniteMoeHybridForCausalLM declares
    # IsHybrid and GraniteSwitchForCausalLM deliberately does not (declaring it
    # would size the KV cache for zero attention layers), so the two sides resolve
    # DIFFERENT prefix-caching defaults -- an asymmetry in a test whose whole job is
    # to have one variable. It also kills a vLLM 0.28-only crash in upstream's
    # model: with prefix caching on, 0.28 logs "Mamba cache mode is set to 'align'"
    # and then asserts "no mamba layers in the model" (v1/worker/mamba_utils.py).
    llm_kwargs.setdefault("enable_prefix_caching", False)
    #
    # Deliberately NOT defaulting enforce_eager, and NOT a free choice -- it decides
    # whether the adapter-present runners can meet get_tolerances' 1e-5 inert-path
    # gate. Measured on vLLM 0.26.0, eager, zero-LoRA pair: switch-vs-upstream is a
    # real, fully deterministic ~1e-3 difference (each engine bit-exact against a
    # second copy of itself), appearing ONLY when the adapter tier is live, because
    # the fused SWITCH kernel reduces in a different order than the native linear.
    # torch.compile normalizes that order and closes the gap while preserving
    # semantics (a real kernel bug still trips 1e-5). The one place eager IS
    # required is _granite4_fullsize_tests.py, which asks for it explicitly (an
    # unpinned autotuner reddened ~1 run in 7 at full size). max_logprobs and
    # gpu_memory_utilization are set in the worker, where torch.cuda is safe to
    # touch. Full rationale table: git history of this file.

    run_id = next(_logprobs_run_counter)
    spec_path = os.path.join(str(tmpdir), f"logprobs_inputs_{run_id}.json")
    out_path = os.path.join(str(tmpdir), f"logprobs_out_{run_id}.pt")
    with open(spec_path, "w") as f:
        json.dump(
            {
                "input_ids": list(input_ids_list),
                "vocab_size": vocab_size,
                "llm_kwargs": llm_kwargs,
            },
            f,
        )

    cmd = [
        sys.executable,
        _LOGPROBS_WORKER,
        "run",
        "--model",
        str(model_dir),
        "--inputs",
        spec_path,
        "--out",
        out_path,
    ]
    # Bounded retry on the transient device-unavailable race ONLY. Each attempt is
    # a FRESH worker process -- the key difference from the old in-process retry: a
    # new process gets a clean CUDA context, which is what rides through the host's
    # sticky-but-transient Exclusive_Process window (an in-process retry could not,
    # and anyway it keyed on a wrapped exception string that never matched). vLLM's
    # non-eager init (torch.compile + CUDA-graph capture) is the path that trips
    # this; eager init does not, but the adapter-live tests require non-eager for
    # the tolerance gate, so retry rather than force eager. Any OTHER failure (a
    # real config/weight/logic error) raises immediately -- it is not retried.
    # Round-robin the pinned GPU across engines (and across retries of one engine)
    # so consecutive spawns do not contend for the same device while a previous
    # worker's EngineCore child is still releasing it under Exclusive_Process. A
    # retry advances the slot too, moving a failed spawn to the OTHER GPU rather
    # than back onto the one that just refused it. No-op on a single-GPU host
    # (slot stays 0). This is a secondary robustness measure; the primary fix is
    # keeping the worker parent CUDA-free (vllm_gpu_mem.gpu_mem_util).
    device_count = _visible_device_count() if _ROUND_ROBIN_DEVICES else 1
    print(f"\n  logprobs worker: {' '.join(cmd)}")
    for attempt in range(1, _WORKER_INIT_RETRIES + 1):
        worker_env = dict(os.environ)
        if device_count > 1:
            slot = (run_id + attempt - 1) % device_count
            worker_env["GS_LOGPROBS_PIN_DEVICE"] = str(slot)
            print(f"  round-robin device slot {slot} (of {device_count})")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_WORKER_TIMEOUT,
            env=worker_env,
        )
        if result.stdout:
            print(result.stdout[-4000:])
        if result.stderr:
            print("  STDERR:", result.stderr[-2000:])
        if result.returncode == 0:
            break

        combined = f"{result.stdout}\n{result.stderr}"
        transient = _is_transient_device_error(combined)
        if not transient or attempt == _WORKER_INIT_RETRIES:
            raise RuntimeError(
                f"vLLM logprobs worker failed for {model_dir} "
                f"(exit {result.returncode}"
                f"{', transient device-unavailable, retries exhausted' if transient else ''}"
                f").\n"
                f"STDOUT (tail):\n{result.stdout[-2000:]}\n"
                f"STDERR (tail):\n{result.stderr[-2000:]}"
            )
        print(
            f"  worker hit the transient device-unavailable race "
            f"(attempt {attempt}/{_WORKER_INIT_RETRIES}); letting the device settle "
            f"{_WORKER_RETRY_DELAY:g}s and re-spawning a fresh worker"
        )
        time.sleep(_WORKER_RETRY_DELAY)

    # Loaded on CPU: the pytest process must stay CUDA-free.
    return torch.load(out_path, map_location="cpu")


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
        tmpdir=tmpdir,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        tmpdir=tmpdir,
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
        tmpdir=tmpdir,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        tmpdir=tmpdir,
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
        tmpdir=tmpdir,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        input_ids,
        vocab_size,
        tmpdir=tmpdir,
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
        tmpdir=tmpdir,
        **llm_kwargs,
    )
    switch_logprobs = run_vllm_logprobs(
        switch_dir,
        switch_ids[0].tolist(),
        vocab_size,
        tmpdir=tmpdir,
        **llm_kwargs,
    )

    return upstream_logprobs, switch_logprobs
