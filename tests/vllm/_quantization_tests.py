# SPDX-License-Identifier: Apache-2.0
"""Quantization tests for GraniteSwitch vLLM backend.
Inner file — run by test_quantization.py in subprocess.

Verifies that quantization:
1. Actually quantizes base model linear layers (weight dtype/shape changes)
2. Keeps LoRA/aLoRA weights in full precision (bfloat16)
3. Adapters still activate (different output with vs without adapter token)

Quantization methods tested:
- BitsAndBytes INT4 (NF4)
- FP8 (vLLM native fp8)

Requires: CUDA GPU, vLLM, bitsandbytes.
Model: ibm-granite/granite-switch-4.1-3b-preview (pre-composed, loaded from HF).
"""

import os

# Force in-process mode so we can inspect model internals directly.
# Must be set BEFORE importing vLLM.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        from vllm.plugins import load_general_plugins  # noqa: F401
        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

MODEL_ID = "ibm-granite/granite-switch-4.1-3b-preview"

# ---------------------------------------------------------------------------
# Test data — real adapter prompts (LoRA + aLoRA)
# ---------------------------------------------------------------------------

ADAPTER_TESTS = [
    {
        "adapter_name": "hallucination_detection",
        "type": "lora",
        "messages": [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "Photosynthesis converts sunlight into glucose."},
            {"role": "user", "content": (
                "<guardian>You are a judge agent. Your role is to assess whether "
                "the provided text meets the given criteria.\n\n"
                "### Criteria: A factually incorrect response.\n\n"
                "### Scoring Schema: If the last assistant's text meets the "
                "criteria, return 'yes'; otherwise, return 'no'."
            )},
        ],
    },
    {
        "adapter_name": "answerability",
        "type": "alora",
        "messages": [
            {"role": "user", "content": "Who created Python?"},
        ],
        "documents": [
            {"doc_id": "1", "text": "Python was created by Guido van Rossum in 1991."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(quantization, load_format, gpu_memory_utilization=0.9):
    """Load model with given quantization config via vLLM."""
    from vllm import LLM
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    llm = LLM(
        model=MODEL_ID,
        quantization=quantization,
        load_format=load_format,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=256,
        enforce_eager=True,
    )
    return llm


def _get_tokenizer():
    """Get tokenizer for the model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


def _make_prompt(tokenizer, messages, adapter_name=None, documents=None):
    """Build a prompt string using the chat template."""
    kwargs = {}
    if adapter_name:
        kwargs["adapter_name"] = adapter_name
    if documents:
        kwargs["documents"] = documents
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )


def _generate(llm, prompt, max_tokens=32):
    """Generate text from a prompt."""
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


def _get_model_from_llm(llm):
    """Extract the actual model from vLLM's LLM wrapper.

    With VLLM_ENABLE_V1_MULTIPROCESSING=0 (set at top of file), the model
    lives in-process. We access it via the engine_core chain.
    """
    engine = llm.llm_engine
    # v1 in-process path (InprocClient)
    if hasattr(engine, 'engine_core'):
        core = engine.engine_core
        # InprocClient wraps EngineCore
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        if hasattr(core, 'model_executor'):
            executor = core.model_executor
            if hasattr(executor, 'driver_worker'):
                worker = executor.driver_worker
                if hasattr(worker, 'worker'):
                    # WorkerWrapperBase wraps the actual GPUWorker
                    worker = worker.worker
                return worker.model_runner.model
    # Fallback: model_executor on engine directly (older vLLM / v0 compat)
    if hasattr(engine, 'model_executor'):
        executor = engine.model_executor
        if hasattr(executor, 'driver_worker'):
            worker = executor.driver_worker
            if hasattr(worker, 'worker'):
                worker = worker.worker
            return worker.model_runner.model
    raise RuntimeError("Cannot access model from vLLM LLM object in this vLLM version")


# ---------------------------------------------------------------------------
# BitsAndBytes INT4 (NF4)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bnb_int4_llm():
    """Load model with BitsAndBytes INT4 (NF4) quantization."""
    pytest.importorskip("bitsandbytes")
    return _load_model(quantization="bitsandbytes", load_format="bitsandbytes")


class TestBnBInt4BaseQuantized:
    """BnB INT4: base linear layers must actually be quantized."""

    def test_base_weights_are_quantized(self, bnb_int4_llm):
        """Verify base linear layer weights are stored as uint8 (4-bit packed)."""
        model = _get_model_from_llm(bnb_int4_llm)

        quantized_count = 0
        total_linear = 0
        for name, module in model.named_modules():
            # vLLM BnB layers have weight in uint8 format
            if hasattr(module, "weight") and hasattr(module, "input_size_per_partition"):
                # This is a vLLM LinearBase — check if quantized
                total_linear += 1
                if module.weight.dtype == torch.uint8:
                    quantized_count += 1

        assert quantized_count > 0, (
            f"No quantized linear layers found (checked {total_linear} LinearBase modules). "
            "BnB INT4 quantization did not apply."
        )
        print(f"\n  BnB INT4: {quantized_count}/{total_linear} base linear layers quantized")


class TestBnBInt4LoRAPrecision:
    """BnB INT4: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, bnb_int4_llm):
        """Verify all LoRA parameters (lora_A, lora_B) stay in bfloat16/float16."""
        model = _get_model_from_llm(bnb_int4_llm)

        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        lora_count = 0

        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_count += 1
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")

        assert lora_count > 0, "No LoRA parameters found in model"
        assert not bad_params, (
            f"LoRA params quantized under BnB INT4 (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )
        print(f"\n  BnB INT4: {lora_count} LoRA params verified as full precision")


class TestBnBInt4AdapterActivation:
    """BnB INT4: adapters must activate (different output with adapter via chat template)."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, bnb_int4_llm, case):
        """Output must differ when adapter is activated via chat template."""
        tokenizer = _get_tokenizer()
        documents = case.get("documents")

        base_prompt = _make_prompt(tokenizer, case["messages"], documents=documents)
        adapter_prompt = _make_prompt(
            tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )

        base_out = _generate(bnb_int4_llm, base_prompt)
        adapter_out = _generate(bnb_int4_llm, adapter_prompt)

        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} ({case['type']}) did not activate under BnB INT4.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )
        print(f"\n  BnB INT4 adapter '{case['adapter_name']}' ({case['type']}) activation verified:")
        print(f"    Base:    {repr(base_out[:80])}")
        print(f"    Adapter: {repr(adapter_out[:80])}")


class TestBnBInt4LoRADimensionsCorrect:
    """BnB INT4: LoRA tensors must have correct dimensions (not corrupted by BnB packing)."""

    def test_lora_shapes_match_config(self, bnb_int4_llm):
        """Verify LoRA A/B shapes use correct in/out features, not packed weight shapes."""
        model = _get_model_from_llm(bnb_int4_llm)
        from granite_switch.vllm.core.lora import SwitchedLoRALinear

        checked = 0
        for name, module in model.named_modules():
            if isinstance(module, SwitchedLoRALinear):
                checked += 1
                # Verify dimensions match config, not packed weight shape
                base = module.base_layer
                expected_in = base.input_size_per_partition
                expected_out = base.output_size_per_partition

                assert module.in_features == expected_in, (
                    f"{name}: in_features={module.in_features} != "
                    f"expected input_size_per_partition={expected_in}"
                )
                assert module.out_features == expected_out, (
                    f"{name}: out_features={module.out_features} != "
                    f"expected output_size_per_partition={expected_out}"
                )

                # Check LoRA tensor shapes
                if hasattr(module, "lora_A"):
                    # [num_adapters, 1, max_rank, in_features]
                    assert module.lora_A.shape[-1] == expected_in, (
                        f"{name}: lora_A last dim={module.lora_A.shape[-1]} != in_features={expected_in}"
                    )
                if hasattr(module, "lora_B"):
                    # [num_adapters, 1, out_features, max_rank]
                    assert module.lora_B.shape[-2] == expected_out, (
                        f"{name}: lora_B dim[-2]={module.lora_B.shape[-2]} != out_features={expected_out}"
                    )

        assert checked > 0, "No SwitchedLoRALinear modules found"
        print(f"\n  BnB INT4: {checked} SwitchedLoRALinear dimension checks passed")


# ---------------------------------------------------------------------------
# FP8 (vLLM native)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fp8_llm():
    """Load model with vLLM native FP8 quantization."""
    from vllm import LLM
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    # FP8 requires compute capability >= 8.9 (Hopper H100 / Ada Lovelace)
    # A100 (8.0) does NOT support native FP8.
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        pytest.skip(
            f"FP8 requires compute capability >= 8.9 (Hopper/Ada). "
            f"Got {major}.{minor} ({torch.cuda.get_device_name(0)})"
        )

    llm = LLM(
        model=MODEL_ID,
        quantization="fp8",
        gpu_memory_utilization=0.9,
        max_model_len=256,
        enforce_eager=True,
    )
    return llm


class TestFP8BaseQuantized:
    """FP8: base linear layers must actually use fp8 weights."""

    def test_base_weights_are_fp8(self, fp8_llm):
        """Verify base linear layer weights are in fp8 format."""
        model = _get_model_from_llm(fp8_llm)

        fp8_count = 0
        total_linear = 0
        fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}

        for name, module in model.named_modules():
            if hasattr(module, "weight") and hasattr(module, "input_size_per_partition"):
                total_linear += 1
                if module.weight.dtype in fp8_dtypes:
                    fp8_count += 1

        assert fp8_count > 0, (
            f"No FP8 linear layers found (checked {total_linear} LinearBase modules). "
            "FP8 quantization did not apply."
        )
        print(f"\n  FP8: {fp8_count}/{total_linear} base linear layers quantized to fp8")


class TestFP8LoRAPrecision:
    """FP8: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, fp8_llm):
        """Verify all LoRA parameters stay in bfloat16/float16."""
        model = _get_model_from_llm(fp8_llm)

        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        lora_count = 0

        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_count += 1
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")

        assert lora_count > 0, "No LoRA parameters found in model"
        assert not bad_params, (
            f"LoRA params quantized under FP8 (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )
        print(f"\n  FP8: {lora_count} LoRA params verified as full precision")


class TestFP8AdapterActivation:
    """FP8: adapters must activate."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, fp8_llm, case):
        """Output must differ when adapter is activated via chat template."""
        tokenizer = _get_tokenizer()
        documents = case.get("documents")

        base_prompt = _make_prompt(tokenizer, case["messages"], documents=documents)
        adapter_prompt = _make_prompt(
            tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )

        base_out = _generate(fp8_llm, base_prompt)
        adapter_out = _generate(fp8_llm, adapter_prompt)

        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} ({case['type']}) did not activate under FP8.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )
        print(f"\n  FP8 adapter '{case['adapter_name']}' ({case['type']}) activation verified:")
        print(f"    Base:    {repr(base_out[:80])}")
        print(f"    Adapter: {repr(adapter_out[:80])}")


# ---------------------------------------------------------------------------
# Memory usage sanity check (BnB INT4 should use significantly less than bf16)
# ---------------------------------------------------------------------------

class TestBnBInt4MemoryReduction:
    """BnB INT4: model weight memory should be less than bf16 equivalent."""

    def test_model_weights_smaller_than_bf16(self, bnb_int4_llm):
        """4-bit quantized 3B model weights should be ~1.5 GiB (bf16 would be ~6 GiB)."""
        model = _get_model_from_llm(bnb_int4_llm)

        total_bytes = 0
        for name, param in model.named_parameters():
            total_bytes += param.nelement() * param.element_size()

        weight_gib = total_bytes / (1024**3)
        # 3B model in bf16 = ~6 GiB weights
        # With BnB 4-bit base + bf16 LoRA, expect ~2-3 GiB total parameter memory
        assert weight_gib < 4.0, (
            f"Model parameter memory {weight_gib:.2f} GiB seems too high for 4-bit quantized 3B model. "
            "Expected < 4 GiB. Quantization may not be working."
        )
        print(f"\n  BnB INT4 model parameter memory: {weight_gib:.2f} GiB")
