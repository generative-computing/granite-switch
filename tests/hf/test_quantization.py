# SPDX-License-Identifier: Apache-2.0
"""Quantization tests for GraniteSwitch HF backend.

Verifies that quantization:
1. Preserves adapter activation (output differs with/without adapter via chat template)
2. Keeps LoRA/aLoRA weights in full precision (bfloat16)
3. Actually quantizes base model linear layers

Quantization methods tested:
- BitsAndBytes: NF4 (INT4), FP4
- Quanto: INT4, FP8

No hardware gating — all methods dequantize to BF16 at compute time on HF backend.
Requires: CUDA GPU, bitsandbytes, optimum-quanto.
Model: ibm-granite/granite-switch-4.1-3b-preview (pre-composed, loaded from HF).
"""

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.gpu]

MODEL_ID = "ibm-granite/granite-switch-4.1-3b-preview"

# ---------------------------------------------------------------------------
# Test data
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
# Helper
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, messages, adapter_name=None, documents=None, max_new_tokens=30):
    """Generate greedy text from messages using the chat template."""
    kwargs = {}
    if adapter_name:
        kwargs["adapter_name"] = adapter_name
    if documents:
        kwargs["documents"] = documents

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bnb_model():
    """Load granite-switch from HF with BitsAndBytes NF4 quantization."""
    bitsandbytes = pytest.importorskip("bitsandbytes")  # noqa: F841
    import granite_switch.hf  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def quanto_model():
    """Load granite-switch from HF with Quanto INT4 quantization."""
    pytest.importorskip("optimum.quanto")
    import granite_switch.hf  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

    quanto_config = QuantoConfig(weights="int4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quanto_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests: Adapter Activation
# ---------------------------------------------------------------------------

class TestBnBAdapterActivation:
    """BitsAndBytes NF4: adapters must activate (output differs with adapter via chat template)."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, bnb_model, case):
        model, tokenizer = bnb_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model, tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under BnB NF4.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )


class TestQuantoAdapterActivation:
    """Quanto INT4: adapters must activate (output differs with adapter via chat template)."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, quanto_model, case):
        model, tokenizer = quanto_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model, tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under Quanto INT4.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )


# ---------------------------------------------------------------------------
# Tests: LoRA Weight Precision
# ---------------------------------------------------------------------------

class TestBnBLoRAPrecision:
    """BitsAndBytes NF4: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, bnb_model):
        model, _ = bnb_model
        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")
        assert not bad_params, (
            f"LoRA params quantized under BnB (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )


class TestQuantoLoRAPrecision:
    """Quanto INT4: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, quanto_model):
        model, _ = quanto_model
        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")
        assert not bad_params, (
            f"LoRA params quantized under Quanto (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )


# ---------------------------------------------------------------------------
# Tests: Base Weight Quantization
# ---------------------------------------------------------------------------

class TestBnBBaseQuantized:
    """BitsAndBytes NF4: base linear layers must actually be quantized."""

    def test_base_layers_are_4bit(self, bnb_model):
        model, _ = bnb_model
        quantized_count = 0
        for name, module in model.named_modules():
            if "Linear4bit" in type(module).__name__:
                quantized_count += 1
        assert quantized_count > 0, (
            "No Linear4bit modules found — BnB quantization did not apply."
        )
        print(f"\n  BnB: {quantized_count} layers quantized to 4-bit")


class TestQuantoBaseQuantized:
    """Quanto INT4: base linear layers must actually be quantized."""

    def test_base_layers_are_quantized(self, quanto_model):
        model, _ = quanto_model
        quantized_count = 0
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if "QLinear" in module_type or "Quantized" in module_type:
                quantized_count += 1
            elif hasattr(module, "weight") and hasattr(module.weight, "qtype"):
                quantized_count += 1
        assert quantized_count > 0, (
            "No quantized modules/weights found — Quanto quantization did not apply."
        )
        print(f"\n  Quanto: {quantized_count} layers quantized")


# ===========================================================================
# FP8 Tests (Quanto float8 — dequantizes to BF16 at compute time)
# ===========================================================================

@pytest.fixture(scope="module")
def fp8_model():
    """Load granite-switch from HF with Quanto FP8 quantization."""
    pytest.importorskip("optimum.quanto")
    import granite_switch.hf  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

    quanto_config = QuantoConfig(weights="float8")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quanto_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


class TestFP8AdapterActivation:
    """FP8 (Quanto float8): adapters must activate."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, fp8_model, case):
        model, tokenizer = fp8_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model, tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under FP8.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )


class TestFP8LoRAPrecision:
    """FP8: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, fp8_model):
        model, _ = fp8_model
        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")
        assert not bad_params, (
            f"LoRA params quantized under FP8 (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )


class TestFP8BaseQuantized:
    """FP8: base linear layers must actually be quantized."""

    def test_base_layers_are_quantized(self, fp8_model):
        model, _ = fp8_model
        quantized_count = 0
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if "QLinear" in module_type or "Quantized" in module_type:
                quantized_count += 1
            elif hasattr(module, "weight") and hasattr(module.weight, "qtype"):
                quantized_count += 1
        assert quantized_count > 0, (
            "No quantized modules/weights found — FP8 quantization did not apply."
        )
        print(f"\n  FP8: {quantized_count} layers quantized")


# ===========================================================================
# FP4 Tests (BnB fp4 — dequantizes to BF16 at compute time)
# ===========================================================================

@pytest.fixture(scope="module")
def fp4_model():
    """Load granite-switch from HF with BitsAndBytes FP4 quantization."""
    bitsandbytes = pytest.importorskip("bitsandbytes")  # noqa: F841
    import granite_switch.hf  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


class TestFP4AdapterActivation:
    """FP4 (BnB fp4): adapters must activate."""

    @pytest.mark.parametrize("case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})")
    def test_adapter_activates(self, fp4_model, case):
        model, tokenizer = fp4_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model, tokenizer, case["messages"],
            adapter_name=case["adapter_name"], documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under FP4.\n"
            f"Base output:    {repr(base_out[:100])}\n"
            f"Adapter output: {repr(adapter_out[:100])}"
        )


class TestFP4LoRAPrecision:
    """FP4: LoRA weights must remain in full precision."""

    def test_lora_weights_full_precision(self, fp4_model):
        model, _ = fp4_model
        full_precision_dtypes = {torch.float16, torch.bfloat16, torch.float32}
        bad_params = []
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                if param.dtype not in full_precision_dtypes:
                    bad_params.append(f"{name}: {param.dtype}")
        assert not bad_params, (
            f"LoRA params quantized under FP4 (should stay full precision):\n"
            + "\n".join(bad_params[:10])
        )


class TestFP4BaseQuantized:
    """FP4: base linear layers must actually be quantized."""

    def test_base_layers_are_4bit(self, fp4_model):
        model, _ = fp4_model
        quantized_count = 0
        for name, module in model.named_modules():
            if "Linear4bit" in type(module).__name__:
                quantized_count += 1
        assert quantized_count > 0, (
            "No Linear4bit modules found — FP4 quantization did not apply."
        )
        print(f"\n  FP4: {quantized_count} layers quantized to 4-bit")
