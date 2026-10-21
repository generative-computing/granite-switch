# SPDX-License-Identifier: Apache-2.0
"""Quantization tests for GraniteSwitch HF backend.

Verifies that quantization:
1. Preserves adapter activation (output differs with/without adapter via chat template)
2. Keeps LoRA/aLoRA weights in full precision (bfloat16)
3. Actually quantizes base model linear layers

Quantization methods tested:
- BitsAndBytes: NF4 (INT4), FP4
- Quanto: INT4, FP8

GATING. This module runs wherever it *can* run — there is no opt-in env flag. It
used to sit behind ``GRANITE_SWITCH_E2E_MODELS=1`` like the e2e suites, which meant
all 16 tests skipped on every CI run and the quantization path was never exercised
at all. What actually constrains them is capability, not cost policy, so each
requirement is now checked where it applies and reported honestly:

- CUDA: module-level skip (every backend here needs a GPU).
- ``bitsandbytes`` / ``optimum-quanto``: ``importorskip`` in the owning fixture, so
  a missing backend skips only its own tests.
- Quanto **float8**: a real probe (``_probe_quanto_float8``) — see
  ``require_quanto_float8`` for why a capability check is needed and what it
  catches. These four used to ERROR in fixture setup on an A100.

Model: a real MultiSwitch checkpoint composed on demand from granitelib-rag, which
carries both an aLoRA intrinsic (``answerability``) and a LoRA one
(``hallucination_detection``), warm-reused under ``GRANITE_SWITCH_E2E_DIR``. The
published previews are legacy SingleSwitch checkpoints and no longer load, so the
suite composes its own — same pattern as ``test_multi_switch_mixed_tech.py``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="every quantization backend here needs a CUDA GPU",
    ),
]

BASE_MODEL = "ibm-granite/granite-4.1-3b"
# granitelib-rag carries BOTH adapters ADAPTER_TESTS exercises: answerability as
# an aLoRA and hallucination_detection as a LoRA. Composing rag alone (no filter,
# so the alora>lora preference is kept) yields a naturally mixed checkpoint with
# exactly those names. A dedicated output dir (not the shared "multi-mixed") is
# deliberate: that shared checkpoint may already exist composed from other
# libraries, whose adapter names would not match ADAPTER_TESTS.
ADAPTER_REPOS = ["ibm-granite/granitelib-rag-r1.0"]
_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))


@pytest.fixture(scope="module")
def model_path():
    """Compose (or warm-reuse) a rag MultiSwitch checkpoint; return its dir.

    The compose is the one genuinely expensive thing in this module (a ~3B base
    model plus adapters). It is NOT behind an opt-in flag: `make test-e2e` is the
    way this module is meant to be run, and the warm-reuse below makes a second run
    free. ``GRANITE_SWITCH_E2E_DIR`` relocates the cache.
    """
    out_dir = _E2E_ROOT / "quant-rag"
    if (out_dir / "config.json").exists():
        print(f"warm-reuse {out_dir}", file=sys.stderr)
        return str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        *[arg for r in ADAPTER_REPOS for arg in ("--adapters", r)],
        "--output",
        str(out_dir),
    ]
    print("composing (mixed):", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        pytest.fail(
            f"mixed compose failed (exit {r.returncode})\n"
            f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}",
            pytrace=False,
        )
    return str(out_dir)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ADAPTER_TESTS = [
    {
        "adapter_name": "hallucination_detection",
        "type": "lora",
        "messages": [
            {"role": "user", "content": "What is photosynthesis?"},
            {
                "role": "assistant",
                "content": "Photosynthesis converts sunlight into glucose.",
            },
            {
                "role": "user",
                "content": (
                    "<guardian>You are a judge agent. Your role is to assess whether "
                    "the provided text meets the given criteria.\n\n"
                    "### Criteria: A factually incorrect response.\n\n"
                    "### Scoring Schema: If the last assistant's text meets the "
                    "criteria, return 'yes'; otherwise, return 'no'."
                ),
            },
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


def _generate(
    model, tokenizer, messages, adapter_name=None, documents=None, max_new_tokens=30
):
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
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bnb_model(model_path):
    """Load granite-switch from HF with BitsAndBytes NF4 quantization."""
    bitsandbytes = pytest.importorskip("bitsandbytes")  # noqa: F841
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import granite_switch.hf  # noqa: F401

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def quanto_model(model_path):
    """Load granite-switch from HF with Quanto INT4 quantization."""
    pytest.importorskip("optimum.quanto")
    from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

    import granite_switch.hf  # noqa: F401

    quanto_config = QuantoConfig(weights="int4")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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

    @pytest.mark.parametrize(
        "case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})"
    )
    def test_adapter_activates(self, bnb_model, case):
        model, tokenizer = bnb_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model,
            tokenizer,
            case["messages"],
            adapter_name=case["adapter_name"],
            documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under BnB NF4.\n"
            f"Base output:    {base_out[:100]!r}\n"
            f"Adapter output: {adapter_out[:100]!r}"
        )


class TestQuantoAdapterActivation:
    """Quanto INT4: adapters must activate (output differs with adapter via chat template)."""

    @pytest.mark.parametrize(
        "case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})"
    )
    def test_adapter_activates(self, quanto_model, case):
        model, tokenizer = quanto_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model,
            tokenizer,
            case["messages"],
            adapter_name=case["adapter_name"],
            documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under Quanto INT4.\n"
            f"Base output:    {base_out[:100]!r}\n"
            f"Adapter output: {adapter_out[:100]!r}"
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
            "LoRA params quantized under BnB (should stay full precision):\n"
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
            "LoRA params quantized under Quanto (should stay full precision):\n"
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


# The probe body. Deliberately contains NO granite-switch code: it quantizes a bare
# nn.Linear, so a failure can only come from optimum-quanto/CUDA, never from us.
# ``freeze()`` is the operative call — that is what materializes the quantized
# weight and what JIT-builds the ``quanto_cuda`` extension.
_FLOAT8_PROBE = """
import torch
from optimum.quanto import freeze, qfloat8, quantize

m = torch.nn.Linear(64, 64).cuda()
quantize(m, weights=qfloat8)
freeze(m)
m(torch.randn(2, 64, device="cuda"))
print("FLOAT8_OK")
"""


def _probe_quanto_float8():
    """Ask a subprocess whether Quanto can actually quantize to float8 here.

    A subprocess for two reasons: the extension build and a CUDA context must not
    be inherited into the test session, and a hard abort (``ptxas`` has been seen
    to abort rather than raise) would otherwise take the whole run down.

    Returns ``(ok, reason)``.
    """
    res = subprocess.run(
        [sys.executable, "-c", _FLOAT8_PROBE],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if "FLOAT8_OK" in res.stdout:
        return True, ""
    tail = (res.stderr or res.stdout or "").strip().splitlines()
    return False, " | ".join(tail[-3:])[:400] or f"exit {res.returncode}"


@pytest.fixture(scope="module")
def require_quanto_float8():
    """Skip the FP8 tests unless Quanto float8 works on this machine.

    These four tests used to ERROR rather than skip. The visible exception was a
    transformers ``RuntimeError: issues during automatic conversion of the weights``
    naming ``base_layer.weight``, which reads like a granite-switch LoRA fault and is
    not one: the real failure is inside ``quanto.convert → freeze() →
    quantize_weight``, where optimum-quanto JIT-builds its ``quanto_cuda`` extension
    and ``ptxas`` aborts with ``Feature 'cp.async' requires .target sm_80 or higher``
    while compiling ``compute_75.ptx``. Reproduced on an A100 (sm_80) with a bare
    ``nn.Linear`` and no granite-switch code at all.

    Note the failing object is ``marlin_cuda.o`` — an INT4 kernel — because
    ``quanto_cuda`` bundles marlin+awq+unpack and builds all-or-nothing, so float8's
    ``freeze()`` pays for compiling a kernel it never uses. That is why INT4 can be
    green while float8 is not.

    The probe does not care which of those is the root cause: it asks the only
    question that matters to these tests — can this machine freeze a float8 linear —
    and skips with the real error when it cannot. A green result here therefore means
    "float8 verified", and a skip means "float8 unavailable", never a misattributed
    LoRA failure.
    """
    pytest.importorskip("optimum.quanto")
    ok, reason = _probe_quanto_float8()
    print(f"\nQuanto float8 skip-guard: ok={ok} reason={reason!r}")
    if not ok:
        pytest.skip(f"Quanto cannot quantize to float8 on this machine: {reason}")


@pytest.fixture(scope="module")
def fp8_model(model_path, require_quanto_float8):
    """Load granite-switch from HF with Quanto FP8 quantization."""
    pytest.importorskip("optimum.quanto")
    from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

    import granite_switch.hf  # noqa: F401

    quanto_config = QuantoConfig(weights="float8")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quanto_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


class TestFP8AdapterActivation:
    """FP8 (Quanto float8): adapters must activate."""

    @pytest.mark.parametrize(
        "case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})"
    )
    def test_adapter_activates(self, fp8_model, case):
        model, tokenizer = fp8_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model,
            tokenizer,
            case["messages"],
            adapter_name=case["adapter_name"],
            documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under FP8.\n"
            f"Base output:    {base_out[:100]!r}\n"
            f"Adapter output: {adapter_out[:100]!r}"
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
            "LoRA params quantized under FP8 (should stay full precision):\n"
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
def fp4_model(model_path):
    """Load granite-switch from HF with BitsAndBytes FP4 quantization."""
    bitsandbytes = pytest.importorskip("bitsandbytes")  # noqa: F841
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import granite_switch.hf  # noqa: F401

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="fp4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


class TestFP4AdapterActivation:
    """FP4 (BnB fp4): adapters must activate."""

    @pytest.mark.parametrize(
        "case", ADAPTER_TESTS, ids=lambda c: f"{c['adapter_name']}({c['type']})"
    )
    def test_adapter_activates(self, fp4_model, case):
        model, tokenizer = fp4_model
        documents = case.get("documents")
        base_out = _generate(model, tokenizer, case["messages"], documents=documents)
        adapter_out = _generate(
            model,
            tokenizer,
            case["messages"],
            adapter_name=case["adapter_name"],
            documents=documents,
        )
        assert base_out != adapter_out, (
            f"Adapter {case['adapter_name']} did not activate under FP4.\n"
            f"Base output:    {base_out[:100]!r}\n"
            f"Adapter output: {adapter_out[:100]!r}"
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
            "LoRA params quantized under FP4 (should stay full precision):\n"
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
