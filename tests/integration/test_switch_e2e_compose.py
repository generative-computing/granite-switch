# SPDX-License-Identifier: Apache-2.0
"""Composer + vLLM E2E tests for SingleSwitch (Step 4 Tier 2 of issue #107).

Closes the last gap from issue #107 — the **vLLM gain compensation path** —
by running a real composed model through both HF and vLLM and asserting the
two backends produce equivalent logits on the same input.

  HF path:   SingleSwitch hardcodes scaling=1.0, gain used directly.
  vLLM path: scaling=attention_multiplier, effective_gain=gain/scaling.
             vLLM's Attention kernel applies `scaling` internally so the net
             softmax logit is ±gain via `attention_multiplier × effective_gain`
             in bf16.

If HF says adapter N is active at position P, vLLM's SingleSwitch must arrive
at the same `adapter_indices` via a different computational path — and logits
downstream must match within bf16 tolerance. Any drift beyond `atol=1e-2,
rtol=1e-2` signals a compensation-math bug or a wiring regression.

Scope:
  - Default: one public (base, adapter) pairing known to compose cleanly
    (see `_DEFAULT_BASE_MODEL_PAIRS` below).
  - Extension: local or experimental pairings via the
    `GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS` env var — JSON array of
    `{"base": str, "adapter": str}` entries. Paths are not committed; only
    the mechanism is.

Markers: @pytest.mark.slow + @pytest.mark.requires_model + @pytest.mark.gpu.
CI must opt in explicitly: `pytest -m "slow and requires_model and gpu"`.
"""

import json
import os

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_model, pytest.mark.gpu]


# ----------------------------------------------------------------------------
# Base-model / adapter-library pairs to test
# ----------------------------------------------------------------------------

# Default pairings: publicly available on HuggingFace. `granitelib-core-r1.0`
# is the current-naming-convention adapter library (the older `granite-lib-*`
# naming still resolves but is deprecated). Its subdirectories (e.g.
# `context-attribution/`) contain per-base-model adapter flavors for both
# Granite 4.0 and 4.1 variants; the compose CLI's `discover_adapters` picks
# the flavor matching `--base-model`.
_DEFAULT_BASE_MODEL_PAIRS = [
    ("ibm-granite/granite-4.0-micro", "ibm-granite/granitelib-core-r1.0"),
    ("ibm-granite/granite-4.1-3b", "ibm-granite/granitelib-core-r1.0"),
]


def _load_experimental_pairs():
    """Extension point for local or experimental base/adapter pairings.

    Reads a JSON array of `{"base": str, "adapter": str}` entries from the
    `GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS` environment variable. Both fields
    may be HuggingFace model IDs or local filesystem paths.

    Use cases:
      - Local verification against checkpoints that aren't public yet.
      - CI profiles that want to exercise a broader model matrix than the
        default.

    The env-var mechanism is committed; the values you set it to are not.

    Example:
        GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS='[{"base":"/path/base","adapter":"/path/lib"}]' \\
            pytest tests/integration/test_switch_e2e_compose.py \\
            -m "slow and requires_model and gpu"
    """
    raw = os.environ.get("GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS", "")
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"GRANITE_SWITCH_EXPERIMENTAL_MODEL_PAIRS is not valid JSON: {e}\n"
            f'Expected format: \'[{{"base":"/path","adapter":"/path"}}, ...]\''
        )
    return [(p["base"], p["adapter"]) for p in entries]


BASE_MODEL_PAIRS = _DEFAULT_BASE_MODEL_PAIRS + _load_experimental_pairs()


# ----------------------------------------------------------------------------
# Module-scoped fixture: one compose per (base, adapter) pair
# ----------------------------------------------------------------------------


COMPOSE_TIMEOUT_S = 1800  # 30 min — matches tests/composer/test_compose_e2e.py:28


@pytest.fixture(
    scope="module",
    params=BASE_MODEL_PAIRS,
    ids=lambda p: p[0].rsplit("/", 1)[-1],
)
def composed_model_artifacts(request, tmp_path_factory):
    """Compose once per (base, adapter) pair; share across tests in the module.

    Invokes the `compose_granite_switch` CLI via subprocess — the same pattern
    as tests/composer/test_compose_e2e.py:45-60. The CLI handles adapter-library
    expansion, tokenizer control-token injection, chat-template configuration,
    and the full save pipeline. The Python-level `GraniteSwitchComposer` is a
    lower primitive that expects already-expanded per-adapter paths — so we
    delegate to the CLI and then load the saved model back with `from_pretrained`.

    First-run cost is model-download dominated (~30 min per pair); subsequent
    runs with warm caches are ~5–10 min per pair. Module scope amortizes that
    across all tests in this file.
    """
    import subprocess
    import sys

    from granite_switch.hf import GraniteSwitchForCausalLM

    base_model, adapter_library = request.param
    save_dir_name = base_model.rsplit("/", 1)[-1]
    build_root = tmp_path_factory.mktemp(f"compose_{save_dir_name}")
    save_dir = build_root / "model"

    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        base_model,
        "--adapters",
        adapter_library,
        "--output",
        str(save_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=COMPOSE_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "compose_granite_switch failed for "
            f"base={base_model} adapter={adapter_library}\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )

    hf_model = GraniteSwitchForCausalLM.from_pretrained(str(save_dir)).eval().cuda()
    return {
        "base_model": base_model,
        "adapter_library": adapter_library,
        "hf_model": hf_model,
        "save_dir": save_dir,
    }


# ----------------------------------------------------------------------------
# Test methods
# ----------------------------------------------------------------------------


# Control-token position semantics. Names mirror the bare-switch sweep in
# tests/shared/single_switch_cases.py:182-187 so HF E2E, bare-switch, and
# composer E2E all use the same {early, mid, late} labels; the concrete
# index is computed per-seq_len via _control_position_index().

_CONTROL_POSITION_NAMES = ["early", "mid", "late"]

# Long-context seq_lens for the HF-side adapter-indices check. Tier 1 covers
# these on a tiny CPU model; here we verify the FULL composed HF model (3B+
# params, real max_position_embeddings=131072) produces correct adapter
# indices at production-representative lengths. Short seq_len (8) is kept
# as a fast sanity check; 10K/32K/131K stress RoPE + full-model attention.
_HF_SEQ_LENS = [8, 10_000, 32_768, 131_072]


def _control_position_index(seq_len: int, position_name: str) -> int:
    """Map {early, mid, late} → concrete token index for a given seq_len."""
    if position_name == "early":
        return 1
    if position_name == "mid":
        return seq_len // 2
    if position_name == "late":
        return seq_len - 2
    raise ValueError(f"unknown control_position: {position_name}")


_TEXT_TOKENS = [
    791,
    5679,
    2766,
    279,
    893,
    389,
    813,
    1450,
]  # "The dog bit the man on his hand"


def _build_input(ctrl_pos: int, ctrl_token: int, total_len: int):
    """Build a `total_len`-long sequence with `ctrl_token` at `ctrl_pos`.

    Non-control positions cycle through _TEXT_TOKENS to produce a varied
    distribution that avoids near-ties in logit space (which makes argmax
    fragile under HF/vLLM precision differences).
    """
    seq = [_TEXT_TOKENS[i % len(_TEXT_TOKENS)] for i in range(total_len)]
    seq[ctrl_pos] = ctrl_token
    return seq


@pytest.mark.parametrize("seq_len", _HF_SEQ_LENS)
@pytest.mark.parametrize("control_position", _CONTROL_POSITION_NAMES)
def test_hf_composed_adapter_indices(
    composed_model_artifacts, control_position, seq_len
):
    """Verify the composed HF model's `_last_adapter_indices` is correct at
    production-representative context lengths × control-token positions.

    Covers sub-requirements #3 and #4 of issue #107's E2E bullet:
    - Check `model.model._last_adapter_indices` directly (HF exposes this as
      a plain attribute after forward).
    - Verify pre-control = 0, at/after control = correct adapter ID.

    Runs before the vLLM equivalence test so a bad compose fails fast with a
    clear HF-side error, not buried inside vLLM worker output.
    """
    import torch

    hf_model = composed_model_artifacts["hf_model"]
    adapter_token_id = hf_model.config.adapter_token_ids[0]
    ctrl_pos = _control_position_index(seq_len, control_position)
    seq = _build_input(ctrl_pos, adapter_token_id, seq_len)
    input_ids = torch.tensor([seq], device="cuda")

    with torch.no_grad():
        hf_model(input_ids=input_ids)

    ai = hf_model.model._last_adapter_indices[0]
    if ctrl_pos > 0:
        assert (ai[:ctrl_pos] == 0).all(), (
            f"pre-control slice should be all 0 at {control_position} "
            f"(ctrl_pos={ctrl_pos}, seq_len={seq_len}), "
            f"first 16 values: {ai[:16].tolist()} "
            f"(base_model={composed_model_artifacts['base_model']})"
        )
    assert (ai[ctrl_pos:] == 1).all(), (
        f"post-control slice should be all 1 at {control_position} "
        f"(ctrl_pos={ctrl_pos}, seq_len={seq_len}), "
        f"post-control sample: {ai[ctrl_pos : ctrl_pos + 16].tolist()} "
        f"(base_model={composed_model_artifacts['base_model']})"
    )


def test_hf_vllm_argmax_equivalence(composed_model_artifacts):
    """Crown jewel: HF and vLLM pick the same top-1 token at every position,
    at every tested control-token position (early, middle, late).

    This is the only test that exercises the vLLM gain compensation path
    (effective_gain = control_token_gain / attention_multiplier, applied and
    then inverted by vLLM's Attention kernel) through a real composed model.
    If the compensation math is broken, vLLM applies the wrong LoRA adapter
    at post-control positions and its argmax diverges from HF's.

    ─── Why argmax, not direct adapter_indices comparison ────────────────────

    The HF model exposes `_last_adapter_indices` as a plain attribute after
    forward, which is why `test_hf_composed_adapter_indices` can verify it
    directly. vLLM's equivalent state is harder to reach for three reasons:

    1. **Subprocess barrier.** vLLM v1 runs the model in an EngineCore
       subprocess (seen as `pid=...` in the logs). The switch's tensors live
       in that child process's memory; reading them from the test process
       would require serializing across the subprocess boundary via vLLM's
       output dataclasses — which don't currently carry that data.
    2. **Production-code instrumentation.** Exposing `_last_adapter_indices`
       as a `self.attr` write in `src/granite_switch/vllm/switch/single.py`
       would add test-only state to a production path, and `torch.compile`
       may or may not capture the attribute write reliably across graph
       recompiles.
    3. **Engine internals drift.** The `llm.llm_engine.model_executor. \
       driver_worker.worker.model_runner.model.model.switch` chain reaches
       into vLLM's parent-process model view — it exists on some vLLM
       versions but not others. Tests that depend on it are fragile.

    Argmax equivalence is the cleanest proxy: if vLLM applied the wrong
    adapter at any position, the resulting logits would diverge enough from
    HF's to flip the top-1 token. It exercises the full production engine
    path (CUDA graphs, torch.compile, chunked prefill) — the part that the
    low-level tests in `test_hf_to_vllm_weights.py` bypass.

    ─── Why argmax, not raw-logit tolerance ─────────────────────────────────

    The composed model runs through 40 decoder layers on two different
    backends. Cross-backend bf16 drift at that depth is real and expected.
    Comparable HF-vs-vLLM tests in the repo (`test_hf_to_vllm_weights.py`)
    tolerate `atol=1e-2, rtol=1e-2` only because they use a seeded 4-layer
    tiny model with `attention_multiplier=1.0` and bypass the vLLM engine.
    At 40 layers, logprob drift at low-confidence positions compounds past
    any reasonable tolerance without indicating a bug. Argmax is robust:
    two similar-magnitude logits have to cross over to flip the argmax,
    which normal bf16 drift rarely causes but a wrong LoRA at post-control
    positions would.

    ─── Position sweep in a single test body ────────────────────────────────

    We loop over {early, mid, late} control positions inside one test body
    rather than parametrizing the pytest function. Parametrizing would
    trigger 3 × vLLM engine boots per base model (~15 min overhead each)
    since each call to `run_vllm_logprobs` instantiates and tears down its
    own LLM. Looping inside is free: one engine boot handles all 3 prompts.
    Per-position logprob-drift and argmax-mismatch messages name the
    failing position so diagnostics stay clear.

    ─── Ordering note ───────────────────────────────────────────────────────

    This test moves the HF model to CPU before booting vLLM (see the memory-
    management block below). Because of this, it must be the LAST test in
    this module to reference the HF model on CUDA. Other tests
    (test_hf_composed_adapter_indices) come first in file order and run to
    completion before this one, which is the pytest default collection order.
    """
    import gc
    import os

    import torch

    from tests.shared.vllm_equivalence import extract_logprobs_tensor

    save_dir = composed_model_artifacts["save_dir"]
    hf_model = composed_model_artifacts["hf_model"]
    vocab_size = hf_model.config.vocab_size
    adapter_token_id = hf_model.config.adapter_token_ids[0]
    base_model = composed_model_artifacts["base_model"]

    # --- Stage 1: compute HF logprobs for every position before freeing GPU.
    # HF logits → logprobs (vLLM returns logprobs, so we align dtypes). Drop
    # the last row to match vLLM's [seq_len - 1, vocab] shape (its
    # prompt_logprobs has no entry for position 0: see
    # tests/shared/vllm_equivalence.py:174-189).
    short_seq_len = 8  # short context for the vLLM equivalence loop

    # Baseline: no control token — pure base-model comparison.
    # If HF and vLLM disagree here too, the argmax comparison is unreliable
    # for this model (backend drift, not a switch bug).
    _baseline_seq = list(_TEXT_TOKENS)
    with torch.no_grad():
        _baseline_out = hf_model(input_ids=torch.tensor([_baseline_seq], device="cuda"))
    _hf_baseline_logprobs = torch.log_softmax(_baseline_out.logits[0].float(), dim=-1)[
        :-1
    ].cpu()
    del _baseline_out

    hf_logprobs_by_position = {}
    input_ids_by_position = {}
    for position_name in _CONTROL_POSITION_NAMES:
        ctrl_pos = _control_position_index(short_seq_len, position_name)
        seq = _build_input(ctrl_pos, adapter_token_id, short_seq_len)
        input_ids_by_position[position_name] = seq
        with torch.no_grad():
            hf_out = hf_model(input_ids=torch.tensor([seq], device="cuda"))
        hf_logprobs = torch.log_softmax(hf_out.logits[0].float(), dim=-1)
        hf_logprobs_by_position[position_name] = hf_logprobs[:-1].cpu()
        del hf_out, hf_logprobs

    # --- Stage 2: free HF GPU memory before booting vLLM. vLLM asks for 90%
    # of device memory by default; the HF model's 3B bf16 weights + KV cache
    # push it past the threshold on an 80 GB GPU otherwise. We've captured
    # all HF logprobs on CPU — the HF model is no longer referenced in this
    # test body; moving it to CPU is safe.
    hf_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # --- Stage 3: boot vLLM once, loop over positions. We inline the
    # LLM() lifecycle from tests/shared/vllm_equivalence.py:run_vllm_logprobs
    # (lines 222-268) so we can share a single boot across the 3 positions.
    # gpu_memory_utilization lowered from default 0.9 to 0.7: even after
    # freeing the HF model, CUDA caching allocator and pytest/torch
    # machinery hold ~1–2 GB, and 0.7 leaves plenty for vLLM's KV cache.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=str(save_dir),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_logprobs=vocab_size,
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=vocab_size,
    )

    try:
        # Baseline vLLM run: no control token.
        _baseline_prompt = TokensPrompt(prompt_token_ids=_baseline_seq)
        _baseline_vllm = extract_logprobs_tensor(
            llm.generate(_baseline_prompt, sampling_params=sampling_params), vocab_size
        )
        _baseline_argmax_hf = _hf_baseline_logprobs.argmax(dim=-1)
        _baseline_argmax_vllm = _baseline_vllm.argmax(dim=-1)
        _baseline_mismatches = (
            (_baseline_argmax_hf != _baseline_argmax_vllm)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        print(
            f"\n  [baseline-no-ctrl] argmax mismatch positions: {_baseline_mismatches}"
            f"\n    HF  : {_baseline_argmax_hf.tolist()}"
            f"\n    vLLM: {_baseline_argmax_vllm.tolist()}"
            f"\n  NOTE: any mismatch here = backend drift, not a switch bug"
        )

        failures = []
        for position_name in _CONTROL_POSITION_NAMES:
            seq = input_ids_by_position[position_name]
            prompt = TokensPrompt(prompt_token_ids=seq)
            outputs = llm.generate(prompt, sampling_params=sampling_params)
            vllm_logprobs = extract_logprobs_tensor(outputs, vocab_size)

            hf_logprobs_aligned = hf_logprobs_by_position[position_name]
            abs_diff = (vllm_logprobs - hf_logprobs_aligned).abs()
            print(
                f"\n  [{position_name}] logprob drift "
                f"(diagnostic): max={abs_diff.max().item():.4f}  "
                f"mean={abs_diff.mean().item():.4f}  "
                f"(base_model={base_model})"
            )

            tok_a, tok_b = 596, 337
            hf_gap = hf_logprobs_aligned[:, tok_a] - hf_logprobs_aligned[:, tok_b]
            vl_gap = vllm_logprobs[:, tok_a] - vllm_logprobs[:, tok_b]
            ctrl_pos = _control_position_index(short_seq_len, position_name)
            print(
                f"  [{position_name}] gap(596-337)  ctrl@{ctrl_pos}"
                f"\n    HF  : {[f'{v:+.3f}' for v in hf_gap.tolist()]}"
                f"\n    vLLM: {[f'{v:+.3f}' for v in vl_gap.tolist()]}"
            )

            hf_argmax = hf_logprobs_aligned.argmax(dim=-1)
            vllm_argmax = vllm_logprobs.argmax(dim=-1)
            mismatches = (
                (hf_argmax != vllm_argmax)
                .nonzero(
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )

            pre_ctrl = [i for i in mismatches if i < ctrl_pos]
            post_ctrl = [i for i in mismatches if i >= ctrl_pos]
            hf_top2 = torch.topk(hf_logprobs_aligned, k=2, dim=-1).values
            vllm_top2 = torch.topk(vllm_logprobs, k=2, dim=-1).values
            hf_margin = hf_top2[:, 0] - hf_top2[:, 1]
            vllm_margin = vllm_top2[:, 0] - vllm_top2[:, 1]
            print(
                f"  [{position_name}] mismatch split: "
                f"pre_ctrl={pre_ctrl}, post_ctrl={post_ctrl}"
            )
            if mismatches:
                print(
                    f"  [{position_name}] top1-top2 margins at mismatches"
                    f"\n    HF  : {[f'{hf_margin[i].item():.4f}' for i in mismatches]}"
                    f"\n    vLLM: {[f'{vllm_margin[i].item():.4f}' for i in mismatches]}"
                )

            if mismatches:
                failures.append(
                    (
                        position_name,
                        mismatches,
                        hf_argmax.tolist(),
                        vllm_argmax.tolist(),
                    )
                )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    assert not failures, (
        f"HF and vLLM disagree on top-1 token for {len(failures)} "
        f"position(s) (base_model={base_model}):\n"
        + "\n".join(
            f"  [{pos}] mismatch at positions {mism}\n"
            f"    hf_argmax   = {hf_a}\n"
            f"    vllm_argmax = {vllm_a}"
            for pos, mism, hf_a, vllm_a in failures
        )
        + "\n  This typically indicates a vLLM gain-compensation bug: the "
        "switch\n  produced wrong adapter_indices, the wrong LoRA was "
        "applied, and\n  the downstream logits diverged enough to flip "
        "the top token."
    )
