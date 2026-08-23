# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch through a LIVE vLLM engine, generating real tokens.

The last uncovered path. Everything else either mocks the switch config, or runs a
single HF ``forward`` and inspects ``_last_adapter_indices``:

  * ``tests/vllm/test_multi_switch.py`` drives the switch in a subprocess harness
    with a mock config -- no engine, no generation.
  * ``tests/hf/test_multi_switch_e2e.py`` uses a real composed checkpoint but the
    HF backend.
  * ``tests/integration/test_switch_e2e_compose.py`` boots a real vLLM engine but
    only for SingleSwitch.

Nothing put a composed MultiSwitch checkpoint through ``llm.generate``. That
matters because two of this branch's three bugs were only reachable through a real
serving path: the codebook loading as all zeros (needs a checkpoint round trip) and
per-request counting anchors (needs a real flattened batch). A mock-config harness
cannot see either.

What these tests assert:
  1. the engine serves a MultiSwitch checkpoint at all (registration + load);
  2. generation is deterministic under greedy sampling;
  3. **batching does not change any request's output** -- solo vs co-batched, which
     is the observable consequence of correct per-request anchoring;
  4. different control tokens produce different continuations, i.e. routing really
     selects distinct adapters instead of silently collapsing to base.

(4) is the guard against a vacuous pass: if the switch were dead (all-base) every
adapter would emit identical text and (1)-(3) would still pass.

Boots vLLM IN-PROCESS, matching this module's sibling
``test_switch_e2e_compose.py`` (``tests/vllm/`` keeps CUDA out of the parent
process, this package does not). Reuses the checkpoint composed by the HF e2e
module via ``GRANITE_SWITCH_E2E_DIR``, so it costs an engine boot, not a compose.

Markers: slow + requires_model + gpu, and skipped unless
``GRANITE_SWITCH_E2E_MODELS=1``.
"""

import gc
import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="composes/loads a real ~3B checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]

MAX_NEW_TOKENS = 16


@pytest.fixture(scope="module")
def multi_checkpoint():
    """Path to a composed ``--switch-type multi`` checkpoint (warm-reused)."""
    # Reuse the HF e2e module's compose helper so both suites share one artifact
    # rather than composing a second 3B model.
    from tests.hf.test_multi_switch_e2e import _compose

    out_dir = _compose("multi")
    assert (out_dir / "config.json").exists(), f"no checkpoint at {out_dir}"
    return out_dir


@pytest.fixture(scope="module")
def engine(multi_checkpoint):
    """A single live vLLM engine shared by the tests in this module."""
    import torch

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM

    # gpu_memory_utilization mirrors test_switch_e2e_compose.py: the default 0.9
    # collides with allocator/pytest overhead on an 80 GB card.
    llm = LLM(
        model=str(multi_checkpoint),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.7,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def control_ids(multi_checkpoint):
    """(A_TOK, B_TOK, C_TOK) control tokens from the composed config."""
    from granite_switch.config import GraniteSwitchConfig

    config = GraniteSwitchConfig.from_pretrained(multi_checkpoint)
    atoks = list(config.adapter_token_ids)
    assert len(atoks) >= 3, f"need >=3 adapters for these tests, got {len(atoks)}"
    return atoks[0], atoks[1], atoks[2]


def _gen(llm, seqs):
    """Generate greedily for a list of token-id sequences; return token-id lists."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)
    prompts = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
    outs = llm.generate(prompts, sampling_params=sp)
    return [[int(t) for t in o.outputs[0].token_ids] for o in outs]


# Real-language prompts, NOT repeated filler tokens. A prompt of the form
# [50, 50, ..., ctrl, 50, 50, 50] is badly out of distribution, and greedy decoding
# on such input sits on near-ties: an earlier version of this module built prompts
# that way and one request produced different (equally meaningless) text solo vs
# batched. A routing probe showed the ADAPTER ROUTING was byte-identical in both
# cases -- only the sampled token broke the tie differently. Asserting on generated
# text over junk input tests the sampler, not the switch.
_QUESTIONS = [
    "Explain in one sentence what a hash function is.",
    "Briefly: what is entropy in thermodynamics?",
    "Summarize what a binary search tree does, briefly.",
]


@pytest.fixture(scope="module")
def tokenizer(multi_checkpoint):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(multi_checkpoint)


def _prompt(tokenizer, question, ctrl):
    """Real-language prompt with ``ctrl`` appended so the adapter fires last."""
    return tokenizer(question)["input_ids"] + [ctrl]


def test_engine_serves_multiswitch_and_generates(engine, control_ids, tokenizer):
    """The engine loads a MultiSwitch checkpoint and produces tokens."""
    a, _b, _c = control_ids
    (out,) = _gen(engine, [_prompt(tokenizer, _QUESTIONS[0], a)])
    assert len(out) == MAX_NEW_TOKENS, (
        f"expected {MAX_NEW_TOKENS} generated tokens, got {len(out)}"
    )


def test_generation_is_deterministic(engine, control_ids, tokenizer):
    """Greedy generation must repeat exactly -- otherwise later comparisons are noise."""
    a, _b, _c = control_ids
    seq = _prompt(tokenizer, _QUESTIONS[0], a)
    (first,) = _gen(engine, [seq])
    (second,) = _gen(engine, [seq])
    assert first == second, (
        "greedy generation is not reproducible, so any solo-vs-batched comparison "
        f"in this module would be meaningless:\n  {first}\n  {second}"
    )


def test_batched_generation_matches_solo(engine, control_ids, tokenizer):
    """Each request must generate the same tokens alone as when co-batched.

    This is the observable consequence of correct per-request counting anchors.
    With a global anchor, a request's routing depends on what else is in the flat
    batch, and its output drifts. Prompt lengths are distinct so requests are not
    accidentally interchangeable.
    """
    a, b, c = control_ids
    seqs = [_prompt(tokenizer, q, t) for q, t in zip(_QUESTIONS, (a, b, c))]

    solo = [_gen(engine, [s])[0] for s in seqs]
    batched = _gen(engine, seqs)

    mismatches = [i for i in range(len(seqs)) if solo[i] != batched[i]]
    assert not mismatches, (
        f"batched generation diverged from solo for request(s) {mismatches}:\n"
        + "\n".join(
            f"  req {i}: solo={solo[i]}\n         batched={batched[i]}"
            for i in mismatches
        )
    )


def test_different_adapters_produce_different_text(engine, control_ids, tokenizer):
    """Distinct control tokens must yield distinct continuations.

    Guards against a vacuous pass: a dead switch that routes everything to base
    would satisfy every other test in this module while producing identical output
    for every adapter. Same prompt body, only the control token differs.
    """
    a, b, c = control_ids
    seqs = [_prompt(tokenizer, _QUESTIONS[0], t) for t in (a, b, c)]
    outs = _gen(engine, seqs)

    unique = {tuple(o) for o in outs}
    assert len(unique) > 1, (
        "all three adapters generated identical text, which is what a switch stuck "
        "on base looks like -- routing is probably not selecting adapters at all:\n"
        + "\n".join(f"  ctrl {t}: {o}" for t, o in zip((a, b, c), outs))
    )
