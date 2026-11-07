# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch through a LIVE vLLM engine, generating real tokens.

The last uncovered path. Everything else either mocks the switch config, or runs a
single HF ``forward`` and inspects ``_last_adapter_indices``:

  * ``tests/vllm/test_multi_switch.py`` drives the switch in a subprocess harness
    with a mock config -- no engine, no generation.
  * ``tests/hf/test_multi_switch_e2e.py`` uses a real composed checkpoint but the
    HF backend.

Nothing put a composed MultiSwitch checkpoint through ``llm.generate``. That
matters because two failure modes are only reachable through a real serving path:
the codebook loading as all zeros (needs a checkpoint round trip) and per-request
counting anchors (needs a real flattened batch). A mock-config harness cannot see
either.

What these tests assert:
  1. the engine serves a MultiSwitch checkpoint at all (registration + load);
  2. generation is deterministic under greedy sampling;
  3. **batching does not change any request's output** -- solo vs co-batched, which
     is the observable consequence of correct per-request anchoring;
  4. different control tokens produce different continuations, i.e. routing really
     selects distinct adapters instead of silently collapsing to base.

(4) is the guard against a vacuous pass: if the switch were dead (all-base) every
adapter would emit identical text and (1)-(3) would still pass.

Boots vLLM IN-PROCESS (this package, unlike ``tests/vllm/``, does not keep CUDA
out of the parent process). Reuses the checkpoint composed by the HF e2e module
via ``GRANITE_SWITCH_E2E_DIR``, so it costs an engine boot, not a compose.

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

    # gpu_memory_utilization is below the default 0.9, which collides with
    # allocator/pytest overhead on an 80 GB card.
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


def _gen_with_logprobs(llm, seqs):
    """Generate greedily, returning (token_ids, per-step sorted logprob list)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        temperature=0.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True, logprobs=5
    )
    prompts = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
    out = []
    for o in llm.generate(prompts, sampling_params=sp):
        comp = o.outputs[0]
        steps = [
            sorted((lp.logprob for lp in d.values()), reverse=True)
            for d in (comp.logprobs or [])
        ]
        out.append(([int(t) for t in comp.token_ids], steps))
    return out


def test_batched_generation_matches_solo(engine, control_ids, tokenizer):
    """Each request must route the same alone as when co-batched.

    This is the observable consequence of correct per-request counting anchors.
    With a global anchor, a request's routing depends on what else is in the flat
    batch, and its output drifts. Prompt lengths are distinct so requests are not
    accidentally interchangeable.

    Tie-tolerant rather than token-exact, and the tolerance is measured, not
    assumed. A token-exact assertion here is not a statement about routing: vLLM
    reduces a co-batched request's logits in a different order than a solo one, so
    a position where the top two candidates are already equal in bf16 can flip on
    sub-ULP noise. Measured on an A100 (vLLM 0.26, bf16) the third request does
    exactly that at step 8 -- solo's top-2 logprobs are the *same bf16 value*
    (-1.07067, -1.07067; gap 0.000000), so which one greedy picks is arbitrary and
    both answers are equally correct.

    Note the earlier attempt recorded above ``_QUESTIONS``: real-language prompts
    replaced junk filler to get off near-ties, and a routing probe confirmed
    routing was already identical in both runs. That reduced the tie rate but
    could not remove it, because no prompt guarantees a strictly-separated argmax
    at every position. So the tie is excluded by *measurement* instead: a
    divergence is accepted only where the top-2 gap is zero.

    What that still catches is the bug the test exists for. A request whose anchor
    came from the wrong place in the flat batch retrieves a different adapter and
    computes a genuinely different distribution, so its divergence sits at a
    separated argmax (nonzero gap) -- which fails here. Only coin-flips pass.
    """
    a, b, c = control_ids
    seqs = [_prompt(tokenizer, q, t) for q, t in zip(_QUESTIONS, (a, b, c))]

    solo = [_gen_with_logprobs(engine, [s])[0] for s in seqs]
    batched = _gen_with_logprobs(engine, seqs)

    real = []
    for i, ((s_toks, s_steps), (b_toks, _)) in enumerate(zip(solo, batched)):
        div = next(
            (j for j in range(min(len(s_toks), len(b_toks))) if s_toks[j] != b_toks[j]),
            None,
        )
        if div is None:
            continue
        # Gap between the best and second-best candidate in the SOLO run at the
        # position that diverged. Exactly 0.0 means the two are indistinguishable
        # in bf16, so the pick is a coin flip; anything above that is a real
        # difference in the distribution and therefore a routing suspect.
        top = s_steps[div] if div < len(s_steps) else []
        gap = (top[0] - top[1]) if len(top) > 1 else float("inf")
        print(f"req {i}: diverged at step {div}, solo top-2 gap {gap:.6f}")
        if gap > 0.0:
            real.append((i, div, gap, s_toks, b_toks))

    assert not real, (
        "batched generation diverged from solo at a position where the argmax was "
        "strictly separated, so it is not a tie-break -- suspect per-request "
        "counting anchors:\n"
        + "\n".join(
            f"  req {i}: step {div}, solo top-2 gap {gap:.6f}\n"
            f"    solo={s}\n    batched={b}"
            for i, div, gap, s, b in real
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
