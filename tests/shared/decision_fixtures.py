# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the decision-adapter tests.

``REFERENCE_BLOCK`` is the io.yaml ``decision:`` block the Granite Switch 4.1-8b
decision adapter needs (the layout it was trained on). ``install_fake_vllm`` puts a
minimal fake of vLLM 0.32's structured-decisions modules into ``sys.modules`` so the
real ``GraniteSwitchDecisionStrategy`` runs on CPU; ``load_strategy_module`` imports
``granite_switch/vllm/structured_decisions.py`` by path so the vLLM backend package
(torch, the model stack) is not pulled in.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

MARKER = "### DECISION"

REFERENCE_BLOCK = {
    "activation": "marker",
    "invocation_marker": MARKER,
    "messages": [
        {"role": "user", "content": "State:\n{state}"},
        {
            "role": "user",
            "content": "{marker}\nQuestion:\n{instructions}\n\nOptions:\n{options}"
            "\n\nReply with exactly one option label.",
        },
    ],
    "labels": "letters",
    "question_types": {
        "choice": {},
        "noul": {
            "order": ["no", "yes"],
            "fallback_text": {"no": "No / false", "yes": "Yes / true"},
        },
        "score": {},
    },
}

STRATEGIES = "vllm.entrypoints.generate.structured_decisions.strategies"
QUESTION_TYPES = "vllm.entrypoints.generate.structured_decisions.question_types"
LABEL_READS = "vllm.entrypoints.generate.label_reads"


def reference_block(**overrides) -> dict:
    block = copy.deepcopy(REFERENCE_BLOCK)
    block.update(overrides)
    return block


def load_strategy_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "src/granite_switch/vllm/structured_decisions.py"
    )
    spec = importlib.util.spec_from_file_location("gs_structured_decisions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStructuredDecisionError(ValueError):
    pass


def install_fake_vllm(monkeypatch, tok, winners=()):
    """Install the fake modules; ``winners[k]`` is the label token id the k-th engine
    read favours. Returns the module dict (``[LABEL_READS].seen`` holds the engine
    inputs, each ``{"messages", "request", "prompt_ids"}``)."""
    winners = list(winners)
    strategies = types.ModuleType(STRATEGIES)
    strategies.READ_STRATEGIES = {}

    class ReadStrategy:
        def __init__(self, context):
            self.context = context

    class NextTokenStrategy(ReadStrategy):
        def __init__(self, context):
            raise AssertionError("the strategy must skip vLLM's own label probe")

        def _tokenizer(self):
            return tok

        def _read_request(self, chat_template_kwargs, cache_salt=None):
            if (chat_template_kwargs or {}).get("enable_thinking"):
                raise FakeStructuredDecisionError("thinking must be off")
            return {
                "chat_template_kwargs": chat_template_kwargs,
                "cache_salt": cache_salt,
            }

        async def _render(self, read_request, messages):
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **read_request["chat_template_kwargs"],
            )
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
            return {
                "messages": messages,
                "request": read_request,
                "prompt_ids": ids,
            }, ids

        def _read_input(self, engine_input, prompt_ids):
            return engine_input

        def _sampling_params(self, label_ids, prompt_ids):
            return label_ids

    def prompt_tail(tokenizer, prompt_ids):
        added = set(tokenizer.get_added_vocab().values())
        start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
        return list(prompt_ids[start:])

    strategies.DecisionLimits = type("DecisionLimits", (SimpleNamespace,), {})
    strategies.QuestionRead = type("QuestionRead", (SimpleNamespace,), {})
    strategies.ReadStrategy = ReadStrategy
    strategies.NextTokenStrategy = NextTokenStrategy
    strategies.ReadContext = SimpleNamespace
    strategies.prompt_tail = prompt_tail

    question_types = types.ModuleType(QUESTION_TYPES)
    question_types.StructuredDecisionError = FakeStructuredDecisionError

    def label_softmax(logprobs):
        top = max(logprobs)
        w = [math.exp(lp - top) for lp in logprobs]
        return [x / sum(w) for x in w]

    question_types.label_softmax = label_softmax

    label_reads = types.ModuleType(LABEL_READS)
    label_reads.seen = []

    async def next_token_label_reads(engine, engine_inputs, params, request_id, **kw):
        label_reads.seen.extend(engine_inputs)
        out = []
        for k, label_ids in enumerate(params):
            win = winners[k] if k < len(winners) else label_ids[0]
            lps = [math.log(0.9) if t == win else math.log(0.05) for t in label_ids]
            out.append(
                SimpleNamespace(
                    logprobs=lps,
                    result=SimpleNamespace(
                        outputs=[SimpleNamespace(token_ids=[win])],
                        prompt_token_ids=[1, 2, 3],
                    ),
                )
            )
        return out

    label_reads.next_token_label_reads = next_token_label_reads

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.entrypoints": types.ModuleType("vllm.entrypoints"),
        "vllm.entrypoints.generate": types.ModuleType("vllm.entrypoints.generate"),
        "vllm.entrypoints.generate.structured_decisions": types.ModuleType(
            "vllm.entrypoints.generate.structured_decisions"
        ),
        STRATEGIES: strategies,
        QUESTION_TYPES: question_types,
        LABEL_READS: label_reads,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules


def build_strategy(sd, hf_config, chat_template=None):
    """The real strategy class over the installed fake modules."""
    context = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=hf_config)
        ),
        chat_template=chat_template,
        default_chat_template_kwargs={},
    )
    return sd._build_strategy_class()(context)


def question(qid, qtype, options, instructions="pick one"):
    return SimpleNamespace(
        id=qid,
        type=SimpleNamespace(name=qtype),
        instructions=instructions,
        options=[SimpleNamespace(name=n, description=d) for n, d in options],
    )


def run_read(
    strategy, questions, instructions=None, state="the state", kwargs=None, media=None
):
    import asyncio

    return asyncio.run(
        strategy.read(
            questions,
            instructions,
            state,
            request_id="r",
            chat_template_kwargs=kwargs,
            lora_request=None,
            priority=0,
            media=media,
        )
    )
