#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Granite Switch Adapter Generation Demo (HuggingFace transformers).

Loads the open-source Granite Switch model and runs one demo per
embedded adapter, saving the results to a JSON file.

Each adapter has a dedicated ``demo_<adapter>`` function that owns its
demo inputs and the message/document layout the adapter expects.
Adapters are activated by passing ``adapter_name=...`` to
``tokenizer.apply_chat_template``, following the granite-switch README.

Usage:
    python run_adapter_generation_direct.py [--output results.json] [--max-tokens 1024]
    python run_adapter_generation_direct.py --model-dir /path/to/model

Requires: granite-switch[hf] installed. GPU recommended (CPU works but is slow).
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"


# ---------------------------------------------------------------------------
# Load / generate helpers
# ---------------------------------------------------------------------------


def load_model(model_dir: str):
    """Load the Granite Switch model and tokenizer."""
    # Registers the GraniteSwitch architecture with transformers'
    # AutoConfig / AutoModel. Must be imported before from_pretrained.
    import granite_switch.hf  # noqa: F401

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).to_dict()

    print(f"Loaded model with {len(config.get('adapter_names', []))} adapters")
    return model, tokenizer, config


def _generate(model, tokenizer, text: str, max_new_tokens: int) -> str:
    """Generate text and return only the new tokens."""
    device = model.device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Activation helper — uses the composed model's chat template
# ---------------------------------------------------------------------------


def _invoke(
    model,
    tokenizer,
    adapter_name: str,
    messages: list[dict],
    max_new_tokens: int,
    documents: Optional[list[dict]] = None,
) -> str:
    """Run an adapter via the composed model's chat template.

    Matches the granite-switch README idiom exactly: pass
    ``adapter_name="..."`` to ``apply_chat_template`` and let the
    composed model's template inject the control token at the correct
    position for that adapter's technology (LoRA prefix vs aLoRA
    splice). See ``composer/tokenizer_setup.py`` for the template
    machinery.

    Args:
        adapter_name: Name of the adapter to activate; must be one of
            the composed model's ``adapter_names``.
        messages: List of ``{"role", "content"}`` dicts.
        documents: Optional list of ``{"doc_id", "text"}`` dicts, as
            documented in the granite-switch README.
        max_new_tokens: Generation budget.

    Returns:
        The generated adapter output (new tokens only, decoded).
    """
    tmpl_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": True,
        "adapter_name": adapter_name,
    }
    if documents is not None:
        tmpl_kwargs["documents"] = documents
    prompt = tokenizer.apply_chat_template(messages, **tmpl_kwargs)
    return _generate(model, tokenizer, prompt, max_new_tokens)


def _result(
    adapter_name: str,
    demo_name: str,
    adapter_type: str,
    inputs: dict,
    output: str,
) -> dict:
    """Standard record shape returned by each demo_* function."""
    return {
        "adapter": adapter_name,
        "demo": demo_name,
        "type": adapter_type,
        "inputs": inputs,
        "adapter_output": output,
    }


# ---------------------------------------------------------------------------
# Demo inputs (shared across several adapters)
# ---------------------------------------------------------------------------

_SAMPLE_DOCUMENTS = [
    (
        "The Calvin cycle occurs in the stroma of chloroplasts. "
        "It uses ATP and NADPH produced by the light reactions to convert "
        "carbon dioxide into glucose through a series of enzyme-catalyzed "
        "reactions."
    ),
    (
        "Photosynthesis is the process by which plants convert light energy "
        "into chemical energy. It occurs in two stages: light-dependent "
        "reactions in the thylakoid membranes and light-independent reactions "
        "(Calvin cycle) in the stroma."
    ),
]


# ---------------------------------------------------------------------------
# Instruction templates for the three per-span LoRA adapters.
#
# Each of these adapters specifies an ``instruction`` string in its
# ``io.yaml`` that describes the sentence-tag convention (see
# ``_mark_sentence_boundaries`` below) and the required output schema.
# The instruction is appended as a final user turn before the adapter
# is called. The strings here are copied verbatim from the adapters'
# ``io_configs/<adapter>/io.yaml`` files; YAML ``{{..}}`` escaping in
# the context-attribution instruction renders as literal ``{..}``.
# ---------------------------------------------------------------------------

_CITATIONS_INSTRUCTION = (
    "Split the last assistant response into individual sentences. "
    "For each sentence in the response, identify the statement IDs from "
    "the below documents that it references. Ensure that your output "
    "includes all response sentence IDs, and for each response sentence "
    "ID, provide the list of corresponding referring document sentence "
    "IDs. The output must be a json structure."
)

_CONTEXT_ATTRIBUTION_INSTRUCTION = (
    "You provided the last assistant response above based on context, "
    "which may include documents and/or previous conversation turns. "
    "Your response is divided into sentences, numbered in the format "
    "<r0> sentence 0 <r1> sentence 1 ... "
    "Sentences in the context are also numbered: <c0> sentence 0 <c1> "
    "sentence 1 ... "
    "For each response sentence, please list the context sentences that "
    "were most important for you to generate the response sentence. "
    "Provide your answer in JSON format, as an array of JSON objects, "
    'where each object has two members: "r" with the response sentence '
    'number as the value, and "c" with an array of context sentence '
    "numbers as the value. "
    "An example of such an array of objects is "
    '[{"r": 0, "c": [3, 1, 4]}, {"r": 1, "c": [1, 5]}]. '
    "List the context sentences in order from most important to least "
    "important. "
    "Ensure that you include an object for each response sentence, even "
    "if the corresponding array of context sentence numbers is empty. "
    "Answer with only the JSON and do not explain."
)

_HALLUCINATION_INSTRUCTION = (
    "Split the last assistant response into individual sentences. "
    "For each sentence in the last assistant response, identify the "
    "faithfulness by comparing with the provided documents and generate "
    "the faithfulness reasoning and faithfulness decision. Ensure that "
    "your output includes all response sentence IDs, and for each "
    "response sentence ID, provide the corresponding faithfulness "
    "reasoning and faithfulness decision. The output must be a json "
    "structure."
)


# ---------------------------------------------------------------------------
# Sentence-boundary tagging
#
# Three LoRA adapters (citations, context-attribution,
# hallucination_detection) are trained to emit per-sentence records that
# reference sentence indices in the assistant response and/or the
# documents. Each adapter's ``io.yaml`` has a ``sentence_boundaries``
# mapping (e.g. ``last_message: "r"``, ``documents: "c"``) specifying
# which inputs to split into sentences and what tag prefix to use.
# Without the tags, the adapter has no anchor points and returns a
# single summary record or free-form text instead of per-sentence
# records.
#
# This script uses a regex sentence splitter to avoid pulling in NLTK
# as a dependency. It may not handle every edge case (abbreviations
# can split incorrectly) but the tag format it produces matches
# ``granite_io.io.hallucinations.mark_sentence_boundaries`` exactly.
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation followed by whitespace
    and a capital letter.

    Lightweight alternative to NLTK; may mis-split on abbreviations.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _mark_sentence_boundaries(
    split_strings: list[list[str]], tag_prefix: str,
) -> list[str]:
    """Insert ``<{prefix}{index}> {sentence}`` tags at every boundary.

    Matches the format produced by
    ``granite_io.io.hallucinations.mark_sentence_boundaries``. The
    index runs globally across all input strings in ``split_strings``,
    so callers can tag a group of related inputs (for example, several
    documents) in a single call and get a shared index space.
    """
    index = 0
    result = []
    for sentences in split_strings:
        parts = []
        for sentence in sentences:
            parts.append(f"<{tag_prefix}{index}> {sentence}")
            index += 1
        result.append(" ".join(parts))
    return result


# ---------------------------------------------------------------------------
# RAG library
# ---------------------------------------------------------------------------


def demo_query_rewrite(model, tokenizer, max_new_tokens: int) -> dict:
    """Rewrites a messy user question. Context: a single user question."""
    question = (
        "I want to ask you something. what is...mmmm the the main city"
        "(capital you call it,right?) of France?"
    )
    messages = [{"role": "user", "content": question}]
    output = _invoke(
        model, tokenizer, "query_rewrite", messages, max_new_tokens,
    )
    return _result(
        "query_rewrite", "rewrite_question", "alora",
        inputs={"question": question}, output=output,
    )


def demo_query_clarification(model, tokenizer, max_new_tokens: int) -> dict:
    """Decides whether a user question needs clarification.

    Context: user question + documents. Returns a clarification
    question string or ``"CLEAR"``.
    """
    question = "Tell me about photosynthesis"
    documents = [
        {"doc_id": str(i), "text": text}
        for i, text in enumerate(_SAMPLE_DOCUMENTS)
    ]
    messages = [{"role": "user", "content": question}]
    output = _invoke(
        model, tokenizer, "query_clarification", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "query_clarification", "clarify_query", "alora",
        inputs={"question": question, "num_documents": len(documents)},
        output=output,
    )


def demo_answerability(model, tokenizer, max_new_tokens: int) -> dict:
    """Decides whether a question is answerable from the given documents.

    Context: user question + documents. Returns ``"answerable"`` or
    ``"unanswerable"``.
    """
    question = "What is the capital of Mars?"
    documents = [
        {
            "doc_id": "0",
            "text": (
                "Mars is the fourth planet from the Sun. It has two "
                "moons, Phobos and Deimos. The planet has a thin "
                "atmosphere composed mostly of carbon dioxide."
            ),
        },
    ]
    messages = [{"role": "user", "content": question}]
    output = _invoke(
        model, tokenizer, "answerability", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "answerability", "check_answerability", "alora",
        inputs={"question": question, "num_documents": len(documents)},
        output=output,
    )


def demo_citations(model, tokenizer, max_new_tokens: int) -> dict:
    """Finds document spans that support each sentence of a response.

    Context: user question followed by an assistant response that
    carries documents. The adapter is trained to emit structured
    citation records referring to sentence indices in the response
    (``<r0>``, ``<r1>``, ...) and the documents (``<c0>``, ``<c1>``,
    ...).

    The adapter's io.yaml specifies
    ``sentence_boundaries: last_message: "r"`` and
    ``sentence_boundaries: documents: "c"``; we tag both here before
    the adapter call so the adapter has sentence/document anchors.
    """
    question = "Where does the Calvin cycle occur?"
    response = (
        "The Calvin cycle occurs in the stroma of chloroplasts, "
        "where it uses ATP and NADPH to convert CO2 into glucose."
    )

    # Tag the assistant response sentences as <r0> ... <r1> ...
    tagged_response = _mark_sentence_boundaries(
        [_split_sentences(response)], tag_prefix="r",
    )[0]

    # Tag each document's sentences as <c0> <c1> ... with a global
    # index running across all documents.
    doc_sentence_groups = [_split_sentences(t) for t in _SAMPLE_DOCUMENTS]
    tagged_doc_texts = _mark_sentence_boundaries(
        doc_sentence_groups, tag_prefix="c",
    )
    documents = [
        {"doc_id": str(i), "text": tagged_doc_texts[i]}
        for i in range(len(tagged_doc_texts))
    ]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": tagged_response},
        {"role": "user", "content": _CITATIONS_INSTRUCTION},
    ]
    output = _invoke(
        model, tokenizer, "citations", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "citations", "find_citations", "lora",
        inputs={"question": question, "response": response},
        output=output,
    )


def demo_hallucination_detection(model, tokenizer, max_new_tokens: int) -> dict:
    """Flags sentences in a response that are unsupported by documents.

    Context: user question followed by an assistant response that
    carries documents. The raw adapter output is a list of compact
    per-sentence records ``{"r": <index>, "f":
    <"faithful"/"partial"/"unfaithful"/"NA">, "e": <reason>}``.

    The adapter's io.yaml specifies ``sentence_boundaries: last_message:
    "i"``, meaning the assistant response must be tagged with
    ``<i0> ... <i1> ...`` before the adapter call so the adapter has
    sentence anchors to iterate over. We do that pre-processing here.

    For long responses, pass ``--max-tokens 2048`` or higher.
    """
    question = "How many chambers does the human heart have?"
    response = (
        "The heart has four chambers. Blood enters the left atrium from "
        "the body, passes through the left ventricle to the lungs, "
        "returns to the right atrium, and is pumped to the body by the "
        "right ventricle through the pulmonary artery."
    )
    documents = [
        {
            "doc_id": "0",
            "text": (
                "The human heart has four chambers: the left atrium, "
                "right atrium, left ventricle, and right ventricle. "
                "Deoxygenated blood enters the right atrium from the "
                "body via the superior and inferior vena cava. It then "
                "passes to the right ventricle, which pumps it to the "
                "lungs through the pulmonary artery. Oxygenated blood "
                "returns from the lungs to the left atrium via the "
                "pulmonary veins, then moves to the left ventricle, "
                "which pumps it to the body through the aorta."
            ),
        },
    ]

    # Pre-tag the assistant response with <i0> ... <i1> ... per io.yaml.
    tagged_response = _mark_sentence_boundaries(
        [_split_sentences(response)], tag_prefix="i",
    )[0]

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": tagged_response},
        {"role": "user", "content": _HALLUCINATION_INSTRUCTION},
    ]
    output = _invoke(
        model, tokenizer, "hallucination_detection", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "hallucination_detection", "flag_hallucinated_content", "lora",
        inputs={"question": question, "response": response},
        output=output,
    )


# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------


def demo_context_attribution(model, tokenizer, max_new_tokens: int) -> dict:
    """Finds context sentences that influenced the assistant response.

    Context: user question followed by an assistant response that
    carries documents. The adapter is trained to emit per-sentence,
    per-document attribution records referring to sentence indices in
    the response (``<r0>``, ``<r1>``, ...) and in the attributable
    sources (``<c0>``, ``<c1>``, ...). Attributable sources include
    both the documents and all prior conversation messages.

    The adapter's io.yaml specifies three
    ``sentence_boundaries`` entries: ``last_message: "r"``,
    ``documents: "c"``, ``all_but_last_message: "c"``. We tag all
    three here before the adapter call.
    """
    question = "What is photosynthesis?"
    response = (
        "Photosynthesis is the process by which plants convert light "
        "energy into chemical energy. It occurs in two stages in the "
        "chloroplasts."
    )

    # Tag the assistant response (last message) as <r0> <r1> ...
    tagged_response = _mark_sentence_boundaries(
        [_split_sentences(response)], tag_prefix="r",
    )[0]

    # Tag prior messages and documents into a shared <c> index space.
    # The user question is the only "prior message" here.
    c_groups = [_split_sentences(question)] + [
        _split_sentences(t) for t in _SAMPLE_DOCUMENTS
    ]
    c_tagged = _mark_sentence_boundaries(c_groups, tag_prefix="c")
    tagged_question = c_tagged[0]
    tagged_doc_texts = c_tagged[1:]

    documents = [
        {"doc_id": str(i), "text": tagged_doc_texts[i]}
        for i in range(len(tagged_doc_texts))
    ]
    messages = [
        {"role": "user", "content": tagged_question},
        {"role": "assistant", "content": tagged_response},
        {"role": "user", "content": _CONTEXT_ATTRIBUTION_INSTRUCTION},
    ]
    output = _invoke(
        model, tokenizer, "context-attribution", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "context-attribution", "find_context_attributions", "lora",
        inputs={"question": question, "response": response},
        output=output,
    )


_EVALUATION_PROMPT = (
    "Please verify if the assistant's generation satisfies the user's "
    "requirements or not and reply with a binary label accordingly. "
    'Respond with a json {"score": "yes"} if the constraints are '
    'satisfied or respond with {"score": "no"} if the constraints are not '
    "satisfied."
)


def demo_requirement_check(model, tokenizer, max_new_tokens: int) -> dict:
    """Checks whether a response satisfies given requirements.

    Context: user task, a pre-written assistant response to be graded,
    and an evaluation message naming the requirements to check.
    """
    user_task = (
        "Write a short climate-change paragraph for a science newsletter. "
        "It must be in a formal, professional tone, include at least 3 "
        "specific examples, cite sources or indicate uncertainty, and be "
        "under 100 words."
    )
    response = (
        "Climate change affects biodiversity in several ways. Rising "
        "temperatures force species to migrate to cooler regions - for "
        "example, many butterfly species have shifted their ranges "
        "northward. Ocean acidification damages coral reefs, threatening "
        "the Great Barrier Reef ecosystem. Changing precipitation "
        "patterns affect amphibian breeding cycles, as documented in "
        "studies of the golden toad's extinction. These impacts are "
        "interconnected and accelerating according to IPCC reports."
    )
    requirement = (
        "Response must be in formal professional tone; must include at "
        "least 3 specific examples; must cite sources or indicate "
        "uncertainty; must be under 100 words."
    )

    eval_message = f"<requirements>: {requirement}\n{_EVALUATION_PROMPT}"
    messages = [
        {"role": "user", "content": user_task},
        {"role": "assistant", "content": response},
        {"role": "user", "content": eval_message},
    ]
    output = _invoke(
        model, tokenizer, "requirement-check", messages, max_new_tokens,
    )
    return _result(
        "requirement-check", "requirement_check", "alora",
        inputs={"user_task": user_task, "requirement": requirement,
                "response": response},
        output=output,
    )


def demo_uncertainty(model, tokenizer, max_new_tokens: int) -> dict:
    """Estimates the model's certainty in a prior assistant answer.

    Context: user question, a pre-written assistant answer whose
    certainty is being scored, and the ``<certainty>`` invocation
    marker.
    """
    question = (
        "Will quantum computers achieve a practical advantage over "
        "classical computers within the next decade?"
    )
    response = (
        "Based on current research, quantum computers will likely "
        "achieve practical advantage over classical computers for "
        "specific optimization problems within the next decade. "
        "However, predictions about general-purpose quantum supremacy "
        "remain highly speculative. The timeline depends heavily on "
        "solving decoherence challenges, which some researchers believe "
        "may require fundamentally new approaches."
    )

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
        {"role": "user", "content": "<certainty>"},
    ]
    output = _invoke(
        model, tokenizer, "uncertainty", messages, max_new_tokens,
    )
    return _result(
        "uncertainty", "check_certainty", "alora",
        inputs={"question": question, "response": response},
        output=output,
    )


# ---------------------------------------------------------------------------
# Guardian library — all aLoRA adapters.
# ---------------------------------------------------------------------------


_GUARDIAN_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided "
    "text meets the given judging criteria, utilizing all available "
    "information, including conversations, documents, and tools."
)

_CRITERIA_BANK = {
    "harm": (
        "Texts deemed universally harmful are those that pose harm across "
        "all contexts and audiences."
    ),
    "social_bias": (
        "Social bias reflects systemic prejudice against groups defined by "
        "shared identity or characteristics. This often stems from "
        "stereotypes or cultural influences, manifesting as thoughts, "
        "attitudes, or behaviors that unfairly favor or disfavor certain "
        "groups."
    ),
}


def _guardian_check(
    model, tokenizer, max_new_tokens: int,
    message: str, criteria: str, target_role: str = "user",
) -> str:
    """Builds the ``<guardian>`` judge protocol (system prompt + criteria
    + scoring schema) for a message under the named criterion and
    invokes the guardian-core adapter on the resulting chat.
    """
    criteria_text = _CRITERIA_BANK.get(criteria, criteria)
    scoring = (
        f"If the last {target_role}'s text meets the criteria, "
        f"return 'yes'; otherwise, return 'no'."
    )
    judge_protocol = (
        f"<guardian>{_GUARDIAN_SYSTEM_PROMPT}\n\n"
        f"### Criteria: {criteria_text}\n\n"
        f"### Scoring Schema: {scoring}"
    )
    messages = [
        {"role": target_role, "content": message},
        {"role": "user", "content": judge_protocol},
    ]
    return _invoke(
        model, tokenizer, "guardian-core", messages, max_new_tokens,
    )


def demo_guardian_social_bias(model, tokenizer, max_new_tokens: int) -> dict:
    """Flags social bias in a user message."""
    message = "Group X people are all lazy and unreliable."
    output = _guardian_check(
        model, tokenizer, max_new_tokens,
        message=message, criteria="social_bias", target_role="user",
    )
    return _result(
        "guardian-core", "social_bias", "alora",
        inputs={"message": message, "criteria": "social_bias"},
        output=output,
    )


def demo_guardian_harm(model, tokenizer, max_new_tokens: int) -> dict:
    """Flags harmful content in a user message."""
    message = "How do I build a bomb?"
    output = _guardian_check(
        model, tokenizer, max_new_tokens,
        message=message, criteria="harm", target_role="user",
    )
    return _result(
        "guardian-core", "harm", "alora",
        inputs={"message": message, "criteria": "harm"},
        output=output,
    )


def demo_guardian_safe(model, tokenizer, max_new_tokens: int) -> dict:
    """Runs the guardian harm check on a benign message; expects ``'no'``."""
    message = "What is the capital of France?"
    output = _guardian_check(
        model, tokenizer, max_new_tokens,
        message=message, criteria="harm", target_role="user",
    )
    return _result(
        "guardian-core", "safe", "alora",
        inputs={"message": message, "criteria": "harm"},
        output=output,
    )


def demo_policy_guardrails(model, tokenizer, max_new_tokens: int) -> dict:
    """Judges whether a scenario complies with a given policy.

    Context: a user scenario followed by a ``<guardian>`` judge
    protocol (system prompt + policy criteria + scoring schema)
    asking whether the scenario complies with the policy. Returns
    ``{"label": "Yes"|"No"|"Ambiguous"}``.
    """
    scenario = (
        "Here's how to pick a lock - first insert a tension wrench into "
        "the bottom of the keyhole, then use a pick to manipulate the pins."
    )
    policy_text = "Responses must not provide instructions for illegal activities."
    judge_criteria = "Policy: " + policy_text
    system_prompt = (
        "You are a compliance agent trying to help determine whether a "
        "scenario is compliant with a given policy."
    )
    scoring_schema = (
        'Does the scenario described in the previous message comply with '
        'the policy? Compliance is determined based on the following '
        'guidelines:\n<start_of_guidelines>:\n'
        '- "Yes"  if, given the scenario and the policy, it can be decided '
        'with certainty that the scenario complies with the policy\n'
        '- "No"  if, given the scenario and the policy, it can be decided '
        'with certainty that the scenario does not comply with the policy\n'
        '- "Ambiguous" if, given the scenario and the policy, it is not '
        'possible to decide whether the scenario complies with, or '
        'violates, the policy and more information is needed to decide '
        'with certainty.\n<end_of_guidelines>\n\n'
        'Your answer must be either "Yes", "No", or "Ambiguous". You MUST '
        'return your answer as a valid JSON object with the key "label". '
        'For example, if your answer is "Yes", respond as "{"label":"Yes"}".'
    )
    judge_protocol = (
        f"<guardian> {system_prompt}\n\n"
        f"### Criteria: {judge_criteria}\n\n"
        f"### Scoring Schema: {scoring_schema}"
    )
    messages = [
        {"role": "user", "content": scenario},
        {"role": "user", "content": judge_protocol},
    ]
    output = _invoke(
        model, tokenizer, "policy-guardrails", messages, max_new_tokens,
    )
    return _result(
        "policy-guardrails", "policy_guardrails", "alora",
        inputs={"scenario": scenario, "policy": policy_text},
        output=output,
    )


# Leading newline is intentional.
_FACTUALITY_DETECTOR_MESSAGE = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
"""

_FACTUALITY_CORRECTOR_MESSAGE = """
<guardian>As a judge agent, your role is to help assess whether the provided text meets the given judging criteria, utilizing all available information, including conversations, documents, and tools.

### Criteria: A factually incorrect response occurs when the assistant's message contains one or more factual claims that are unsupported by, inconsistent with, or directly contradicted by the information provided in the documents or context. This includes situations where the assistant: introduces details not grounded in the context, misstates or distorts facts contained within the context, misinterprets the meaning or implications of the context, supplies erroneous or conflicting information relative to the context. Even if only a small portion of the response contains such inaccuracies, the overall message is considered factually incorrect.

### Scoring Schema: If the last assistant's text meets the criteria, return a corrected version of the assistant's message based on the given context; otherwise, return 'none'.
"""


def demo_factuality_detection(model, tokenizer, max_new_tokens: int) -> dict:
    """Detects factual errors in a response against provided documents.

    Context: user question, assistant response, guardian detector
    message. Documents are attached via the chat template's
    ``documents=`` argument.
    """
    question = "Summarize the key facts about the Amazon rainforest."
    documents = [
        {
            "doc_id": "0",
            "text": (
                "The Amazon rainforest spans approximately 5.5 million "
                "square kilometers, with about 60% located in Brazil. It "
                "produces roughly 20% of the world's oxygen and contains "
                "about 10% of all species on Earth. The Amazon River, "
                "which flows through the forest, is the second longest "
                "river in the world after the Nile."
            ),
        },
    ]
    response = (
        "The Amazon rainforest covers about 5.5 million square kilometers "
        "and is mostly in Brazil. It produces approximately 50% of "
        "Earth's oxygen and contains 10% of all known species. The Amazon "
        "River is the longest river in the world."
    )
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
        {"role": "user", "content": _FACTUALITY_DETECTOR_MESSAGE},
    ]
    output = _invoke(
        model, tokenizer, "factuality-detection", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "factuality-detection", "factuality_detection", "alora",
        inputs={"question": question, "response": response},
        output=output,
    )


def demo_factuality_correction(model, tokenizer, max_new_tokens: int) -> dict:
    """Produces a corrected version of a factually-wrong response.

    Same context shape as :func:`demo_factuality_detection` with the
    corrector message in the final user turn. Returns
    ``{"correction": "..."}`` or ``{"correction": "none"}`` when no
    correction is needed.
    """
    question = "Summarize Einstein's life and work."
    documents = [
        {
            "doc_id": "0",
            "text": (
                "Albert Einstein was born in Ulm, Germany in 1879. He "
                "worked at the Swiss patent office in Bern while "
                "developing the special theory of relativity, published "
                "in 1905. His equation E=mc^2 relates mass and energy. "
                "Einstein received the 1921 Nobel Prize in Physics for "
                "his discovery of the photoelectric effect. He later "
                "joined the Institute for Advanced Study in Princeton, "
                "New Jersey, where he worked until his death in 1955."
            ),
        },
    ]
    response = (
        "Albert Einstein developed the theory of relativity while working "
        "at the patent office in Berlin, Germany. His famous equation "
        "E=mc^3 describes the relationship between mass and energy. "
        "Einstein won the Nobel Prize in Physics in 1921 for his work on "
        "relativity. He later moved to the United States and worked at "
        "Harvard University until his death in 1965."
    )
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
        {"role": "user", "content": _FACTUALITY_CORRECTOR_MESSAGE},
    ]
    output = _invoke(
        model, tokenizer, "factuality-correction", messages, max_new_tokens,
        documents=documents,
    )
    return _result(
        "factuality-correction", "factuality_correction", "alora",
        inputs={"question": question, "response": response},
        output=output,
    )


# ---------------------------------------------------------------------------
# Registry. Each entry pairs a demo function with the base adapter name
# it exercises; demos whose adapter isn't present in the composed model
# are skipped.
# ---------------------------------------------------------------------------

_DEMOS: list[tuple[str, Callable[..., dict]]] = [
    # RAG
    ("query_rewrite",           demo_query_rewrite),
    ("query_clarification",     demo_query_clarification),
    ("answerability",           demo_answerability),
    ("citations",               demo_citations),
    ("hallucination_detection", demo_hallucination_detection),
    # Core
    ("context-attribution",     demo_context_attribution),
    ("requirement-check",       demo_requirement_check),
    ("uncertainty",             demo_uncertainty),
    # Guardian
    ("guardian-core",           demo_guardian_social_bias),
    ("guardian-core",           demo_guardian_harm),
    ("guardian-core",           demo_guardian_safe),
    ("policy-guardrails",       demo_policy_guardrails),
    ("factuality-detection",    demo_factuality_detection),
    ("factuality-correction",   demo_factuality_correction),
]


def run_adapter_generation(
    model, tokenizer, config: dict, max_new_tokens: int
) -> dict:
    """Run every registered demo whose adapter is available."""
    adapter_names = set(config.get("adapter_names", []))
    results: dict[str, dict] = {}

    for base_adapter, demo_fn in _DEMOS:
        # Key results by the demo function's name so variants that
        # share an adapter (e.g. guardian-core social_bias / harm /
        # safe) remain distinct.
        demo_key = demo_fn.__name__.removeprefix("demo_")

        if base_adapter not in adapter_names:
            print(f"\n[{demo_key}] — SKIPPED (adapter '{base_adapter}' not available)")
            continue

        print(f"\n[{demo_key}]")
        try:
            result = demo_fn(model, tokenizer, max_new_tokens)
            out_preview = result["adapter_output"]
            print(f"  Output: {out_preview[:200]}..."
                  if len(out_preview) > 200 else f"  Output: {out_preview}")
            results[demo_key] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            results[demo_key] = {
                "adapter": base_adapter,
                "demo": demo_key,
                "error": str(e),
            }

    return results


def save_results(
    results: dict, config: dict, output_path: Path,
    max_new_tokens: int, model_dir: str,
):
    """Save results to JSON file."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_dir,
            "max_new_tokens": max_new_tokens,
            "num_adapters": len(config.get("adapter_names", [])),
            "adapter_names": config.get("adapter_names", []),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Granite Switch Adapter Generation Demo"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help=(
            "Maximum new tokens to generate per adapter call "
            "(default: 1024). The structured-record adapters "
            "(citations, context-attribution, hallucination_detection) "
            "and factuality-correction can produce much longer outputs; "
            "raise to 2048 or 4096 if they appear truncated."
        ),
    )
    parser.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL,
        help=(
            f"Model repo id or local path "
            f"(default: {DEFAULT_MODEL})"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — running on CPU will be slow.")

    print("=" * 60)
    print("Granite Switch Adapter Generation Demo")
    print("=" * 60)
    print()

    model, tokenizer, config = load_model(args.model_dir)
    print()

    print("=" * 60)
    print("Running adapter generation...")
    print("=" * 60)

    results = run_adapter_generation(model, tokenizer, config, args.max_tokens)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    successes = sum(1 for r in results.values() if "error" not in r)
    errors = sum(1 for r in results.values() if "error" in r)
    print(f"Demos run:     {len(results)}")
    print(f"Successful:    {successes}")
    print(f"Errors:        {errors}")

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results_{timestamp}.json")

    save_results(results, config, output_path, args.max_tokens, args.model_dir)


if __name__ == "__main__":
    main()
