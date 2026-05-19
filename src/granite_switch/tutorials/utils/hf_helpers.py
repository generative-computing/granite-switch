"""Helper utilities for HuggingFace backend tutorials.

This module provides reusable helper functions for granite_switch_with_hf.ipynb
and other HuggingFace-based tutorial notebooks. Functions handle common patterns
like adapter invocation, conversation management, and complex adapter protocols.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from IPython.display import Markdown, display


def generate_turn(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: List[Dict[str, str]],
    adapter: Optional[str] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: int = 64,
) -> str:
    """Render a chat prompt with the named adapter active and greedy-decode.

    Args:
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
        device: Device string ("cuda" or "cpu")
        messages: List of message dicts with "role" and "content" keys
        adapter: Optional adapter name to activate via chat template
        documents: Optional documents list for RAG adapters
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text (decoded, with special tokens stripped)
    """
    kwargs = {"adapter_name": adapter} if adapter else {}
    if documents:
        kwargs["documents"] = documents
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def screen_user_message(
    model: Any,
    tokenizer: Any,
    device: str,
    candidate: str,
    judge_system: str,
    jailbreak_criteria: str,
) -> str:
    """Run guardian-core against a candidate user message for jailbreak/harm.

    Args:
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
        device: Device string ("cuda" or "cpu")
        candidate: User message to screen
        judge_system: System prompt for judge agent
        jailbreak_criteria: Jailbreak detection criteria string

    Returns:
        Parsed score ("yes" means criterion fired -> block, "no" means safe)
    """
    schema = (
        "If the last user's text meets the criteria, return 'yes'; otherwise, "
        "return 'no'."
    )
    guardian_block = (
        f"<guardian>{judge_system}\n\n### Criteria: {jailbreak_criteria}\n\n"
        f"### Scoring Schema: {schema}"
    )
    judge_msgs = [
        {"role": "user", "content": candidate},
        {"role": "user", "content": guardian_block},
    ]
    raw = generate_turn(
        model, tokenizer, device, judge_msgs, adapter="guardian-core", max_new_tokens=15
    )
    return json.loads(raw)["score"]


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for tutorial purposes.

    Note: This is a basic regex-based splitter suitable for demos.
    Production code should use nltk.sent_tokenize() or spacy for robustness.

    Args:
        text: Input text to split

    Returns:
        List of sentence strings
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def run_context_attribution(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: List[Dict[str, str]],
    documents: List[Dict[str, Any]],
    attribution_instruction: str,
) -> Tuple[str, List[str], Dict[int, Tuple[str, str]]]:
    """Invoke context-attribution adapter with sentence tagging.

    This adapter requires:
    1. Response sentences tagged with <r0>, <r1>, ...
    2. Context sentences tagged with <c0>, <c1>, ...
    3. Instruction turn asking for attribution mapping

    Args:
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
        device: Device string ("cuda" or "cpu")
        messages: Conversation history (last message is assistant response to attribute)
        documents: Document list for context
        attribution_instruction: Instruction prompt for attribution adapter

    Returns:
        Tuple of:
        - raw: Raw JSON string from adapter
        - response_sents: List of response sentences
        - tagged_context: Dict mapping c_id -> (source_label, original_text)
    """
    # Clone the conversation with sentences tagged; keep a reverse map of the
    # tags back to the original sentence text for printing the audit trail.
    c_counter = 0
    tagged_context = {}  # c_id -> (source, original_text)
    tagged_documents = []
    for doc in documents:
        parts = []
        for sent in _split_sentences(doc["text"]):
            parts.append(f"<c{c_counter}> {sent}")
            tagged_context[c_counter] = (f"doc {doc['doc_id']}", sent)
            c_counter += 1
        tagged_documents.append({"doc_id": doc["doc_id"], "text": " ".join(parts)})

    tagged_messages = []
    for msg in messages[:-1]:
        new_content = " ".join(
            f"<c{c_counter + i}> {s}"
            for i, s in enumerate(_split_sentences(msg["content"]))
        )
        for i, s in enumerate(_split_sentences(msg["content"])):
            tagged_context[c_counter + i] = (f"{msg['role']} turn", s)
        c_counter += len(_split_sentences(msg["content"]))
        tagged_messages.append({"role": msg["role"], "content": new_content})

    # Last message (the assistant response) is tagged with <r0>, <r1>, ...
    last = messages[-1]
    response_sents = _split_sentences(last["content"])
    tagged_last = " ".join(f"<r{i}> {s}" for i, s in enumerate(response_sents))
    tagged_messages.append({"role": last["role"], "content": tagged_last})

    # Instruction turn at the end.
    tagged_messages.append({"role": "user", "content": attribution_instruction})

    raw = generate_turn(
        model,
        tokenizer,
        device,
        tagged_messages,
        adapter="context-attribution",
        documents=tagged_documents,
        max_new_tokens=200,
    )
    return raw, response_sents, tagged_context


def show_conversation_as_markdown(messages: List[Dict[str, str]]) -> None:
    """Render the entire conversation as Markdown with role-labeled blocks.

    Call this once at the end of a cell after mutating messages, so the
    reader always sees the latest full context without duplicated prints.

    Args:
        messages: List of message dicts with "role" and "content" keys
    """
    n = len(messages)
    md = [f"*conversation so far - {n} turn{'s' if n != 1 else ''}*", ""]
    for msg in messages:
        label = "**User**" if msg["role"] == "user" else "**Assistant**"
        md.append(f"{label}\n\n> {msg['content']}")
    display(Markdown("\n\n".join(md)))


def say_user(messages: List[Dict[str, str]], content: str) -> None:
    """Append a user turn to the conversation.

    Does NOT print - call show_conversation_as_markdown() at the end of the cell.

    Args:
        messages: Conversation history to mutate
        content: User message content
    """
    messages.append({"role": "user", "content": content})


def say_assistant(messages: List[Dict[str, str]], content: str) -> None:
    """Append an assistant turn to the conversation.

    Does NOT print - call show_conversation_as_markdown() at the end of the cell.

    Args:
        messages: Conversation history to mutate
        content: Assistant message content
    """
    messages.append({"role": "assistant", "content": content})
