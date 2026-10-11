# SPDX-License-Identifier: Apache-2.0
"""How a model's prompts carry their documents, and its chat-template options.

Used by both generation paths, so they render the same prompts:
``generate.py`` (granite-switch under vLLM) and ``hf_generate.py`` (HF + PEFT,
and the base model). Standard library only.

Granite 4.1's chat template renders a ``documents=`` argument itself, into the
system turn. Granite 4.2's has no such argument, and opens a reasoning block
unless ``enable_thinking`` is False. So 4.2 checkpoints are trained with
reasoning off and their documents placed in the messages: in the system turn,
as 4.1 put them, for the adapters trained on 4.1's data; in ``tool`` messages
for the SR checkpoints, whose trainer placed them there. A model's ``prompt``
entry in ``adapters.yaml`` names a style for its base model, and one per
technology where that differs (``common.Model.prompt_for``). Styles, for a row
whose messages end with the user's question:

* ``native``: ``documents=`` to the chat template.
* ``system``: the block Granite 4.1's template writes for ``documents=``, in
  the system turn: its preamble, each document as one JSON line within
  ``<documents>`` tags, and its closing instruction. After a system message of
  the row's own, as 4.1 puts it.
* ``tool_json_after_question``: one tool message after the last user turn,
  with every document in a JSON list of ``{source, document_id, content}``.
* ``tool_text_before_question``: one tool message per document, holding its
  text, before the last user turn.

The base-model column also gets an instruction: a final user turn naming the
output format its scorer expects, since a base model without an adapter does
not know it. The instruction texts are private (``vela/local``) and reach the
pod with the reference job; ``with_instruction`` applies one. Its ``mode``:

* ``append``: after the conversation.
* ``replace_last_user``: in place of the row's own last user turn, a terse
  task request the adapter was trained on.
"""

from __future__ import annotations

import json

DOCUMENT_STYLES = (
    "native",
    "system",
    "tool_json_after_question",
    "tool_text_before_question",
)
INSTRUCTION_MODES = ("append", "replace_last_user")
# Granite 4.1's chat template writes documents= into the system turn between
# these two texts, one JSON line per document (its documents_system_message_*).
SYSTEM_DOCUMENTS_PREFIX = (
    "You are a helpful assistant with access to the following documents. You may use one or "
    "more documents to assist with the user query.\n\nYou are given a list of documents within "
    "<documents></documents> XML tags:\n<documents>"
)
SYSTEM_DOCUMENTS_SUFFIX = (
    "\n</documents>\n\nWrite the response to the user's input by strictly aligning with the "
    "facts in the provided documents. If the information needed to answer the question is not "
    "available in the documents, inform the user that the question cannot be answered based on "
    "the available data."
)


def fix_documents(docs):
    """Documents as the chat templates take them: dicts with a ``text``."""
    if not docs:
        return docs
    return [
        d if isinstance(d, dict) else {"title": "Context", "text": str(d)} for d in docs
    ]


def _record(doc: dict, index: int) -> dict:
    return {
        "source": doc.get("source", "knowledge_base"),
        "document_id": str(doc.get("document_id", doc.get("doc_id", index))),
        "content": doc.get("content", doc.get("text", "")),
    }


def _text(doc: dict) -> str:
    for key in ("text", "content"):
        if isinstance(doc.get(key), str) and doc[key]:
            return doc[key]
    return json.dumps(doc, ensure_ascii=False)


def system_documents(documents: list) -> str:
    """The system-turn block Granite 4.1's template writes for ``documents=``:
    each document as one JSON line (its ``tojson``). The documents are dicts, as
    transformers requires of ``documents=``; ``fix_documents`` makes them so."""
    lines = "".join("\n" + json.dumps(d, ensure_ascii=False) for d in documents)
    return SYSTEM_DOCUMENTS_PREFIX + lines + SYSTEM_DOCUMENTS_SUFFIX


def _with_system_documents(messages: list[dict], documents: list) -> list[dict]:
    block = system_documents(documents)
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(
            content, list
        ):  # text entries, joined as 4.1's template joins them
            content = "\n".join(
                e.get("text", "")
                for e in content
                if isinstance(e, dict) and e.get("type") == "text"
            )
        return [{**messages[0], "content": f"{content}\n\n{block}"}, *messages[1:]]
    return [{"role": "system", "content": block}, *messages]


def with_documents(
    messages: list[dict], documents, style: str
) -> tuple[list[dict], list | None]:
    """The messages, and the ``documents=`` argument, for a document style."""
    if style not in DOCUMENT_STYLES:
        raise ValueError(f"unknown document style {style!r}")
    documents = fix_documents(documents)
    if style == "system":
        return (
            _with_system_documents(messages, documents) if documents else messages
        ), None
    if style == "native" or not documents:
        return messages, documents
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if style == "tool_json_after_question":
        payload = [_record(d, i) for i, d in enumerate(documents)]
        tool = [{"role": "tool", "content": json.dumps(payload, indent=2)}]
        if users:
            at = users[-1] + 1
        else:  # before the first answer, if any
            roles = [m.get("role") for m in messages]
            at = roles.index("assistant") if "assistant" in roles else len(messages)
    else:
        tool = [{"role": "tool", "content": _text(d)} for d in documents]
        at = users[-1] if users else len(messages)
    return [*messages[:at], *tool, *messages[at:]], None


def without_last_user(messages: list[dict]) -> list[dict]:
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    return [m for i, m in enumerate(messages) if not users or i != users[-1]]


def chat_text(
    tokenizer,
    row: dict,
    documents: str,
    template_kwargs: dict,
    instruction: dict | None = None,
    **extra,
) -> str:
    """A row's prompt, ending in the generation prompt.

    ``documents`` is the row's document style, ``template_kwargs`` its model's
    chat-template options, ``instruction`` a base-model instruction
    (``{"mode", "text"}``), and ``extra`` more template arguments (the
    composed model's ``adapter_name``).

    The documents go in before the instruction, so they keep their place
    next to the question and the instruction is the last thing read.
    """
    messages = row["messages"]
    if instruction:
        if instruction["mode"] not in INSTRUCTION_MODES:
            raise ValueError(f"unknown instruction mode {instruction['mode']!r}")
        if instruction["mode"] == "replace_last_user":
            messages = without_last_user(messages)
    messages, documents_arg = with_documents(messages, row.get("documents"), documents)
    if instruction:
        messages = [*messages, {"role": "user", "content": instruction["text"]}]
    return tokenizer.apply_chat_template(
        messages,
        tools=row.get("tools"),
        documents=documents_arg,
        add_generation_prompt=True,
        tokenize=False,
        **template_kwargs,
        **extra,
    )
