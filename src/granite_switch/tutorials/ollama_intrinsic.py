# SPDX-License-Identifier: Apache-2.0
"""Granite Switch adapter invocation through a local ``ollama serve``.

The Granite Switch tutorials invoke the embedded LoRA adapters through Mellea's
``OpenAIBackend`` against **vLLM**, which renders the model's chat template
server-side so the adapter's control token lands in the prompt. Ollama does not
render that template server-side and must be driven through the **raw**
``/api/generate`` endpoint (``raw: true``) with the control token already in the
prompt. See https://huggingface.co/barha/granite-switch-4.1-3b-preview-GGUF

:class:`OllamaIntrinsicBackend` bridges the two: it reuses Mellea's
``IntrinsicsRewriter`` / ``IntrinsicsResultProcessor`` to build each adapter's
request envelope and parse its output, fetches the model's chat template from
Ollama's ``/api/show`` and renders it client-side with ``adapter_name=...``,
then POSTs the rendered prompt to the raw endpoint. Runs on ``ollama serve``
with no GPU (Metal on Apple Silicon).
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import requests
import yaml
from jinja2 import BaseLoader, Environment

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "barvhaim/granite-switch-4.1-3b-preview:latest"

# Selects which io.yaml overlay variant Mellea ships for this checkpoint.
CANONICAL_MODEL = "granite-4.1-3b"


def load_chat_template_from_ollama(model: str, ollama_url: str) -> str:
    """Fetch the Granite Switch chat template from Ollama's ``/api/show``."""
    resp = requests.post(f"{ollama_url}/api/show", json={"model": model}, timeout=30)
    resp.raise_for_status()
    template = resp.json().get("template", "")
    if not template or "adapter_map" not in template:
        raise RuntimeError(
            f"Ollama /api/show for {model!r} did not return the Granite Switch "
            "chat template (no 'adapter_map' found). Is this a granite-switch "
            "model created from the GGUF?"
        )
    return template


def make_template_env() -> Environment:
    """Jinja env matching how transformers/vLLM render chat templates.

    ``autoescape=False`` plus a non-HTML-escaping ``tojson`` keep the ``<|...|>``
    control tokens and the document JSON literal intact — the model was not
    trained on HTML-escaped ``&lt;|...|&gt;``.
    """
    env = Environment(
        loader=BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["tojson"] = lambda value, indent=None: json.dumps(
        value, ensure_ascii=False, indent=indent
    )
    return env


def _overlay_root() -> pathlib.Path:
    import mellea

    return pathlib.Path(mellea.__file__).parent / "backends" / "adapters" / "_overlays"


# Adapters whose io.yaml ships in-repo with Mellea (no network needed).
_LOCAL_OVERLAY_ADAPTERS = {
    "guardian-core": "alora",
    "policy-guardrails": "alora",
    "factuality-detection": "alora",
    "factuality-correction": "alora",
    "requirement-check": "alora",
    "uncertainty": "alora",
}

# RAG-library adapters: io.yaml is fetched from the HF repo below and cached.
_RAG_REPO = "ibm-granite/granitelib-rag-r1.0"
_RAG_ADAPTERS = {
    "query_rewrite": "alora",
    "answerability": "alora",
    "query_clarification": "alora",
    "citations": "lora",
    "hallucination_detection": "lora",
}
_RAG_CACHE = pathlib.Path(__file__).parent / ".rag_io_cache"


def _fetch_rag_io_config(intrinsic_name: str, subdir: str) -> dict:
    """Fetch (and cache) a RAG adapter's io.yaml from the HF RAG library."""
    rel = f"{intrinsic_name}/{CANONICAL_MODEL}/{subdir}/io.yaml"
    cache_path = _RAG_CACHE / rel
    if cache_path.exists():
        return yaml.safe_load(cache_path.read_text())

    url = f"https://huggingface.co/{_RAG_REPO}/raw/main/{rel}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        raise ValueError(
            f"No io.yaml for {intrinsic_name!r} at {CANONICAL_MODEL}/{subdir} "
            f"in {_RAG_REPO} — check the adapter name and variant."
        )
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text)
    return yaml.safe_load(resp.text)


def load_io_config(intrinsic_name: str) -> dict | None:
    """Return the parsed io.yaml for an intrinsic (local overlay or HF), or None."""
    subdir = _LOCAL_OVERLAY_ADAPTERS.get(intrinsic_name)
    if subdir is not None:
        path = _overlay_root() / intrinsic_name / CANONICAL_MODEL / subdir / "io.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text())

    subdir = _RAG_ADAPTERS.get(intrinsic_name)
    if subdir is not None:
        return _fetch_rag_io_config(intrinsic_name, subdir)

    return None


@dataclass
class OllamaIntrinsicBackend:
    """Invoke Granite Switch embedded adapters over Ollama's raw endpoint."""

    model: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    verbose: bool = False

    _template: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        src = load_chat_template_from_ollama(self.model, self.ollama_url)
        self._template = make_template_env().from_string(src)

    def _dump_prompt(self, prompt: str, adapter_name: str | None) -> None:
        if not self.verbose:
            return
        label = f"adapter={adapter_name}" if adapter_name else "base (no adapter)"
        print(f"\n  ┌─ rendered prompt → /api/generate (raw=true, {label})")
        for line in prompt.splitlines() or [""]:
            print(f"  │ {line}")
        print("  └─" + "─" * 60)

    def render(
        self,
        messages: list[dict],
        *,
        adapter_name: str | None = None,
        documents: list[dict] | None = None,
    ) -> str:
        """Render the model's chat template with an optional adapter token."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "add_generation_prompt": True,
        }
        if documents:
            kwargs["documents"] = documents
        if adapter_name:
            kwargs["adapter_name"] = adapter_name
        prompt = self._template.render(**kwargs)
        self._dump_prompt(prompt, adapter_name)
        return prompt

    def generate(
        self,
        prompt: str,
        *,
        num_predict: int = 256,
        logprobs: bool = False,
        top_logprobs: int = 10,
    ) -> dict:
        """POST a raw prompt to Ollama and return the full JSON response."""
        body: dict[str, Any] = {
            "model": self.model,
            "raw": True,
            "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict},
            "prompt": prompt,
        }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        resp = requests.post(f"{self.ollama_url}/api/generate", json=body, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def call_adapter(
        self,
        intrinsic_name: str,
        messages: list[dict],
        *,
        documents: list[dict] | None = None,
        rewriter_kwargs: dict | None = None,
        num_predict: int = 256,
    ) -> dict:
        """Invoke an embedded adapter and return a structured result.

        Mellea's ``IntrinsicsRewriter`` builds the request envelope and its
        ``IntrinsicsResultProcessor`` parses the output. Raises if the adapter
        has no io.yaml wired (add it to the maps above) — running it without an
        envelope silently produces wrong output.
        """
        cfg = load_io_config(intrinsic_name)
        if cfg is None:
            raise ValueError(
                f"No io.yaml wired for {intrinsic_name!r}. Add it to "
                "_RAG_ADAPTERS or _LOCAL_OVERLAY_ADAPTERS."
            )

        from mellea.formatters import granite as g

        rewriter = g.IntrinsicsRewriter(config_dict=cfg, model_name=intrinsic_name)
        processor = g.IntrinsicsResultProcessor(config_dict=cfg)

        request: dict[str, Any] = {
            "messages": list(messages),
            "extra_body": {"documents": documents or []},
        }
        rewritten = rewriter.transform(request, **(rewriter_kwargs or {}))

        rendered_messages = [
            {"role": m.role, "content": m.model_dump(exclude_unset=True).get("content")}
            for m in rewritten.messages
        ]
        rendered_documents = documents
        if rewritten.extra_body and rewritten.extra_body.documents:
            rendered_documents = [
                {"doc_id": d.doc_id, "text": d.text}
                for d in rewritten.extra_body.documents
            ]

        prompt = self.render(
            rendered_messages,
            adapter_name=intrinsic_name,
            documents=rendered_documents,
        )

        params = rewriter.parameters or {}
        max_tokens = int(params.get("max_completion_tokens", num_predict))
        wants_logprobs = bool(params.get("logprobs", False)) or _needs_likelihood(cfg)

        resp = self.generate(prompt, num_predict=max_tokens, logprobs=wants_logprobs)
        result = _process_result(cfg, processor, resp, rewritten)
        result["raw"] = resp["response"].strip()
        return result

    def answer(
        self,
        messages: list[dict],
        *,
        documents: list[dict] | None = None,
        num_predict: int = 512,
    ) -> str:
        """Generate a grounded answer from the base model (no adapter token)."""
        prompt = self.render(messages, documents=documents)
        resp = self.generate(prompt, num_predict=num_predict)
        return resp["response"].strip()


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _needs_likelihood(cfg: dict) -> bool:
    return any(
        t.get("type") == "likelihood" for t in (cfg.get("transformations") or [])
    )


def _process_result(cfg: dict, processor, resp: dict, rewritten=None) -> dict:
    """Turn an Ollama response into the intrinsic's structured result.

    Handles the likelihood-scored guardian adapters (via token logprobs),
    citation-style structural transformations (via Mellea's processor), and
    plain-JSON adapters.
    """
    text = resp["response"].strip()

    if _needs_likelihood(cfg):
        like = next(t for t in cfg["transformations"] if t.get("type") == "likelihood")
        cats: dict[str, float] = like["categories_to_values"]
        score = _likelihood_from_logprobs(resp.get("logprobs"), cats)
        if score is None:
            parsed = _try_json(text) or {}
            label = str(parsed.get("score", "")).strip().lower()
            score = cats.get(label, 0.0)
        nested = any(
            t.get("type") == "nest" and t.get("field_name") == "guardian"
            for t in cfg.get("transformations", [])
        )
        return {"guardian": {"score": score}} if nested else {"score": score}

    if _needs_processor(cfg):
        parsed = _run_processor(processor, text, rewritten)
        return {"parsed": parsed}

    parsed = _try_json(text)
    for t in cfg.get("transformations") or []:
        if t.get("type") == "nest" and not t.get("input_path"):
            parsed = {t["field_name"]: parsed}
    return {"parsed": parsed}


def _needs_processor(cfg: dict) -> bool:
    structural = {
        "explode",
        "decode_sentences",
        "merge_spans",
        "project",
        "drop_duplicates",
    }
    return any(t.get("type") in structural for t in (cfg.get("transformations") or []))


def _run_processor(processor, text: str, rewritten) -> Any:
    from mellea.formatters.granite.base.types import ChatCompletionResponse

    response = ChatCompletionResponse.model_validate(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    out = processor.transform(response, rewritten)
    return json.loads(out.choices[0].message.content)


def _likelihood_from_logprobs(
    logprobs: list[dict] | None, categories: dict[str, float]
) -> float | None:
    """Probability-weighted value from the first matching category token."""
    if not logprobs:
        return None
    cats = {k.lower(): v for k, v in categories.items()}
    for entry in logprobs:
        tok = entry.get("token", "").strip().lower()
        if tok in cats:
            p = math.exp(entry["logprob"])
            this_val = cats[tok]
            others = [v for k, v in cats.items() if k != tok]
            other_val = others[0] if others else 0.0
            return p * this_val + (1.0 - p) * other_val
    return None
