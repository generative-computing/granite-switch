# Using Mellea with Granite Switch

This guide shows how to use **Mellea** to invoke embedded adapters in a Granite Switch model served by vLLM.

## Overview

**Mellea** is a Python framework that exposes a programming model for LLMs with an opinionated view on how modern software design practices and LLMs ought to be cross-leveraged to maximize accuracy, control and robustness in applications leveraging LLMs. In the context of Granite Switch, Mellea provides high-level intrinsic functions (Guardian, RAG, Core) that automatically route through the correct control tokens. Instead of manually constructing prompts with control tokens, you call simple Python functions.

**Granite Switch** is the model architecture - an instruction-following model from the Granite Family with multiple LoRA adapters embedded as a single checkpoint.

**vLLM** is a high-performance inference engine for LLMs. Granite Switch currently supports vLLM as the inference backend; other inference engines may be supported in the future.

Together, Mellea + Granite Switch + vLLM provide a production-ready inference stack for adapter-based AI applications.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Code     │────>│     Mellea      │────>│  vLLM Server    │
│                 │     │  (intrinsics)   │     │ (Granite Switch)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              │ Adds control tokens    │ Routes to
                              │ automatically          │ embedded adapters
                              v                        v
                        guardian_check()         <|guardian-core|>
                        rag.rewrite_question()   <|query_rewrite|>
```

## Prerequisites

1. **A composed Granite Switch model** with the adapters you need
2. **A running vLLM server** serving that model
3. **Mellea** installed

See [PREREQUISITES.md](../PREREQUISITES.md) for detailed setup instructions.

## Installation

```bash
# Install Mellea from source
pip install "git+https://github.com/generative-computing/mellea.git@main"
```

## Quick Example

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components.chat import Message as MelleaMessage
from mellea.stdlib.components.intrinsic.guardian import guardian_check

# 1. Connect to vLLM server
backend = OpenAIBackend(
    model_id="path/to/your/granite-switch-model",
    base_url="http://localhost:8000/v1",
    api_key="unused",  # vLLM doesn't require auth by default
)

# 2. Register embedded adapters (tells Mellea which adapters are available)
backend.register_embedded_adapter_model("path/to/your/granite-switch-model")
print(f"Available adapters: {backend.list_adapters()}")

# 3. Use Guardian adapter for harm detection
ctx = ChatContext().add(MelleaMessage("user", "Is this message safe?"))
score = guardian_check(ctx, backend, "harm", target_role="user")
print(f"Harm score: {score:.3f}")  # 0.0 = safe, 1.0 = harmful
```

## Available Intrinsics

Mellea provides wrappers for three categories of intrinsics:

### Guardian Intrinsics

```python
from mellea.stdlib.components.intrinsic.guardian import (
    guardian_check,        # Score messages against criteria
    policy_guardrails,     # Evaluate against policy documents
    factuality_detection,  # Flag factual errors
    factuality_correction, # Fix factual errors
)

# Harm check with pre-baked criteria
score = guardian_check(ctx, backend, "harm", target_role="user")

# Or custom criteria
custom_criteria = "The message contains personal information like SSN or credit card numbers."
score = guardian_check(ctx, backend, custom_criteria, target_role="user")
```

**Pre-baked criteria** (from `CRITERIA_BANK`): `harm`, `social_bias`, `jailbreak`, `profanity`, `unethical_behavior`, `violence`, `groundedness`, `answer_relevance`, `context_relevance`, `function_call`

### RAG Intrinsics

```python
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.components import Document as MelleaDocument

# Create documents
documents = [
    MelleaDocument(doc_id="0", text="Paris is the capital of France."),
    MelleaDocument(doc_id="1", text="Berlin is the capital of Germany."),
]

# Rewrite a messy query
query = "what is...the capital of france??"
rewritten = rag.rewrite_question(query, ctx, backend)
# Output: "What is the capital of France?"

# Check if documents can answer the query
answerability = rag.check_answerability(rewritten, documents, ctx, backend)
# Output: "answerable" or "unanswerable"

# Get clarifying question if needed
clarification = rag.clarify_query(rewritten, documents, ctx, backend)
# Output: "CLEAR" or a follow-up question

# Find citations for an answer
answer = "The capital of France is Paris."
citations = rag.find_citations(answer, documents, ctx, backend)
# Output: [{"doc_id": "0", "span": "Paris is the capital of France."}]
```

### Core Intrinsics

```python
from mellea.stdlib.components.intrinsic.core import (
    check_certainty,           # Model's confidence
    requirement_check,         # Verify requirements met
    find_context_attributions, # Attribute to sources
)

# Check model certainty
certainty = check_certainty(ctx, backend)

# Verify response meets requirements
requirements = ["Must be formal", "Must cite sources"]
check = requirement_check(response, requirements, ctx, backend)
```

## Low-Level Adapter Invocation

For adapters not yet wrapped by Mellea, use the `Intrinsic` AST node directly:

```python
import json
from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic
import mellea.stdlib.functional as mfuncs
from mellea.backends import ModelOption

# Invoke adapter by name
ctx = ChatContext().add(MelleaMessage("user", "Your prompt here"))

out, _ = mfuncs.act(
    Intrinsic("query_rewrite"),  # adapter name
    ctx, backend,
    model_options={ModelOption.TEMPERATURE: 0.0},
    strategy=None,
)
result = json.loads(str(out))
print(result)
```

## Generating with the Base Model

For non-intrinsic generation (regular chat), use `mfuncs.act` with a message:

```python
import mellea.stdlib.functional as mfuncs
from mellea.backends import ModelOption

# Regular generation (no adapter)
ctx = ChatContext().add(MelleaMessage("user", "What is 2+2?"))
out, _ = mfuncs.act(
    MelleaMessage("user", "What is 2+2?"),
    ctx, backend,
    model_options={ModelOption.TEMPERATURE: 0.0},
)
answer = str(out)
```

## Full Example: RAG Pipeline

```python
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.context import ChatContext
from mellea.stdlib.components import Document as MelleaDocument
from mellea.stdlib.components.chat import Message as MelleaMessage
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.components.intrinsic.guardian import guardian_check
import mellea.stdlib.functional as mfuncs
from mellea.backends import ModelOption

# Setup
backend = OpenAIBackend(
    model_id="your-model",
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
backend.register_embedded_adapter_model("your-model")

# Documents (from your retriever)
documents = [
    MelleaDocument(doc_id="0", text="The Eiffel Tower is 330 meters tall."),
    MelleaDocument(doc_id="1", text="It was built in 1889 for the World's Fair."),
]

# User query
query = "how tall is the eiffel tower?"
ctx = ChatContext()

# 1. Check for harmful content
harm_score = guardian_check(
    ctx.add(MelleaMessage("user", query)),
    backend, "harm", target_role="user"
)
if harm_score >= 0.5:
    print("Query blocked for safety")
    exit()

# 2. Rewrite query
rewritten = rag.rewrite_question(query, ctx, backend)
print(f"Rewritten: {rewritten}")

# 3. Check answerability
answerability = rag.check_answerability(rewritten, documents, ctx, backend)
if answerability == "unanswerable":
    clarification = rag.clarify_query(rewritten, documents, ctx, backend)
    print(f"Need clarification: {clarification}")
    exit()

# 4. Generate answer
out, _ = mfuncs.act(
    MelleaMessage("user", rewritten, documents=documents),
    ctx, backend,
    model_options={ModelOption.TEMPERATURE: 0.0},
)
answer = str(out)
print(f"Answer: {answer}")

# 5. Get citations
ctx_with_q = ctx.add(MelleaMessage("user", rewritten))
citations = rag.find_citations(answer, documents, ctx_with_q, backend)
print(f"Citations: {citations}")
```

## Next Steps

- **[Hello Adapter](../notebooks/00_hello_adapter.ipynb)** - Minimal embedded-adapter invocation via the HuggingFace backend
- **[Bring Your Own Adapter](bring_your_own_adapter.md)** - Train a custom adapter and compose it in
- **[Compare Inference Throughput](compare_inference_throughput.md)** - Benchmark ALORA vs LoRA on a 6-step RAG pipeline
- **[Mellea Repository](https://github.com/generative-computing/mellea)** - Full documentation
- **[Granite Models](https://huggingface.co/ibm-granite)**
- **[vLLM Documentation](https://docs.vllm.ai/)**
