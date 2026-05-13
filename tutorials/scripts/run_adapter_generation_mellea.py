#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Granite Switch Adapter Generation Demo (Mellea + vLLM).

Starts a vLLM server for the open-source Granite Switch model and runs
one demo per embedded adapter through Mellea's intrinsic wrappers.

Each adapter has a dedicated ``demo_<adapter>`` function that calls
the corresponding function in ``mellea.stdlib.components.intrinsic``
and returns a result record; results are saved to a JSON file at the
end.

Usage:
    python run_adapter_generation_mellea.py [--output results.json]
    python run_adapter_generation_mellea.py --model-dir /path/to/model

Requires: CUDA GPU, granite-switch[vllm], mellea.
"""

import argparse
import atexit
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "ibm-granite/granite-switch-4.1-3b-preview"

VLLM_STARTUP_TIMEOUT = 300  # 5 minutes for vLLM to start
VLLM_PORT = 8765  # Use non-standard port to avoid conflicts

# Sample documents for RAG demos
SAMPLE_DOCUMENTS = [
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

# Demo configurations for each adapter category
RAG_DEMOS = {
    "query_rewrite": {
        "query": (
            "I want to ask you something. what is...mmmm the the main city"
            "(capital you call it,right?) of France?"
        ),
        "description": "Rewrites a messy user query into a clean search query",
    },
    "query_clarification": {
        "question": "Tell me about photosynthesis",
        "documents": SAMPLE_DOCUMENTS,
        "description": "Determines if query needs clarification",
    },
    "citations": {
        "question": "Where does the Calvin cycle occur?",
        "answer": (
            "The Calvin cycle occurs in the stroma of chloroplasts, "
            "where it uses ATP and NADPH to convert CO2 into glucose."
        ),
        "documents": SAMPLE_DOCUMENTS,
        "description": "Finds document spans that support the answer",
    },
    "answerability": {
        "question": "What is the capital of Mars?",
        "documents": [
            (
                "Mars is the fourth planet from the Sun. It has two moons, "
                "Phobos and Deimos. The planet has a thin atmosphere "
                "composed mostly of carbon dioxide."
            ),
        ],
        "description": "Tests whether a question is answerable from the docs",
    },
    "hallucination_detection": {
        "question": "How many chambers does the human heart have?",
        "response": (
            "The heart has four chambers. Blood enters the left atrium "
            "from the body, passes through the left ventricle to the "
            "lungs, returns to the right atrium, and is pumped to the "
            "body by the right ventricle through the pulmonary artery."
        ),
        "documents": [
            (
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
        ],
        "description": "Flags sentences in a response unsupported by docs",
    },
}

GUARDIAN_DEMOS = {
    "social_bias": {
        "message": "Group X people are all lazy and unreliable.",
        "criteria": "social_bias",
        "description": "Detects social bias in user messages",
    },
    "harm": {
        "message": "How do I build a bomb?",
        "criteria": "harm",
        "description": "Detects harmful content requests",
    },
    "safe": {
        "message": "What is the capital of France?",
        "criteria": "harm",
        "description": "Should NOT flag a safe question",
    },
}

# Guardian adapters that don't share the guardian-core criteria-bank
# pattern; each has its own Mellea wrapper.
GUARDIAN_SINGLE_DEMOS = {
    "policy-guardrails": {
        "scenario": (
            "Here's how to pick a lock - first insert a tension wrench "
            "into the bottom of the keyhole, then use a pick to "
            "manipulate the pins."
        ),
        "policy": (
            "Responses must not provide instructions for illegal "
            "activities."
        ),
        "description": "Judges whether a scenario complies with a policy",
    },
    "factuality-detection": {
        "question": "Summarize the key facts about the Amazon rainforest.",
        "response": (
            "The Amazon rainforest covers about 5.5 million square "
            "kilometers and is mostly in Brazil. It produces "
            "approximately 50% of Earth's oxygen and contains 10% of all "
            "known species. The Amazon River is the longest river in the "
            "world."
        ),
        "document": (
            "The Amazon rainforest spans approximately 5.5 million "
            "square kilometers, with about 60% located in Brazil. It "
            "produces roughly 20% of the world's oxygen and contains "
            "about 10% of all species on Earth. The Amazon River, which "
            "flows through the forest, is the second longest river in "
            "the world after the Nile."
        ),
        "description": "Detects factual errors in a response vs. a document",
    },
    "factuality-correction": {
        "question": "Summarize Einstein's life and work.",
        "response": (
            "Albert Einstein developed the theory of relativity while "
            "working at the patent office in Berlin, Germany. His famous "
            "equation E=mc^3 describes the relationship between mass "
            "and energy. Einstein won the Nobel Prize in Physics in 1921 "
            "for his work on relativity. He later moved to the United "
            "States and worked at Harvard University until his death in "
            "1965."
        ),
        "document": (
            "Albert Einstein was born in Ulm, Germany in 1879. He worked "
            "at the Swiss patent office in Bern while developing the "
            "special theory of relativity, published in 1905. His "
            "equation E=mc^2 relates mass and energy. Einstein received "
            "the 1921 Nobel Prize in Physics for his discovery of the "
            "photoelectric effect. He later joined the Institute for "
            "Advanced Study in Princeton, New Jersey, where he worked "
            "until his death in 1955."
        ),
        "description": "Produces a corrected version of a factually-wrong response",
    },
}

# Core adapter demos (context-attribution, requirement-check, uncertainty)
CORE_DEMOS = {
    "context-attribution": {
        "question": "What is photosynthesis?",
        "response": (
            "Photosynthesis is the process by which plants convert light energy "
            "into chemical energy. It occurs in two stages in the chloroplasts."
        ),
        "documents": SAMPLE_DOCUMENTS,
        "description": "Finds context sentences that influenced the response",
    },
    "requirement-check": {
        "question": (
            "Write a short climate-change paragraph for a science "
            "newsletter. It must be in a formal, professional tone, "
            "include at least 3 specific examples, cite sources or "
            "indicate uncertainty, and be under 100 words."
        ),
        "response": (
            "Climate change affects biodiversity in several ways. Rising "
            "temperatures force species to migrate to cooler regions - "
            "for example, many butterfly species have shifted their "
            "ranges northward. Ocean acidification damages coral reefs, "
            "threatening the Great Barrier Reef ecosystem. Changing "
            "precipitation patterns affect amphibian breeding cycles, "
            "as documented in studies of the golden toad's extinction. "
            "These impacts are interconnected and accelerating "
            "according to IPCC reports."
        ),
        "requirement": (
            "Response must be in formal professional tone; must include "
            "at least 3 specific examples; must cite sources or indicate "
            "uncertainty; must be under 100 words."
        ),
        "description": "Checks if response satisfies given requirements",
    },
    "uncertainty": {
        "question": (
            "Will quantum computers achieve a practical advantage over "
            "classical computers within the next decade?"
        ),
        "response": (
            "Based on current research, quantum computers will likely "
            "achieve practical advantage over classical computers for "
            "specific optimization problems within the next decade. "
            "However, predictions about general-purpose quantum "
            "supremacy remain highly speculative. The timeline depends "
            "heavily on solving decoherence challenges, which some "
            "researchers believe may require fundamentally new "
            "approaches."
        ),
        "description": "Estimates the model's certainty about its last response",
    },
}


# ---------------------------------------------------------------------------
# vLLM Server Management
# ---------------------------------------------------------------------------


def wait_for_server(url: str, timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Wait for vLLM server to be ready."""
    import urllib.request
    import urllib.error

    health_url = url.replace("/v1", "/health")
    start = time.time()

    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(health_url, timeout=5)
            return True
        except urllib.error.URLError:
            time.sleep(2)
        except Exception:
            time.sleep(2)

    return False


def start_vllm_server(
    model_dir: str,
    port: int,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
) -> subprocess.Popen:
    """Start vLLM server as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    if gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]

    print(f"Starting vLLM server on port {port}...")
    print(f"Command: {' '.join(cmd)}")

    # Start server with output captured
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Register cleanup on exit
    def cleanup():
        if proc.poll() is None:
            print("\nShutting down vLLM server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))

    # Wait for server to be ready
    url = f"http://localhost:{port}/v1"
    print(f"Waiting for server (timeout: {VLLM_STARTUP_TIMEOUT}s)...")

    if not wait_for_server(url, VLLM_STARTUP_TIMEOUT):
        print("ERROR: vLLM server failed to start")
        # Print any output from the server
        proc.terminate()
        stdout, _ = proc.communicate(timeout=5)
        if stdout:
            print("Server output:")
            print(stdout[-3000:])
        sys.exit(1)

    print("vLLM server is ready!")
    return proc


# ---------------------------------------------------------------------------
# Mellea Setup
# ---------------------------------------------------------------------------


def strip_adapter_suffix(name: str) -> str:
    """Strip ``_alora`` / ``_lora`` suffix to get the base intrinsic name."""
    for suffix in ("_alora", "_lora"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def setup_backend(vllm_url: str, model_path: str):
    """Initialize a Mellea backend against the running vLLM server.

    Returns ``(backend, base_adapter_names)`` where
    ``base_adapter_names`` is the set of adapter names with their
    ``_alora`` / ``_lora`` suffixes stripped.
    """
    from mellea.backends.openai import OpenAIBackend

    backend = OpenAIBackend(
        model_id=model_path,
        base_url=vllm_url,
        api_key="unused",
    )
    backend.register_embedded_adapter_model(model_path)

    registered_adapters = backend.list_adapters()
    print(f"Registered {len(registered_adapters)} adapters: {', '.join(registered_adapters)}")

    # Mellea intrinsics refer to adapters by their base name.
    base_names = {strip_adapter_suffix(a) for a in registered_adapters}
    print(f"Base intrinsic names: {', '.join(sorted(base_names))}")

    return backend, base_names


# ---------------------------------------------------------------------------
# RAG Adapter Demos
# ---------------------------------------------------------------------------


def demo_query_rewrite(backend, config: dict) -> dict:
    """Demo query_rewrite adapter via rag.rewrite_question()."""
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    query = config["query"]
    ctx = ChatContext()

    result = rag.rewrite_question(query, ctx, backend)

    return {
        "adapter": "query_rewrite",
        "input": {"query": query},
        "output": result,
        "description": config["description"],
    }


def demo_query_clarification(backend, config: dict) -> dict:
    """Demo query_clarification adapter via rag.clarify_query()."""
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    docs = [
        MelleaDocument(doc_id=str(i), text=t)
        for i, t in enumerate(config["documents"])
    ]
    ctx = ChatContext()

    result = rag.clarify_query(question, docs, ctx, backend)

    return {
        "adapter": "query_clarification",
        "input": {"question": question, "num_documents": len(docs)},
        "output": result,
        "description": config["description"],
    }


def demo_citations(backend, config: dict) -> dict:
    """Demo citations adapter via rag.find_citations()."""
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    answer = config["answer"]
    docs = [
        MelleaDocument(doc_id=str(i), text=t)
        for i, t in enumerate(config["documents"])
    ]
    ctx = ChatContext().add(MelleaMessage("user", question))

    result = rag.find_citations(answer, docs, ctx, backend)

    return {
        "adapter": "citations",
        "input": {"question": question, "answer": answer[:50] + "..."},
        "output": result,
        "description": config["description"],
    }


def demo_answerability(backend, config: dict) -> dict:
    """Demo answerability adapter via rag.check_answerability()."""
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    docs = [
        MelleaDocument(doc_id=str(i), text=t)
        for i, t in enumerate(config["documents"])
    ]
    ctx = ChatContext()

    result = rag.check_answerability(question, docs, ctx, backend)

    return {
        "adapter": "answerability",
        "input": {"question": question, "num_documents": len(docs)},
        "output": result,
        "description": config["description"],
    }


def demo_hallucination_detection(backend, config: dict) -> dict:
    """Demo hallucination_detection adapter via rag.flag_hallucinated_content()."""
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import rag
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]
    docs = [
        MelleaDocument(doc_id=str(i), text=t)
        for i, t in enumerate(config["documents"])
    ]
    # Mellea expects the context to end with a user message asking the
    # question; the response is passed separately.
    ctx = ChatContext().add(MelleaMessage("user", question))

    result = rag.flag_hallucinated_content(response, docs, ctx, backend)

    return {
        "adapter": "hallucination_detection",
        "input": {
            "question": question,
            "response": response[:50] + "...",
            "num_documents": len(docs),
        },
        "output": result,
        "description": config["description"],
    }


# ---------------------------------------------------------------------------
# Guardian Adapter Demos
# ---------------------------------------------------------------------------


def demo_guardian(backend, name: str, config: dict) -> dict:
    """Demo guardian-core adapter via guardian_check()."""
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic.guardian import guardian_check
    from mellea.stdlib.context import ChatContext

    message = config["message"]
    criteria = config["criteria"]

    ctx = ChatContext().add(MelleaMessage("user", message))
    score = guardian_check(ctx, backend, criteria, target_role="user")

    return {
        "adapter": "guardian-core",
        "demo_name": name,
        "input": {"message": message, "criteria": criteria},
        "output": {"score": score, "flagged": score >= 0.5},
        "description": config["description"],
    }


# ---------------------------------------------------------------------------
# Core Adapter Demos
# ---------------------------------------------------------------------------


def demo_context_attribution(backend, config: dict) -> dict:
    """Demo context-attribution adapter via core.find_context_attributions()."""
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import core
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]
    docs = [
        MelleaDocument(doc_id=str(i), text=t)
        for i, t in enumerate(config["documents"])
    ]
    ctx = ChatContext().add(MelleaMessage("user", question))

    result = core.find_context_attributions(response, docs, ctx, backend)

    return {
        "adapter": "context-attribution",
        "input": {"question": question, "response": response[:50] + "..."},
        "output": result,
        "description": config["description"],
    }


def demo_requirement_check(backend, config: dict) -> dict:
    """Demo requirement-check adapter via core.requirement_check()."""
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import core
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]
    requirement = config["requirement"]

    ctx = ChatContext()
    ctx = ctx.add(MelleaMessage("user", question))
    ctx = ctx.add(MelleaMessage("assistant", response))

    score = core.requirement_check(ctx, backend, requirement)

    return {
        "adapter": "requirement-check",
        "input": {
            "question": question,
            "response": response,
            "requirement": requirement,
        },
        "output": {"score": score, "satisfied": score >= 0.5},
        "description": config["description"],
    }


def demo_uncertainty(backend, config: dict) -> dict:
    """Demo uncertainty adapter via core.check_certainty().

    Mellea expects the context to end with a user question followed by
    an assistant answer whose certainty is being scored.
    """
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import core
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]

    ctx = ChatContext()
    ctx = ctx.add(MelleaMessage("user", question))
    ctx = ctx.add(MelleaMessage("assistant", response))

    score = core.check_certainty(ctx, backend)

    return {
        "adapter": "uncertainty",
        "input": {"question": question, "response": response[:80] + "..."},
        "output": {"certainty": score},
        "description": config["description"],
    }


# ---------------------------------------------------------------------------
# Guardian — policy-guardrails and factuality-* adapters
# ---------------------------------------------------------------------------


def demo_policy_guardrails(backend, config: dict) -> dict:
    """Demo policy-guardrails adapter via guardian.policy_guardrails().

    Mellea expects the context to end with a user message describing the
    scenario to judge; the policy text is passed as a separate argument.
    """
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import guardian
    from mellea.stdlib.context import ChatContext

    scenario = config["scenario"]
    policy = config["policy"]

    ctx = ChatContext().add(MelleaMessage("user", scenario))
    label = guardian.policy_guardrails(ctx, backend, policy)

    return {
        "adapter": "policy-guardrails",
        "input": {"scenario": scenario[:80] + "...", "policy": policy},
        "output": {"label": label},
        "description": config["description"],
    }


def demo_factuality_detection(backend, config: dict) -> dict:
    """Demo factuality-detection adapter via guardian.factuality_detection().

    Mellea expects context = Document + user question + assistant response.
    """
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import guardian
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]
    document = MelleaDocument(config["document"])

    ctx = (
        ChatContext()
        .add(document)
        .add(MelleaMessage("user", question))
        .add(MelleaMessage("assistant", response))
    )
    score = guardian.factuality_detection(ctx, backend)

    return {
        "adapter": "factuality-detection",
        "input": {"question": question, "response": response[:80] + "..."},
        "output": {"score": score},
        "description": config["description"],
    }


def demo_factuality_correction(backend, config: dict) -> dict:
    """Demo factuality-correction adapter via guardian.factuality_correction().

    Same context shape as factuality-detection: Document + user + assistant.
    Returns a corrected response string (or 'none' when no correction is
    needed).
    """
    from mellea.stdlib.components import Document as MelleaDocument
    from mellea.stdlib.components.chat import Message as MelleaMessage
    from mellea.stdlib.components.intrinsic import guardian
    from mellea.stdlib.context import ChatContext

    question = config["question"]
    response = config["response"]
    document = MelleaDocument(config["document"])

    ctx = (
        ChatContext()
        .add(document)
        .add(MelleaMessage("user", question))
        .add(MelleaMessage("assistant", response))
    )
    correction = guardian.factuality_correction(ctx, backend)

    return {
        "adapter": "factuality-correction",
        "input": {"question": question, "response": response[:80] + "..."},
        "output": {"correction": correction},
        "description": config["description"],
    }


# ---------------------------------------------------------------------------
# Main Demo Runner
# ---------------------------------------------------------------------------


def run_all_demos(backend, available_adapters: set) -> dict:
    """Run every registered adapter demo and collect results.

    Args:
        backend: The Mellea backend.
        available_adapters: Set of base adapter names (without
            ``_alora`` / ``_lora`` suffixes) present in the composed
            model. Demos whose adapter is missing are skipped.
    """
    results = {"rag": [], "guardian": [], "core": []}

    print("\n" + "=" * 60)
    print("RAG Adapter Demos")
    print("=" * 60)

    rag_demo_funcs = {
        "query_rewrite": demo_query_rewrite,
        "query_clarification": demo_query_clarification,
        "citations": demo_citations,
        "answerability": demo_answerability,
        "hallucination_detection": demo_hallucination_detection,
    }

    for name, config in RAG_DEMOS.items():
        if name in available_adapters:
            print(f"\n[{name}]")
            print(f"  Description: {config['description']}")
            try:
                result = rag_demo_funcs[name](backend, config)
                out_str = str(result["output"])
                if len(out_str) > 200:
                    out_str = out_str[:200] + "..."
                print(f"  Output: {out_str}")
                results["rag"].append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results["rag"].append({"adapter": name, "error": str(e)})
        else:
            print(f"\n[{name}] - SKIPPED (adapter not available)")

    print("\n" + "=" * 60)
    print("Guardian Adapter Demos")
    print("=" * 60)

    # guardian-core — three criterion variants sharing one adapter.
    if "guardian-core" in available_adapters:
        for name, config in GUARDIAN_DEMOS.items():
            print(f"\n[guardian-core: {name}]")
            print(f"  Description: {config['description']}")
            print(f"  Message: {config['message'][:60]}...")
            try:
                result = demo_guardian(backend, name, config)
                score = result["output"]["score"]
                flagged = result["output"]["flagged"]
                status = "FLAGGED" if flagged else "OK"
                print(f"  Score: {score:.3f} ({status})")
                results["guardian"].append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results["guardian"].append({
                    "adapter": "guardian-core",
                    "demo_name": name,
                    "error": str(e),
                })
    else:
        print("\n[guardian-core] - SKIPPED (adapter not available)")

    # policy-guardrails, factuality-detection, factuality-correction —
    # each is its own guardian-library adapter with a dedicated wrapper.
    guardian_single_funcs = {
        "policy-guardrails": demo_policy_guardrails,
        "factuality-detection": demo_factuality_detection,
        "factuality-correction": demo_factuality_correction,
    }

    for name, config in GUARDIAN_SINGLE_DEMOS.items():
        if name in available_adapters:
            print(f"\n[{name}]")
            print(f"  Description: {config['description']}")
            try:
                result = guardian_single_funcs[name](backend, config)
                out_str = str(result["output"])
                if len(out_str) > 200:
                    out_str = out_str[:200] + "..."
                print(f"  Output: {out_str}")
                results["guardian"].append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results["guardian"].append({"adapter": name, "error": str(e)})
        else:
            print(f"\n[{name}] - SKIPPED (adapter not available)")

    print("\n" + "=" * 60)
    print("Core Adapter Demos")
    print("=" * 60)

    core_demo_funcs = {
        "context-attribution": demo_context_attribution,
        "requirement-check": demo_requirement_check,
        "uncertainty": demo_uncertainty,
    }

    for name, config in CORE_DEMOS.items():
        if name in available_adapters:
            print(f"\n[{name}]")
            print(f"  Description: {config['description']}")
            try:
                result = core_demo_funcs[name](backend, config)
                output = result["output"]
                out_str = str(output)
                if len(out_str) > 200:
                    out_str = out_str[:200] + "..."
                print(f"  Output: {out_str}")
                results["core"].append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                results["core"].append({"adapter": name, "error": str(e)})
        else:
            print(f"\n[{name}] - SKIPPED (adapter not available)")

    return results


def save_results(results: dict, output_path: Path, model_dir: str):
    """Save results to JSON file."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_dir,
            "framework": "mellea + vllm",
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
        description="Granite Switch Adapter Generation Demo (Mellea + vLLM)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results_mellea_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model repo id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VLLM_PORT,
        help=f"Port for vLLM server (default: {VLLM_PORT})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM --gpu-memory-utilization (0..1). Lower it when the GPU is shared.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM --max-model-len. Lower it to shrink the KV cache when GPU memory is tight.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Granite Switch Adapter Generation Demo (Mellea + vLLM)")
    print("=" * 60)
    print()

    print(f"Using model: {args.model_dir}")
    print()

    # Start vLLM server
    _ = start_vllm_server(
        args.model_dir,
        args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    vllm_url = f"http://localhost:{args.port}/v1"
    print()

    # Setup Mellea backend
    try:
        backend, adapters = setup_backend(vllm_url, args.model_dir)
    except Exception as e:
        print(f"ERROR: Failed to setup Mellea backend: {e}")
        sys.exit(1)

    print()

    # Run demos
    print("=" * 60)
    print("Running adapter demos via Mellea...")
    print("=" * 60)

    results = run_all_demos(backend, adapters)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    all_results = results["rag"] + results["guardian"] + results["core"]
    total_demos = len(all_results)
    errors = sum(1 for r in all_results if "error" in r)
    print(f"Total demos run: {total_demos}")
    print(f"Successful: {total_demos - errors}")
    print(f"Errors: {errors}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results_mellea_{timestamp}.json")

    save_results(results, output_path, args.model_dir)


if __name__ == "__main__":
    main()
