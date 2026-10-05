# SPDX-License-Identifier: Apache-2.0
"""Decode throughput, measured by the switch benchmark's own driver.

``benchmarks/bench_switch_repro.py`` is the switch benchmark's driver, copied
unchanged; this module only builds its command line for this benchmark's
adapters and reads its record back. ``run_benchmark.py`` runs it once per
engine, one after the other on the same GPU, with the compile caches cleared
before each, as that benchmark's sweep does (``run_switch_repro_sweep.sh``,
its ``run_block``).

Per technology, two of its arms, on this model's staged adapters (N of them):

======== ============================== ==========================================
tech     granite-switch                 stock vLLM
======== ============================== ==========================================
lora     ``gs-lora-vllm``: a checkpoint  ``native-lora``: the same LoRA checkpoints
         of the LoRA adapters
alora    ``gs-lora-vllm``: a checkpoint  ``native-lora``: the aLoRA checkpoints
         of the aLoRA adapters           without their invocation tokens
sr       ``gs-sr-vllm``: a checkpoint    ``native-sr``: the SR checkpoints, their
         of the SR adapters              ``cross_stream`` skipped at load
======== ============================== ==========================================

and its prompt-1 decode cell, the one its batch_decode figure plots: one-token
prompts, every request on one of the N adapters (``--adapter-fractions 100``),
exactly ``generated_tokens`` new tokens each, ``warmup_runs`` untimed and
``timed_runs`` timed runs, a 4,096-token context and CUDA graphs up to batch
1,024, with its gates checked first. Only the batch is fixed here, where the
figure sweeps it.

Its record becomes the cell's entry, with that benchmark's formula: tok/s =
batch x generated tokens / the median run's seconds.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .common import error

DRIVER = "benchmarks/bench_switch_repro.py"
# The cell's fixed settings, as that benchmark's sweep passes them.
PINNED = {
    "prompt_tokens": 1,
    "adapter_fraction": 100,
    "max_model_len": 4096,
    "cudagraph_capture_size": 1024,
    "tensor_parallel_size": 1,
}
# Its arm for each (technology, engine).
ARMS = {
    ("lora", "gs"): "gs-lora-vllm",
    ("lora", "native"): "native-lora",
    ("alora", "gs"): "gs-lora-vllm",
    ("alora", "native"): "native-lora",
    ("sr", "gs"): "gs-sr-vllm",
    ("sr", "native"): "native-sr",
}
# Recorded by the driver, but not for a public page.
PRIVATE_PROVENANCE = ("hostname",)


def command(
    tech: str, engine: str, model: str, paths: list[str], settings: dict, dump: Path
) -> list[str]:
    """The driver's arguments for one engine, as ``run_block`` passes them."""
    arm = ARMS[tech, engine]
    n = len(paths)
    args = ["--arm", arm, "--model", model, "--num-adapters", str(n)]
    if engine == "native":
        args += ["--lora-path", ",".join(paths)]
        if tech == "sr":  # stock vLLM has no module for SR's shunt
            args += ["--lora-skip-prefixes", "cross_stream"]
    return [
        *args,
        "--decode-sweep",
        "--batch-sizes", str(settings["batch"]),
        "--adapter-fractions", str(PINNED["adapter_fraction"]),
        "--decode-tokens", str(settings["generated_tokens"]),
        "--input-tokens", str(PINNED["prompt_tokens"]),
        "--max-model-len", str(PINNED["max_model_len"]),
        "--tensor-parallel-size", str(PINNED["tensor_parallel_size"]),
        "--num-runs", str(settings["timed_runs"]),
        "--warmup-runs", str(settings["warmup_runs"]),
        "--cudagraph-capture-size", str(PINNED["cudagraph_capture_size"]),
        "--tag", f"{arm}_N{n}",
        "--dump-iters", str(dump),
    ]  # fmt: skip


def entry(record: dict) -> dict:
    """A cell's throughput entry from the driver's decode record."""
    times = [ms / 1000 for ms in record["iters_ms"]]
    median = statistics.median(times)
    provenance = {
        k: v
        for k, v in (record.get("provenance") or {}).items()
        if k not in PRIVATE_PROVENANCE
    }
    return {
        "tokens_per_s": round(record["batch"] * record["decode_tokens"] / median, 1),
        "median_s": round(median, 4),
        "runs_s": [round(t, 4) for t in times],
        "batch": record["batch"],
        "generated_tokens": record["decode_tokens"],
        "prompt_tokens": record["input_tokens"],
        "prompt_tokens_realized": record.get("prompt_tokens_realized"),
        "adapters": record["N"],
        "arm": record["arm"],
        "gpu": provenance.get("gpu_name"),
        "preflight": record.get("preflight"),
        "provenance": provenance,
        "smi_before": record.get("smi_before"),
        "smi_after": record.get("smi_after"),
    }


def read_entry(dump: Path, settings: dict) -> dict:
    """The cell's entry from the driver's dump file, or an error."""
    for line in dump.read_text().splitlines():
        record = json.loads(line) if line.strip() else {}
        if (
            record.get("phase") == "decode"
            and record.get("frac") == PINNED["adapter_fraction"]
            and record.get("batch") == settings["batch"]
        ):
            return entry(record)
    return error("the driver wrote no decode record")


def failure(output_tail: list[str]) -> str:
    """Why the driver failed, from its last output lines: its FATAL line if any."""
    for line in reversed(output_tail):
        if "FATAL" in line:
            reason = line.split("FATAL", 1)[1].lstrip(" :")
            return "refused: " + reason.split(". ")[0][:200]
    return "throughput run failed"
