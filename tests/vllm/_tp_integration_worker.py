# SPDX-License-Identifier: Apache-2.0
"""TP integration test worker — runs in subprocess to avoid CUDA fork issues.

Commands:
  build            — Compose a zero-adapter switch model and save to disk (CPU)
  build-compose    — Compose via the CLI compose script with adapter repos (CPU)
  build-granitemoe — Compose a tiny synthetic pure-sparse MoE checkpoint (CPU)
  run              — Load model in vLLM with given TP size, generate, save output (GPU)

``run --fault <name>`` injects a sharding bug (see :data:`FAULTS`) so the test can
prove a TP comparison is able to fail. Each fault lives only under tensor
parallelism -- it acts on one rank's shard -- so a TP=1 run is unaffected and only
the sharded path breaks, as with a real slicing or reduce bug.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAIN_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    # >512 tokens to exercise FA3 non-CUDA-graph path under TP (#104)
    ("Summarize the following document in detail. " * 100).strip(),
]

CHAT_MESSAGES = [
    {"role": "user", "content": "Is this document relevant to the query?"},
]

# Used with ``--token-prompts``: a synthetic checkpoint ships no tokenizer, so
# the string prompts above are unusable.  The two rows differ only at position 2
# -- control token vs. an ordinary id -- so the switch's own attention layer runs
# under sharding on one of them and not the other.
_CONTROL_TOKEN_ID = 250
TOKEN_PROMPTS = [
    [10, 11, _CONTROL_TOKEN_ID, 12, 13, 14, 15, 16],
    [10, 11, 30, 12, 13, 14, 15, 16],
]


# ── Sharding fault injection ──────────────────────────────────────────

#: Sharding bugs the sensitivity guard injects at TP=2. Each patches OUR code
#: (src/granite_switch/vllm), never vLLM's, so it does not depend on vLLM
#: internals that move between the supported lines.
FAULTS = {
    # Column-parallel LoRA-B: a rank loads the other rank's output rows, the
    # shape of an off-by-one slice offset in slice_lora_b_weight.
    "lora_b_wrong_shard": "SwitchedLoRALinear.slice_lora_b_weight takes rank r+1's rows",
    # Row-parallel LoRA-A: a rank loads the other rank's input columns.
    "lora_a_wrong_shard": "SwitchedLoRALinear.slice_lora_a_weight takes rank r+1's columns",
    # Row-parallel layers return their rank-local partial sum, unreduced.
    "skip_row_reduce": "SwitchedLoRALinear skips the row-parallel all-reduce",
    # A base (non-LoRA) row-parallel weight: rank 0's o_proj shard is negated
    # after load. Unlike the LoRA faults this bites with no adapter active, so it
    # checks the sharded layers reach the output at all.
    "base_o_proj_shard": "rank 0's o_proj base weight shard is negated after load",
}

#: Written to a directory put on PYTHONPATH, so it runs at interpreter start in
#: every process -- vLLM's TP worker processes are spawned, not forked, and
#: would not inherit a patch applied here. Mirrors the trace hooks in
#: tests/integration/_conversation_concurrency_worker.py.
_FAULT_SITECUSTOMIZE = r"""
import os

_FAULT = os.environ.get("GS_TP_FAULT")


def _install():
    from granite_switch.vllm.core import lora as _lora

    cls = _lora.SwitchedLoRALinear

    def _other_rank_slice(orig, dim):
        def patched(self, full_weight, slice_idx=0):
            if self.tp_size <= 1:
                return orig(self, full_weight, slice_idx)
            # Only touch the dimension this slicer actually shards.
            if dim == "a" and not self._is_row_parallel:
                return orig(self, full_weight, slice_idx)
            if dim == "b" and not self._is_column_parallel:
                return orig(self, full_weight, slice_idx)
            saved = self.tp_rank
            self.tp_rank = (saved + 1) % self.tp_size
            try:
                return orig(self, full_weight, slice_idx)
            finally:
                self.tp_rank = saved

        return patched

    if _FAULT == "lora_b_wrong_shard":
        cls.slice_lora_b_weight = _other_rank_slice(cls.slice_lora_b_weight, "b")
    elif _FAULT == "lora_a_wrong_shard":
        cls.slice_lora_a_weight = _other_rank_slice(cls.slice_lora_a_weight, "a")
    elif _FAULT in ("skip_row_reduce", "base_o_proj_shard"):
        import torch

        from granite_switch.vllm import granite_switch_model as _gsm

        orig_load = _gsm.GraniteSwitchForCausalLM.load_weights

        def load_weights(self, weights):
            loaded = orig_load(self, weights)
            hit = 0
            for name, mod in self.named_modules():
                if _FAULT == "skip_row_reduce":
                    if isinstance(mod, cls) and mod._row_parallel_reduce:
                        mod._row_parallel_reduce = False
                        hit += 1
                elif name.endswith(".o_proj"):
                    base = getattr(mod, "base_layer", mod)
                    rank = getattr(mod, "tp_rank", None)
                    if rank is None:
                        from vllm.distributed import get_tensor_model_parallel_rank

                        rank = get_tensor_model_parallel_rank()
                    if rank == 0 and getattr(base, "weight", None) is not None:
                        # Negate exactly ONCE. After finalize_weights a
                        # SwitchedLoRALinear's base_layer.weight is a VIEW of
                        # w_ext[:N_total] (forward reads w_ext), so negating both
                        # would cancel out and leave the weights untouched.
                        w_ext = getattr(mod, "w_ext", None)
                        n = getattr(mod, "_N_total", None)
                        target = w_ext[:n] if w_ext is not None and n is not None else base.weight
                        with torch.no_grad():
                            target.neg_()
                        hit += 1
            print(f"GS_TP_FAULT {_FAULT}: applied to {hit} module(s)", flush=True)
            return loaded

        _gsm.GraniteSwitchForCausalLM.load_weights = load_weights
    print(f"GS_TP_FAULT {_FAULT}: installed in pid {os.getpid()}", flush=True)


if _FAULT:
    # Install once granite_switch's vLLM model module is imported, not now:
    # sitecustomize runs at interpreter start, before the package is
    # importable in every process, and importing it eagerly here would also
    # load vLLM into processes that never use it.
    import importlib.abc
    import sys

    _TRIGGER = "granite_switch.vllm.granite_switch_model"

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != _TRIGGER:
                return None
            sys.meta_path.remove(self)
            import importlib.util

            spec = importlib.util.find_spec(name)
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _install()

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
"""


def _fault_env(fault, work_dir):
    """Env for a run with ``fault`` injected via sitecustomize, or {} for none."""
    if not fault:
        return {}
    if fault not in FAULTS:
        raise SystemExit(f"unknown --fault {fault!r}; known: {sorted(FAULTS)}")
    hook_dir = Path(work_dir) / "fault_hook"
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "sitecustomize.py").write_text(_FAULT_SITECUSTOMIZE)
    path = os.environ.get("PYTHONPATH", "")
    return {
        "GS_TP_FAULT": fault,
        "PYTHONPATH": f"{hook_dir}{os.pathsep}{path}" if path else str(hook_dir),
    }


def cmd_build(args):
    """Build a zero-adapter GraniteSwitch model."""
    from granite_switch.composer import GraniteSwitchComposer

    base_model = args.base_model
    output_dir = args.output_dir

    base_config = AutoConfig.from_pretrained(base_model)
    vocab_size = base_config.vocab_size

    adapter_token_id = vocab_size - 100
    muted_token_id = vocab_size - 101

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model,
        built_in_adapter_names=["test"],
        adapter_names=["test"],
        adapter_token_ids=[adapter_token_id],
        adapter_substitute_token_ids=[1],
        muted_adapter_token_ids=[muted_token_id],
    )

    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    del model

    print("BUILD_OK")
    return 0


def cmd_build_compose(args):
    """Build a GraniteSwitch model using the CLI compose script."""
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        args.base_model,
        "--output",
        args.output_dir,
    ]
    for repo in args.adapter_repos:
        cmd.extend(["--adapters", repo])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        print(f"Compose failed (exit {result.returncode})")
        return result.returncode

    print("BUILD_COMPOSE_OK")
    return 0


def cmd_build_granitemoe(args):
    """Compose a tiny synthetic pure-sparse MoE checkpoint (no download).

    Deliberately not built with ``save_switch_model``: ``SwitchedLoRALinear``
    zero-initializes ``lora_B``, so a synthetic switch model has an identically
    zero adapter delta and a TP comparison over it degenerates into comparing two
    base-only runs.  Composing a real PEFT adapter is what makes the adapter live,
    and the guards below fail the build rather than let that pass silently.
    """
    from tests.shared.granitemoe_compose import (
        ADAPTER_BUILDERS,
        GPU_GEOMETRY,
        compose_granitemoe,
    )

    output_dir = Path(args.output_dir)
    work = output_dir.parent

    build = compose_granitemoe(
        name=args.variant,
        base_path=work / f"{args.variant}_moe_base",
        adapter_path=work / f"{args.variant}_moe_adapter",
        output_dir=output_dir,
        make_adapter=ADAPTER_BUILDERS[args.variant],
        geometry=GPU_GEOMETRY,
    )

    live = [
        name
        for name, param in build.model.named_parameters()
        if "lora_B" in name and param.abs().sum() > 0
    ]
    if not live:
        print("BUILD_FAIL: every lora_B is zero; the adapter is dead", file=sys.stderr)
        return 1
    if args.variant == "sr" and not any("cross_stream" in n for n in live):
        print(
            "BUILD_FAIL: cross_stream.lora_B is zero; the SR cross-stream "
            "injection is dead and only the q/o deltas would be compared",
            file=sys.stderr,
        )
        return 1

    config = json.loads((output_dir / "config.json").read_text())
    if config.get("shared_intermediate_size") != 0:
        print(
            f"BUILD_FAIL: shared_intermediate_size is "
            f"{config.get('shared_intermediate_size')!r}, not 0 — this is not the "
            f"pure-sparse path under test",
            file=sys.stderr,
        )
        return 1
    if not config.get("num_local_experts"):
        print("BUILD_FAIL: no expert bank in the composed config", file=sys.stderr)
        return 1

    del build
    print(f"BUILD_GRANITEMOE_OK live_lora_B={len(live)}")
    return 0


def cmd_run(args):
    """Load model in vLLM and generate."""
    if args.fault and os.environ.get("GS_TP_FAULT") != args.fault:
        # The fault must be in place in vLLM's spawned TP worker processes, which
        # only inherit it through the environment at interpreter start: re-run
        # this command with the sitecustomize hook on PYTHONPATH.
        env = {**os.environ, **_fault_env(args.fault, Path(args.output_path).parent)}
        return subprocess.run([sys.executable, *sys.argv], env=env).returncode
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    model_path = args.model_path
    tp_size = args.tp_size
    output_path = args.output_path

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
    )
    if args.token_prompts:
        # A synthetic checkpoint has no tokenizer, so vLLM must not try to load
        # one.  bf16 keeps the numerics on the same footing as the real-checkpoint
        # arms above, which is what the tolerances downstream were measured for.
        llm_kwargs.update(
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            gpu_memory_utilization=0.3,
        )

    llm = LLM(**llm_kwargs)

    # Request top-K logprobs at the first generated token of each prompt.
    # We compare distributions across TP sizes (within a tolerance) instead
    # of asserting byte-equality of greedy text — the latter is not a
    # well-defined invariant in bf16 across different all-reduce orderings.
    # See docs/TENSOR_PARALLEL_FIX.md for the analysis.
    sampling = SamplingParams(temperature=0.0, max_tokens=20, logprobs=20)

    def _top_logprobs(first_step_logprobs):
        """Convert the first-step Logprob dict to a JSON-serialisable list of
        (token_id, logprob) sorted descending by logprob."""
        if first_step_logprobs is None:
            return None
        items = [
            (int(tid), float(lp.logprob)) for tid, lp in first_step_logprobs.items()
        ]
        items.sort(key=lambda x: -x[1])
        return items

    records = []

    if args.token_prompts:
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in TOKEN_PROMPTS]
    else:
        prompts = PLAIN_PROMPTS

    outputs = llm.generate(prompts, sampling)
    for o in outputs:
        completion = o.outputs[0]
        first_step = completion.logprobs[0] if completion.logprobs else None
        records.append(
            {
                "text": completion.text,
                "first_token_topk": _top_logprobs(first_step),
            }
        )

    # One chat prompt per adapter. The composed chat template gates control-token
    # injection on ``adapter_name`` (composer/tokenizer_setup.py::
    # configure_chat_template emits "{%- if adapter_name is defined and
    # adapter_name in adapter_map %}"). This call previously passed
    # ``intrinsic_name``; Jinja silently treats an unknown kwarg as not-defined,
    # so no control token was ever injected and the chat arm of the TP comparison
    # agreed vacuously -- two servers both running plain base. Guard it rather
    # than trust the key.
    if args.intrinsic_name:
        tok = AutoTokenizer.from_pretrained(model_path)
    for name in args.intrinsic_name or []:
        rendered = tok.apply_chat_template(
            CHAT_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            adapter_name=name,
        )
        expected_token = f"<|{name}|>"
        if expected_token not in rendered:
            print(
                f"RUN_FAIL: chat template did not inject {expected_token} for "
                f"adapter_name={name!r}; the chat arm would compare "
                f"two base-only runs and agree vacuously. Rendered: {rendered[:300]!r}",
                file=sys.stderr,
            )
            return 1
        chat_outputs = llm.chat(
            CHAT_MESSAGES,
            sampling_params=sampling,
            chat_template_kwargs={"adapter_name": name},
        )
        for o in chat_outputs:
            completion = o.outputs[0]
            first_step = completion.logprobs[0] if completion.logprobs else None
            records.append(
                {
                    "text": completion.text,
                    "first_token_topk": _top_logprobs(first_step),
                }
            )

    with open(output_path, "w") as f:
        json.dump(records, f)

    del llm
    print("RUN_OK")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build")
    p_build.add_argument("--base-model", required=True)
    p_build.add_argument("--output-dir", required=True)

    p_compose = sub.add_parser("build-compose")
    p_compose.add_argument("--base-model", required=True)
    p_compose.add_argument("--output-dir", required=True)
    p_compose.add_argument("--adapter-repos", nargs="+", required=True)

    p_moe = sub.add_parser("build-granitemoe")
    p_moe.add_argument("--output-dir", required=True)
    p_moe.add_argument("--variant", choices=("lora", "sr"), required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--tp-size", type=int, required=True)
    p_run.add_argument("--output-path", required=True)
    p_run.add_argument(
        "--intrinsic-name",
        action="append",
        default=None,
        help="Adds a chat-template prompt activating this adapter; repeatable",
    )
    p_run.add_argument(
        "--token-prompts",
        action="store_true",
        help="Feed raw token ids instead of strings (checkpoint has no tokenizer)",
    )
    p_run.add_argument(
        "--fault",
        choices=sorted(FAULTS),
        default=None,
        help="Inject a sharding bug (only bites at TP>1); see FAULTS",
    )

    args = parser.parse_args()
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "build-compose":
        return cmd_build_compose(args)
    elif args.command == "build-granitemoe":
        return cmd_build_granitemoe(args)
    elif args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
