# SPDX-License-Identifier: Apache-2.0
"""Worker for Granite Switch vLLM pipeline-parallel generation tests.

This runs outside the parent pytest process so the parent does not create a
CUDA context before vLLM starts its distributed workers.
"""

import argparse
import faulthandler
import gc
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from tests.shared.generation_models import (
    DENSE_CFG,
    save_switch_model,
    single_overrides,
)

DECODER_LAYERS = 40
CONTROL_TOKEN_ID = 250
GENERATE_TIMEOUT_SECONDS = 180


def _log(message: str) -> None:
    print(f"PP_GENERATION_PHASE {message}", flush=True)


def _build_model(tmpdir):
    """Build a tiny model whose config has 41 layers and decoder has 40."""
    base_cfg = {
        **DENSE_CFG,
        "num_hidden_layers": DECODER_LAYERS,
        "layer_types": ["attention"] * DECODER_LAYERS,
    }
    return save_switch_model(
        base_cfg,
        single_overrides(base_cfg),
        tmpdir=tmpdir,
    )


def run_pp_generation(tmpdir):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    _log("build_model_start")
    model_dir = _build_model(tmpdir)
    _log("build_model_done")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    _log("llm_init_start")
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=64,
        gpu_memory_utilization=0.3,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
        enforce_eager=True,
    )
    _log("llm_init_done")

    prompt = TokensPrompt(
        prompt_token_ids=[10, 11, CONTROL_TOKEN_ID, 12, 13, 14, 15, 16],
    )
    sampling_params = SamplingParams(max_tokens=4, temperature=0.0)
    _log("generate_start")
    faulthandler.dump_traceback_later(
        GENERATE_TIMEOUT_SECONDS,
        file=sys.stderr,
        exit=True,
    )
    try:
        outputs = llm.generate(
            prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()
    _log("generate_done")
    generated = outputs[0].outputs[0].token_ids

    assert len(generated) == 4, f"Expected 4 generated tokens, got {len(generated)}"

    _log("cleanup_start")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    _log("cleanup_done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", required=True)
    args = parser.parse_args()

    run_pp_generation(args.tmpdir)
    print("PP_GENERATION_OK")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

    # This worker is isolated from pytest specifically to contain vLLM's
    # multiprocessing state. Avoid hanging in interpreter teardown if a vLLM
    # helper process/thread survives after the test assertion has passed.
    os._exit(exit_code)
