# SPDX-License-Identifier: Apache-2.0
"""Small composed checkpoints shared by CPU and vLLM stream tests."""

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from granite_switch.composer import GraniteSwitchComposer
from tests.shared.base_models import dims_from_config, write_tiny_granite_base
from tests.shared.synthetic_adapters import (
    PROFILES,
    attn_modules,
    base_weight_norms,
    mlp_modules,
    synthesize_adapter,
)


def compose_stream_model(root, streams, *, dual_stream=False):
    """Compose real weights with vLLM-compatible head dimensions and LoRA rank."""
    base = write_tiny_granite_base(
        root / "base",
        num_heads=2,
        num_kv_heads=2,
        head_dim=32,
        vocab_size=max(300, 250 + len(streams)),
    )
    dims = dims_from_config(AutoConfig.from_pretrained(base))
    norms = base_weight_norms(base)
    paths = []
    for slot, spec in enumerate(streams):
        path = root / f"slot_{slot}"
        path.mkdir()
        paths.append(str(path))
        if spec["type"] == "lora":
            synthesize_adapter(
                path,
                dims,
                norms,
                PROFILES["light"],
                seed=slot,
                modules=attn_modules(with_kv=not dual_stream) + mlp_modules(dims),
                cross_stream_rank=16 if dual_stream else None,
            )
            continue
        outputs = len(spec["output"]["token_ids"])
        if spec["type"] in ("probe", "test_buffered"):
            if spec["type"] == "test_buffered":
                from tests.shared.stream_plugins.buffered_head import BufferedHeadType

                filename = BufferedHeadType.weights_file
            else:
                filename = "probe_head.safetensors"
            save_file(
                {
                    "weight": torch.zeros(outputs, dims.hidden),
                    "bias": torch.zeros(outputs),
                },
                str(path / filename),
            )
        elif spec["type"] == "mlp_head":
            from tests.shared.stream_plugins.mlp_head import MLP_HEAD_FILE

            width = spec["settings"]["width"]
            save_file(
                {
                    "w1": torch.ones(width, dims.hidden) * 0.01,
                    "w2": torch.ones(outputs, width) * 0.02,
                },
                str(path / MLP_HEAD_FILE),
            )
        else:
            raise ValueError(f"No test artifact writer for {spec['type']!r}.")
    n = len(streams)
    return GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base),
        adapter_paths=paths,
        adapter_names=[f"adapter_{i + 1}" for i in range(n)],
        adapter_token_ids=[250 + i for i in range(n)],
        adapter_substitute_token_ids=[1] * n,
        streams=streams,
        switch_head_dim=32,
        max_position_embeddings=512,
    ).eval()
