# SPDX-License-Identifier: Apache-2.0
"""Tiny stand-ins for the real Granite bases, built from captured copies of them.

PR runs must not download real models (see docs/CICD.md, "Release test tier"). The
composer, though, depends on the real bases' *format*: the chat template it injects
control tokens into, the special-token ids it probes, the reserved
``<|unused_N|>`` rows it borrows, the config fields it reads. A tiny random base
with a made-up tokenizer passes none of those checks, so the compose CLI refuses
it, and a test that got past them would prove nothing about the real thing.

So each fake base is the real repo's small files, captured verbatim into
``tests/shared/fixtures/bases/<name>/`` by ``capture()``, plus three things
generated here at test time:

* **config.json:** the captured config with only the size fields shrunk
  (hidden size, layers, heads, intermediate). Every other field, including
  ``vocab_size``, ``tie_word_embeddings``, ``torch_dtype`` and the Granite
  multipliers, is the real value.
* **tokenizer.json:** a byte-level BPE with no merges, padded so every captured
  added token keeps its real id. Text tokenizes to one token per byte, and every
  special token, control-token placement and template render behaves as on the
  real tokenizer. Only ordinary-word token ids differ.
* **model.safetensors:** seeded random weights of the shrunk shape.

The release tier runs the same tests on the real bases, which is what catches a
capture that has fallen behind its Hub repo. Refresh with
``python -m tests.shared.fake_bases --capture``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.shared.base_models import _randn

FIXTURES = Path(__file__).parent / "fixtures" / "bases"

#: name -> Hub repo the capture came from. The name is what tests pass around.
BASES = {
    "granite-4.0-micro": "ibm-granite/granite-4.0-micro",
    "granite-4.1-3b": "ibm-granite/granite-4.1-3b",
    "granite-4.2-3b": "ibm-granite/granite-4.2-3b",
}

#: Copied verbatim from the Hub repo. Everything a compose reads except weights
#: and the 7 MB vocabulary.
CAPTURED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)

#: The size fields shrunk in the fake config. The geometry the vLLM tiny-model
#: tests already serve (tests/shared/granitemoe_compose.py GPU_GEOMETRY): head_dim
#: 32, the smallest the vLLM attention backends accept. Keeps GQA (more query than
#: KV heads), which the real bases all use.
TINY_DIMS = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}


def capture(names=None, revision=None) -> None:
    """Refresh the captured files from the Hub. Needs network; not run in tests."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    for name in names or BASES:
        dest = FIXTURES / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in CAPTURED_FILES:
            try:
                src = hf_hub_download(BASES[name], filename, revision=revision)
            except EntryNotFoundError:
                (dest / filename).unlink(missing_ok=True)
                continue
            shutil.copyfile(src, dest / filename)
        print(f"captured {name} from {BASES[name]}")


def _tiny_config(captured: dict) -> dict:
    config = dict(captured)
    config.update(TINY_DIMS)
    layers = TINY_DIMS["num_hidden_layers"]
    config["head_dim"] = TINY_DIMS["hidden_size"] // TINY_DIMS["num_attention_heads"]
    if "layer_types" in config:
        config["layer_types"] = config["layer_types"][:layers]
    if "shared_intermediate_size" in config:
        config["shared_intermediate_size"] = TINY_DIMS["intermediate_size"]
    if "mamba_d_head" in config:
        # Unused on an all-attention base, but transformers validates
        # mamba_n_heads * mamba_d_head == mamba_expand * hidden_size.
        expand = config.get("mamba_expand", 2)
        config["mamba_d_head"] = (
            expand * TINY_DIMS["hidden_size"] // config["mamba_n_heads"]
        )
        if config["mamba_d_head"] == 0:
            config["mamba_n_heads"] = 1
            config["mamba_d_head"] = expand * TINY_DIMS["hidden_size"]
    config["max_position_embeddings"] = 4096
    return config


def _added_tokens(tokenizer_config: dict) -> dict[int, dict]:
    return {int(i): t for i, t in tokenizer_config["added_tokens_decoder"].items()}


#: Ordinary words that are single tokens on the real Granite 4 tokenizers (the same
#: id on 4.0, 4.1 and 4.2), and that the tests rely on being single tokens, e.g.
#: as probe labels, which the composer requires to encode to exactly one token.
#: The fake tokenizer keeps each at its real id.
SINGLE_TOKEN_WORDS = {
    "safe": 19193,
    "unsafe": 39257,
    "yes": 9891,
    "no": 2201,
    "true": 1904,
    "false": 3934,
    "good": 19045,
    "bad": 14176,
    "neutral": 60668,
    "high": 12156,
    "low": 10516,
    "medium": 27178,
    "left": 2414,
    "right": 1315,
    "up": 455,
}


def _write_tokenizer(dest: Path, tokenizer_config: dict) -> None:
    """Byte-level BPE whose added tokens, and :data:`SINGLE_TOKEN_WORDS`, keep
    their real ids. Every other word tokenizes to one token per byte."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    added = _added_tokens(tokenizer_config)
    first_added = min(added)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    # BPE joins bytes only through merges, so each single-token word gets a
    # left-to-right merge chain. Its intermediate prefixes take filler ids, which
    # are never otherwise produced; the word itself takes its real id.
    words = {word: token_id for word, token_id in SINGLE_TOKEN_WORDS.items()}
    reserved = set(words.values())
    merges: list[tuple[str, str]] = []
    pieces: list[str] = []
    # Longest first: BPE applies merges by rank, so a word that contains another
    # ("unsafe" holds "safe") must have its chain ranked ahead, or the shorter
    # word's merges fire first and split it.
    for word in sorted(words, key=len, reverse=True):
        for k in range(2, len(word) + 1):
            merges.append((word[: k - 1], word[k - 1]))
            if k < len(word) and word[:k] not in pieces:
                pieces.append(word[:k])
    next_free = iter(i for i in range(len(vocab), first_added) if i not in reserved)
    for piece in pieces:
        vocab[piece] = next(next_free)
    vocab.update(words)
    # Fillers occupy every remaining id below the first added token, so the added
    # tokens land on the ids the real tokenizer gives them.
    used = set(vocab.values())
    for i in range(len(alphabet), first_added):
        if i not in used:
            vocab[f"<|filler_{i}|>"] = i
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=list(dict.fromkeys(merges))))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    for token_id in sorted(added):
        spec = added[token_id]
        token = AddedToken(
            spec["content"],
            lstrip=spec.get("lstrip", False),
            rstrip=spec.get("rstrip", False),
            normalized=spec.get("normalized", False),
            single_word=spec.get("single_word", False),
            special=spec.get("special", True),
        )
        got = tokenizer.add_special_tokens([token])
        assert got == 1 and tokenizer.token_to_id(spec["content"]) == token_id, (
            f"added token {spec['content']!r} did not land on id {token_id}"
        )
    tokenizer.save(str(dest / "tokenizer.json"))


@dataclass(frozen=True)
class WeightPreset:
    """Random-weight scales for a fake base.

    ``body``: std of the decoder-layer projections. ``embed``: std of the token
    embedding (and the untied head). ``final_norm``: the final RMSNorm's gain,
    which sets how peaked the next-token distribution is.
    """

    body: float
    embed: float
    final_norm: float


#: Two presets, because on a 2-layer, hidden-128 model with the REAL Granite
#: multipliers (embedding_multiplier 12, residual_multiplier 0.22, logits_scaling
#: 10) no single set of scales gives both properties a trained 40-layer model has.
#: Measured on the tiny geometry:
#:
#:   ``REALISTIC`` (the default): body 0.02, embed 0.02, final norm 1. Activations
#:       stay small (hidden-state rms ~0.24), which the tests comparing EXACT
#:       scores need: test_probe_chunked_prefill checks probe scores to +/- 0.3
#:       against an fp32 reference, test_tp_integration TP=1 vs TP=2 logprobs.
#:       But the layers barely reach the output (zeroing every attention or MLP
#:       output changes it by JSD 0.000) and the output is near-uniform.
#:   ``LIVE_LAYERS``: body 1.0, embed 1.0, final norm 4. Every sublayer matters
#:       (dropping attention or the MLP moves the output by JSD 0.98 / 0.99) and
#:       the output is peaked (entropy ~4 nats), which tests that need an adapter
#:       to MOVE the output depend on: test_multi_adapter_equivalence's `distinct`
#:       check. But intermediate hidden states reach rms ~340, where bf16 rounding
#:       (~+/-16 at a score of ~2000) breaks those exact-score tolerances.
#:
#: Between the two, the layers either do not matter or the residual grows: the
#: 12x embedding has to be matched for a layer to count. The real model gets both
#: through depth. So a test asks for the one its assertions need.
REALISTIC = WeightPreset(body=0.02, embed=0.02, final_norm=1.0)
LIVE_LAYERS = WeightPreset(body=1.0, embed=1.0, final_norm=4.0)


def _scale(key: str, preset: WeightPreset) -> float:
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        return preset.embed
    return preset.body


def _write_weights(dest: Path, config: dict, seed: int, preset: WeightPreset) -> None:
    """Seeded random weights of the shrunk shape, keyed per tensor name."""
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    head_dim = config["head_dim"]
    q_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim
    shared = config["model_type"] == "granitemoehybrid"
    vocab = config["vocab_size"]

    shapes = {"model.embed_tokens.weight": (vocab, hidden)}
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (q_width, hidden)
        shapes[f"{p}.self_attn.k_proj.weight"] = (kv_width, hidden)
        shapes[f"{p}.self_attn.v_proj.weight"] = (kv_width, hidden)
        shapes[f"{p}.self_attn.o_proj.weight"] = (hidden, q_width)
        if shared:
            shapes[f"{p}.shared_mlp.input_linear.weight"] = (2 * inter, hidden)
            shapes[f"{p}.shared_mlp.output_linear.weight"] = (hidden, inter)
        else:
            shapes[f"{p}.mlp.gate_proj.weight"] = (inter, hidden)
            shapes[f"{p}.mlp.up_proj.weight"] = (inter, hidden)
            shapes[f"{p}.mlp.down_proj.weight"] = (hidden, inter)
    if not config.get("tie_word_embeddings", True):
        shapes["lm_head.weight"] = (vocab, hidden)

    # The captured dtype (bfloat16 on every Granite 4 base), so compose, save and
    # load round the same way they do on the real checkpoint.
    dtype = getattr(torch, config.get("torch_dtype") or "float32")
    sd = {
        key: _randn(key, seed, *shape, scale=_scale(key, preset)).to(dtype)
        for key, shape in shapes.items()
    }
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}"
        sd[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=dtype)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=dtype)
    sd["model.norm.weight"] = torch.full((hidden,), preset.final_norm, dtype=dtype)
    save_file(sd, str(dest / "model.safetensors"))


def build_fake_base(
    name: str, dest, *, seed: int = 0, preset: WeightPreset = REALISTIC
) -> Path:
    """Write a loadable tiny ``name`` base to ``dest`` and return its path.

    ``name`` is a key of :data:`BASES`. Composes with the real CLI and loads with
    ``from_pretrained``; see the module docstring for what is real and what is not.
    ``preset`` picks the random-weight scales (see :data:`REALISTIC`).
    """
    src = FIXTURES / name
    if not src.is_dir():
        raise FileNotFoundError(f"no captured base {name!r} under {FIXTURES}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for filename in CAPTURED_FILES:
        if (src / filename).exists() and filename != "config.json":
            shutil.copyfile(src / filename, dest / filename)

    config = _tiny_config(json.loads((src / "config.json").read_text()))
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text())
    _write_tokenizer(dest, tokenizer_config)
    _write_weights(dest, config, seed, preset)
    _write_repo_extras(dest)
    return dest


def _write_repo_extras(dest: Path) -> None:
    """Placeholders for the repo files a compose must NOT carry over.

    Every Granite repo ships them, and the composer's upstream-file copy skips
    each by a different rule (dotfile, signature extension, excluded name), so a
    fake without them would never exercise those exclusions.
    """
    (dest / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs\n")
    (dest / "model.sig").write_bytes(b"fake-signature")
    (dest / "README.md").write_text("# Fake base for granite-switch tests\n")
    # A real, complete index for the single shard: loaders (vLLM among them)
    # trust an index when one exists and load only the files it names, so an
    # empty weight_map would leave no weights to load at all.
    from safetensors import safe_open

    with safe_open(str(dest / "model.safetensors"), framework="pt") as f:
        weight_map = {key: "model.safetensors" for key in f.keys()}
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the captured Granite base files."
    )
    parser.add_argument("--capture", action="store_true", help="refresh from Hub")
    parser.add_argument("--names", nargs="*", choices=sorted(BASES))
    args = parser.parse_args()
    if args.capture:
        capture(args.names)


if __name__ == "__main__":
    main()


__all__ = [
    "BASES",
    "LIVE_LAYERS",
    "REALISTIC",
    "WeightPreset",
    "build_fake_base",
    "capture",
]
