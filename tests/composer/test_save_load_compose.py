# SPDX-License-Identifier: Apache-2.0
"""Save/load roundtrip tests for the compose pipeline.

Calls ``compose_granite_switch.build()`` and
``compose_granite_switch.save_and_validate_model_artifacts()`` directly —
tests exercise the actual production code paths, not a replica.

Two phases, each with full validation (inference, config, weights, files):

  **Phase 1 — build → save → load**
    build() produces in-memory model → save_and_validate_model_artifacts()
    writes to disk → from_pretrained() loads back.  Compare built vs loaded.

  **Phase 2 — load → save → load (double serialization)**
    Load from Phase 1 output → save to second dir → load again.
    Compare the two loaded models AND the two directory contents.

Requires base model + RAG adapters download.  Marked slow + requires_model.
"""

import filecmp
import gc
import sys
from pathlib import Path
from unittest.mock import patch

import json
import pytest
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

import granite_switch.hf  # noqa: F401 — registers AutoModel

SEED = 42



# ── Helpers ──────────────────────────────────────────────────────────


EXPECTED_ONLY_IN_DIR_1 = {
    "BUILD.md",
    "adapter_index.json",
    "compose_report.json",
    "io_configs/",  # entire directory
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
}

def _is_expected_pipeline_file(rel_path: Path) -> bool:
    """Return True if rel_path is a known pipeline-only file/dir."""
    s = str(rel_path).replace("\\", "/")  # normalize for Windows
    for entry in EXPECTED_ONLY_IN_DIR_1:
        if entry.endswith("/"):
            # Directory prefix match
            if s.startswith(entry) or s == entry.rstrip("/"):
                return True
        else:
            # Exact filename match
            if s == entry:
                return True
    return False


EXCLUDE_FROM_BINARY_COMPARE = {
    "special_tokens_map.json",  # JSON key ordering may differ between save methods
    "io_configs/",              # pipeline-only dir; explicit for clarity
}

def _is_excluded_from_binary_compare(rel_path: Path) -> bool:
    """Return True if rel_path should be skipped during binary comparison."""
    s = str(rel_path).replace("\\", "/")
    for entry in EXCLUDE_FROM_BINARY_COMPARE:
        if entry.endswith("/"):
            if s.startswith(entry) or s == entry.rstrip("/"):
                return True
        else:
            if s == entry:
                return True
    return False

def _forward_logits(model, input_ids):
    """Run forward pass and return logits on CPU."""
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits.cpu()


def _call_build(output_dir, extra_args=None):
    """Call compose_granite_switch.build() with mocked sys.argv."""
    from granite_switch.composer.compose_granite_switch import build

    fake_argv = [
        "compose_granite_switch",
        "--adapters", "ibm-granite/granite-lib-rag-r1.0",
        "--output", str(output_dir),
    ]
    if extra_args:
        fake_argv.extend(extra_args)
    with patch.object(sys, "argv", fake_argv):
        return build()


def _call_save(build_result):
    """Call save_and_validate_model_artifacts() with the build result."""
    from granite_switch.composer.compose_granite_switch import (
        save_and_validate_model_artifacts,
    )

    (
        model, tokenizer, args, base_model_local_path, base_model_size_gb,
        adapter_paths, all_discovered, adapter_token_ids,
        start_time, new_vocab_size, original_vocab_size,
    ) = build_result

    save_and_validate_model_artifacts(
        model=model,
        tokenizer=tokenizer,
        args=args,
        base_model_local_path=base_model_local_path,
        all_discovered=all_discovered,
        adapter_token_ids=adapter_token_ids,
        base_model_size_gb=base_model_size_gb,
        adapter_paths=adapter_paths,
        start_time=start_time,
        new_vocab_size=new_vocab_size,
        original_vocab_size=original_vocab_size,
    )


def _make_inputs(tokenizer, adapter_token_id):
    """Create base-mode and adapter-mode input tensors with realistic text.

    Base mode: a plain question with context (no control token).
    Adapter mode: same text with adapter control token prepended, matching the
    LoRA activation pattern (control token at the beginning of the sequence).
    """
    prompt = (
        "Document: The Eiffel Tower was built in 1889 for the World's Fair "
        "in Paris. It stands 330 meters tall and was the tallest man-made "
        "structure in the world until 1930.\n"
        "Question: When was the Eiffel Tower built?"
    )
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    base_input = torch.tensor([base_ids], dtype=torch.long)

    # Adapter mode: prepend the control token (LoRA activation pattern)
    adapter_ids = [adapter_token_id] + base_ids
    adapter_input = torch.tensor([adapter_ids], dtype=torch.long)

    return base_input, adapter_input

# ── Golden Set: structured collection of test strings covering edge cases ──
# Each entry stresses a different aspect of the tokenizer's behavior.
# When a regression is found in the wild, ADD a new entry here — the suite
# improves over time.
def _make_golden_set(adapter_name: str) -> list[str]:
    """Build the golden set of test strings, parameterized by an adapter name."""
    control = f"<|{adapter_name}|>"
    return [
        # Plain text
        "The Eiffel Tower was built in 1889.",
        "Hello, world!",
        "Short.",
        # Adapter control token (atomic tokenization)
        f"{control}What is the capital of France?",
        # Whitespace edge cases
        "hello   world",                  # multiple internal spaces
        "   leading whitespace",          # leading spaces
        "trailing whitespace   ",         # trailing spaces
        "line1\nline2",                   # newline
        "with\ttab",                      # tab
        # Adapter token with surrounding whitespace variations
        f"prefix {control} suffix",       # spaces around
        f"prefix{control}suffix",         # no spaces (lstrip/rstrip behavior)
        # Long-ish text
        "word " * 50,
        # Unicode (accents and non-Latin scripts)
        "café résumé naïve",
    ]
# ════════════════════════════════════════════════════════════════════
# Phase 1 fixture: build() → save → load
# ════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def phase1(tmp_path_factory):
    """build() → save_and_validate_model_artifacts() → from_pretrained().

    Returns everything needed to compare built vs loaded:
    logits, configs, state_dicts, and the save directory.
    """
    save_dir = str(tmp_path_factory.mktemp("phase1") / "model")

    # ── build() ──
    build_result = _call_build(save_dir)
    (
        model, tokenizer, args, base_model_local_path, base_model_size_gb,
        adapter_paths, all_discovered, adapter_token_ids,
        start_time, new_vocab_size, original_vocab_size,
    ) = build_result

    model.eval()
    base_input, adapter_input = _make_inputs(tokenizer, adapter_token_ids[0])

    # Built model artifacts
    logits_built_base = _forward_logits(model, base_input)
    logits_built_adapter = _forward_logits(model, adapter_input)
    built_config = model.config
    built_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Capture built tokenizer info
    built_tokenizer_len = len(tokenizer)
    built_tokenizer_ids = {
        tok: tokenizer.convert_tokens_to_ids(tok)
        for tok in tokenizer.all_special_tokens
    }
    # Encode/decode roundtrip sample
    test_text = "The Eiffel Tower was built in 1889."
    built_encoded = tokenizer.encode(test_text, add_special_tokens=False)
    built_decoded = tokenizer.decode(built_encoded)

    # Golden Set encoding (structured edge-case suite)
    golden_strings = _make_golden_set(list(built_config.adapter_names)[0])
    built_golden_encoded = [
        tokenizer.encode(s, add_special_tokens=False) for s in golden_strings
    ]

    rng = random.Random(SEED)
    sample_size = min(100, original_vocab_size)
    sample_ids = rng.sample(range(original_vocab_size), sample_size)
    built_base_vocab_sample = {
        tid: tokenizer.convert_ids_to_tokens(tid) for tid in sample_ids
    }
    built_added_vocab = dict(tokenizer.get_added_vocab()) 

    # ── save ──
    _call_save(build_result)
    dtype = model.dtype
    del model
    gc.collect()

    # ── load ──
    loaded = AutoModelForCausalLM.from_pretrained(
        save_dir, torch_dtype=dtype,
    ).eval()
    loaded.model.rotary_emb.to(torch.bfloat16)
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)

    logits_loaded_base = _forward_logits(loaded, base_input)
    logits_loaded_adapter = _forward_logits(loaded, adapter_input)
    loaded_config = loaded.config
    loaded_state_dict = {k: v.cpu().clone() for k, v in loaded.state_dict().items()}

    # Capture loaded tokenizer info
    loaded_tokenizer_len = len(loaded_tokenizer)
    loaded_tokenizer_ids = {
        tok: loaded_tokenizer.convert_tokens_to_ids(tok)
        for tok in loaded_tokenizer.all_special_tokens
    }
    loaded_encoded = loaded_tokenizer.encode(test_text, add_special_tokens=False)
    loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

    loaded_golden_encoded = [
        loaded_tokenizer.encode(s, add_special_tokens=False) for s in golden_strings
    ]

    del loaded
    gc.collect()

    return {
        "built_base": logits_built_base,
        "built_adapter": logits_built_adapter,
        "loaded_base": logits_loaded_base,
        "loaded_adapter": logits_loaded_adapter,
        "built_config": built_config,
        "loaded_config": loaded_config,
        "built_state_dict": built_state_dict,
        "loaded_state_dict": loaded_state_dict,
        "save_dir": Path(save_dir),
        "built_tokenizer_len": built_tokenizer_len,
        "loaded_tokenizer_len": loaded_tokenizer_len,
        "built_tokenizer_ids": built_tokenizer_ids,
        "loaded_tokenizer_ids": loaded_tokenizer_ids,
        "built_encoded": built_encoded,
        "loaded_encoded": loaded_encoded,
        "built_decoded": built_decoded,
        "loaded_decoded": loaded_decoded,
        "adapter_token_ids": adapter_token_ids,
        "adapter_names": list(built_config.adapter_names),
        "original_vocab_size": original_vocab_size,        
        "built_base_vocab_sample": built_base_vocab_sample,   
        "built_added_vocab": built_added_vocab,   
        "golden_strings": golden_strings,
        "built_golden_encoded": built_golden_encoded,
        "loaded_golden_encoded": loaded_golden_encoded,

    }


# ════════════════════════════════════════════════════════════════════
# Phase 2 fixture: load → save → load (double serialization)
# ════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def phase2(tmp_path_factory):
    """build()+save → load → save to second dir → load again.

    Returns everything needed to compare the two loads and two directories.
    """
    # ── First save: build() + save ──
    save_dir_1 = str(tmp_path_factory.mktemp("phase2a") / "model")
    build_result = _call_build(save_dir_1)
    adapter_token_ids = build_result[7]  # index 7 = adapter_token_ids
    _call_save(build_result)
    del build_result
    gc.collect()

    # ── First load ──
    loaded_1 = AutoModelForCausalLM.from_pretrained(
        save_dir_1, torch_dtype=torch.bfloat16,
    ).eval()
    loaded_1.model.rotary_emb.to(torch.bfloat16)
    tokenizer_1 = AutoTokenizer.from_pretrained(save_dir_1)

    base_input, adapter_input = _make_inputs(tokenizer_1, adapter_token_ids[0])

    logits_1_base = _forward_logits(loaded_1, base_input)
    logits_1_adapter = _forward_logits(loaded_1, adapter_input)
    config_1 = loaded_1.config
    state_dict_1 = {k: v.cpu().clone() for k, v in loaded_1.state_dict().items()}

    # Tokenizer 1 info
    test_text = "The Eiffel Tower was built in 1889."
    tok1_len = len(tokenizer_1)
    tok1_encoded = tokenizer_1.encode(test_text, add_special_tokens=False)
    tok1_control_ids = {
        tok: tokenizer_1.convert_tokens_to_ids(tok)
        for tok in tokenizer_1.all_special_tokens
    }
    
    adapter_name_for_golden = loaded_1.config.adapter_names[0]
    golden_strings = _make_golden_set(adapter_name_for_golden)
    tok1_golden_encoded = [
        tokenizer_1.encode(s, add_special_tokens=False)
        for s in golden_strings
    ]    

    # ── Second save ──
    save_dir_2 = str(tmp_path_factory.mktemp("phase2b") / "model")
    loaded_1.save_pretrained(save_dir_2, max_shard_size="5GB")
    tokenizer_1.save_pretrained(save_dir_2)
    del loaded_1
    gc.collect()

    # ── Second load ──
    loaded_2 = AutoModelForCausalLM.from_pretrained(
        save_dir_2, torch_dtype=torch.bfloat16,
    ).eval()
    loaded_2.model.rotary_emb.to(torch.bfloat16)
    tokenizer_2 = AutoTokenizer.from_pretrained(save_dir_2)

    logits_2_base = _forward_logits(loaded_2, base_input)
    logits_2_adapter = _forward_logits(loaded_2, adapter_input)
    config_2 = loaded_2.config
    state_dict_2 = {k: v.cpu().clone() for k, v in loaded_2.state_dict().items()}

    # Tokenizer 2 info
    tok2_len = len(tokenizer_2)
    tok2_encoded = tokenizer_2.encode(test_text, add_special_tokens=False)
    tok2_control_ids = {
        tok: tokenizer_2.convert_tokens_to_ids(tok)
        for tok in tokenizer_2.all_special_tokens
    }

    tok2_golden_encoded = [
        tokenizer_2.encode(s, add_special_tokens=False) for s in golden_strings
    ]

    del loaded_2
    gc.collect()

    return {
        "dir_1": Path(save_dir_1),
        "dir_2": Path(save_dir_2),
        "logits_1_base": logits_1_base,
        "logits_1_adapter": logits_1_adapter,
        "logits_2_base": logits_2_base,
        "logits_2_adapter": logits_2_adapter,
        "config_1": config_1,
        "config_2": config_2,
        "state_dict_1": state_dict_1,
        "state_dict_2": state_dict_2,
        "tok1_len": tok1_len,
        "tok2_len": tok2_len,
        "tok1_encoded": tok1_encoded,
        "tok2_encoded": tok2_encoded,
        "tok1_control_ids": tok1_control_ids,
        "tok2_control_ids": tok2_control_ids,
        "golden_strings": golden_strings,
        "tok1_golden_encoded": tok1_golden_encoded,
        "tok2_golden_encoded": tok2_golden_encoded,
    }


# ════════════════════════════════════════════════════════════════════
# Phase 1: build → save → load (built vs loaded)
# ════════════════════════════════════════════════════════════════════


class TestPhase1_BuildSaveLoad:
    """build() → save → load: full validation of built vs loaded model."""

    # ── Inference ──

    def test_inference_base_mode_exact(self, phase1):
        """Base mode logits: built must equal loaded (bit-exact)."""
        built = phase1["built_base"]
        loaded = phase1["loaded_base"]

        diff = (built.float() - loaded.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\n  Phase 1 base mode: max={max_diff:.2e}, mean={mean_diff:.2e}")

        assert torch.equal(built, loaded), (
            f"Base mode logits NOT bit-exact. max={max_diff:.2e}, mean={mean_diff:.2e}"
        )

    def test_inference_adapter_mode_exact(self, phase1):
        """Adapter mode logits: built must equal loaded (bit-exact)."""
        built = phase1["built_adapter"]
        loaded = phase1["loaded_adapter"]

        diff = (built.float() - loaded.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\n  Phase 1 adapter mode: max={max_diff:.2e}, mean={mean_diff:.2e}")

        assert torch.equal(built, loaded), (
            f"Adapter mode logits NOT bit-exact. max={max_diff:.2e}, mean={mean_diff:.2e}"
        )

    # ── Config ──

    def test_io_structure_matches_adapter_index(self, phase1):
        """Verify io_configs structure matches adapter_index.json."""

        save_dir = phase1["save_dir"]

        # ── load adapter_index.json ──
        adapter_index_path = save_dir / "adapter_index.json"
        assert adapter_index_path.exists(), "adapter_index.json missing"

        with open(adapter_index_path) as f:
            data = json.load(f)

        adapters = data["adapters"]

        # ── expected names ──
        expected_names = {a["adapter_name"] for a in adapters}

        io_dir = save_dir / "io_configs"
        assert io_dir.exists(), "io_configs directory missing"

        # ── actual names ──
        actual_names = {p.name for p in io_dir.iterdir() if p.is_dir()}

        # ── check exact match ──
        assert actual_names == expected_names, (
            f"io_configs mismatch:\n"
            f"  expected: {sorted(expected_names)}\n"
            f"  actual:   {sorted(actual_names)}"
        )

        # ── check each has io.yaml ──
        for name in expected_names:
            io_file = io_dir / name / "io.yaml"
            assert io_file.exists(), f"{name}/io.yaml missing"

    def test_pipeline_metadata_files_exist(self, phase1):
        """BUILD.md, compose_report.json must be generated by the pipeline."""
        save_dir = phase1["save_dir"]
        required = ["BUILD.md", "compose_report.json"]
        missing = [name for name in required if not (save_dir / name).exists()]
        assert not missing, f"Pipeline did not generate: {missing}"
        # Upstream README.md must NOT be present — it's replaced by BUILD.md.
        assert not (save_dir / "README.md").exists(), (
            "Upstream README.md should not be copied into the composed output"
        )
        
    def test_config_adapter_identity(self, phase1):
        """num_adapters, token IDs, names, substitute IDs survive save→load."""
        built = phase1["built_config"]
        loaded = phase1["loaded_config"]

        assert loaded.num_adapters == built.num_adapters
        assert loaded.adapter_token_ids == built.adapter_token_ids
        assert loaded.adapter_names == built.adapter_names
        assert loaded.adapter_substitute_token_ids == built.adapter_substitute_token_ids

    def test_config_lora(self, phase1):
        """adapter_ranks, max_lora_rank, lora_target_modules survive save→load."""
        built = phase1["built_config"]
        loaded = phase1["loaded_config"]

        assert loaded.adapter_ranks == built.adapter_ranks
        assert loaded.max_lora_rank == built.max_lora_rank
        assert loaded.lora_target_modules == built.lora_target_modules

    def test_config_switch(self, phase1):
        """switch_head_dim and control_token_gain survive save→load."""
        built = phase1["built_config"]
        loaded = phase1["loaded_config"]

        assert loaded.switch_head_dim == built.switch_head_dim
        assert loaded.control_token_gain == built.control_token_gain

    def test_config_granite_scaling(self, phase1):
        """Granite-specific scaling parameters survive save→load."""
        built = phase1["built_config"]
        loaded = phase1["loaded_config"]

        assert loaded.logits_scaling == built.logits_scaling
        assert loaded.attention_multiplier == built.attention_multiplier
        assert loaded.residual_multiplier == built.residual_multiplier
        assert loaded.embedding_multiplier == built.embedding_multiplier

    def test_config_architecture(self, phase1):
        """Core architecture dimensions survive save→load."""
        built = phase1["built_config"]
        loaded = phase1["loaded_config"]

        assert loaded.hidden_size == built.hidden_size
        assert loaded.num_hidden_layers == built.num_hidden_layers
        assert loaded.num_attention_heads == built.num_attention_heads
        assert loaded.num_key_value_heads == built.num_key_value_heads
        assert loaded.intermediate_size == built.intermediate_size
        assert loaded.vocab_size == built.vocab_size

    # ── Weights ──

    def test_weights_keys_match(self, phase1):
        """Same set of state_dict keys before and after."""
        built_keys = set(phase1["built_state_dict"].keys())
        loaded_keys = set(phase1["loaded_state_dict"].keys())

        assert not (built_keys - loaded_keys), f"Missing: {built_keys - loaded_keys}"
        assert not (loaded_keys - built_keys), f"Extra: {loaded_keys - built_keys}"

    def test_weights_shapes_match(self, phase1):
        """Every tensor has the same shape before and after."""
        built_sd = phase1["built_state_dict"]
        loaded_sd = phase1["loaded_state_dict"]

        mismatched = [
            f"{k}: {built_sd[k].shape} -> {loaded_sd[k].shape}"
            for k in built_sd
            if k in loaded_sd and built_sd[k].shape != loaded_sd[k].shape
        ]
        assert not mismatched, f"Shape mismatches:\n" + "\n".join(f"  {m}" for m in mismatched)

    def test_weights_values_match(self, phase1):
        """Every tensor is bit-exact identical before and after."""
        built_sd = phase1["built_state_dict"]
        loaded_sd = phase1["loaded_state_dict"]

        mismatched = []
        for key in built_sd:
            if key in loaded_sd and not torch.equal(built_sd[key], loaded_sd[key]):
                diff = (built_sd[key].float() - loaded_sd[key].float()).abs().max().item()
                mismatched.append(f"{key}: max_diff={diff:.2e}")

        assert not mismatched, (
            f"{len(mismatched)} tensor(s) differ:\n"
            + "\n".join(f"  {m}" for m in mismatched[:10])
        )

    # ── Tokenizer ──

    def test_tokenizer_vocab_size(self, phase1):
        """Tokenizer vocabulary size preserved after save→load."""
        built_len = phase1["built_tokenizer_len"]
        loaded_len = phase1["loaded_tokenizer_len"]
        print(f"\n  built vocab: {built_len}, loaded vocab: {loaded_len}")
        assert loaded_len == built_len, (
            f"Vocab size differs: built={built_len}, loaded={loaded_len} "
            f"(diff={loaded_len - built_len})"
        )

    def test_tokenizer_encode_decode_roundtrip(self, phase1):
        """Same text encodes to same token IDs after save→load."""
        built_enc = phase1["built_encoded"]
        loaded_enc = phase1["loaded_encoded"]
        print(f"\n  built encoded: {len(built_enc)} tokens, loaded encoded: {len(loaded_enc)} tokens")

        assert loaded_enc == built_enc, (
            f"Encoded IDs differ ({len(built_enc)} vs {len(loaded_enc)} tokens):\n"
            f"  built:  {built_enc[:15]}...\n"
            f"  loaded: {loaded_enc[:15]}..."
        )
        assert phase1["loaded_decoded"] == phase1["built_decoded"], (
            f"Decoded text differs:\n"
            f"  built:  {phase1['built_decoded']!r}\n"
            f"  loaded: {phase1['loaded_decoded']!r}"
        )

    def test_tokenizer_control_token_ids(self, phase1):
        """Every adapter control token maps to the correct ID after save→load."""
        adapter_names = phase1["adapter_names"]
        adapter_token_ids = phase1["adapter_token_ids"]
        loaded_ids = phase1["loaded_tokenizer_ids"]

        print(f"\n  Checking {len(adapter_names)} adapter control tokens:")
        for i, name in enumerate(adapter_names):
            act_tok = f"<|{name}|>"
            act_expected = adapter_token_ids[i]
            act_actual = loaded_ids.get(act_tok, "MISSING")
            status = "ok" if act_actual == act_expected else "MISMATCH"
            print(f"    {name}: {act_tok}={act_actual} (expect {act_expected}) [{status}]")

            assert act_tok in loaded_ids, f"Missing control token: {act_tok}"
            assert loaded_ids[act_tok] == act_expected, (
                f"{act_tok}: expected {act_expected}, got {act_actual}"
            )
    def test_added_tokens_in_added_vocab(self, phase1):
        """Every adapter control token must be registered in tokenizer.get_added_vocab().

        convert_tokens_to_ids() always returns *some* ID — even for non-existent tokens
        (it falls back to UNK). get_added_vocab() returns only tokens explicitly
        registered as added tokens, which is the only guarantee that they will be
        treated atomically during encoding (i.e. <|adapter|> stays as one ID instead
        of being split into <, |, adapter, |, >).
        """
        save_dir = phase1["save_dir"]
        adapter_names = phase1["adapter_names"]

        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
        added_vocab = loaded_tokenizer.get_added_vocab()

        print(f"\n  Total added tokens in loaded tokenizer: {len(added_vocab)}")
        print(f"  Checking {len(adapter_names)} adapter control tokens:")

        missing = []
        for name in adapter_names:
            control_token = f"<|{name}|>"
            present = control_token in added_vocab
            status = "ok" if present else "MISSING"
            token_id = added_vocab.get(control_token, "—")
            print(f"    {control_token}: id={token_id} [{status}]")
            if not present:
                missing.append(control_token)

        assert not missing, (
            f"{len(missing)} adapter control token(s) not in added vocab: {missing}. "
            f"Available added tokens: {sorted(added_vocab.keys())[:10]}..."
        )

    def test_base_vocab_integrity(self, phase1):
        """Random sample of base-vocab tokens must keep their IDs after save→load.

        Adding new tokens to a tokenizer must APPEND them to the end of the
        vocabulary, never shift or overwrite existing entries. If a base-vocab
        token's ID changed, the model's embedding layer would point to the wrong
        token at inference time, producing garbage output.

        This test samples 100 random IDs from the base (pre-extension) range and
        verifies that each ID maps to the same token in the built tokenizer
        (captured pre-save) and in the loaded tokenizer (reloaded from disk).
        """
        save_dir = phase1["save_dir"]
        built_sample = phase1["built_base_vocab_sample"]   # {id: token_str}
        original_vocab_size = phase1["original_vocab_size"]

        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        mismatches = []
        for token_id, built_token in built_sample.items():
            loaded_token = loaded_tokenizer.convert_ids_to_tokens(token_id)
            if built_token != loaded_token:
                mismatches.append((token_id, built_token, loaded_token))

        print(
            f"\n  Sampled {len(built_sample)} base-vocab IDs "
            f"from range 0..{original_vocab_size - 1}"
        )

        assert not mismatches, (
            f"Base vocab shifted at {len(mismatches)} of {len(built_sample)} "
            f"sampled positions. First 5 mismatches "
            f"(id, built_token, loaded_token):\n"
            + "\n".join(f"  {m}" for m in mismatches[:5])
        )

    def test_added_token_id_mapping_consistency(self, phase1):
        """Every added token must have the same ID in built and loaded tokenizers.

        Iterates over the FULL set of added tokens (not just adapter control tokens),
        so this catches any drift in chat template tokens, role markers, or any
        other special tokens the pipeline registers — beyond just the adapters.
        """
        save_dir = phase1["save_dir"]
        built_added = phase1["built_added_vocab"]   # {token_str: id} — needs fixture support

        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)
        loaded_added = loaded_tokenizer.get_added_vocab()

        print(f"\n  built added tokens: {len(built_added)}")
        print(f"  loaded added tokens: {len(loaded_added)}")

        # Same set of keys
        only_in_built = set(built_added) - set(loaded_added)
        only_in_loaded = set(loaded_added) - set(built_added)
        assert not only_in_built, f"Added tokens missing in loaded: {only_in_built}"
        assert not only_in_loaded, f"Extra added tokens in loaded: {only_in_loaded}"

        # Same IDs for shared keys
        mismatches = {
            tok: (built_added[tok], loaded_added[tok])
            for tok in built_added
            if built_added[tok] != loaded_added[tok]
        }
        assert not mismatches, (
            f"{len(mismatches)} added token(s) have different IDs:\n"
            + "\n".join(f"  {t}: built={a}, loaded={b}" for t, (a, b) in mismatches.items())
        )

    def test_tokenizer_golden_set(self, phase1):
        """Golden Set: every string in the curated suite encodes identically before/after save→load.

        This is the broadest tokenizer behavior test. The golden set covers plain text,
        adapter control tokens, whitespace edge cases, and Unicode. A regression in any
        one entry indicates a class of inputs the tokenizer no longer handles consistently.
        When new tokenizer bugs surface, add the offending string to _make_golden_set().
        """
        strings = phase1["golden_strings"]
        built = phase1["built_golden_encoded"]
        loaded = phase1["loaded_golden_encoded"]

        print(f"\n  Golden Set: {len(strings)} test strings")

        mismatches = []
        for s, b, l in zip(strings, built, loaded):
            if b != l:
                mismatches.append((s, b, l))

        if mismatches:
            for s, b, l in mismatches:
                print(f"  MISMATCH on {s!r}:")
                print(f"    built:  {b}")
                print(f"    loaded: {l}")

        assert not mismatches, (
            f"Golden set: {len(mismatches)} of {len(strings)} entries differ. "
            f"First mismatch: {mismatches[0][0]!r}"
        )

    def test_tokenizer_atomic_control_tokens(self, phase1):
        """Each adapter control token must encode to exactly ONE token ID.

        Control tokens like <|context_relevance|> activate the corresponding LoRA
        adapter at the position where they appear. If the tokenizer splits them
        into sub-tokens (e.g. ['<', '|', 'context', '_', 'relevance', '|', '>']),
        the activation never fires and the model sees garbage instead.

        Atomic tokenization requires that:
        1. The control token is registered as an added token (not just a string in vocab)
        2. The tokenizer treats it as a single unit during encoding
        """
        save_dir = phase1["save_dir"]
        adapter_names = phase1["adapter_names"]

        loaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)

        print(f"\n  Checking atomic encoding of {len(adapter_names)} control tokens:")

        non_atomic = []
        for name in adapter_names:
            control_token = f"<|{name}|>"
            # Encode without any wrapping text and without special tokens
            encoded = loaded_tokenizer.encode(control_token, add_special_tokens=False)
            is_atomic = len(encoded) == 1
            status = "ok" if is_atomic else f"SPLIT into {len(encoded)} tokens"
            print(f"    {control_token}: encoded as {encoded} [{status}]")

            if not is_atomic:
                sub_tokens = loaded_tokenizer.convert_ids_to_tokens(encoded)
                non_atomic.append(
                    f"{control_token} → {len(encoded)} tokens: {sub_tokens}"
                )

        assert not non_atomic, (
            f"{len(non_atomic)} control token(s) were not encoded atomically:\n  "
            + "\n  ".join(non_atomic)
        )

# ════════════════════════════════════════════════════════════════════
# Phase 2: load → save → load (double serialization)
# ════════════════════════════════════════════════════════════════════


class TestPhase2_DoubleSerialization:
    """load → save → load: full validation of loaded_1 vs loaded_2."""

    # ── Inference ──

    def test_inference_base_mode(self, phase2):
        """Base mode: loaded_1 vs loaded_2 must be bit-exact."""
        logits_1 = phase2["logits_1_base"]
        logits_2 = phase2["logits_2_base"]

        assert torch.equal(logits_1, logits_2), (
            f"Base mode logits differ. "
            f"Max diff: {(logits_1.float() - logits_2.float()).abs().max().item():.2e}"
        )

    def test_inference_adapter_mode(self, phase2):
        """Adapter mode: loaded_1 vs loaded_2 must be bit-exact."""
        logits_1 = phase2["logits_1_adapter"]
        logits_2 = phase2["logits_2_adapter"]

        assert torch.equal(logits_1, logits_2), (
            f"Adapter mode logits differ. "
            f"Max diff: {(logits_1.float() - logits_2.float()).abs().max().item():.2e}"
        )

    # ── Config ──

    def test_config_matches(self, phase2):
        """All critical config fields identical between loaded_1 and loaded_2."""
        c1 = phase2["config_1"]
        c2 = phase2["config_2"]

        assert c2.num_adapters == c1.num_adapters
        assert c2.adapter_token_ids == c1.adapter_token_ids
        assert c2.adapter_names == c1.adapter_names
        assert c2.adapter_ranks == c1.adapter_ranks
        assert c2.max_lora_rank == c1.max_lora_rank
        assert c2.lora_target_modules == c1.lora_target_modules
        assert c2.adapter_substitute_token_ids == c1.adapter_substitute_token_ids
        assert c2.logits_scaling == c1.logits_scaling
        assert c2.attention_multiplier == c1.attention_multiplier
        assert c2.vocab_size == c1.vocab_size

    # ── Weights ──

    def test_weights_match(self, phase2):
        """All tensors bit-exact between loaded_1 and loaded_2."""
        sd1 = phase2["state_dict_1"]
        sd2 = phase2["state_dict_2"]

        assert set(sd1.keys()) == set(sd2.keys()), (
            f"Key mismatch. Missing: {set(sd1.keys()) - set(sd2.keys())}"
        )

        mismatched = []
        for key in sd1:
            if not torch.equal(sd1[key], sd2[key]):
                diff = (sd1[key].float() - sd2[key].float()).abs().max().item()
                mismatched.append(f"{key}: max_diff={diff:.2e}")

        assert not mismatched, (
            f"{len(mismatched)} tensor(s) differ:\n"
            + "\n".join(f"  {m}" for m in mismatched[:10])
        )

    # ── Files ──
    
    # def test_file_content_matches(self, phase2):
    #     """All shared files must be binary-identical."""
    #     dir_1 = phase2["dir_1"]
    #     dir_2 = phase2["dir_2"]

    #     files_2 = {p.relative_to(dir_2) for p in dir_2.rglob("*") if p.is_file()}
    #     shared = {f for f in files_2 if (dir_1 / f).exists()}

    #     mismatched = []
    #     for rel_path in sorted(shared):
    #         f1 = dir_1 / rel_path
    #         f2 = dir_2 / rel_path
    #         if not filecmp.cmp(str(f1), str(f2), shallow=False):
    #             size_1 = f1.stat().st_size
    #             size_2 = f2.stat().st_size
    #             mismatched.append((str(rel_path), size_1, size_2))

    #     assert not mismatched, (
    #         f"{len(mismatched)} file(s) differ: "
    #         + ", ".join(f"{name} ({s1} vs {s2} bytes)" for name, s1, s2 in mismatched)
    #     )

    #     print(f"\n  {len(shared)} shared files — all binary-identical")

    def test_file_content_matches(self, phase2):
        """All shared files must be binary-identical, except those explicitly excluded.

        Uses an expel approach: by default every shared file is compared byte-for-byte.
        Files listed in EXCLUDE_FROM_BINARY_COMPARE are skipped (e.g. special_tokens_map.json
        where HuggingFace's save_pretrained may produce different but semantically
        equivalent output). Any new shared file is compared automatically — fail-safe.

        Safetensor shard files are compared by total size across all shards rather than
        per-file, because shard boundaries may shift between saves while total content
        remains identical.
        """
        dir_1 = phase2["dir_1"]
        dir_2 = phase2["dir_2"]

        files_2 = {p.relative_to(dir_2) for p in dir_2.rglob("*") if p.is_file()}
        shared_all = {f for f in files_2 if (dir_1 / f).exists()}

        # Apply expel filter
        shared_to_compare = {f for f in shared_all if not _is_excluded_from_binary_compare(f)}
        excluded = shared_all - shared_to_compare

        if excluded:
            print(f"\n  Excluded from binary comparison ({len(excluded)}):")
            for f in sorted(excluded):
                print(f"    {f}")

        # Separate safetensor files and their index from other files — shard
        # boundaries can shift between saves, so compare aggregate size instead
        # of per-file bytes. The index JSON records shard assignments and will
        # also differ when boundaries shift.
        safetensor_files = {f for f in shared_to_compare if f.name.endswith(".safetensors")}
        shard_index_files = {f for f in shared_to_compare if f.name == "model.safetensors.index.json"}
        other_files = shared_to_compare - safetensor_files - shard_index_files

        mismatched = []

        # Compare non-safetensor files byte-for-byte
        for rel_path in sorted(other_files):
            f1 = dir_1 / rel_path
            f2 = dir_2 / rel_path
            if not filecmp.cmp(str(f1), str(f2), shallow=False):
                size_1 = f1.stat().st_size
                size_2 = f2.stat().st_size
                mismatched.append((str(rel_path), size_1, size_2))

        # Compare safetensor files by total tensor data size (excluding headers).
        # Shard boundaries may shift between saves, changing which tensors land in
        # which file and thus changing header sizes — but raw tensor data is identical.
        if safetensor_files:
            import struct

            def _tensor_data_size(path):
                # Safetensor file layout:
                #   [8 bytes: little-endian uint64 header length]
                #   [header_len bytes: JSON header with tensor metadata]
                #   [remaining bytes: raw tensor data]
                # Subtract both the 8-byte length prefix and the header itself.
                with open(path, "rb") as f:
                    header_len = struct.unpack("<Q", f.read(8))[0]
                return path.stat().st_size - 8 - header_len

            total_1 = sum(_tensor_data_size(dir_1 / f) for f in safetensor_files)
            total_2 = sum(_tensor_data_size(dir_2 / f) for f in safetensor_files)
            if total_1 != total_2:
                mismatched.append(
                    (f"safetensors tensor data ({len(safetensor_files)} shards)", total_1, total_2)
                )
            else:
                print(
                    f"  {len(safetensor_files)} safetensor shard(s) — "
                    f"tensor data matches: {total_1:,} bytes"
                )

        assert not mismatched, (
            f"{len(mismatched)} file(s) differ: "
            + ", ".join(f"{name} ({s1} vs {s2} bytes)" for name, s1, s2 in mismatched)
        )

        print(f"  {len(other_files)} non-tensor files compared — all binary-identical")
    # ── Tokenizer ──

    def test_tokenizer_vocab_size(self, phase2):
        """Vocab size identical between loaded_1 and loaded_2."""
        len_1 = phase2["tok1_len"]
        len_2 = phase2["tok2_len"]
        print(f"\n  loaded_1 vocab: {len_1}, loaded_2 vocab: {len_2}")
        assert len_2 == len_1, (
            f"Vocab size differs: loaded_1={len_1}, loaded_2={len_2} "
            f"(diff={len_2 - len_1})"
        )
    def test_file_structure_matches(self, phase2):
        """Both directories must have the same files, modulo known pipeline-only files.

        dir_1 contains extra metadata files written by the compose pipeline
        (adapter_index.json, io_configs/, compose_report.json, README.md)
        that HuggingFace's save_pretrained does not replicate. These files are
        listed in EXPECTED_ONLY_IN_DIR_1 and excluded from the comparison —
        they are reported in the diagnostic output for visibility but do not
        fail the test.

        The test fails if:
        - dir_2 contains any file not in dir_1 (save_pretrained wrote something extra)
        - dir_1 contains an unexpected file not covered by EXPECTED_ONLY_IN_DIR_1
        """
        dir_1 = phase2["dir_1"]
        dir_2 = phase2["dir_2"]

        files_1 = {p.relative_to(dir_1) for p in dir_1.rglob("*") if p.is_file()}
        files_2 = {p.relative_to(dir_2) for p in dir_2.rglob("*") if p.is_file()}

        raw_only_in_1 = sorted(files_1 - files_2)
        only_in_2 = sorted(files_2 - files_1)
        shared = files_1 & files_2

        # Split dir_1-only files into expected (pipeline metadata) and unexpected
        expected_only_in_1 = [f for f in raw_only_in_1 if _is_expected_pipeline_file(f)]
        unexpected_only_in_1 = [f for f in raw_only_in_1 if not _is_expected_pipeline_file(f)]

        print(f"\n  dir_1: {len(files_1)} files")
        print(f"  dir_2: {len(files_2)} files")
        print(f"  shared: {len(shared)} files")

        if expected_only_in_1:
            print(f"  expected pipeline-only in dir_1 ({len(expected_only_in_1)}) — IGNORED:")
            for f in expected_only_in_1:
                size = (dir_1 / f).stat().st_size
                print(f"    {f}  ({size:,} bytes)")

        if unexpected_only_in_1:
            print(f"  UNEXPECTED only in dir_1 ({len(unexpected_only_in_1)}):")
            for f in unexpected_only_in_1:
                size = (dir_1 / f).stat().st_size
                print(f"    {f}  ({size:,} bytes)")

        if only_in_2:
            print(f"  only in dir_2 ({len(only_in_2)}):")
            for f in only_in_2:
                size = (dir_2 / f).stat().st_size
                print(f"    {f}  ({size:,} bytes)")

        assert not unexpected_only_in_1, (
            f"Unexpected files in dir_1 not covered by EXPECTED_ONLY_IN_DIR_1: "
            f"{[str(f) for f in unexpected_only_in_1]}. "
            f"Either add them to EXPECTED_ONLY_IN_DIR_1 or fix the pipeline."
        )
        assert not only_in_2, (
            f"save_pretrained wrote files not present in the pipeline output: "
            f"{[str(f) for f in only_in_2]}"
        )

    def test_tokenizer_encode_matches(self, phase2):
        """Same text encodes to same IDs from both loaded tokenizers."""
        enc_1 = phase2["tok1_encoded"]
        enc_2 = phase2["tok2_encoded"]
        print(f"\n  loaded_1 encoded: {len(enc_1)} tokens, loaded_2 encoded: {len(enc_2)} tokens")
        assert enc_2 == enc_1, (
            f"Encoded IDs differ:\n"
            f"  loaded_1 ({len(enc_1)} tokens): {enc_1[:15]}...\n"
            f"  loaded_2 ({len(enc_2)} tokens): {enc_2[:15]}..."
        )

    def test_tokenizer_control_token_ids(self, phase2):
        """All control token IDs identical between loaded_1 and loaded_2."""
        ids_1 = phase2["tok1_control_ids"]
        ids_2 = phase2["tok2_control_ids"]

        print(f"\n  loaded_1 special tokens: {len(ids_1)}, loaded_2 special tokens: {len(ids_2)}")

        missing = set(ids_1.keys()) - set(ids_2.keys())
        extra = set(ids_2.keys()) - set(ids_1.keys())

        if missing:
            print(f"  Missing in loaded_2: {missing}")
        if extra:
            print(f"  Extra in loaded_2: {extra}")

        assert not missing, f"Control tokens missing in loaded_2: {missing}"
        assert not extra, f"Extra control tokens in loaded_2: {extra}"

        mismatches = {
            tok: (ids_1[tok], ids_2[tok])
            for tok in ids_1
            if tok in ids_2 and ids_1[tok] != ids_2[tok]
        }
        if mismatches:
            for tok, (id1, id2) in mismatches.items():
                print(f"  ID mismatch: {tok} -> {id1} vs {id2}")

        assert not mismatches, f"Control token ID mismatches: {mismatches}"


    def test_tokenizer_golden_set(self, phase2):
        """Golden Set: every string encodes identically across double serialization."""
        strings = phase2["golden_strings"]
        enc_1 = phase2["tok1_golden_encoded"]
        enc_2 = phase2["tok2_golden_encoded"]

        print(f"\n  Golden Set: {len(strings)} test strings")

        mismatches = []
        for s, a, b in zip(strings, enc_1, enc_2):
            if a != b:
                mismatches.append((s, a, b))

        if mismatches:
            for s, a, b in mismatches:
                print(f"  MISMATCH on {s!r}:")
                print(f"    loaded_1: {a}")
                print(f"    loaded_2: {b}")

        assert not mismatches, (
            f"Golden set: {len(mismatches)} of {len(strings)} entries differ. "
            f"First mismatch: {mismatches[0][0]!r}"
        )