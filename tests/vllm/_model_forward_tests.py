# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward pass tests for GraniteSwitchForCausalLM (vLLM backend).
Inner file — run by test_model_forward.py in subprocess.

Tests the full model wiring: switch -> adapter_indices -> decoder layers.
All tests run on a single CUDA GPU with random weights (no pretrained checkpoint needed).

Requires CUDA GPU and vLLM installed. All tests are skipped otherwise.
"""

import json
import os
import tempfile

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm.config import VllmConfig  # noqa: F401
        from vllm.forward_context import (  # noqa: F401
            ForwardContext,
            override_forward_context,
        )
        from vllm.model_executor.layers.attention.attention import (
            Attention,  # noqa: F401
        )

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import ForwardContext, override_forward_context

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM
    from granite_switch.vllm.switch.multi import MultiSwitch

# ── Constants ────────────────────────────────────────────────────────

BLOCK_SIZE = 16
MAX_TOKENS = 1024
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────


def _tiny_vllm_config(
    adapter_kinds=None,
    probe_label_token_ids=None,
    streams=None,
    **extra,
):
    """Minimal GraniteSwitchConfig for single-GPU vLLM tests.

    Two slots by default; ``streams`` (one entry per slot) sets the slot count.
    """
    n = len(streams) if streams is not None else 2
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,  # 2 MultiSwitch cache slots + 2 decoder
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=n,
        adapter_token_ids=[250 + i for i in range(n)],
        adapter_names=[f"adapter_{i + 1}" for i in range(n)],
        adapter_substitute_token_ids=[1] * n,
        # Rank must be in the fused kernel's SUPPORTED_RANKS (16, 32, ...);
        # finalize_weights asserts this for applicable adapters.
        max_lora_rank=16,
        adapter_ranks=[16] * n,
        adapter_kinds=adapter_kinds,
        probe_label_token_ids=probe_label_token_ids,
        streams=streams,
        **extra,
        switch_head_dim=32,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


def _tiny_vllm_config_no_adapters():
    """Minimal GraniteSwitchConfig with no adapters."""
    return GraniteSwitchConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_adapters=0,
        max_position_embeddings=512,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )


from tests.shared.vllm_distributed import ensure_distributed as _ensure_distributed
from tests.shared.vllm_kv_cache import cascade_kwargs, setup_kv_cache


def _make_vllm_config(config):
    """Create a VllmConfig with a real ModelConfig from our GraniteSwitchConfig."""
    from vllm.config import ModelConfig

    from granite_switch.vllm import register

    register()

    tmpdir = tempfile.mkdtemp(prefix="granite_switch_test_")
    config_dict = config.to_dict()
    config_dict["architectures"] = ["GraniteSwitchForCausalLM"]
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    model_config = ModelConfig(
        model=tmpdir,
        dtype="bfloat16",
        max_model_len=config.max_position_embeddings,
        enforce_eager=True,
    )

    vllm_config = VllmConfig(model_config=model_config)
    return vllm_config


def _init_model_weights(model):
    """Initialize base model weights that vLLM leaves uninitialized."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if "lora_A" in name or "lora_B" in name:
                continue
            if "layernorm" in name or "norm" in name:
                continue
            param.data.normal_(0, 0.02)


def _set_nonzero_lora(model, scale=0.1):
    """Set non-zero lora_A and lora_B on every LoRA layer."""
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.shared_mlp
            if hasattr(attn.qkv_proj, "lora_A_slices"):
                for a in attn.qkv_proj.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
                for b in attn.qkv_proj.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(attn.o_proj, "lora_A"):
                attn.o_proj.lora_A.data = torch.randn_like(attn.o_proj.lora_A) * scale
                attn.o_proj.lora_B.data = torch.randn_like(attn.o_proj.lora_B) * scale
            if hasattr(mlp.input_linear, "lora_A_slices"):
                for a in mlp.input_linear.lora_A_slices:
                    a.data = torch.randn_like(a) * scale
                for b in mlp.input_linear.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            if hasattr(mlp.output_linear, "lora_A"):
                mlp.output_linear.lora_A.data = (
                    torch.randn_like(mlp.output_linear.lora_A) * scale
                )
                mlp.output_linear.lora_B.data = (
                    torch.randn_like(mlp.output_linear.lora_B) * scale
                )


# ── Base test class ──────────────────────────────────────────────────


class _VLLMModelTestBase:
    """Base class providing model creation and full forward pass machinery."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        _ensure_distributed()

        self.device = torch.device("cuda")
        self.config = self._make_config()

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        try:
            self.vllm_config = _make_vllm_config(self.config)
            with set_current_vllm_config(self.vllm_config):
                # Construct on-device, mirroring vLLM's model loader (which
                # builds inside `with target_device:`). Each SwitchedLoRALinear
                # caches its device at construction, so building on CUDA lets
                # finalize build the fused weights on CUDA. (Constructing on CPU
                # then .to() leaves that cached device stale and the kernel sees
                # CPU kernel-metadata tensors.)
                with torch.device(self.device):
                    self.model = GraniteSwitchForCausalLM(
                        vllm_config=self.vllm_config,
                    )
        finally:
            torch.set_default_dtype(old_dtype)

        torch.manual_seed(SEED)
        _init_model_weights(self.model)  # base weights
        _set_nonzero_lora(self.model)  # adapter weights — set BEFORE finalize

        # The fused design bakes LoRA weights into w_ext at finalize time, so
        # finalize must run AFTER the weights are set — exactly as vLLM's
        # load_weights() does as its post-load step.
        #
        # #83 moved this off the model: each DecoderInterface owns the post-load
        # step and load_weights() ends by calling finalize_modules(). Go through
        # the interface so this harness exercises the same path as a real load.
        self.model.model.decoder_interface.finalize_modules(
            self.model, self.model.config
        )

        self.model_config = self.model.config
        self._setup_kv_caches()

        yield

        sfc = self.vllm_config.compilation_config.static_forward_context
        for name in list(self._layer_names):
            sfc.pop(name, None)

    def _make_config(self):
        return _tiny_vllm_config()

    def _setup_kv_caches(self):
        self._kv_caches = []
        self._attention_map = {}

        num_decoder_layers = self.config.num_hidden_layers - 2
        num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

        # MultiSwitch owns TWO attention modules (counting + memory), each with
        # its own KV-cache slot; register both under their real vLLM layer names.
        for switch_attn in (
            self.model.model.switch.counting_attn,
            self.model.model.switch.memory_attn,
        ):
            self._setup_single_attn(switch_attn, switch_attn.layer_name, num_blocks)

        for i in range(num_decoder_layers):
            layer_attn = self.model.model.layers[i].self_attn.attn
            layer_name = f"model.layers.{i}.self_attn.attn"
            self._setup_single_attn(layer_attn, layer_name, num_blocks)

        self._layer_names = list(self._attention_map.keys())

    def _setup_single_attn(self, attn, layer_name, num_blocks):
        kv_cache = setup_kv_cache(
            attn, self.vllm_config, num_blocks, BLOCK_SIZE, self.device
        )
        self._kv_caches.append(kv_cache)
        self._attention_map[layer_name] = attn

    def _build_metadata(self, seq_len):
        device = self.device
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
        num_blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table = torch.arange(
            num_blocks_needed,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0)
        query_start_loc = torch.tensor(
            [0, seq_len],
            dtype=torch.int32,
            device=device,
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        backend_name = next(iter(self._attention_map.values())).attn_backend.get_name()
        if backend_name == "FLASH_ATTN":
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

            # scheduler_metadata is FA3-only — passing it on FA2 (Ampere/A100)
            # forces FA3 kernel dispatch and crashes with "no kernel image
            # is available". Only compute it when get_flash_attn_version() == 3
            # (Hopper SM90+).
            scheduler_metadata = None
            try:
                from vllm.v1.attention.backends.fa_utils import (
                    get_flash_attn_version,
                    get_scheduler_metadata,
                )

                if get_flash_attn_version() == 3:
                    first_attn = next(iter(self._attention_map.values()))
                    scheduler_metadata = get_scheduler_metadata(
                        batch_size=1,
                        max_seqlen_q=seq_len,
                        max_seqlen_k=seq_len,
                        num_heads_q=first_attn.num_heads,
                        num_heads_kv=first_attn.num_kv_heads,
                        headdim=first_attn.head_size,
                        cache_seqlens=seq_lens,
                        qkv_dtype=torch.bfloat16,
                        cu_seqlens_q=query_start_loc,
                        page_size=BLOCK_SIZE,
                        causal=True,
                        num_splits=0,
                    )
            except ImportError:
                pass

            metadata = FlashAttentionMetadata(
                num_actual_tokens=seq_len,
                max_query_len=seq_len,
                query_start_loc=query_start_loc,
                max_seq_len=seq_len,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=slot_mapping,
                **cascade_kwargs(),
                causal=True,
                scheduler_metadata=scheduler_metadata,
            )
        else:
            pytest.skip(
                f"Backend {backend_name}: metadata not implemented for this test"
            )

        return metadata, slot_mapping

    def _run_forward(self, input_ids_list):
        for kv_cache in self._kv_caches:
            kv_cache.zero_()

        seq_len = len(input_ids_list)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)

        metadata, slot_mapping = self._build_metadata(seq_len)

        attn_metadata = {name: metadata for name in self._layer_names}
        slot_mapping_dict = {name: slot_mapping for name in self._layer_names}

        forward_ctx = ForwardContext(
            no_compile_layers=self.vllm_config.compilation_config.static_forward_context,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping_dict,
        )

        old_direct = {}
        for name, attn in self._attention_map.items():
            old_direct[name] = attn.use_direct_call
            attn.use_direct_call = True

        try:
            with override_forward_context(forward_ctx):
                hidden_states = self.model.forward(
                    input_ids=input_ids,
                    positions=positions,
                )
        finally:
            for name, attn in self._attention_map.items():
                attn.use_direct_call = old_direct[name]

        return hidden_states

    def _run_forward_and_logits(self, input_ids_list):
        hidden_states = self._run_forward(input_ids_list)
        logits = self.model.compute_logits(hidden_states)
        return logits


# ════════════════════════════════════════════════════════════════════
# 1. Model instantiation
# ════════════════════════════════════════════════════════════════════


class TestModelInstantiation(_VLLMModelTestBase):
    def test_switch_model_creates(self):
        assert isinstance(self.model.model.switch, MultiSwitch)
        num_decoder_layers = self.config.num_hidden_layers - 2
        assert len(self.model.model.layers) == num_decoder_layers

    def test_no_adapter_model_creates(self):
        config = _tiny_vllm_config_no_adapters()
        vllm_config = _make_vllm_config(config)

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vllm_config):
                model = GraniteSwitchForCausalLM(
                    vllm_config=vllm_config,
                ).to(self.device)
        finally:
            torch.set_default_dtype(old_dtype)

        assert model.model.switch is None
        assert len(model.model.layers) == 2


# ════════════════════════════════════════════════════════════════════
# 2. Forward output shape
# ════════════════════════════════════════════════════════════════════


class TestForwardOutputShape(_VLLMModelTestBase):
    def test_basic_output_shape(self):
        self.model.eval()
        input_ids_list = [10, 20, 30, 40, 50, 60, 70, 80]
        with torch.no_grad():
            hidden_states = self._run_forward(input_ids_list)
        assert hidden_states.shape == (8, self.config.hidden_size)

    def test_longer_sequence_shape(self):
        self.model.eval()
        input_ids_list = list(range(1, 33))
        with torch.no_grad():
            hidden_states = self._run_forward(input_ids_list)
        assert hidden_states.shape == (32, self.config.hidden_size)


# ════════════════════════════════════════════════════════════════════
# 4. Adapter indices wiring
# ════════════════════════════════════════════════════════════════════


class TestAdapterIndicesWiring(_VLLMModelTestBase):
    def test_control_token_activates_adapter(self):
        # LoRA weights were set non-zero and baked into w_ext at finalize time
        # in setup_model; setting them now (post-finalize) would have no effect.
        self.model.eval()

        with_ctrl = [10, 20, 250, 30, 40, 50, 60, 70]
        no_ctrl = [10, 20, 100, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_ctrl = self._run_forward_and_logits(with_ctrl)
            logits_text = self._run_forward_and_logits(no_ctrl)

        torch.testing.assert_close(logits_ctrl[:2], logits_text[:2])

        assert not torch.allclose(logits_ctrl[3:], logits_text[3:]), (
            "Post-control logits should differ when adapter is active"
        )

    def test_different_adapters_produce_different_logits(self):
        # LoRA weights baked into w_ext at finalize (setup_model); the two
        # adapters get distinct random rows there, so they diverge downstream.
        self.model.eval()

        seq_a1 = [10, 20, 250, 30, 40, 50, 60, 70]
        seq_a2 = [10, 20, 251, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_a1 = self._run_forward_and_logits(seq_a1)
            logits_a2 = self._run_forward_and_logits(seq_a2)

        torch.testing.assert_close(logits_a1[:2], logits_a2[:2])

        assert not torch.allclose(logits_a1[3:], logits_a2[3:]), (
            "Different adapters should produce different post-control logits"
        )


class TestProbeSlotRouting(_VLLMModelTestBase):
    def _make_config(self):
        return _tiny_vllm_config(
            adapter_kinds=["lora", "probe"],
            probe_label_token_ids=[None, [100, 101]],
        )

    def test_marker_routes_out_of_lora_and_emits_label(self, monkeypatch):
        self.model.eval()
        seen = {}
        switch = self.model.model.switch
        original_split = switch.split_indices

        def record_split(adapter_indices):
            lora_indices, head_indices = original_split(adapter_indices)
            seen["lora"] = lora_indices.detach().cpu()
            seen["probe"] = head_indices.detach().cpu()
            return lora_indices, head_indices

        monkeypatch.setattr(switch, "split_indices", record_split)

        with torch.no_grad():
            logits = self._run_forward_and_logits([10, 20, 30, 251])

        assert seen["lora"].tolist() == [0, 0, 0, 0]
        assert seen["probe"].tolist() == [0, 0, 0, 2]
        final_logits = logits[-1]
        assert torch.isfinite(final_logits[[100, 101]]).all()
        non_label_ids = [
            i for i in range(self.config.vocab_size) if i not in (100, 101)
        ]
        assert torch.isneginf(final_logits[non_label_ids]).all()


# ════════════════════════════════════════════════════════════════════
# 4b. Stream API: head types on the generic hooks (granite_switch.streams)
# ════════════════════════════════════════════════════════════════════

_HEAD_LABELS = [100, 101]


def _probe_entry(row, **extra):
    return {
        "type": "probe",
        "head_row": row,
        "output": {"mode": "select", "token_ids": list(_HEAD_LABELS)},
        **extra,
    }


def _non_labels(vocab_size):
    return [i for i in range(vocab_size) if i not in _HEAD_LABELS]


class TestStreamCompactRows(_VLLMModelTestBase):
    """[probe, lora, probe]: one LoRA row, a two-row probe bank, any slot order."""

    def _make_config(self):
        return _tiny_vllm_config(
            streams=[_probe_entry(0), {"type": "lora"}, _probe_entry(1)]
        )

    def test_rows_and_routing(self):
        self.model.eval()
        assert self.model.model.probe_head.weight.shape[0] == 2
        qkv = self.model.model.layers[0].self_attn.qkv_proj
        assert qkv.num_adapters == 1  # LoRA tensors hold the adapter slot only
        assert self.model.model.lora_row_lut_t.tolist() == [0, 0, 1, 0]

        with torch.no_grad():
            self.model.model.probe_head.bias[1, :2] = torch.tensor([-5.0, 5.0])
            # The last probe slot (3) reads its own bank row (1).
            logits = self._run_forward_and_logits([10, 20, 30, 252])
        last = logits[-1]
        assert int(last.argmax()) == _HEAD_LABELS[1]
        assert torch.isneginf(last[_non_labels(self.config.vocab_size)]).all()

        with torch.no_grad():
            # The LoRA slot (2, in the middle) runs its LoRA row: it changes the
            # post-control logits and leaves every row finite.
            plain = self._run_forward_and_logits([10, 20, 30, 40])
            adapted = self._run_forward_and_logits([10, 251, 30, 40])
        assert torch.isfinite(adapted).all()
        assert not torch.allclose(plain[2:], adapted[2:])


class TestStreamHeadsOnly(_VLLMModelTestBase):
    """Every slot a head: no LoRA layers and no kernel metadata at all."""

    def _make_config(self):
        return _tiny_vllm_config(streams=[_probe_entry(0), _probe_entry(1)])

    def test_heads_only_serves_the_verdict(self):
        self.model.eval()
        assert self.config.lora_target_modules == []
        assert self.model.model.lora_meta is None
        assert not [n for n, _ in self.model.named_parameters() if "lora_" in n]
        with torch.no_grad():
            self.model.model.probe_head.bias[0, :2] = torch.tensor([5.0, -5.0])
            logits = self._run_forward_and_logits([10, 20, 30, 250])
        assert int(logits[-1].argmax()) == _HEAD_LABELS[0]


class TestStreamPluginMerge(_VLLMModelTestBase):
    """The example plugin type, with ``merge``: the LM row plus its scores."""

    def _make_config(self):
        import tests.shared.stream_plugins.mlp_head  # noqa: F401 (registers it)

        return _tiny_vllm_config(
            streams=[
                {"type": "lora"},
                {
                    "type": "mlp_head",
                    "head_row": 0,
                    "output": {
                        "mode": "merge",
                        "token_ids": list(_HEAD_LABELS),
                        "weight": 0.5,
                    },
                    "settings": {"width": 8},
                },
            ]
        )

    def test_merge_adds_the_head_scores(self):
        self.model.eval()
        module = self.model.model.stream_mlp_head
        with torch.no_grad():
            module.w1.normal_(0, 0.3)
            module.w2.normal_(0, 1.0)
            # The same prompt and marker, scored by the plugin, and a copy
            # with the head zeroed: they may differ only on the label ids.
            merged = self._run_forward_and_logits([10, 20, 30, 251])[-1].float()
            w1, w2 = module.w1.clone(), module.w2.clone()
            module.w2.zero_()
            plain = self._run_forward_and_logits([10, 20, 30, 251])[-1].float()
            module.w1.copy_(w1)
            module.w2.copy_(w2)

        others = _non_labels(self.config.vocab_size)
        torch.testing.assert_close(merged[others], plain[others])
        assert torch.isfinite(merged).all()
        assert not torch.allclose(merged[_HEAD_LABELS], plain[_HEAD_LABELS])


class TestStreamHeadOutputFit(_VLLMModelTestBase):
    """A head module may return extra columns, in another float dtype.

    Here the module returns fp32 (the model runs bf16) and two columns more
    than its outputs: the scores are fitted to the score row, column k landing
    on token_ids[k].
    """

    def _make_config(self):
        return _tiny_vllm_config(streams=[{"type": "lora"}, _probe_entry(0)])

    def test_wide_fp32_scores_fit_the_row(self, monkeypatch):
        self.model.eval()
        n = len(_HEAD_LABELS)
        wide = torch.arange(1.0, n + 3, device=self.device)

        def forward(hidden, rows):
            return wide.expand(hidden.shape[0], -1).float()

        monkeypatch.setattr(self.model.model.probe_head, "forward", forward)
        with torch.no_grad():
            last = self._run_forward_and_logits([10, 20, 30, 251])[-1].float()
        torch.testing.assert_close(last[_HEAD_LABELS], wide[:n])
        assert torch.isneginf(last[_non_labels(self.config.vocab_size)]).all()


class TestStreamHeadOnAdapter(_VLLMModelTestBase):
    """A probe that names an adapter runs it at its marker (Shadow Residual).

    The adapter writes no K/V, so the marker alone gives the adapter's state:
    the probe on the adapter must read what the probe on the base model reads,
    plus the adapter's change.
    """

    def _make_config(self):
        return _tiny_vllm_config(
            streams=[
                {"type": "lora"},
                _probe_entry(0),
                _probe_entry(1, adapter="adapter_1"),
            ],
            dual_stream=True,
            cross_stream_rank=16,
        )

    def test_marker_routes_through_the_named_adapter(self):
        self.model.eval()
        assert self.model.model.adapter_lut_t.tolist() == [0, 1, 0, 1]
        head = self.model.model.probe_head
        with torch.no_grad():
            # Both rows score the same way, so only the read state differs.
            head.weight[0, :2].normal_(0, 0.5)
            head.weight[1, :2] = head.weight[0, :2]
            on_base = self._run_forward_and_logits([10, 20, 30, 251])[-1]
            on_adapter = self._run_forward_and_logits([10, 20, 30, 252])[-1]
        assert torch.isfinite(on_base[_HEAD_LABELS]).all()
        assert torch.isfinite(on_adapter[_HEAD_LABELS]).all()
        assert not torch.allclose(on_base[_HEAD_LABELS], on_adapter[_HEAD_LABELS])


class TestStreamHeadOnLoRAAdapter(_VLLMModelTestBase):
    """A probe on a single-stream LoRA adapter, with the adapter's own token.

    The prompt carries the adapter's token first (as the chat template places
    it) and the probe's marker last. The probe must read the state the adapter
    computes when it runs alone: with an identity-like head, its scores equal
    the head applied to the adapter-alone state at that position.
    """

    def _make_config(self):
        return _tiny_vllm_config(
            streams=[
                {"type": "lora"},
                _probe_entry(0),
                _probe_entry(1, adapter="adapter_1"),
            ]
        )

    def test_paired_read_is_the_adapter_state(self):
        self.model.eval()
        assert self.model.model.adapter_lut_t.tolist() == [0, 1, 0, 1]
        head = self.model.model.probe_head
        n = len(_HEAD_LABELS)
        with torch.no_grad():
            head.weight[1, :n].normal_(0, 0.5)
            body = [10, 20, 30]
            # The adapter alone, then a plain token that embeds as the marker
            # does (its substitute, id 1): the reference state.
            alone = self._run_forward([250, *body, 1])[-1, : self.config.hidden_size]
            reference = head.weight[1, :n] @ alone.to(head.weight.dtype)
            paired = self._run_forward_and_logits([250, *body, 252])[-1]
            marker_only = self._run_forward_and_logits([1, *body, 252])[-1]
        torch.testing.assert_close(
            paired[_HEAD_LABELS].float(), reference.float(), atol=0.05, rtol=0.02
        )
        # Without the adapter's token, the marker runs it on one token only:
        # a different state, so different scores.
        assert not torch.allclose(
            paired[_HEAD_LABELS].float(), marker_only[_HEAD_LABELS].float(), atol=0.05
        )


# ════════════════════════════════════════════════════════════════════
# 5. KV visibility tests
# ════════════════════════════════════════════════════════════════════


class TestKVVisibility(_VLLMModelTestBase):
    def test_adapter_token_kv_invisible(self):
        torch.manual_seed(SEED)
        self.model.eval()

        seq = [10, 20, 250, 30, 40, 50, 60, 70]

        with torch.no_grad():
            logits_a = self._run_forward_and_logits(seq)

        with torch.no_grad():
            perturbation = (
                torch.randn(
                    self.config.hidden_size, device=self.device, dtype=torch.bfloat16
                )
                * 10.0
            )
            self.model.model.embed_tokens.weight.data[250] += perturbation

        with torch.no_grad():
            logits_b = self._run_forward_and_logits(seq)

        torch.testing.assert_close(
            logits_a[3:],
            logits_b[3:],
            msg="Post-adapter-token logits should be identical",
        )
