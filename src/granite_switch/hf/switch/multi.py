# SPDX-License-Identifier: Apache-2.0
"""Coded-memory MultiSwitch (HuggingFace backend) — Kerdock/DG codes on token-exchange.

Performs coarse-grained, multi-transition adapter routing (base <-> exp1 <->
exp2 <-> base <-> ... arbitrarily many times per request) via two tiny
attention heads:

1. **Counting head (single query head).** A position-0 anchor holds ``V=1``;
   every control token holds ``V=0`` and an un-masked key. A one-hot query
   reads back ``1/(1+n)`` where ``n`` is the number of control tokens seen so
   far (causally). ``recover_count_from_signal`` inverts this to the integer
   *write address* ``n``. Non-participating tokens are masked with a large
   finite negative key value (``-1e9``, not literal ``-inf``: ``0 * -inf =
   NaN`` under IEEE-754, and our one-hot queries have zeros where the mask
   sits). The count is sequence-length-independent — it depends only on how
   many control tokens precede a position. FP32 is used for the counting Q/K/V
   so the ``1/(1+n)`` inversion stays precise (bf16 loses precision past a few
   hundred transitions).

2. **Memory head (single head, Kerdock/DG coded).** At each control token the
   key is ``code(n) * memory_gain`` and the value is the ``expert_id``. Every
   token queries with ``code(n)``. Kerdock/DG codewords have provably low
   mutual coherence, so the softmax over coded keys concentrates on the
   matching address and the attention output is the ``expert_id`` most recently
   written at the current address. ``round`` + ``clamp[0, num_adapters]`` yields
   the integer adapter index.

The codebook is precomputed once into a registered buffer, so the forward path
does a plain index lookup (``self.codebook[write_addresses]``) — no numpy, no
per-call for-loops. Both this buffer and the token-exchange LUT are registered
``persistent=True``: ``from_pretrained`` fills only tensors present in the
checkpoint, so a non-persistent buffer is silently returned as ALL ZEROS,
which collapses the retrieval softmax to a uniform average over adapters.

Token-exchange: adapter *selection* reads the ORIGINAL ``input_ids``; at the
end of ``forward`` the control-token ids are rewritten to their substitute ids
via a precomputed LUT (``apply_token_exchange``) so the decoder embeds a clean
sequence and never knows a control token existed. The LUT is built in
``__init__`` from the config's ``adapter_token_ids`` /
``adapter_substitute_token_ids`` (``build_control_to_substitute_lut``).

``num_cache_layers == 2``: the switch owns two logical cache slots (counting +
memory) at ``layer_idx`` and ``layer_idx + 1``. The property returns 2 so the
model glue can account for both.

The cache params below are the LIVE ``generate()`` path, not legacy scaffolding.
``GraniteSwitchModel.forward`` creates a ``DynamicCache`` whenever ``use_cache``
is set (the config default) and passes it to this switch on every call, so at
decode ``q_len == 1`` while both heads attend over the whole cached key length.
That asymmetry is load-bearing: building the internal mask ``q_len x q_len``
instead of over ``kv_len`` crashed a real composed 3B, which is what
``tests/hf/test_multi_switch_generate.py`` was added to pin. The same file
asserts that a control token seen only during prefill still routes every later
decode step to its adapter -- the KV cache is how the switch carries the adapter
forward, not an obstacle to it.

memory_gain: the default 28.0 (``ms_memory_gain``) is validated for exact
retrieval at the codebook's full capacity with margin. A gain of 16.0 is the
smallest that still retrieves exactly at capacity; 8.0 does not. If you change
the code or the gain, re-validate against the codes' exact-retrieval unit test —
the right value is the smallest gain that gives exact retrieval with margin.
"""

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...token_exchange import apply_token_exchange, build_control_to_substitute_lut
from .codes import KerdockDGCodeGenerator, recover_count_from_signal

# Large negative finite value for masking. NOT literal -inf because IEEE 754
# defines 0 * ±inf = NaN, and our one-hot Q vectors have zeros at dimensions
# where K has the mask value. Using -1e9 gives 0*(-1e9)=0 (clean) and
# exp(-1e9) ≈ 0 in softmax (same masking effect).
_NEG_INF = -1e9


class _CountingAttention(nn.Module):
    """Single-head counting attention module (nn.Module for HF backends).

    One query head, one KV head. Head dim is ``counting_head_dim`` (independent
    of the model head_dim). Only dim 0 carries signal; the rest is zero padding.
    """

    def __init__(self, head_dim: int, layer_idx: int):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = layer_idx


class _MemoryAttention(nn.Module):
    """Single-head Kerdock/DG memory attention module (nn.Module for HF backends).

    Uses ``memory_head_dim`` (>= code/memory dimension). Stores a reference to
    the model config so backends like flash_attention_2 can read
    ``config._pre_quantization_dtype`` / ``config._attn_implementation``.
    """

    def __init__(self, head_dim: int, layer_idx: int, config=None):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = layer_idx
        self.config = config


class MultiSwitch(nn.Module):
    """Coded-memory multi-transition switch (HF attention backends).

    Uses Kerdock/DG codes for exact integer adapter retrieval across an
    arbitrary number of transitions per request. See the module docstring for
    the engine.

    Args:
        num_adapters: Number of real LoRA adapters (does NOT include base).
            Index 0 always means base/no-adapter; valid adapter indices are
            ``0..num_adapters``. Matches ``GraniteSwitchConfig.num_adapters``.
        config: Model configuration. Provides backbone head geometry, the
            token-exchange substitute ids, and the ``ms_*`` coded-engine params.
        control_token_gain: Accepted for signature parity with the modeling
            glue. The coded engine's key scaling is ``ms_memory_gain``, so this
            argument is not used by the memory head.
        switch_head_dim: Fallback head_dim (>= 32) for standalone/test mode
            when no backbone geometry is available on ``config``.
        layer_idx: Base cache slot index. Counting uses ``layer_idx``, memory
            uses ``layer_idx + 1`` (num_cache_layers == 2).
    """

    def __init__(
        self,
        num_adapters: int,
        config=None,
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.control_token_gain = control_token_gain
        self.config = config
        self.layer_idx = layer_idx
        self.counting_layer_idx = layer_idx
        self.memory_layer_idx = layer_idx + 1

        # ── Expert-id offset (two accepted adapter_token_ids layouts).
        #   * num_adapters entries (no base-reset slot):
        #     adapter_token_ids[i] fires adapter i+1 -> expert_id = argmax + 1.
        #   * num_adapters + 1 entries (base-reset layout): adapter_token_ids[0]
        #     is the base-reset token (fires 0) and [1..] fire 1.. -> expert_id
        #     = argmax (no shift). This lets a request transition back to base.
        #
        # The base-reset layout comes from composing with `--base-reset-token`
        # (opt-in): `add_control_tokens(base_reset=True)` prepends `<|base_reset|>`,
        # so `len(adapter_token_ids) == num_adapters + 1` and the offset below is 0.
        # Without the flag nothing changes (one token per adapter, offset 1) --
        # still the default. tests/composer/test_base_reset_token.py drives this
        # engine with exactly what compose emits; TestBaseResetLayout covers the
        # engine in isolation against the hand-built ATOK_BASE_RESET fixture.
        #
        # Ordinary aLoRA chat does NOT need it: the chat template emits one control
        # token for the CURRENT turn only, so prior turns carry none and route to
        # base already. An explicit base token is needed when a single sequence
        # holds several control tokens and must return to base between them --
        # agentic per-step switching, or a preserved multi-turn history.
        #
        # The token is placed by the caller, not the template: configure_chat_template
        # only emits control tokens for adapters, and base-reset is not an adapter.
        # The offset is a static (shape-derived) Python constant computed once
        # here, so forward stays branch-free / torch.compile-safe.
        ctrl_ids = (
            getattr(config, "adapter_token_ids", None) if config is not None else None
        )
        if ctrl_ids is not None and len(ctrl_ids) == num_adapters + 1:
            self._expert_id_offset = 0  # base-reset layout; argmax is the id
        else:
            self._expert_id_offset = 1  # no base slot; adapter i+1 for slot i

        # ── Coded-engine parameters (read via getattr; config carries them as
        # plain attributes, no dict). code_m=6 Kerdock => N=64, capacity=2048.
        code_m = getattr(config, "ms_code_m", 6) if config is not None else 6
        code_type = (
            getattr(config, "ms_code_type", "kerdock")
            if config is not None
            else "kerdock"
        )
        self.memory_gain = (
            getattr(config, "ms_memory_gain", 28.0) if config is not None else 28.0
        )
        counting_head_dim = (
            getattr(config, "ms_counting_head_dim", 32) if config is not None else 32
        )

        # ── Kerdock/DG code generator + precomputed codebook buffer.
        # precompute_codebook returns [capacity, N] unit-norm codewords; N is
        # the code (= memory) dimension.
        #
        # MUST be persistent=True. ``from_pretrained`` materializes modules on
        # the meta device and then fills only tensors present in the checkpoint;
        # a NON-persistent buffer is absent from the state_dict, so whatever
        # ``__init__`` computed here is discarded and the buffer comes back as
        # ALL ZEROS. A zeroed codebook makes every memory key/query zero, so all
        # attention logits collapse to 0, the retrieval softmax goes UNIFORM, and
        # each position averages the expert ids it can see instead of selecting
        # the most recent one (e.g. [T,A,T,B,T] -> memory_raw
        # [0, .504, .337, .756, .605] -> rounds to [0,1,0,1,1] instead of
        # [0,1,1,2,2]). Storing the tensor also makes a missing/incompatible
        # checkpoint fail loudly as a missing key rather than silently routing
        # to averaged adapters. Cost is ~512 KB on a multi-GB checkpoint.
        self.code_gen = KerdockDGCodeGenerator(
            m=code_m, code_type=code_type, verbose=False
        )
        self.capacity = self.code_gen.capacity
        self.memory_dim = self.code_gen.N
        codebook = self.code_gen.precompute_codebook(dtype=torch.float32)
        self.register_buffer("codebook", codebook, persistent=True)

        # ── Head geometry.
        # Counting is a SINGLE query head over a SINGLE KV head.
        # FlashAttention requires head_dim >= 32. Counting head_dim: only dim 0
        # carries the one-hot signal; the rest is zero padding.
        self.counting_head_dim = max(int(counting_head_dim), 32)
        if self.counting_head_dim < 32:
            raise ValueError(
                f"counting_head_dim must be >= 32 for FlashAttention, got {self.counting_head_dim}"
            )

        # Memory head_dim: must hold the full code vector (>= memory_dim). For
        # Kerdock m=6, memory_dim=64 already exceeds 32. Prefer the backbone
        # projection_head_dim when it is >= memory_dim (so all attention layers
        # share one head_dim); otherwise use memory_dim directly. Extra dims (if
        # any) are zero-padded.
        backbone_head_dim = None
        if config is not None:
            backbone_head_dim = getattr(config, "projection_head_dim", None)
            if backbone_head_dim is None and hasattr(config, "num_attention_heads"):
                backbone_head_dim = config.hidden_size // config.num_attention_heads
        if backbone_head_dim is not None and backbone_head_dim >= self.memory_dim:
            self.memory_head_dim = backbone_head_dim
        else:
            self.memory_head_dim = max(self.memory_dim, 32)
        if self.memory_head_dim < self.memory_dim:
            raise ValueError(
                f"memory_head_dim ({self.memory_head_dim}) must be >= code/memory "
                f"dimension ({self.memory_dim}) to hold a full codeword"
            )

        # Attention submodules (proper nn.Modules for HF backend dispatch).
        # Counting: single head, cache slot = counting_layer_idx.
        self.counting_attn_module = _CountingAttention(
            head_dim=self.counting_head_dim,
            layer_idx=self.counting_layer_idx,
        )
        # Memory: single head, cache slot = memory_layer_idx; gets config for
        # flash_attention_2 dtype/impl introspection.
        self.memory_attn_module = _MemoryAttention(
            head_dim=self.memory_head_dim,
            layer_idx=self.memory_layer_idx,
            config=config,
        )

        # Token-exchange LUT (control id -> substitute id), None if unconfigured.
        # persistent=True for the same reason as ``codebook`` above: a
        # non-persistent buffer is zeroed by ``from_pretrained``. This LUT uses
        # -1 as the "not a control token" sentinel, so an all-zero LUT would
        # rewrite EVERY token id to 0 in apply_token_exchange.
        lut = build_control_to_substitute_lut(config)
        if lut is not None:
            self.register_buffer("control_to_substitute_lut", lut, persistent=True)
        else:
            self.control_to_substitute_lut = None

    @property
    def num_cache_layers(self) -> int:
        """Cache slots used by this switch: counting slot + memory slot."""
        return 2

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token adapter indices and rewrite control tokens.

        Adapter selection reads the ORIGINAL ``input_ids``. Token-exchange is
        applied at the very end: ``modified_input_ids = apply_token_exchange(
        lut, input_ids)``.

        Args:
            input_ids: ``[batch, seq_len]`` input token ids.
            adapter_token_ids: ``[num_adapters (+1)]`` activating control token
                ids. ``adapter_token_ids[i]`` fires adapter ``i`` (index 0 is
                the base/no-adapter slot when present).
            attention_mask: ``[batch, 1, seq_len, seq_len]`` optional 4D mask.
            past_key_values: Cache shared with the decoder layers. The model glue
                passes the live cache on every call, so this is the normal
                ``generate()`` path; ``None`` only for a single full-sequence
                forward. Both heads attend over the whole cache, so the internal
                mask is built over ``kv_len``, not ``q_len``.
            cache_position: ``[seq_len]`` positions; position 0 is the counting
                anchor. Defaults to ``arange(seq_len)``.

        Returns:
            ``(adapter_indices, modified_input_ids)`` — both ``[batch, seq_len]``.
            ``adapter_indices``: 0 = base, 1+ = adapters.
            ``modified_input_ids``: control ids rewritten to substitute ids.
        """
        bsz, q_len = input_ids.shape
        device = input_ids.device

        # Position-0 anchor for the 1/(1+n) counting (replaces a special
        # <|init|> token). Derived from cache_position; NOT KV-hidden.
        if cache_position is None:
            cache_position = torch.arange(q_len, device=device)
        is_counting_anchor = (
            (cache_position == 0).unsqueeze(0).expand(bsz, -1)
        )  # [B, S]

        # Vectorized control-token matching against ORIGINAL input_ids.
        # [B, S, 1] == [1, 1, A] -> [B, S, A]
        matches = input_ids.unsqueeze(2) == adapter_token_ids.unsqueeze(0).unsqueeze(0)
        is_control_token = matches.any(dim=2)  # [B, S]
        # expert_id = argmax + offset (offset selects the layout; see __init__).
        expert_ids = torch.where(
            is_control_token,
            matches.long().argmax(dim=2) + self._expert_id_offset,
            torch.zeros_like(input_ids, dtype=torch.long),
        )  # [B, S]

        # ==================================================================
        # Step 1: Counting via single-head attention -> write address n.
        # ==================================================================
        # FP32 is mandatory for precise 1/(1+n) recovery; SDPA handles fp32.
        key_states_count = torch.full(
            (bsz, 1, q_len, self.counting_head_dim),
            _NEG_INF,
            device=device,
            dtype=torch.float32,
        )
        # Un-mask (0) dim 0 at anchor + control tokens (arithmetic where).
        anchor_or_control = is_counting_anchor | is_control_token  # [B, S]
        key_states_count[:, 0, :, 0] = torch.where(
            anchor_or_control,
            torch.zeros_like(key_states_count[:, 0, :, 0]),
            key_states_count[:, 0, :, 0],
        )

        value_states_count = torch.zeros(
            (bsz, 1, q_len, self.counting_head_dim),
            device=device,
            dtype=torch.float32,
        )
        value_states_count[:, 0, :, 0] = is_counting_anchor.to(
            torch.float32
        )  # v=1 at anchor

        query_states_count = torch.zeros(
            (bsz, 1, q_len, self.counting_head_dim),
            device=device,
            dtype=torch.float32,
        )
        query_states_count[:, 0, :, 0] = 1.0  # one-hot => Q·K = K[0]

        # Optional KV cache (standalone/legacy use; model glue passes none).
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states_count, value_states_count = past_key_values.update(
                key_states_count,
                value_states_count,
                self.counting_layer_idx,
                cache_kwargs,
            )

        # Counting head ALWAYS uses SDPA: fp32 precision is required, and
        # flash/other backends downcast fp32 and destroy 1/(1+n) recovery.
        counting_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

        # Memory head ALSO uses SDPA (fp32), NOT the model's configured backend.
        # Its Q/K/V are built in fp32 (below), and the memory read is a softmax
        # over Kerdock/DG-coded keys whose low mutual coherence only survives in
        # fp32 — the same precision requirement as the counting head. Dispatching
        # fp32 tensors through the model's real backend (e.g. flash_attention_2,
        # which requires fp16/bf16) silently corrupts the read and routes every
        # token to base: the isolated switch tests pass, but the full model
        # returns all-zeros adapter indices. Forcing SDPA keeps the read exact and
        # backend-independent. (An earlier version delegated to
        # config._attn_implementation on the mistaken assumption that a downcast
        # was harmless; it is not — it destroys the codebook separation.)
        memory_interface = counting_interface

        # Build an INTERNAL fp32 causal mask instead of consuming the model's
        # ``attention_mask``. The model's mask is created in the model dtype
        # (bf16 in real deployments); feeding a bf16 mask to these fp32-Q/K/V
        # SDPA calls forces the attention to run in bf16, and bf16's coarse
        # resolution corrupts the counting head's ``1/(1+n)`` recovery — the
        # recovered segment index ``n`` comes out off by one, so routing lags a
        # position (e.g. ``[T,A,T,B,T]`` -> ``[0,0,0,1,1]`` instead of
        # ``[0,1,1,2,2]``). The switch's attention is plain causal (each position
        # attends to all positions at-or-before it); which keys actually count is
        # already encoded in the fp32 key masking above, so this positional mask
        # just needs to be causal and fp32. Building it here makes both heads
        # fully independent of the model's mask dtype/backend.
        # The mask must span the KEY length, not the query length. With a KV cache
        # (i.e. during generation) the heads attend over the whole cached history,
        # so ``kv_len > q_len`` and a ``q_len x q_len`` mask is simply the wrong
        # shape: at a decode step with q_len=1 and kv_len=6 SDPA received a
        # (1,1,1,1) bias for a 1-query-over-6-keys attention and raised
        # "(*bias): last dimension must be contiguous" (a shape complaint, not an
        # actual stride problem -- every tensor was contiguous). Prefill has
        # q_len == kv_len, which is why every forward-only test passed.
        #
        # The causal offset shifts by the cached prefix: the ``kv_len - q_len``
        # cached positions are all strictly in the past, so query i may attend to
        # keys 0..(kv_len - q_len + i). At prefill the offset is 1 and this reduces
        # exactly to the previous behavior.
        def _causal_mask(kv_len: int) -> torch.Tensor:
            return torch.triu(
                torch.full(
                    (q_len, kv_len), _NEG_INF, device=device, dtype=torch.float32
                ),
                diagonal=1 + (kv_len - q_len),
            ).view(1, 1, q_len, kv_len)

        # Each head owns its own cache slot, so their key lengths are tracked
        # independently (the memory cache is updated further down, after this).
        counting_mask = _causal_mask(key_states_count.shape[2])

        # Both heads run with autocast DISABLED. The enclosing model commonly runs
        # under bf16 autocast; if these fp32 attention ops are allowed to autocast,
        # SDPA casts the fp32 Q/K/V (and the fp32 causal mask) down to bf16. For the
        # memory head that is fatal: its keys are ``code(n) * memory_gain`` (logits
        # ~28) and the mask is a large negative — in bf16 these overflow the softmax
        # to NaN (intermittently, per kernel), which rounds/clamps to 0 and routes
        # every token to base. Forcing fp32 here keeps both heads exact and
        # deterministic regardless of the model's autocast context.
        _autocast_off = (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type in ("cuda", "cpu")
            else torch.autocast(device_type="cuda", enabled=False)
        )
        with _autocast_off:
            count_output, _ = counting_interface(
                self.counting_attn_module,
                query_states_count,
                key_states_count,
                value_states_count,
                counting_mask,
                dropout=0.0,
                scaling=self.counting_attn_module.scaling,
                sliding_window=None,
            )
        # count_output: [B, S, num_heads=1, counting_head_dim]
        counting_signal = count_output[:, :, 0, 0]  # [B, S] = 1/(1+n)

        write_addresses = recover_count_from_signal(
            counting_signal, capacity=self.capacity
        )  # [B, S]

        # ==================================================================
        # Step 2: Coded memory -> expert id.
        # ==================================================================
        flat_addresses = write_addresses.reshape(-1)
        all_code_vectors = self.codebook.to(device)[flat_addresses].reshape(
            bsz, q_len, self.memory_dim
        )  # [B, S, memory_dim], fp32

        write_mask = is_control_token.unsqueeze(2).to(torch.float32)  # [B, S, 1]

        key_states_memory = torch.zeros(
            (bsz, 1, q_len, self.memory_head_dim),
            device=device,
            dtype=torch.float32,
        )
        # code(n) * memory_gain at control tokens, zero elsewhere (arithmetic).
        key_states_memory[:, 0, :, : self.memory_dim] = (
            all_code_vectors * self.memory_gain * write_mask
        )

        value_states_memory = torch.zeros(
            (bsz, 1, q_len, self.memory_head_dim),
            device=device,
            dtype=torch.float32,
        )
        value_states_memory[:, 0, :, 0] = expert_ids.to(
            torch.float32
        ) * is_control_token.to(torch.float32)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states_memory, value_states_memory = past_key_values.update(
                key_states_memory,
                value_states_memory,
                self.memory_layer_idx,
                cache_kwargs,
            )

        query_states_memory = torch.zeros(
            (bsz, 1, q_len, self.memory_head_dim),
            device=device,
            dtype=torch.float32,
        )
        query_states_memory[:, 0, :, : self.memory_dim] = (
            all_code_vectors  # code(n) everywhere
        )

        with _autocast_off:  # fp32; see the counting-head call above (bf16 -> NaN)
            memory_output, _ = memory_interface(
                self.memory_attn_module,
                query_states_memory,
                key_states_memory,
                value_states_memory,
                # Built from the MEMORY head's own key length: its cache slot is
                # separate from the counting head's and is updated above, so the
                # two key lengths must be tracked independently.
                _causal_mask(key_states_memory.shape[2]),
                dropout=0.0,
                scaling=self.memory_attn_module.scaling,
                sliding_window=None,
            )

        # ==================================================================
        # Step 3: Extract, round, clamp adapter indices.
        # ==================================================================
        # memory_output: [B, S, num_heads=1, memory_head_dim]
        adapter_indices = memory_output[:, :, 0, 0]  # [B, S] raw expert id
        adapter_indices = torch.round(adapter_indices).long()
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        assert adapter_indices.shape == input_ids.shape, (
            f"adapter_indices shape {adapter_indices.shape} must match "
            f"input_ids shape {input_ids.shape}"
        )

        # Token-exchange rewrite (branch-free).
        modified_input_ids = apply_token_exchange(
            self.control_to_substitute_lut, input_ids
        )

        return adapter_indices, modified_input_ids
