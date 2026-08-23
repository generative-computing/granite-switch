# SPDX-License-Identifier: Apache-2.0
"""Coded-memory MultiSwitch (vLLM backend) — Kerdock/DG codes on token-exchange.

vLLM counterpart of the HF ``MultiSwitch``. Performs coarse-grained,
multi-transition adapter routing (base <-> exp1 <-> exp2 <-> base <-> ...
arbitrarily many times per request) via two tiny attention heads:

1. **Counting head (single query head).** A position-0 anchor holds ``V=1``;
   every control token holds ``V=0`` and an un-masked key. A one-hot query
   reads back ``1/(1+n)`` where ``n`` is the number of control tokens seen so
   far (causally). ``recover_count_from_signal`` inverts this to the integer
   *write address* ``n``. Non-participating tokens are masked with a large
   finite negative key value (``-1e9``, not literal ``-inf``: ``0 * -inf =
   NaN`` under IEEE-754, and our one-hot queries have zeros where the mask
   sits). This count is sequence-length-independent — it depends only on how
   many control tokens precede a position, not on absolute position.

2. **Memory head (single head, Kerdock/DG coded).** At each control token the
   key is ``code(n) * memory_gain`` and the value is the ``expert_id``. Every
   token queries with ``code(n)``. Because Kerdock/DG codewords have provably
   low mutual coherence, the softmax over coded keys concentrates on the
   matching address, so the attention output is the ``expert_id`` most recently
   written at the current address. ``round`` + ``clamp[0, num_adapters]`` yields
   the integer adapter index.

The codebook is precomputed once into a registered buffer so the forward path
does a plain index lookup (``self.codebook[write_addresses]``) — no numpy, no
lazy init — which keeps the whole switch AOT-autograd / ``@support_torch_compile``
safe. All masking is arithmetic (``torch.where`` / multiply), there is no
boolean-index assignment, no data-dependent branching, and the only debug
writes sit behind ``not torch.compiler.is_compiling()``, so they are absent from a
compiled graph entirely -- ``_debug_write_addresses`` is available under
``enforce_eager`` only, never in a default-configured server.

The Kerdock/DG codes and the token-exchange helpers are imported from the HF
backend (``granite_switch.hf.switch``) as the single source of truth; the codes
are pure torch/numpy and backend-agnostic, so there is no vLLM copy.

Token-exchange: adapter *selection* reads the ORIGINAL ``input_ids``; at the
end of ``forward`` the control-token ids are rewritten to their substitute ids
via a precomputed LUT (``apply_token_exchange``) so the decoder embeds a clean
sequence and never knows a control token existed.

``num_cache_layers == 2``: this switch constructs two ``vllm.Attention`` layers
(the counting slot + the memory slot), each consuming one KV-cache group. This
is the rationale for the property returning 2 (SingleSwitch returns 1).

Batching (vLLM): the counting/memory attention runs over the flat token stream
vLLM hands the model, and the counting anchor is derived from the ``positions``
tensor (``positions == 0``). Because vLLM's ``positions`` restart at 0 for each
request, a flattened multi-request batch carries one anchor PER REQUEST, which is
exactly what the ``1/(1+n)`` counting needs. The two heads are real paged-KV
``vllm.Attention`` modules, so vLLM already confines each request's attention to
its own tokens. This only holds when the caller forwards the real ``positions``:
``forward`` therefore REQUIRES it and raises rather than synthesizing an
``arange``, which would anchor only the first request in the batch and silently
mis-route the others.

Decode reads the control token back out of the cache. That is the mechanism rather
than a gap, and it is measured, not argued: see the verdict below. A control token
seen during prefill is absent from a decode step's ``input_ids``, but both heads
are real paged-KV ``vllm.Attention`` modules,
so a decode query still attends over the cached anchor and control-token keys: the
counting head recovers the same ``n`` from ``1/(1+n)``, and the memory head reads
back the codeword written at address ``n``, and with it the active expert id. The
HF twin relies on exactly this and asserts it over a growing ``DynamicCache``
(``tests/hf/test_multi_switch_generate.py``).

Routing correctness here is pinned by
``tests/integration/test_multi_switch_serving_e2e.py``, which traces every
``forward`` from INSIDE the engine process (a ``sitecustomize.py`` on
``PYTHONPATH`` -- an in-process monkeypatch cannot reach the EngineCore child) and
compares the adapter index at every position against a ground-truth array. Do not
try to establish this from generated text: a control token routes the prefill
region, so continuations differ between adapters even when decode reverts to base.
That file records the failure it was built for -- identical output with 23/23
wrong decode routing.

VERDICT (granite-switch-internal, vela_yamls/ci/multiswitch-serving-routing.yaml, 1
GPU): decode routing is correct. Zero of 1623 checked positions mis-routed, across
five prompts at 100% trace coverage each (24/24, 61/61, 413/413, 614/614, 396/396),
over 23 pure-decode and 6 mixed forwards with up to 5 concurrent requests in one
forward. The 23 is the same count as the historical 23/23 failure above, so this
exercises the workload that once broke rather than an easier one. Two prompts
generated different TEXT solo vs co-batched; their routing was identical, which is
the bf16 sampling near-tie and not a routing leak -- the reason that test reports
text differences and asserts only on routing.

memory_gain: the default 28.0 (``ms_memory_gain``) is validated for exact
retrieval at the codebook's full capacity with margin. A gain of 16.0 is the
smallest that still retrieves exactly at capacity; 8.0 does not. If you change
the code or the gain, re-validate against the codes' exact-retrieval unit test —
the right value is the smallest gain that gives exact retrieval with margin.
"""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.attention import Attention

from granite_switch.hf.switch._token_exchange import (
    apply_token_exchange,
    build_substitute_lut,
)
from granite_switch.hf.switch.codes import (
    KerdockDGCodeGenerator,
    recover_count_from_signal,
)

# Large negative finite value for masking. NOT literal -inf because IEEE 754
# defines 0 * ±inf = NaN, and our one-hot Q vectors have zeros at dimensions
# where K has the mask value. Using -1e9 gives 0*(-1e9)=0 (clean) and
# exp(-1e9) ≈ 0 in softmax (same masking effect).
_NEG_INF = -1e9


class MultiSwitch(nn.Module):
    """Coded-memory multi-transition switch (vLLM Attention backend).

    Uses Kerdock/DG codes for exact integer adapter retrieval across an
    arbitrary number of transitions per request. See module docstring for the
    engine.

    Args:
        num_adapters: Number of real LoRA adapters (does NOT include base).
            Index 0 always means base/no-adapter; valid adapter indices are
            ``0..num_adapters``. Matches ``GraniteSwitchConfig.num_adapters``.
        vllm_config: vLLM configuration (provides dtype, cache/quant config).
        control_token_gain: Accepted for signature parity with SingleSwitch /
            the modeling glue. The coded engine's key scaling is
            ``ms_memory_gain`` (the codes carry the addressing, not a single
            gain dim), so this argument is not used by the memory head.
        switch_head_dim: Fallback head_dim (>= 32) for standalone/test mode
            when no backbone geometry is available on ``config``.
        config: GraniteSwitchConfig (provides backbone head geometry + the
            token-exchange substitute ids + the ``ms_*`` coded-engine params).
    """

    def __init__(
        self,
        num_adapters: int,
        vllm_config: VllmConfig | None = None,
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        config=None,
        attn_backend=None,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.control_token_gain = control_token_gain

        if vllm_config is not None and vllm_config.model_config is not None:
            self.dtype = vllm_config.model_config.dtype
        else:
            self.dtype = torch.get_default_dtype()

        hf_config = config
        if (
            hf_config is None
            and vllm_config is not None
            and vllm_config.model_config is not None
        ):
            hf_config = vllm_config.model_config.hf_config

        # ── Expert-id offset (two accepted adapter_token_ids layouts).
        #   * num_adapters entries (SingleSwitch-style, no base-reset slot):
        #     adapter_token_ids[i] fires adapter i+1 -> expert_id = argmax + 1.
        #   * num_adapters + 1 entries (base-reset layout): adapter_token_ids[0]
        #     is the base-reset token (fires 0) and [1..] fire 1.. -> expert_id
        #     = argmax (no shift). This lets a request transition back to base.
        #
        # The base-reset layout comes from composing with `--base-reset-token`
        # (opt-in): `add_control_tokens(base_reset=True)` prepends `<|base_reset|>`,
        # so `len(adapter_token_ids) == num_adapters + 1` and the offset below is 0.
        # Without the flag nothing changes (one token per adapter, offset 1) --
        # still the default.
        #
        # Ordinary aLoRA chat does NOT need it: the chat template emits one control
        # token for the CURRENT turn only, so prior turns carry none and route to
        # base already. An explicit base token is needed when a single sequence
        # holds several control tokens and must return to base between them --
        # agentic per-step switching, or a preserved multi-turn history.
        #
        # The token is placed by the caller, not the template: configure_chat_template
        # only emits control tokens for adapters, and base-reset is not an adapter.
        # Static (shape-derived) Python constant computed once here, so forward
        # stays branch-free / @support_torch_compile-safe.
        ctrl_ids = (
            getattr(hf_config, "adapter_token_ids", None)
            if hf_config is not None
            else None
        )
        if ctrl_ids is not None and len(ctrl_ids) == num_adapters + 1:
            self._expert_id_offset = 0  # base-reset layout; argmax is the id
        else:
            self._expert_id_offset = 1  # no base slot; adapter i+1 for slot i

        # ── Coded-engine parameters (read via getattr; config carries them as
        # plain attributes, no dict). code_m=6 Kerdock => N=64, capacity=2048.
        code_m = getattr(hf_config, "ms_code_m", 6) if hf_config is not None else 6
        code_type = (
            getattr(hf_config, "ms_code_type", "kerdock")
            if hf_config is not None
            else "kerdock"
        )
        self.memory_gain = (
            getattr(hf_config, "ms_memory_gain", 28.0)
            if hf_config is not None
            else 28.0
        )
        counting_head_dim = (
            getattr(hf_config, "ms_counting_head_dim", 32)
            if hf_config is not None
            else 32
        )

        # ── Kerdock/DG code generator + precomputed codebook buffer.
        # precompute_codebook returns [capacity, N] unit-norm codewords; N is
        # the code (= memory) dimension.
        #
        # MUST be persistent=True. A NON-persistent buffer is absent from the
        # state_dict, so checkpoint loaders that materialize modules on the meta
        # device (HF ``from_pretrained``; vLLM's weight loader likewise fills
        # only checkpoint tensors) discard what ``__init__`` computed and leave
        # the buffer ALL ZEROS. A zeroed codebook makes every memory key/query
        # zero, so all logits collapse to 0, the retrieval softmax goes UNIFORM,
        # and each position averages the visible expert ids instead of selecting
        # the most recent one. Keeping it persistent also turns a missing or
        # incompatible checkpoint into a loud missing-key error instead of
        # silent mis-routing. Cost is ~512 KB on a multi-GB checkpoint.
        self.code_gen = KerdockDGCodeGenerator(
            m=code_m, code_type=code_type, verbose=False
        )
        self.capacity = self.code_gen.capacity
        self.memory_dim = self.code_gen.N
        codebook = self.code_gen.precompute_codebook(dtype=torch.float32)
        self.register_buffer("codebook", codebook, persistent=True)

        # ── Head geometry.
        # Counting is a SINGLE query head over a SINGLE KV head. Both counting
        # and memory heads use num_heads == num_kv_heads == 1. FlashAttention
        # requires head_dim >= 32. Counting head_dim: only dim 0 (the one-hot
        # signal channel) is used; the rest is zero padding.
        self.counting_head_dim = max(int(counting_head_dim), 32)
        if self.counting_head_dim < 32:
            raise ValueError(
                f"counting_head_dim must be >= 32 for FlashAttention, got {self.counting_head_dim}"
            )

        # Memory head_dim: must hold the full code vector (>= memory_dim) AND
        # satisfy the >= 32 kernel constraint. Kerdock m=6 gives memory_dim=64,
        # so the code already exceeds 32; extra dims (if any) are zero-padded.
        # Prefer aligning to the backbone projection_head_dim when it is >=
        # memory_dim, so all Attention layers share one head_dim (page-size
        # compatibility); otherwise use memory_dim directly.
        backbone_head_dim = None
        if hf_config is not None:
            backbone_head_dim = getattr(hf_config, "projection_head_dim", None)
            if backbone_head_dim is None and hasattr(hf_config, "num_attention_heads"):
                backbone_head_dim = (
                    hf_config.hidden_size // hf_config.num_attention_heads
                )
        if backbone_head_dim is not None and backbone_head_dim >= self.memory_dim:
            self.memory_head_dim = backbone_head_dim
        else:
            self.memory_head_dim = max(self.memory_dim, 32)
        if self.memory_head_dim < self.memory_dim:
            raise ValueError(
                f"memory_head_dim ({self.memory_head_dim}) must be >= code/memory "
                f"dimension ({self.memory_dim}) to hold a full codeword"
            )

        # Single counting + single memory KV head. Under TP each rank builds
        # identical one-hot / coded Q/K/V locally, and a single head has nothing
        # to shard — so no TP head division is applied.
        self.num_heads = 1
        self.num_kv_heads = 1

        cache_config = vllm_config.cache_config if vllm_config is not None else None
        quant_config = vllm_config.quant_config if vllm_config is not None else None

        # Counting head: single query head, single KV head.
        counting_kwargs = dict(
            num_heads=self.num_heads,
            head_size=self.counting_head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix="switch.multi.0",
        )
        if attn_backend is not None:
            counting_kwargs["attn_backend"] = attn_backend
        self.counting_attn = Attention(**counting_kwargs)

        # Memory head: single head, Kerdock/DG-coded keys for exact retrieval.
        memory_kwargs = dict(
            num_heads=self.num_heads,
            head_size=self.memory_head_dim,
            scale=1.0,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix="switch.multi.1",
        )
        if attn_backend is not None:
            memory_kwargs["attn_backend"] = attn_backend
        self.memory_attn = Attention(**memory_kwargs)

        # Token-exchange LUT (control id -> substitute id), None if unconfigured.
        # persistent=True for the same reason as ``codebook`` above: a
        # non-persistent buffer is zeroed by checkpoint loading. This LUT uses -1
        # as the "not a control token" sentinel, so an all-zero LUT would rewrite
        # EVERY token id to 0 in apply_token_exchange.
        lut = build_substitute_lut(hf_config) if hf_config is not None else None
        if lut is not None:
            self.register_buffer("control_to_substitute_lut", lut, persistent=True)
        else:
            self.control_to_substitute_lut = None

    @property
    def num_cache_layers(self) -> int:
        """KV-cache slots used by this switch: counting slot + memory slot."""
        return 2

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token adapter indices and rewrite control tokens.

        Adapter selection reads the ORIGINAL ``input_ids``. Token-exchange is
        applied at the very end: ``modified_input_ids = apply_token_exchange(
        lut, input_ids)``.

        Args:
            input_ids: ``[total_tokens]`` flattened token ids (vLLM scheduler).
            adapter_token_ids: ``[num_adapters (+1)]`` activating control token
                ids. ``adapter_token_ids[i]`` fires adapter ``i`` (index 0 is
                the base/no-adapter slot when present).
            positions: ``[total_tokens]`` REAL per-request token positions, as
                supplied by the vLLM model forward. Each request's first token
                must carry position 0 -- that is the ``1/(1+n)`` counting anchor.
                REQUIRED: passing ``None`` raises, because synthesizing
                ``arange(total_tokens)`` over a flattened batch would anchor only
                the first request and silently mis-route the rest.

        Returns:
            ``(adapter_indices, modified_input_ids)`` — both ``[total_tokens]``.
            ``adapter_indices``: 0 = base, 1+ = adapters.
            ``modified_input_ids``: control ids rewritten to substitute ids.
        """
        total_tokens = input_ids.shape[0]
        device = input_ids.device
        # The KV-cache dtype, not a choice: both heads are paged-KV vllm.Attention
        # modules, so their Q/K/V must match the cache. The HF twin forces fp32 for
        # the counting head and is exact past n=4095; a bf16 cache quantizes the
        # 1/(1+n) signal and inverts exactly only to n=188 (189 aliases). See
        # docs/MULTISWITCH_EXPLAINED.html section 8; the bound itself is pinned by
        # tests/unit/test_counting_ceiling.py and enforced (client-side only) by
        # conversation.MAX_RETAINED_CONTROL_TOKENS.
        dtype = self.dtype

        # Position-0 anchor for the 1/(1+n) counting (replaces a special
        # <|init|> token). Derived from positions; NOT KV-hidden.
        #
        # ``positions`` must be the REAL per-request positions supplied by vLLM.
        # There is deliberately no ``arange(total_tokens)`` fallback: vLLM
        # flattens a batch into one flat tensor, so a fabricated arange places an
        # anchor only in the first request and every later request counts against
        # a missing baseline, recovering a wrong write address and retrieving an
        # arbitrary adapter. That failure is silent -- routing looks plausible and
        # only diverges once a request carries >=3 control tokens -- so a loud
        # error here is much cheaper than the mis-routing it replaces.
        if positions is None:
            raise ValueError(
                "MultiSwitch.forward() requires per-request `positions`: the "
                "1/(1+n) counting head places its anchor at `positions == 0`. "
                "vLLM flattens batches into one [total_tokens] tensor, so a "
                "synthesized arange() would give only the first request an "
                "anchor and silently mis-route every other request. Pass the "
                "`positions` that the model forward already receives."
            )
        is_counting_anchor = positions == 0  # [total_tokens]

        # Vectorized control-token matching against ORIGINAL input_ids.
        matches = input_ids.unsqueeze(1) == adapter_token_ids.unsqueeze(0)  # [T, A]
        is_control_token = matches.any(dim=1)  # [total_tokens]
        # expert_id = argmax + offset (offset selects the layout; see __init__).
        # 0 for non-control tokens.
        expert_ids = torch.where(
            is_control_token,
            matches.long().argmax(dim=1) + self._expert_id_offset,
            torch.zeros_like(input_ids, dtype=torch.long),
        )  # [total_tokens]

        # ==================================================================
        # Step 1: Counting via single-head attention -> write address n.
        # ==================================================================
        _zero = torch.tensor(0.0, dtype=dtype, device=device)
        _one = torch.tensor(1.0, dtype=dtype, device=device)

        # Keys: default masked (-1e9); un-mask (0) at anchor + control tokens
        # so the one-hot query attends only to those. Arithmetic (torch.where),
        # no boolean-index assignment.
        k_count = torch.full(
            (total_tokens, self.num_kv_heads, self.counting_head_dim),
            _NEG_INF,
            device=device,
            dtype=dtype,
        )
        anchor_or_control = is_counting_anchor | is_control_token
        k_count[:, 0, 0] = torch.where(anchor_or_control, _zero, k_count[:, 0, 0])

        # Values: v=1 at the position-0 anchor only (the 1/(1+n) numerator).
        v_count = torch.zeros(
            (total_tokens, self.num_kv_heads, self.counting_head_dim),
            device=device,
            dtype=dtype,
        )
        v_count[:, 0, 0] = torch.where(is_counting_anchor, _one, _zero)

        # Query: one-hot on dim 0 => Q·K = K[0] (0 for attended, -1e9 masked).
        q_count = torch.zeros(
            (total_tokens, self.num_heads, self.counting_head_dim),
            device=device,
            dtype=dtype,
        )
        q_count[:, 0, 0] = _one

        count_output = self.counting_attn(q_count, k_count, v_count)
        count_output = count_output.reshape(
            total_tokens, self.num_heads, self.counting_head_dim
        )
        counting_signal = count_output[:, 0, 0]  # [total_tokens] = 1/(1+n)

        # n = round(1/signal - 1), clamped to codebook capacity.
        write_addresses = recover_count_from_signal(
            counting_signal, capacity=self.capacity
        )

        # Debug-only side write. Evaluated once, at trace time, where
        # is_compiling() is True -- so the body is not in the compiled graph and
        # these attributes never appear in a default-configured server. Readers
        # (tests/integration/*_worker.py) must set enforce_eager=True. The same
        # mechanism is why a host-side budget check cannot live in this method.
        if not torch.compiler.is_compiling():
            self._debug_write_addresses = write_addresses
            self._debug_counting_signal = counting_signal

        # ==================================================================
        # Step 2: Coded memory -> expert id.
        # ==================================================================
        # Look up the codeword for each token's address (compile-safe gather).
        all_code_vectors = self.codebook[write_addresses].to(dtype)  # [T, memory_dim]
        write_mask = is_control_token.unsqueeze(-1).to(dtype)  # [T, 1]

        # Keys: code(n) * memory_gain at control tokens, zero elsewhere
        # (arithmetic masking). Value: expert_id at control tokens.
        k_memory = torch.zeros(
            (total_tokens, self.num_kv_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        k_memory[:, 0, : self.memory_dim] = (
            all_code_vectors * self.memory_gain * write_mask
        )

        v_memory = torch.zeros(
            (total_tokens, self.num_kv_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        v_memory[:, 0, 0] = expert_ids.to(dtype) * is_control_token.to(dtype)

        # Query: code(n) for every token (reads back the value at address n).
        q_memory = torch.zeros(
            (total_tokens, self.num_heads, self.memory_head_dim),
            device=device,
            dtype=dtype,
        )
        q_memory[:, 0, : self.memory_dim] = all_code_vectors

        memory_output = self.memory_attn(q_memory, k_memory, v_memory)
        memory_output = memory_output.reshape(
            total_tokens, self.num_heads, self.memory_head_dim
        )

        # ==================================================================
        # Step 3: Extract, round, clamp adapter indices.
        # ==================================================================
        adapter_indices = memory_output[:, 0, 0]  # [total_tokens]
        adapter_indices = torch.round(adapter_indices).long()
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        # Token-exchange rewrite (branch-free; runs every step under compile).
        modified_input_ids = apply_token_exchange(
            self.control_to_substitute_lut, input_ids
        )

        return adapter_indices, modified_input_ids
