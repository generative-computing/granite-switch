# SPDX-License-Identifier: Apache-2.0
"""Fixed-point loop head for Granite Switch (HF backend).

A loop slot is a sibling of a probe slot: it shares the adapter index, the
control token (placed last, like a probe marker) and the read of one layer of
the frozen base model, and it rewrites logits. Where a probe scores a few labels
at its marker, a loop head runs a small looped transformer over a fixed-size
grid of positions before the marker and writes one score per output token at
each of them.

The looped block follows the Fixed-Point Reasoning Model (FPRM; Movahedi et al.,
"Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers", 2026):

* the state ``z`` is updated by ``z <- z + eta * (f(z; x) - z)`` until the
  relative change ``||z - f(z)||_inf / ||f(z)||_inf`` drops below ``fp_thresh``;
* ``f`` is a short stack of pre-norm transformer layers with residual scaling,
  ``z <- alpha_1 * z + beta_1 * sublayer(norm(z))``, after an input injection
  ``z <- alpha_2 * z + beta_2 * x`` and a depthwise convolution over the grid;
* ``beta_1`` and ``beta_2`` are tied to the learnable per-channel ``alpha_1`` and
  ``alpha_2`` so the iterates stay bounded;
* the step size ``eta`` decays when the residual stops improving.

Here ``x`` is the base model's hidden state at the grid positions, projected
down to the head's width, rather than a learned token embedding.

The head only reads: it never writes back into the base model, so the base
model's K/V and every non-grid logit row are untouched.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Fixed-point solve stops when the step size has decayed below this.
MIN_STEPSIZE = 1e-3


def trunc_normal_(tensor: torch.Tensor, std: float, generator=None) -> torch.Tensor:
    """Truncated normal at +-2 sigma whose standard deviation is exactly ``std``.

    FPRM's initializer (a port of JAX's): ``nn.init.trunc_normal_`` keeps the
    pre-truncation std, so its samples come out about 12% narrower.
    """
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        lo, hi = -2.0, 2.0
        a, b = math.erf(lo / math.sqrt(2)), math.erf(hi / math.sqrt(2))
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_lo, pdf_hi = c * math.exp(-0.5 * lo**2), c * math.exp(-0.5 * hi**2)
        comp_std = std / math.sqrt(
            1 - (hi * pdf_hi - lo * pdf_lo) / z - ((pdf_hi - pdf_lo) / z) ** 2
        )
        tensor.uniform_(a, b, generator=generator)
        tensor.erfinv_().mul_(math.sqrt(2) * comp_std)
        return tensor.clip_(lo * comp_std, hi * comp_std)


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS norm in fp32, as FPRM uses before every sub-layer."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(seq_len: int, head_dim: int, theta: float, device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class _LoopAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE over the grid."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        b, n, w = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, N, d]
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(o.transpose(1, 2).reshape(b, n, w))


class _LoopMLP(nn.Module):
    """SwiGLU MLP; the hidden width is rounded up to a multiple of 256."""

    def __init__(self, width: int, expansion: float):
        super().__init__()
        inner = round(expansion * width * 2 / 3)
        inner = -(inner // -256) * 256
        self.gate_up = nn.Linear(width, 2 * inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class _LoopLayer(nn.Module):
    """One pre-norm layer with residual scaling on both sub-layers."""

    def __init__(self, width: int, num_heads: int, expansion: float, eps: float):
        super().__init__()
        self.attn = _LoopAttention(width, num_heads)
        self.mlp = _LoopMLP(width, expansion)
        self.eps = eps

    def forward(self, z, alpha_1, beta_1, cos, sin):
        z = alpha_1 * z + beta_1 * self.attn(_rms_norm(z, self.eps), cos, sin)
        z = alpha_1 * z + beta_1 * self.mlp(_rms_norm(z, self.eps))
        return z


class LoopHead(nn.Module):
    """One fixed-point loop head: a looped transformer over a grid of positions.

    Args:
        hidden_size: The base model's hidden size (the read state's width).
        width: The looped block's width.
        num_layers: Layers in the looped block (``f`` applies all of them).
        num_heads: Attention heads in the looped block.
        expansion: SwiGLU width multiplier.
        grid_size: Number of grid positions the head solves (e.g. 81).
        num_outputs: Scores per grid position (e.g. 9 digits).
        prefix_len: Learned prefix tokens prepended to the grid.
        conv_kernel_size: Depthwise convolution kernel over a square grid; 0
            disables it. A non-square grid uses a 1D convolution.
        alpha_1_init, alpha_2_init: Initial residual-scaling values in (0, 1).
        rope_theta: RoPE base for the looped block's attention.
        init_std: Standard deviation of the solver's random starting state.
        eps: RMS-norm epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        width: int,
        num_layers: int,
        num_heads: int,
        expansion: float,
        grid_size: int,
        num_outputs: int,
        prefix_len: int = 16,
        conv_kernel_size: int = 3,
        alpha_1_init: float = 0.75,
        alpha_2_init: float = 0.25,
        rope_theta: float = 10000.0,
        init_std: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        # Every submodule initializes through reset_parameters; the switch
        # model's _init_weights leaves modules carrying this flag alone.
        self._owns_init = True
        if width % num_heads != 0:
            raise ValueError(
                f"width {width} must be divisible by num_heads {num_heads}"
            )
        for name, value in (
            ("alpha_1_init", alpha_1_init),
            ("alpha_2_init", alpha_2_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if conv_kernel_size and conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd, got {conv_kernel_size}")

        self.width = width
        self.grid_size = grid_size
        self.num_outputs = num_outputs
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.init_std = init_std
        self.eps = eps

        self.in_proj = nn.Linear(hidden_size, width, bias=False)
        self.prefix = nn.Parameter(torch.zeros(prefix_len, width))
        self.layers = nn.ModuleList(
            _LoopLayer(width, num_heads, expansion, eps) for _ in range(num_layers)
        )
        self.out_proj = nn.Linear(width, num_outputs, bias=False)
        # FPRM's halt head (from TRM): reads the first prefix token and is
        # trained to predict "the current answer is fully correct". Fixed-point
        # halting does not use it; it is part of the training objective.
        self.q_head = nn.Linear(width, 1, bias=True)

        side = math.isqrt(grid_size)
        self.grid_side = side if side * side == grid_size else 0
        self.conv = None
        if conv_kernel_size:
            conv_cls = nn.Conv2d if self.grid_side else nn.Conv1d
            self.conv = conv_cls(
                width,
                width,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=width,
                bias=False,
            )

        # Learnable per-channel scales, kept in (0, 1) through a sigmoid.
        self.alpha_1_logit = nn.Parameter(
            torch.full((width,), math.log(alpha_1_init / (1 - alpha_1_init)))
        )
        self.alpha_2_logit = nn.Parameter(
            torch.full((width,), math.log(alpha_2_init / (1 - alpha_2_init)))
        )
        # FPRM leaves these out of weight decay.
        for param in (self.alpha_1_logit, self.alpha_2_logit):
            param._no_weight_decay = True
        if self.conv is not None:
            for param in self.conv.parameters():
                param._no_weight_decay = True

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset_parameters(self) -> None:
        """FPRM's initialization.

        Linears: truncated LeCun normal (std 1/sqrt(fan_in), exact). The read
        projection plays the role of FPRM's token embedding (std 1/sqrt(width),
        scaled by sqrt(width)), so the injection is unit-scale either way. The
        prefix stands in for FPRM's puzzle embedding, which starts at zero. The
        conv keeps PyTorch's default init, as in FPRM; the halt head starts at
        zero weight and bias -5.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.q_head:
                trunc_normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
        if self.conv is not None:
            self.conv.reset_parameters()
        nn.init.zeros_(self.prefix)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

    # ------------------------------------------------------------------ #
    # The fixed-point map
    # ------------------------------------------------------------------ #
    def scales(self):
        """``(alpha_1, beta_1, alpha_2, beta_2)``, each broadcastable to ``[B, N, W]``."""
        alpha_1 = torch.sigmoid(self.alpha_1_logit).view(1, 1, -1)
        alpha_2 = torch.sigmoid(self.alpha_2_logit).view(1, 1, -1)
        decay = alpha_1.pow(2 * self.num_layers)
        beta_2 = 1 - alpha_2 * decay
        beta_1 = beta_2 * (1 - alpha_1) / (1 - decay + 1e-5)
        return alpha_1, beta_1, alpha_2, beta_2

    def inject(self, read: torch.Tensor) -> torch.Tensor:
        """The input injection ``x``: ``[B, grid, hidden] -> [B, prefix + grid, W]``.

        FPRM injects ``sqrt(width) * E[token]`` with ``E`` initialized at std
        ``1/sqrt(width)``: unit scale at the start, magnitude learned. Here the
        read is RMS-normalized (base hidden states differ in scale from layer to
        layer) and projected by a LeCun-initialized linear, which is unit scale
        at the start with a learned magnitude too. The prefix plays FPRM's
        puzzle embedding (one shared id for Sudoku) and is scaled the same way.
        Positions come from RoPE inside the looped block.
        """
        x = self.in_proj(_rms_norm(read, self.eps).to(self.in_proj.weight.dtype))
        prefix = (self.prefix * math.sqrt(self.width)).to(x.dtype)
        return torch.cat([prefix.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)

    def _convolve(self, z: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return z
        b = z.shape[0]
        head, grid = z[:, : self.prefix_len], z[:, self.prefix_len :]
        if self.grid_side:
            g = grid.transpose(1, 2).reshape(
                b, self.width, self.grid_side, self.grid_side
            )
            grid = self.conv(g).reshape(b, self.width, -1).transpose(1, 2)
        else:
            grid = self.conv(grid.transpose(1, 2)).transpose(1, 2)
        return torch.cat([head, grid], dim=1)

    def f(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One application of the looped block: ``f(z; x)``."""
        alpha_1, beta_1, alpha_2, beta_2 = self.scales()
        head_dim = self.layers[0].attn.head_dim
        cos, sin = _rope_cos_sin(z.shape[1], head_dim, self.rope_theta, z.device)
        z = self._convolve(z)
        z = alpha_2 * z + beta_2 * x
        for layer in self.layers:
            z = layer(z, alpha_1, beta_1, cos, sin)
        return z

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        """Scores per grid position: ``[B, prefix + grid, W] -> [B, grid, outputs]``."""
        return self.out_proj(z[:, self.prefix_len :])

    def halt_logit(self, z: torch.Tensor) -> torch.Tensor:
        """The halt head's logit per row: ``[B, prefix + grid, W] -> [B]``."""
        return self.q_head(z[:, 0]).squeeze(-1).float()

    def init_state(self, x: torch.Tensor, generator=None) -> dict:
        """A fresh solver state for a batch whose injection is ``x``.

        The starting point is random: a truncated normal with std ``init_std``,
        as in FPRM. Pass a seeded ``generator`` for a reproducible solve.
        """
        b = x.shape[0]
        z = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        z = trunc_normal_(z, std=self.init_std, generator=generator).to(x.dtype)
        return {
            "z": z,
            "stepsize": x.new_ones(b, 1, 1, dtype=torch.float32),
            "best": x.new_full((b,), float("inf"), dtype=torch.float32),
            "patience": torch.zeros(b, dtype=torch.long, device=x.device),
            "residual": x.new_full((b,), float("inf"), dtype=torch.float32),
            "iters": torch.zeros(b, dtype=torch.long, device=x.device),
        }

    def step(
        self,
        state: dict,
        x: torch.Tensor,
        decay: float,
        patience: int,
        fp_thresh: float,
        active: torch.Tensor | None = None,
    ) -> dict:
        """One damped fixed-point update, with FPRM's step-size control.

        ``active`` (``[B]`` bool) freezes converged rows: their state, step size
        and counters are left as they are.
        """
        z = state["z"]
        fz = self.f(z, x)
        with torch.no_grad():
            diff = (z - fz).detach().float().abs().amax(dim=-1)
            scale = fz.detach().float().abs().amax(dim=-1) + 1e-8
            residual = (diff / scale).amax(dim=1)  # [B]
        eta = state["stepsize"].to(z.dtype)
        z_new = z + eta * (fz - z)

        improved = residual < state["best"]
        best = torch.where(improved, residual, state["best"])
        wait = torch.where(
            improved, torch.zeros_like(state["patience"]), state["patience"] + 1
        )
        shrink = (wait >= patience) & (residual >= fp_thresh)
        stepsize = torch.where(
            shrink.view(-1, 1, 1), state["stepsize"] * decay, state["stepsize"]
        )
        wait = torch.where(shrink, torch.zeros_like(wait), wait)

        if active is None:
            active = torch.ones_like(improved)
        keep = ~active
        return {
            "z": torch.where(keep.view(-1, 1, 1), z, z_new),
            "stepsize": torch.where(keep.view(-1, 1, 1), state["stepsize"], stepsize),
            "best": torch.where(keep, state["best"], best),
            "patience": torch.where(keep, state["patience"], wait),
            "residual": torch.where(keep, state["residual"], residual),
            "iters": state["iters"] + active.long(),
        }

    @torch.no_grad()
    def solve(
        self,
        read: torch.Tensor,
        max_iter: int,
        fp_thresh: float,
        decay: float,
        patience: int,
        generator=None,
        min_iter: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """Iterate to a fixed point; returns ``(logits [B, grid, outputs], state)``.

        FPRM's evaluation rule: every row keeps iterating until the whole batch
        has converged (the largest residual is below ``fp_thresh``) or every
        row's step size has decayed below :data:`MIN_STEPSIZE`, after at least
        ``min_iter`` and at most ``max_iter`` updates.
        """
        x = self.inject(read)
        state = self.init_state(x, generator)
        for k in range(max_iter):
            if k >= min_iter and (
                bool((state["residual"] < fp_thresh).all())
                or bool((state["stepsize"] < MIN_STEPSIZE).all())
            ):
                break
            state = self.step(state, x, decay, patience, fp_thresh)
        return self.logits(state["z"]), state

    def fixed_start(self, seed: int = 0) -> torch.Tensor:
        """A reproducible starting state ``[prefix + grid, W]`` for :meth:`solve_each`.

        Drawn like :meth:`init_state` draws a random one (truncated normal,
        std ``init_std``), but from a CPU generator seeded with ``seed``, so
        every backend and device gets the same start.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Explicit device and dtype: vLLM builds models under a default CUDA
        # device and a bf16 default dtype, and the start must not follow them.
        z0 = torch.empty(
            self.prefix_len + self.grid_size,
            self.width,
            device="cpu",
            dtype=torch.float32,
        )
        return trunc_normal_(z0, std=self.init_std, generator=gen)

    @torch.no_grad()
    def solve_each(
        self,
        read: torch.Tensor,
        z0: torch.Tensor,
        max_iter: int,
        fp_thresh: float,
        decay: float,
        patience: int,
        min_iter: int = 0,
        check_every: int = 50,
    ) -> tuple[torch.Tensor, dict]:
        """Solve every row on its own: ``(logits [B, grid, outputs], state)``.

        For serving, where a batch is whatever requests were scheduled
        together, so one request's answer must not depend on the others:

        * every row starts from the same state ``z0`` (``[prefix + grid, W]``)
          rather than a random draw, so a request gets the same answer whatever
          it is batched with and however often it is re-run;
        * a row stops updating once it converges (its residual is below
          ``fp_thresh``) or its step size has decayed below
          :data:`MIN_STEPSIZE`, instead of waiting for the whole batch.

        Convergence is checked on the host every ``check_every`` updates; in
        between, converged rows are frozen on the device.
        """
        x = self.inject(read)
        state = self.init_state(x)
        state["z"] = z0.to(device=x.device, dtype=x.dtype).expand_as(x).clone()
        active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for k in range(max_iter):
            if k >= min_iter:
                active = (state["residual"] >= fp_thresh) & (
                    state["stepsize"].view(-1) >= MIN_STEPSIZE
                )
                if k % check_every == 0 and not bool(active.any()):
                    break
            state = self.step(state, x, decay, patience, fp_thresh, active=active)
        return self.logits(state["z"]), state


class SwitchedLoopHeads(nn.Module):
    """The loop heads of a checkpoint, one per loop slot, keyed by adapter index.

    Unlike the probe bank, loop heads are not stacked into one tensor: each head
    is a small transformer, and a request selects one slot. Heads are kept in a
    ``ModuleDict`` whose keys are the 0-based slot numbers as strings, so the
    checkpoint names are ``loop_heads.heads.<slot>.*``.
    """

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleDict()
        for slot, (kind, spec) in enumerate(
            zip(config.adapter_kinds, config.loop_specs or [None] * config.num_adapters)
        ):
            if kind != "loop":
                continue
            self.heads[str(slot)] = LoopHead(
                hidden_size=config.hidden_size,
                width=spec["width"],
                num_layers=spec["num_layers"],
                num_heads=spec["num_heads"],
                expansion=spec["expansion"],
                grid_size=spec["grid_size"],
                num_outputs=len(spec["output_token_ids"]),
                prefix_len=spec["prefix_len"],
                conv_kernel_size=spec["conv_kernel_size"],
                alpha_1_init=spec["alpha_1_init"],
                alpha_2_init=spec["alpha_2_init"],
                rope_theta=spec["rope_theta"],
                init_std=spec["init_std"],
            )
        for head in self.heads.values():
            head.reset_parameters()
            for module in head.modules():
                module._owns_init = True

    def head(self, slot: int) -> LoopHead:
        """The head serving 1-based adapter index ``slot``."""
        return self.heads[str(slot - 1)]
