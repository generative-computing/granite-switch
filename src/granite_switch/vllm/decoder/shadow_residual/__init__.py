# SPDX-License-Identifier: Apache-2.0
"""Shadow-Residual decoder tier (SWITCH kernel).

Serves Granite-Switch Shadow-Residual models: two per-token streams sharing the
base weights, stacked on the token dim as ``[2M, H]`` through the SWITCH fused
GEMM + expand; base-only K/V; doubled-Q attention; a shrink-only W_cross shunt
(``base -> adapter``); per-token output select.

SR is an *adaptation* of the shared host model
(``granite_switch.vllm.granite_switch_model:GraniteSwitchForCausalLM``): its
divergence lives in ``granite_switch.vllm.decoder.interface.SRDecoderInterface``
(stream doubling, terminal merge, fuse-at-load, kernel meta) plus the decoder tier
here. See the module docstrings in :mod:`.decoder`, :mod:`.kernel_meta`, and
:mod:`.wcross_shunt`.

Registration of the ``SRSwitchForCausalLM`` / ``ShadowResidualForCausalLM`` arch
strings (all pointing at the shared class) is handled by
``granite_switch.vllm.register()`` — there is no separate SR registration.
"""

__all__: list[str] = []
