# SPDX-License-Identifier: Apache-2.0
"""What ``granite_switch.vllm.register()`` puts in place before any model loads.

In-process on purpose, unlike most of ``tests/vllm/``: ``register()`` only
mutates registries and creates no engine, so it never opens a CUDA context and
needs no subprocess wrapper.
"""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="requires vLLM installed"
)


class TestLegacyLayerTypeAlias:
    """``register()`` must make vLLM accept the post-5.16 layer-type spelling.

    transformers 5.16 renamed the layer type ``attention`` to ``full_attention``
    and rewrites it inside ``PreTrainedConfig.__init__``, so it is now the only
    spelling a written config can carry. vLLM <=0.25 keys
    ``ALL_DECODER_LAYER_TYPES`` on the old name and dies with
    ``KeyError: 'full_attention'`` while building decoder layers.

    The alias lives in ``register()`` because the lookup happens in vLLM's
    spawned engine-core process, which loads plugins itself -- a fixture in the
    test process would not reach it. Upstream added the key in 0.26.0, where this
    becomes a no-op; the assertions hold either way, which is what lets the block
    be deleted on a version bump without touching this test.
    """

    def test_full_attention_resolves_to_the_attention_layer(self):
        from vllm.model_executor.models import granitemoehybrid as gmh

        from granite_switch.vllm import register

        register()

        table = gmh.ALL_DECODER_LAYER_TYPES
        assert "attention" in table, "upstream renamed or removed the legacy key"
        assert "full_attention" in table, (
            "register() did not alias the post-5.16 layer-type spelling; a config "
            "written by transformers >=5.16 will KeyError at engine init"
        )
        assert table["full_attention"] is table["attention"], (
            "full_attention must resolve to the very same layer class, not a copy "
            "-- aliasing is meant to add a name, not a behaviour"
        )

    def test_register_is_re_entrant(self):
        """``register()`` is called once per process by vLLM, but its docstring
        promises re-entrancy and ``setdefault`` is what keeps that true here."""
        from vllm.model_executor.models import granitemoehybrid as gmh

        from granite_switch.vllm import register

        register()
        first = gmh.ALL_DECODER_LAYER_TYPES["full_attention"]
        register()
        assert gmh.ALL_DECODER_LAYER_TYPES["full_attention"] is first
