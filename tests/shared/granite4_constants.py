# SPDX-License-Identifier: Apache-2.0
"""Derived production constants for Granite 4.x / 4.1 test parametrizations.

Single source of truth so tests don't hardcode values that change per model
family. All values derive from GRANITE4_FULLSIZE in granite4_equivalence.py
(or GraniteSwitchConfig defaults) — adding a new Granite variant there
automatically updates this file's exports.
"""

from granite_switch.config import GraniteSwitchConfig

# TODO: consider moving the GRANITE4_FULLSIZE definition into this file —
# it's the natural owner of Granite 4 production constants, and other tests
# that need production geometry would find it here rather than in the
# equivalence-test helper.
# TODO: if possible, derive these values directly from the HuggingFace model
# configs (e.g. via AutoConfig.from_pretrained("ibm-granite/granite-4.0-1b"))
# instead of maintaining a static dict. Would eliminate drift when Granite
# releases new variants or updates existing ones.
from tests.shared.granite4_equivalence import GRANITE4_FULLSIZE

DEFAULT_CONTROL_TOKEN_GAIN = (
    GraniteSwitchConfig().control_token_gain
)  # derived from GraniteSwitchConfig default (15.0)

PRODUCTION_ATTENTION_MULTIPLIERS = sorted(
    {cfg["attention_multiplier"] for cfg in GRANITE4_FULLSIZE.values()}
)  # [0.0078125, 0.015625]

MAX_POSITION_EMBEDDINGS = GRANITE4_FULLSIZE["4.0-1b"][
    "max_position_embeddings"
]  # 131072
