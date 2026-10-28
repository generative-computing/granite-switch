# SPDX-License-Identifier: Apache-2.0
"""The agent race: granite-switch SR against stock vLLM LoRA, agents switching adapters.

``lane.py`` runs one lane and records its events, ``run_race.py`` runs both on
a pod, and ``build_page.py`` puts the recorded race into its page,
``docs/agent_race.html``, which lives on main. See docs/AGENT_RACE.md.
"""
