# SPDX-License-Identifier: Apache-2.0
"""vLLM config hook for Granite Switch checkpoints (registered by ``register``).

vLLM calls an architecture's ``verify_and_update_config`` while it builds the
engine config, before it chooses the scheduler. For a checkpoint with a loop
slot this selects the scheduler that books a loop request's one-step answer
(``granite_switch.vllm.loop.make_loop_scheduler``). Any other checkpoint is
left untouched. Kept apart from ``.loop`` so registering the plugin imports no
model-runner internals.
"""

from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

logger = init_logger(__name__)

LOOP_SCHEDULER = "granite_switch.vllm.loop.make_loop_scheduler"


class GraniteSwitchConfigHook(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if "loop" not in (getattr(hf_config, "adapter_kinds", None) or []):
            return
        scheduler_config = vllm_config.scheduler_config
        if scheduler_config.scheduler_cls is None:
            scheduler_config.scheduler_cls = LOOP_SCHEDULER
        elif scheduler_config.scheduler_cls != LOOP_SCHEDULER:
            logger.warning(
                "Scheduler %s is set: loop requests may read the prefix cache "
                "(skipping the grid read), and the async scheduler cannot book "
                "their one-step answers.",
                scheduler_config.scheduler_cls,
            )
