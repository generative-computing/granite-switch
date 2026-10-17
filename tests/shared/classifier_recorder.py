# SPDX-License-Identifier: Apache-2.0
"""Importable model-RPC callbacks for recording sampled classifier rows.

PP's multiprocessing executor uses standard pickle, so callbacks must live at
module scope. Each worker owns its records; the driver reads them through RPC.
"""


def install_classifier_recorder(model):
    """Record after compute_logits, outside the compiled/captured forward."""
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    records = []
    target._classifier_test_records = records
    original = target.compute_logits
    config = target.config
    head = target.model.classifier_head

    def compute_logits(hidden_states):
        logits = original(hidden_states)
        slots = hidden_states[:, -1].round().long()
        for row in (slots > 0).nonzero(as_tuple=True)[0].tolist():
            slot = int(slots[row])
            labels = config.classifier_label_token_ids[slot - 1]
            n = len(labels)
            h = config.hidden_size
            record = {
                "slot": slot,
                "packed_scores": hidden_states[row, h : h + n].float().cpu(),
                "rewritten_label_logits": logits[row, labels].float().cpu(),
            }
            # Only a final-layer head reads the packed hidden-state prefix.
            if config.classifier_read_layers[slot - 1] is None:
                record["served_hidden"] = hidden_states[row, :h].float().cpu()
                record["loaded_weight"] = head.weight[slot - 1, :n].float().cpu()
                record["loaded_bias"] = head.bias[slot - 1, :n].float().cpu()
            records.append(record)
        return logits

    target.compute_logits = compute_logits


def read_classifier_records(model):
    """Return and clear this worker's rows; non-sampling PP ranks return []."""
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    records = target._classifier_test_records
    result = [
        {k: v if k == "slot" else v.tolist() for k, v in r.items()} for r in records
    ]
    records.clear()
    return result
