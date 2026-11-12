# SPDX-License-Identifier: Apache-2.0
"""Importable model-RPC callbacks for recording sampled probe rows.

PP's multiprocessing executor uses standard pickle, so callbacks must live at
module scope. Each worker owns its records; the driver reads them through RPC.
"""


def install_probe_recorder(model):
    """Record after compute_logits, outside the compiled/captured forward."""
    from granite_switch.streams import stream_specs
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    records = []
    target._probe_test_records = records
    original = target.compute_logits
    config = target.config
    # Only the probe's bank is read back below; a checkpoint may have none.
    head = getattr(target.model, "probe_head", None)
    specs = {s.slot: s for s in stream_specs(config)}

    def compute_logits(hidden_states):
        logits = original(hidden_states)
        if logits is None:  # a tensor-parallel rank that gathers no logits
            return logits
        slots = hidden_states[:, -1].round().long()
        for row in (slots > 0).nonzero(as_tuple=True)[0].tolist():
            spec = specs[int(slots[row])]
            labels = list(spec.token_ids)
            n = len(labels)
            h = config.hidden_size
            record = {
                "slot": spec.slot,
                "packed_scores": hidden_states[row, h : h + n].float().cpu(),
                "rewritten_label_logits": logits[row, labels].float().cpu(),
            }
            # Only a final-layer head reads the packed hidden-state prefix. The
            # slot's head lives at its own row of the probe bank.
            if spec.read_layer is None and spec.type == "probe" and head is not None:
                record["served_hidden"] = hidden_states[row, :h].float().cpu()
                record["loaded_weight"] = head.weight[spec.head_row, :n].float().cpu()
                record["loaded_bias"] = head.bias[spec.head_row, :n].float().cpu()
            records.append(record)
        return logits

    target.compute_logits = compute_logits


def read_probe_records(model):
    """Return and clear this worker's rows; non-sampling PP ranks return []."""
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    records = target._probe_test_records
    result = [
        {k: v if k == "slot" else v.tolist() for k, v in r.items()} for r in records
    ]
    records.clear()
    return result
