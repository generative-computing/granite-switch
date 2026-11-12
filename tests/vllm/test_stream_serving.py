# SPDX-License-Identifier: Apache-2.0
"""A checkpoint with stream entries, served by a real vLLM engine.

The other engine tests serve probe checkpoints in the pre-streams layout, with
every head on the base model. This one serves the stream API's layout through
the engine, eager and under FULL CUDA graphs, and on two GPUs:

* compact LoRA rows: the heads take no LoRA rows;
* a head on an adapter (option B): ``tox_sr`` runs the Shadow Residual adapter
  at its marker, so it must read a different state than ``tox`` on the base;
* a plugin head type, found only through its ``granite_switch.streams`` entry
  point, in merge mode: the example ``mlp_head``.

Identity probes expose the state they read: their packed scores must be the
served hidden state, and their rewritten label logits the scores. Every head is
compared with the HF fp32 reference against the gap a routing bug would leave:
``tox`` and ``tox_sr`` differ only by the adapter at the marker, and ``nudge``
merged differs from its LM row only by the head.

Requires GPU + vLLM; the two-GPU cases need two GPUs.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vLLM stream serving test requires a GPU"
)

WORKER = os.path.join(os.path.dirname(__file__), "_stream_serving_worker.py")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(WORKER)))
HEADS = ("tox", "tox_sr", "nudge")


@pytest.fixture(scope="module")
def plugin_site(tmp_path_factory):
    """A dist-info that exposes the example plugin only as an entry point."""
    site = tmp_path_factory.mktemp("plugin_site")
    info = site / "gs_stream_test_plugins-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: gs-stream-test-plugins\nVersion: 0.0.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[granite_switch.streams]\n"
        "mlp_head = tests.shared.stream_plugins.mlp_head:MLPHeadType\n"
    )
    return str(site)


def _env(plugin_site):
    env = dict(os.environ)
    env["VLLM_USE_V1"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Extend, don't replace, the caller's PYTHONPATH: it may name the source
    # tree under test.
    paths = [_REPO, plugin_site, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    return env


def _run(argv, plugin_site, out_path, timeout=1800):
    proc = subprocess.run(
        [sys.executable, WORKER, *argv, out_path],
        env=_env(plugin_site),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for stream in ("stdout", "stderr"):
        with open(out_path.removesuffix(".json") + f".{stream}.log", "w") as log:
            log.write(getattr(proc, stream))
    label = " ".join(argv[:1] + argv[2:])
    if not os.path.exists(out_path):
        pytest.fail(
            f"{label}: worker wrote no result\nstdout={proc.stdout[-1500:]}\n"
            f"stderr={proc.stderr[-1500:]}",
            pytrace=False,
        )
    res = json.load(open(out_path))
    if "fatal" in res:
        pytest.fail(f"{label}: worker crashed:\n{res['fatal'][:2000]}", pytrace=False)
    return res


@pytest.fixture(scope="module")
def composed(tmp_path_factory, plugin_site):
    out = tmp_path_factory.mktemp("stream_ckpt")
    return _run(["compose", str(out)], plugin_site, str(out / "compose.json"))


def _serve(composed, plugin_site, tmp_path, mode, tp=1, pp=1):
    return _run(
        ["serve", composed["ckpt"], mode, str(tp), str(pp)],
        plugin_site,
        str(tmp_path / f"serve_{mode}_tp{tp}_pp{pp}.json"),
    )


def _dist(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _check(served, composed):
    ref = composed["reference"]
    # The gaps a routing bug would leave: the adapter missing at tox_sr's
    # marker reads tox's state; a missing merge leaves the LM row.
    gap_pair = _dist(ref["tox_sr"], ref["tox"])
    gap_merge = _dist(ref["nudge"], ref["nudge_lm"])
    assert gap_pair > 0.1 and gap_merge > 0.1, (gap_pair, gap_merge)

    seq = served["sequential"]
    by_slot = {}
    for slot, name in enumerate(HEADS, start=2):
        first, replay = seq[name]
        assert [r["slot"] for r in first["records"]] == [slot]
        assert replay["records"] == first["records"]
        record, output = first["records"][0], first["outputs"][0]
        by_slot[slot] = record
        if name in ("tox", "tox_sr"):
            # Identity probe: the scores are the state it read, exactly.
            assert record["packed_scores"] == record["served_hidden"]
            assert record["rewritten_label_logits"] == record["packed_scores"]
            scores = record["packed_scores"]
            assert output["token_id"] == 100 + scores.index(max(scores))
            assert not output["has_non_label_support"]
            err = _dist(scores, ref[name])
            assert err < gap_pair / 3, (name, err, gap_pair)
        else:
            err = _dist(record["rewritten_label_logits"], ref["nudge"])
            assert err < gap_merge / 3, (err, gap_merge)
            assert output["has_non_label_support"]  # merge keeps the LM row

    # The SR request is an ordinary adapter request: no head row.
    assert seq["sr"][0]["records"] == []
    assert seq["sr"][0]["outputs"][0]["has_non_label_support"]

    # One mixed batch: the same three head rows, and the SR request.
    mixed = served["mixed"]
    assert sorted(r["slot"] for r in mixed["records"]) == [2, 3, 4]
    for record in mixed["records"]:
        alone = by_slot[record["slot"]]
        gap = gap_merge if record["slot"] == 4 else gap_pair
        assert (
            _dist(record["rewritten_label_logits"], alone["rewritten_label_logits"])
            < gap / 3
        )
    assert [o["has_non_label_support"] for o in mixed["outputs"]] == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize("mode", ["eager", "full"])
def test_streams_checkpoint_serves_every_head(composed, plugin_site, tmp_path, mode):
    _check(_serve(composed, plugin_site, tmp_path, mode), composed)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)], ids=["tp2", "pp2"])
@pytest.mark.parametrize("mode", ["eager", "full"])
def test_streams_checkpoint_on_two_gpus(composed, plugin_site, tmp_path, mode, tp, pp):
    """Heads on two GPUs: every rank holds the head banks, and the scores
    cross pipeline stages, so each run matches the HF reference as one GPU does."""
    _check(_serve(composed, plugin_site, tmp_path, mode, tp=tp, pp=pp), composed)
