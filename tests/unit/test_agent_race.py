# SPDX-License-Identifier: Apache-2.0
"""The agent race (benchmarks/adapter_eval/race/), on CPU.

The switch benchmark's own workload generator and agent loop run against a fake
engine that streams tokens without a model, so the slots, the recording and the
page's data are checked end to end; only the engine is not real.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from benchmarks.adapter_eval.race import build_page, lane, run_race

REPO_ROOT = Path(__file__).resolve().parents[2]
# The race's workload, as the switching block sets it (adapters.yaml).
WORKLOAD = {
    "concurrency": 8,
    "span": 32,
    "decode_tokens": 512,
    "adapters": 64,
    "rank": 32,
    "min_agents": 16,
    "warmup_runs": 1,
}


class FakeEngine:
    """Streams ``max_tokens`` tokens per request, with a fixed share of the prompt
    reported as cached, and counts its prefix-cache resets."""

    def __init__(self):
        self.requests: list[tuple[str, int, str | None]] = []
        self.active = 0
        self.max_active = 0
        self.resets = 0

    def generate(self, prompt, params, request_id, lora_request=None):
        ids = prompt["prompt_token_ids"]
        name = getattr(lora_request, "lora_name", None)
        self.requests.append((request_id, len(ids), name))
        return self._stream(params.max_tokens, len(ids))

    async def _stream(self, n, prompt_len):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)  # the prefill
            out = types.SimpleNamespace(
                outputs=[types.SimpleNamespace(token_ids=[])],
                num_cached_tokens=prompt_len // 4,
            )
            for _ in range(n):
                await asyncio.sleep(0)
                out.outputs[0].token_ids.append(7)
                yield out
        finally:
            self.active -= 1

    def reset_prefix_cache(self):
        self.resets += 1
        return True


@pytest.fixture
def stub_vllm(monkeypatch):
    """The two names the agent loop imports from vLLM."""
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda **kw: dict(kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


@pytest.fixture(scope="module")
def fleet(tmp_path_factory) -> list[dict]:
    """The race's workload from the switch benchmark's generator: 16 agents."""
    out = tmp_path_factory.mktemp("grid") / "grid.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "gen_switching_grid.py"),
            "--concurrency",
            "8",
            "--span",
            "32",
            "--decode",
            "512",
            "--adapters",
            "64",
            "--min-agents",
            "16",
            "--out",
            str(out),
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )
    return lane.load_fleet(out)


def agent_args(**kw):
    defaults = dict(
        vocab_lo=1000,
        vocab_hi=30000,
        deciles=10,
        sleep_dist="uniform",
        sr_control_id=100352,
        lane="lane",
        concurrency=8,
        warmup_runs=1,
        metrics_every=0.05,
    )
    return types.SimpleNamespace(**{**defaults, **kw})


def race(engine, fleet, arm):
    def adapter_of(ids):
        return "ad0" if arm == "shadow-residual" else None

    def lora_for(name):
        return types.SimpleNamespace(lora_name=name)

    args = agent_args(lane="gs-sr-vllm" if arm == "shadow-residual" else "lora-vllm")
    return asyncio.run(
        lane.race_lane(engine, "v1", fleet, arm, args, lora_for, adapter_of)
    )


@pytest.mark.parametrize("arm", ["native-lora", "shadow-residual"])
def test_units_are_the_requests_the_agent_loop_issues(stub_vllm, fleet, arm):
    engine = FakeEngine()
    asyncio.run(
        lane.run_slots(engine, "v1", fleet, arm, agent_args(), lambda n: None, 8)
    )
    for agent in fleet:
        issued = [
            int(rid.rsplit("-u", 1)[1])
            for rid, _, _ in engine.requests
            if f"-a{agent['agent_id']}-" in rid
        ]
        gen = [i for i, u in enumerate(lane.units_of(agent, arm)) if u["kind"] == "gen"]
        assert issued == gen
    if arm == "native-lora":  # a request per segment, 16 switches each
        assert len(engine.requests) == sum(a["n_segments"] for a in fleet)


def test_slots_start_the_next_agent_when_one_finishes(stub_vllm, fleet):
    report = race(FakeEngine(), fleet, "native-lora")
    events = report["events"]
    running, most = 0, 0
    for e in events:
        if e["ev"] == "conv_start":
            running += 1
        elif e["ev"] == "conv_done":
            running -= 1
        most = max(most, running)
    assert most == 8
    starts = {e["conv"]: e["t"] for e in events if e["ev"] == "conv_start"}
    dones = sorted(e["t"] for e in events if e["ev"] == "conv_done")
    assert sorted(starts) == list(range(16)) and len(dones) == 16
    # The first eight start together; each later one when a slot frees.
    assert max(starts[k] for k in range(8)) < dones[0]
    for k in range(8, 16):
        assert starts[k] >= dones[k - 8]


def test_the_recording_names_every_phase(stub_vllm, fleet):
    engine = FakeEngine()
    report = race(engine, fleet, "native-lora")
    assert engine.resets == 2  # before the warm-up run and before the timed one
    assert len(report["warmup_wall_s"]) == 1
    steps = [e["step"] for e in report["events"] if e["ev"] == "step_start"]
    assert {"prefill", "decode", "rtt"} <= set(steps)
    assert any("tool" in s for s in steps)  # the fleet's tool calls
    agent0 = [e for e in report["events"] if e.get("conv") == 0]
    first = next(e for e in agent0 if e["ev"] == "step_start")
    assert first["step"] == "prefill" and first["adapter"].startswith("ad")
    progress = [e["tokens"] for e in agent0 if e["ev"] == "progress"]
    assert progress == sorted(progress) and progress[-1] == 512
    done = next(e for e in agent0 if e["ev"] == "conv_done")
    row = next(r for r in report["rows"] if r["agent_id"] == fleet[0]["agent_id"])
    assert done["wall_time"] == pytest.approx(row["elapsed_s"], abs=1e-3)
    # The engine's accounting: a quarter of every prompt cached, by the fake.
    assert report["prompt_tokens"] == sum(n for _, n, _ in engine.requests[-256:])
    assert report["recomputed_tokens"] == sum(
        n - n // 4 for _, n, _ in engine.requests[-256:]
    )
    # Each agent's first request is its first prefill; every later one re-sends
    # context, and what the cache did not hold of it is the re-prefill.
    timed, firsts = engine.requests[-256:], set()
    first = later = resent = 0
    for rid, n, _ in timed:
        agent = rid.rsplit("-u", 1)[0]
        if agent in firsts:
            later, resent = later + n - n // 4, resent + n
        else:
            firsts.add(agent)
            first += n - n // 4
    assert len(firsts) == 16
    assert report["first_prefill_tokens"] == first
    assert report["reprefilled_tokens"] == later
    assert report["resent_tokens"] == resent
    samples = [e["m"] for e in report["events"] if e["ev"] == "metrics"]
    assert samples and max(m["running"] for m in samples) <= 8
    assert samples[-1]["done"] == 16 and samples[-1]["generated"] == 16 * 512
    assert samples[-1]["reprefilled"] == later
    assert samples[-1]["context_hit"] == pytest.approx(
        report["resent_cached_tokens"] / resent, abs=1e-4
    )


def test_sr_runs_one_request_between_tool_calls(stub_vllm, fleet):
    report = race(FakeEngine(), fleet, "shadow-residual")
    steps = [e for e in report["events"] if e["ev"] == "step_start"]
    assert "rtt" not in {e["step"] for e in steps}  # no switch, no round trip
    assert {e["adapter"] for e in steps if e["step"] == "prefill"} == {"ad0"}
    per_agent = len([e for e in steps if e["step"] == "prefill"]) / len(fleet)
    tools = sum(a["n_tool_nodes"] for a in fleet) / len(fleet)
    assert per_agent <= tools + 1


def test_race_record_and_its_block(stub_vllm, fleet):
    lanes = [
        race(FakeEngine(), fleet, arm) for arm in ("shadow-residual", "native-lora")
    ]
    spec = types.SimpleNamespace(
        model_id="granite-4.1-3b", base_model="ibm-granite/granite-4.1-3b"
    )
    record = run_race.race_record(
        spec, WORKLOAD, {"sha": "b7e5" * 10, "subject": "s", "date": "d"}, lanes
    )
    assert record["servers"] == ["gs-sr-vllm", "lora-vllm"]
    assert record["race_wall"] == max(lane["wall_s"] for lane in lanes)
    times = [e["t"] for e in record["events"]]
    assert times == sorted(times)
    meta = record["metadata"]
    assert (meta["runs"], meta["concurrency"]) == (16, 8)
    assert meta["workload"]["decode_tokens"] == 512
    assert set(meta["lanes"]) == {"gs-sr-vllm", "lora-vllm"}
    # Through the log and back, in short lines a pod log carries whole.
    block = run_race.encode_block(record)
    assert max(len(line) for line in block.splitlines()) <= 76 + 10
    log = f"[pod] start\n{block}\n[submit] done\n"
    assert run_race.decode_block(log) == json.loads(json.dumps(record))
    assert build_page.private_strings(record) == []


def test_lane_commands_take_the_lane_flags(tmp_path):
    for name, arm, _ in run_race.LANES:
        cmd = run_race.lane_command(
            name, arm, tmp_path / "g.jsonl", tmp_path / "o.json", WORKLOAD,
            base_model="ibm-granite/granite-4.1-3b", lora_root=tmp_path / "fleet",
            checkpoint=tmp_path / "ckpt",
        )  # fmt: skip
        assert cmd[:2] == ["-m", run_race.LANE_MODULE]
        args = lane.parser().parse_args(cmd[2:])
        assert (args.lane, args.arm, args.concurrency, args.warmup_runs) == (
            name, arm, 8, 1,
        )  # fmt: skip
    gs, native = (dict(zip(c[2::2], c[3::2], strict=False)) for c in (
        run_race.lane_command(*run_race.LANES[0][:2], tmp_path, tmp_path, WORKLOAD, checkpoint=tmp_path),
        run_race.lane_command(*run_race.LANES[1][:2], tmp_path, tmp_path, WORKLOAD, base_model="b", lora_root=tmp_path),
    ))  # fmt: skip
    assert "--sr-checkpoint" in gs and "--lora-root" not in gs
    assert native["--lora-rank"] == "32" and "--sr-checkpoint" not in native


def test_sr_adapters_are_named_as_the_fleet_names_them(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"adapter_names": ["adapter_00", "adapter_17"], "adapter_token_ids": [5, 6]}
        )
    )
    assert lane.sr_adapters(tmp_path) == {5: "ad0", 6: "ad17"}


def test_build_page_embeds_the_race(tmp_path):
    # The page, emptied of the race it holds.
    page = build_page.embed((REPO_ROOT / "docs" / "agent_race.html").read_text(), None)
    assert "const RACE_EVENTS_EMBEDDED = null; // <<RACE_EVENTS>>" in page
    record = {"metadata": {"lanes": {"gs-sr-vllm": {"wall_s": 1.0}}}, "events": []}
    log = tmp_path / "pod.log"
    log.write_text("noise\n" + run_race.encode_block(record) + "\n")
    out = tmp_path / "agent_race.html"
    out.write_text(page)
    assert build_page.main([str(log), "--page", str(out)]) == 0
    embedded = out.read_text()
    line = next(x for x in embedded.splitlines() if "<<RACE_EVENTS>>" in x)
    assert json.loads(line.split(" = ", 1)[1].rsplit(";", 1)[0]) == record
    # Built again, the data line is replaced, not added.
    assert build_page.main([str(log), "--page", str(out)]) == 0
    assert out.read_text().count("<<RACE_EVENTS>>") == 1
    # A path from the run's storage never reaches the page.
    record["metadata"]["checkpoint"] = "/mnt/storage/run/race_sr"
    log.write_text(run_race.encode_block(record))
    with pytest.raises(SystemExit, match="absolute paths"):
        build_page.main([str(log), "--page", str(out)])
