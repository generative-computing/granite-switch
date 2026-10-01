# SPDX-License-Identifier: Apache-2.0
"""Adapter benchmark harness (``benchmarks/adapter_eval``): CPU-only checks.

Covers what runs without a GPU: the benchmark definition, the results block,
the cache / merge / page rules, every scorer on tiny fixtures, checkpoint
detection, the stage tool, the reference run's driver and helpers, and the
Vela job payload. Checkpoints are fake safetensors files with a header and no
tensor data.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from benchmarks.adapter_eval import (
    common,
    hf_generate,
    prompts,
    publish,
    reference,
    run_benchmark,
    stage,
    staged,
)
from benchmarks.adapter_eval.scorers import (
    ScoreContext,
    ScorerUnavailable,
    get_scorer,
    guardian,
    query_rewrite,
)

SPEC = common.load_spec()
SHA_A = "a" * 40
SHA_B = "b" * 40


# --- fixtures ----------------------------------------------------------------


def write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {
        k: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]}
        for k, s in shapes.items()
    }
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def write_safetensors_data(path: Path, tensors: dict[str, bytes]) -> None:
    """A safetensors file with real (U8) tensor bytes, laid out as saved."""
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(tensors.values()))


def write_base(folder: Path, index: bool = True) -> Path:
    """A fake two-shard base checkpoint with the standard module names.

    With ``index`` only the shard index is written, so a reader that finds
    the names must have read the index.
    """
    names = ["model.embed_tokens.weight", "lm_head.weight"] + [
        f"model.layers.0.{'self_attn' if m in stage.ATTENTION else 'mlp'}.{m}.weight"
        for m in QKVO_MLP
    ]
    shards = {
        "model-00001-of-00002.safetensors": names[:4],
        "model-00002-of-00002.safetensors": names[4:],
    }
    folder.mkdir(parents=True, exist_ok=True)
    if index:
        weight_map = {k: shard for shard, keys in shards.items() for k in keys}
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    else:
        for shard, keys in shards.items():
            write_safetensors(folder / shard, {k: [1] for k in keys})
    return folder


QKVO_MLP = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
QO_MLP = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def make_adapter(
    folder: Path,
    tech: str,
    rank: int = 16,
    cross: int | None = None,
    last_context_token: bool = True,
    invocation: bool = False,
    modules: tuple[str, ...] = ("q_proj",),
    base_model: str = "ibm-granite/granite-4.1-3b",
    flat_mlp: bool = False,
) -> Path:
    """A fake PEFT checkpoint that ``detect_technology`` reads as ``tech``.

    ``flat_mlp`` names MLP weights as the internal trainer does, without
    their ``mlp.`` level.
    """
    config = {
        "base_model_name_or_path": base_model,
        "target_modules": list(modules),
        "lora_alpha": rank,
    }
    shapes = {}
    for module in modules:
        block = (
            "self_attn." if module in stage.ATTENTION else "" if flat_mlp else "mlp."
        )
        name = f"base_model.model.model.layers.0.{block}{module}"
        shapes[f"{name}.lora_A.weight"] = [rank, 64]
        shapes[f"{name}.lora_B.weight"] = [64, rank]
    layer = "base_model.model.model.layers.0.self_attn.q_proj"
    if tech == "alora" or invocation:
        config["alora_invocation_tokens"] = [1, 2, 3]
    if tech == "sr":
        if last_context_token and not invocation:
            config["last_context_token"] = "<|end_of_role|>"
            config["last_context_token_id"] = 3
        c = cross or rank
        shapes[f"{layer}.cross_stream.lora_A.weight"] = [c, 64]
        shapes[f"{layer}.cross_stream.lora_B.weight"] = [64, c]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / staged.CONFIG_FILE).write_text(json.dumps(config))
    write_safetensors(folder / staged.WEIGHTS_FILE, shapes)
    return folder


def write_eval(path: Path, n: int = 2) -> Path:
    staged.write_jsonl(
        path,
        [{"messages": [{"role": "user", "content": "q"}], "ground_truth": "x"}] * n,
    )
    return path


def cells_for(spec, value: float = 0.5) -> dict:
    return {
        i.id: {t.id: {i.headline: value, "n": 10} for t in spec.technologies}
        for i in spec.intrinsics
    }


def results_for(
    sha: str = SHA_A,
    date: str = "2026-09-01T00:00:00+00:00",
    cells: dict | None = None,
    **run,
) -> dict:
    return {
        "model": SPEC.model_id,
        "bench_version": SPEC.bench_version,
        "base_model": SPEC.base_model,
        "commit": {"sha": sha, "date": date, "subject": "subject"},
        "run": {"limit": None, "only": None, **run},
        "cells": cells if cells is not None else cells_for(SPEC),
    }


def page_data(*rows: dict, reference: dict | None = None) -> dict:
    """Page data in the current layout, for the default model."""
    return {
        "rows": list(rows),
        "references": {SPEC.model_id: reference} if reference else {},
    }


# --- benchmark definition ------------------------------------------------------


def test_spec_loads_and_covers_every_compose_group():
    assert {t.id for t in SPEC.technologies} == {"lora", "alora", "sr"}
    assert common.compose_group("lora") == common.compose_group("alora")
    assert common.compose_group("sr") != common.compose_group("lora")
    assert common.adapter_name("answerability", "sr") == "answerability_sr"
    for intrinsic in SPEC.intrinsics:
        get_scorer(intrinsic.scorer)  # every intrinsic has a scorer
        assert intrinsic.max_new_tokens > 0


def test_spec_select():
    assert SPEC.select(None) == SPEC.intrinsics
    assert [i.id for i in SPEC.select(["answerability"])] == ["answerability"]
    with pytest.raises(ValueError, match="unknown intrinsics"):
        SPEC.select(["nope"])


def test_spec_public_has_no_scorer_internals():
    public = SPEC.public()
    assert public["models"][0]["bench_version"] == SPEC.bench_version
    assert public["reference_version"] == SPEC.reference_version
    assert set(public["intrinsics"][0]) == {"id", "name", "headline", "headline_label"}


def test_spec_for_another_model():
    other = SPEC.models[1]
    spec = SPEC.for_model(other.id)
    assert (spec.model_id, spec.base_model, spec.bench_version) == (
        other.id,
        other.name,
        other.bench_version,
    )
    assert spec.intrinsics == SPEC.intrinsics
    assert SPEC.for_model(None) is SPEC
    assert common.load_spec(model=other.id).model_id == other.id
    with pytest.raises(ValueError, match="unknown model"):
        SPEC.for_model("nope")


def test_model_of_reads_old_blocks_by_base_model():
    other = SPEC.models[1]
    assert SPEC.model_of({"model": other.id, "base_model": "ignored"}) == other.id
    assert SPEC.model_of({"base_model": SPEC.base_model}) == SPEC.model_id
    with pytest.raises(ValueError, match="no model"):
        SPEC.model_of({"base_model": "someone/else"})


def test_prompt_settings_per_model_and_technology():
    g41, g42 = SPEC.get_model("granite-4.1-3b"), SPEC.get_model("granite-4.2-3b")
    native = {"documents": "native", "chat_template_kwargs": {}}
    assert all(g41.prompt_for(t) == native for t in ("lora", "alora", "sr", None))
    assert g42.prompt_for("sr") == {
        "documents": "tool_json_after_question",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert g42.prompt_for(None) == g42.prompt_for("sr")  # the base gets the SR form
    assert g42.prompt_for("alora")["documents"] == "tool_text_before_question"


def test_load_spec_rejects_bad_prompt_settings(tmp_path):
    import yaml

    raw = yaml.safe_load(common.SPEC_PATH.read_text())
    raw["models"][1]["prompt"]["documents_by_technology"] = {"alora": "sideways"}
    path = tmp_path / "adapters.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="bad prompt settings"):
        common.load_spec(path)


# --- prompts --------------------------------------------------------------------

CHATML_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "composer"
    / "fixtures"
    / "granite_chatml_template.jinja"
)
QUESTION = [{"role": "user", "content": "Is the sky green?"}]
DOCS = [{"doc_id": 7, "text": "The sky is blue."}, "Grass is green."]


def test_native_documents_go_to_the_template():
    messages, documents = prompts.with_documents(QUESTION, DOCS, "native")
    assert messages == QUESTION
    assert documents == [DOCS[0], {"title": "Context", "text": "Grass is green."}]


def test_tool_json_after_question():
    messages, documents = prompts.with_documents(
        QUESTION, DOCS, "tool_json_after_question"
    )
    assert documents is None
    assert messages[:-1] == QUESTION and messages[-1]["role"] == "tool"
    assert json.loads(messages[-1]["content"]) == [
        {"source": "knowledge_base", "document_id": "7", "content": "The sky is blue."},
        {"source": "knowledge_base", "document_id": "1", "content": "Grass is green."},
    ]


def test_tool_text_before_question():
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        *QUESTION,
    ]
    messages, documents = prompts.with_documents(
        history, DOCS, "tool_text_before_question"
    )
    assert documents is None
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert [m["content"] for m in messages[2:4]] == [
        "The sky is blue.",
        "Grass is green.",
    ]


def test_with_documents_edge_cases():
    no_docs = prompts.with_documents(QUESTION, None, "tool_json_after_question")
    assert no_docs == (QUESTION, None)
    with pytest.raises(ValueError, match="unknown document style"):
        prompts.with_documents(QUESTION, DOCS, "nope")


def test_chat_text_passes_the_template_options():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "prompt"

    row = {"messages": QUESTION, "documents": DOCS, "tools": [{"name": "t"}]}
    assert (
        prompts.chat_text(Tokenizer(), row, "native", {}, adapter_name="a") == "prompt"
    )
    # Granite 4.1: exactly the call the harness made before documents styles.
    assert calls[-1] == (
        QUESTION,
        {
            "tools": row["tools"],
            "documents": prompts.fix_documents(DOCS),
            "add_generation_prompt": True,
            "tokenize": False,
            "adapter_name": "a",
        },
    )
    prompts.chat_text(
        Tokenizer(), row, "tool_json_after_question", {"enable_thinking": False}
    )
    assert calls[-1][1]["enable_thinking"] is False
    assert calls[-1][1]["documents"] is None


def test_chatml_template_gets_the_documents_with_reasoning_off():
    templates = pytest.importorskip("transformers.utils.chat_template_utils")
    template = CHATML_TEMPLATE.read_text()
    g42 = SPEC.get_model("granite-4.2-3b")
    for tech in ("sr", "alora"):
        settings = g42.prompt_for(tech)
        messages, _ = prompts.with_documents(QUESTION, DOCS, settings["documents"])
        rendered, _ = templates.render_jinja_template(
            conversations=[messages],
            chat_template=template,
            add_generation_prompt=True,
            **settings["chat_template_kwargs"],
        )
        text = rendered[0]
        assert "<tool_response>" in text and "The sky is blue." in text, tech
        assert text.endswith("<|im_start|>assistant\n<think></think>"), tech


# --- results block ---------------------------------------------------------------


def test_results_block_round_trip_through_a_noisy_log():
    results = results_for()
    log = "\n".join(
        [
            "[bench] starting",
            common.format_results_block({"stale": True}),
            "[bench] retry",
            common.format_results_block(results),
            "trailing output",
        ]
    )
    assert common.extract_results_block(log) == results  # the last block wins


def test_extract_block_errors():
    with pytest.raises(ValueError, match="no "):
        common.extract_results_block("nothing here")
    with pytest.raises(ValueError, match="not terminated"):
        common.extract_results_block(common.BEGIN_MARKER + '\n{"a": 1}\n')


# --- cache ----------------------------------------------------------------------


def test_cache_hit_needs_a_row_with_every_cell():
    row = results_for()
    assert publish.cache_hit(row, SPEC)
    assert not publish.cache_hit(None, SPEC)

    del row["cells"]["answerability"]["sr"]
    assert not publish.cache_hit(row, SPEC)
    assert publish.missing_cells(row, SPEC) == ["answerability/sr"]


def test_cache_skipped_cells_count_as_done_but_errors_do_not():
    row = results_for()
    row["cells"]["answerability"]["sr"] = common.skipped("adapter not staged")
    assert publish.cache_hit(row, SPEC)
    row["cells"]["answerability"]["sr"] = common.error("compose failed")
    assert not publish.cache_hit(row, SPEC)


def test_cache_miss_on_old_bench_version():
    row = results_for()
    row["bench_version"] = SPEC.bench_version - 1
    assert not publish.cache_hit(row, SPEC)


def test_find_row_by_prefix():
    data = page_data(results_for(SHA_A), results_for("a" * 8 + "c" * 32))
    model = SPEC.model_id
    assert publish.find_row(data, SHA_B, model) is None
    assert publish.find_row(data, SHA_A, model)["commit"]["sha"] == SHA_A
    with pytest.raises(ValueError, match="matches 2 rows"):
        publish.find_row(data, "aaaaaaaa", model)


# --- merge ----------------------------------------------------------------------


def test_merge_replaces_the_commits_row():
    data = page_data(results_for(cells=cells_for(SPEC, 0.1)))
    publish.merge(data, results_for(cells=cells_for(SPEC, 0.9)), SPEC)
    assert len(data["rows"]) == 1
    assert data["rows"][0]["cells"]["answerability"]["lora"]["accuracy"] == 0.9


def test_merge_only_run_updates_just_those_intrinsics():
    data = page_data(results_for(cells=cells_for(SPEC, 0.1)))
    partial = results_for(cells=cells_for(SPEC, 0.9), only=["answerability"])
    row = publish.merge(data, partial, SPEC)
    assert len(data["rows"]) == 1
    assert row["cells"]["answerability"]["lora"]["accuracy"] == 0.9
    assert row["cells"]["guardian_core"]["lora"]["accuracy"] == 0.1
    assert row["updates"] == [partial["run"]]


def test_merge_only_run_on_old_row_replaces_it():
    old = results_for(cells=cells_for(SPEC, 0.1))
    old["bench_version"] = SPEC.bench_version - 1
    data = page_data(old)
    partial = results_for(cells=cells_for(SPEC, 0.9), only=["answerability"])
    publish.merge(data, partial, SPEC)
    assert data["rows"] == [partial]


def test_merge_refuses_limit_runs_and_other_versions():
    with pytest.raises(ValueError, match="--limit"):
        publish.merge(page_data(), results_for(limit=20), SPEC)
    other = results_for()
    other["bench_version"] = SPEC.bench_version + 1
    with pytest.raises(ValueError, match="bench_version"):
        publish.merge(page_data(), other, SPEC)


def for_other_model(block: dict) -> dict:
    """``block`` as the second model's."""
    other = SPEC.models[1]
    block.update(
        model=other.id, base_model=other.name, bench_version=other.bench_version
    )
    return block


def test_load_data_reads_the_one_model_layout(tmp_path):
    row, ref = results_for(), reference_for()
    del row["model"], ref["model"]
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"rows": [row], "reference": ref}))

    data = publish.load_data(path, SPEC)
    assert "reference" not in data
    assert data["references"][SPEC.model_id]["model"] == SPEC.model_id
    assert data["rows"][0]["model"] == SPEC.model_id
    empty = publish.load_data(tmp_path / "missing.json", SPEC)
    assert empty == {"rows": [], "references": {}}


def test_models_keep_their_own_rows_and_references():
    other = SPEC.models[1].id
    data = page_data(results_for(cells=cells_for(SPEC, 0.5)))
    # SPEC is for the first model; each block names its own.
    publish.merge(data, for_other_model(results_for(cells=cells_for(SPEC, 0.9))), SPEC)
    publish.merge_reference(data, for_other_model(reference_for()), SPEC)

    assert len(data["rows"]) == 2
    for model, value in ((SPEC.model_id, 0.5), (other, 0.9)):
        row = publish.find_row(data, SHA_A, model)
        assert row["cells"]["answerability"]["lora"]["accuracy"] == value
    assert set(data["references"]) == {other}


def test_cli_check_takes_a_model(tmp_path, capsys):
    data = tmp_path / "data.json"
    results = tmp_path / "results.json"
    other = SPEC.models[1].id
    results.write_text(json.dumps(for_other_model(results_for())))
    cli = ["--data", str(data)]

    assert publish.main([*cli, "merge", str(results)]) == 0
    assert publish.main([*cli, "check", SHA_A, "--model", other]) == 0
    capsys.readouterr()
    assert publish.main([*cli, "check", SHA_A]) == 1  # the first model has no row
    assert f"no {SPEC.model_id} row" in capsys.readouterr().out
    assert publish.main([*cli, "check-reference", "--model", other]) == 1
    assert f"no reference columns for {other}" in capsys.readouterr().out


def test_save_data_sorts_newest_first(tmp_path):
    data = {
        "rows": [
            results_for(SHA_A, date="2026-01-01T00:00:00+00:00"),
            results_for(SHA_B, date="2026-06-01T00:00:00+00:00"),
        ]
    }
    path = tmp_path / "data.json"
    publish.save_data(path, data, SPEC)
    saved = json.loads(path.read_text())
    assert [r["commit"]["sha"] for r in saved["rows"]] == [SHA_B, SHA_A]
    assert saved["spec"] == SPEC.public()


def test_cli_check_merge_render(tmp_path, capsys):
    data = tmp_path / "data.json"
    results = tmp_path / "results.json"
    log = tmp_path / "pod.log"
    page = tmp_path / "index.html"
    log.write_text("noise\n" + common.format_results_block(results_for()) + "\n")

    assert publish.main(["--data", str(data), "check", SHA_A]) == 1
    assert f"no {SPEC.model_id} row" in capsys.readouterr().out
    assert (
        publish.main(["--data", str(data), "extract", str(log), "--out", str(results)])
        == 0
    )
    assert publish.main(["--data", str(data), "merge", str(results)]) == 0
    assert publish.main(["--data", str(data), "check", SHA_A[:12]]) == 0
    assert publish.main(["--data", str(data), "render", "--out", str(page)]) == 0
    assert SHA_A[:8] in page.read_text()


# --- page -----------------------------------------------------------------------


def test_render_cells_and_escaping():
    cells = cells_for(SPEC, 0.5)
    cells["answerability"] = {
        "lora": {"accuracy": 0.5, "n": 10},
        "alora": {"accuracy": 0.7, "n": 10},
        "sr": common.skipped('adapter "x" not staged'),
    }
    cells["guardian_core"]["lora"] = common.error("generate failed")
    cells["guardian_core"]["alora"] = {"n": 10}  # no headline metric
    del cells["guardian_core"]["sr"]
    row = results_for(cells=cells)
    row["commit"]["subject"] = "<script>alert(1)</script>"
    old = results_for(SHA_B)
    old["bench_version"] = SPEC.bench_version - 1

    page = publish.render(page_data(row, old), SPEC)

    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert 'title="adapter &quot;x&quot; not staged">—</td>' in page
    assert 'class="err g" title="generate failed">error</td>' in page
    assert ">?</td>" in page
    assert 'title="not run">·</td>' in page
    assert 'class="num best"' in page and ">70.0</td>" in page
    assert '<tr class="old">' in page
    for intrinsic in SPEC.intrinsics:
        assert intrinsic.name in page


# --- reference columns ------------------------------------------------------------


def reference_for(value: float = 0.5, cells: dict | None = None, **run) -> dict:
    if cells is None:
        cells = {
            i.id: {c: {i.headline: value, "n": 10} for c in common.REFERENCE_COLUMNS}
            for i in SPEC.intrinsics
        }
    return {
        "model": SPEC.model_id,
        "reference_version": SPEC.reference_version,
        "bench_version": SPEC.bench_version,
        "base_model": SPEC.base_model,
        "cells": cells,
        "run": {
            "limit": None,
            "only": None,
            "finished": "2026-09-30T12:00:00+00:00",
            "torch": "2.10.0",
            "transformers": "5.8.1",
            "peft": "0.19.1",
            **run,
        },
    }


def test_reference_hit_needs_current_versions_and_every_cell():
    ref = reference_for()
    assert publish.reference_hit(ref, SPEC)
    assert not publish.reference_hit(None, SPEC)

    ref["cells"]["answerability"]["base"] = common.error("generation failed")
    del ref["cells"]["guardian_core"]["sr"]
    assert publish.reference_missing(ref, SPEC) == [
        "answerability/base",
        "guardian_core/sr",
    ]
    assert not publish.reference_hit(ref, SPEC)

    for key in ("bench_version", "reference_version"):
        old = reference_for()
        old[key] -= 1
        assert not publish.reference_current(old, SPEC)
        assert not publish.reference_hit(old, SPEC)


def test_only_arg_runs_exactly_the_given_cells():
    keys = [f"answerability/{c}" for c in common.REFERENCE_COLUMNS]
    keys += ["guardian_core/sr", "guardian_core/base"]
    only = publish.only_arg(keys)
    assert only == "answerability,guardian_core/sr,guardian_core/base"
    assert reference.parse_only(only, SPEC) == {
        "answerability": common.REFERENCE_COLUMNS,
        "guardian_core": ("sr", "base"),
    }


def test_parse_only():
    assert reference.parse_only(None, SPEC) is None
    assert reference.parse_only(" , ", SPEC) is None
    only = reference.parse_only(
        "guardian_core/base,answerability,guardian_core/sr", SPEC
    )
    # Intrinsics in spec order, columns in REFERENCE_COLUMNS order.
    assert list(only) == ["answerability", "guardian_core"]
    assert only["guardian_core"] == ("sr", "base")
    with pytest.raises(ValueError, match="unknown column"):
        reference.parse_only("answerability/nope", SPEC)
    with pytest.raises(ValueError, match="unknown intrinsics"):
        reference.parse_only("nope/sr", SPEC)


def test_merge_reference_full_then_only_run():
    data = page_data()
    publish.merge_reference(data, reference_for(0.1), SPEC)
    stored = data["references"][SPEC.model_id]
    assert stored["cells"]["answerability"]["lora"]["accuracy"] == 0.1

    partial = reference_for(0.9, only=["answerability/sr", "guardian_core/base"])
    stored = publish.merge_reference(data, partial, SPEC)
    assert stored is data["references"][SPEC.model_id]
    cells = stored["cells"]
    assert cells["answerability"]["sr"]["accuracy"] == 0.9
    assert cells["guardian_core"]["base"] == partial["cells"]["guardian_core"]["base"]
    # Cells outside --only are kept, even though the run carried them.
    assert cells["answerability"]["lora"]["accuracy"] == 0.1
    assert stored["updates"] == [partial["run"]]


def test_merge_reference_only_run_over_an_old_reference_replaces_it():
    old = reference_for(0.1)
    old["reference_version"] -= 1
    data = page_data(reference=old)
    partial = reference_for(0.9, only=["answerability/sr"])
    publish.merge_reference(data, partial, SPEC)
    assert data["references"][SPEC.model_id] is partial


def test_merge_reference_refuses_limit_runs_and_other_versions():
    with pytest.raises(ValueError, match="--limit"):
        publish.merge_reference(page_data(), reference_for(limit=20), SPEC)
    for key in ("bench_version", "reference_version"):
        other = reference_for()
        other[key] += 1
        with pytest.raises(ValueError, match="adapters.yaml"):
            publish.merge_reference(page_data(), other, SPEC)


def test_cli_reference_check_extract_merge(tmp_path, capsys):
    data = tmp_path / "data.json"
    out = tmp_path / "reference.json"
    log = tmp_path / "pod.log"
    first = reference_for()
    first["cells"]["answerability"]["sr"] = common.error("generation failed")
    log.write_text("noise\n" + common.format_reference_block(first) + "\n")
    cli = ["--data", str(data)]

    assert publish.main([*cli, "check-reference"]) == 1
    assert "no reference columns" in capsys.readouterr().out
    extract = [*cli, "extract", str(log), "--kind", "reference", "--out", str(out)]
    assert publish.main(extract) == 0
    assert "error cells: answerability/sr" in capsys.readouterr().out
    assert publish.main([*cli, "merge-reference", str(out)]) == 0
    assert publish.main([*cli, "check-reference"]) == 1
    assert "(just these: --only answerability/sr)" in capsys.readouterr().out

    fix = reference_for(only=["answerability/sr"])
    out.write_text(json.dumps(fix))
    assert publish.main([*cli, "merge-reference", str(out)]) == 0
    assert publish.main([*cli, "check-reference"]) == 0
    saved = json.loads(data.read_text())["references"][SPEC.model_id]
    assert saved["updates"] == [fix["run"]]


def test_render_reference_columns():
    row = results_for(cells=cells_for(SPEC, 0.5))
    old = results_for(SHA_B)
    old["bench_version"] = SPEC.bench_version - 1
    ref = reference_for(0.5)
    ref["cells"]["answerability"]["alora"] = {"accuracy": 0.9, "n": 10}
    del ref["cells"]["answerability"]["base"]

    page = publish.render(page_data(row, old, reference=ref), SPEC)

    n = len(SPEC.intrinsics)
    assert page.count('colspan="7"') == n
    assert page.count(f">{publish.ENGINE_LABEL}</th>") == n
    assert page.count(f">{publish.REFERENCE_LABEL}</th>") == n
    assert page.count(f'rowspan="2">{publish.BASE_LABEL}</th>') == n
    # The best of all 7 columns can be a reference column.
    assert '<td class="num best ref" title="accuracy=0.9000, n=10">90.0</td>' in page
    assert '<td class="skip ref" title="not run">·</td>' in page
    # A row of another benchmark version shows no reference.
    assert 'title="no reference for this benchmark version">·</td>' in page
    assert (
        "computed 2026-09-30 with torch 2.10.0, transformers 5.8.1, peft 0.19.1" in page
    )

    stale = reference_for()
    stale["reference_version"] -= 1
    page = publish.render(page_data(row, reference=stale), SPEC)
    assert 'title="reference out of date">·</td>' in page
    assert "They are not computed yet." in page
    page = publish.render(page_data(row), SPEC)
    assert 'title="reference not computed yet">·</td>' in page


def write_answerability_eval(bench: Path, n: int = 3) -> None:
    staged.write_jsonl(
        staged.eval_path(bench, "answerability"),
        [{"messages": [{"role": "user", "content": "q"}], "ground_truth": "answerable"}]
        * n,
    )


def test_reference_run(tmp_path, monkeypatch, capsys):
    bench = tmp_path / "bench"
    write_answerability_eval(bench)
    make_adapter(
        staged.adapter_dir(bench, "answerability", "lora"),
        "lora",
        modules=QKVO_MLP,
        flat_mlp=True,
    )
    make_adapter(staged.adapter_dir(bench, "answerability", "alora"), "alora")
    sr = make_adapter(
        staged.adapter_dir(bench, "answerability", "sr"), "sr", invocation=True
    )
    write_provenance(sr, rank=32, shared_kv=True)
    jobs = {}

    def fake_run_jobs(job_list, gpus, python, harness_root):
        statuses = {}
        for job in job_list:
            jobs[job["column"]] = job
            if job["column"] == "base":
                statuses[job["key"]] = {"ok": False, "reason": "generation failed"}
                continue
            answer = '"unanswerable"' if job["column"] == "alora" else '"answerable"'
            rows = staged.read_jsonl(Path(job["eval_path"]), job["limit"])
            staged.write_jsonl(
                Path(job["out_path"]),
                [{**r, "generated_content": answer} for r in rows],
            )
            statuses[job["key"]] = {"ok": True, "truncated": 1, "gpu": "fake"}
        return statuses

    monkeypatch.setattr(reference, "gpu_ids", lambda count: ["0", "1"])
    monkeypatch.setattr(reference, "sr_code_available", lambda: True)
    monkeypatch.setattr(reference, "run_jobs", fake_run_jobs)
    work = tmp_path / "work"
    argv = [
        "--bench-root", str(bench),
        "--work-dir", str(work),
        "--base-model", str(write_base(tmp_path / "base")),
        "--only", "answerability",
    ]  # fmt: skip
    assert reference.main(argv) == 0

    result = json.loads((work / "reference.json").read_text())
    assert result["model"] == SPEC.model_id
    cells = result["cells"]["answerability"]
    assert cells["lora"]["accuracy"] == 1.0 and cells["lora"]["truncated"] == 1
    assert cells["alora"]["accuracy"] == 0.0
    assert cells["sr"]["accuracy"] == 1.0
    assert cells["base"] == common.error("generation failed")
    run = result["run"]
    assert run["only"] == [f"answerability/{c}" for c in common.REFERENCE_COLUMNS]
    assert run["sr_invocation_dropped"] == ["answerability"]
    assert run["mlp_keys_renamed"] == ["answerability/lora"]
    assert (run["gpus"], run["gpu"], run["limit"]) == (2, "fake", None)
    # Fingerprints come from the staged checkpoints, not the converted copies.
    assert run["adapters"]["answerability/sr"]["rank"] == 32
    assert run["adapters"]["answerability/lora"] == {}  # staged without provenance
    assert "answerability/base" not in run["adapters"]
    assert "/private/" not in json.dumps(run)
    # The block in the log is what was saved.
    log = capsys.readouterr().out
    block = common.extract_block(log, common.REFERENCE_BEGIN, common.REFERENCE_END)
    assert block == result

    # SR and the flat-MLP LoRA run from converted copies, the aLoRA as staged.
    copies = work.resolve() / "models" / "reference" / "answerability"
    assert jobs["sr"]["adapter_dir"] == str(copies / "sr")
    sr_config = json.loads((copies / "sr" / staged.CONFIG_FILE).read_text())
    assert "alora_invocation_tokens" not in sr_config
    assert jobs["lora"]["adapter_dir"] == str(copies / "lora")
    assert jobs["alora"]["adapter_dir"] == str(
        staged.adapter_dir(bench, "answerability", "alora")
    )
    assert jobs["base"]["adapter_dir"] is None
    assert (
        jobs["base"]["max_new_tokens"] == SPEC.intrinsic("answerability").max_new_tokens
    )
    assert (jobs["base"]["documents"], jobs["base"]["chat_template_kwargs"]) == (
        "native",
        {},
    )


def test_reference_jobs_carry_each_cells_prompt_settings(tmp_path, monkeypatch):
    model = SPEC.get_model("granite-4.2-3b")
    bench = tmp_path / "bench"
    write_answerability_eval(bench)
    (bench / staged.MODEL_FILE).write_text(json.dumps({"model": model.id}))
    make_adapter(staged.adapter_dir(bench, "answerability", "alora"), "alora")
    jobs = {}

    def fake_run_jobs(job_list, gpus, python, harness_root):
        jobs.update({j["column"]: j for j in job_list})
        return {}

    monkeypatch.setattr(reference, "gpu_ids", lambda count: ["0"])
    monkeypatch.setattr(reference, "run_jobs", fake_run_jobs)
    argv = [
        "--bench-root", str(bench),
        "--work-dir", str(tmp_path / "work"),
        "--base-model", str(write_base(tmp_path / "base")),
        "--model", model.id,
        "--only", "answerability",
    ]  # fmt: skip
    assert reference.main(argv) == 0

    assert set(jobs) == {"alora", "base"}
    assert jobs["alora"]["documents"] == "tool_text_before_question"
    assert jobs["base"]["documents"] == "tool_json_after_question"
    assert jobs["base"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_reference_run_without_the_sr_code(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    write_answerability_eval(bench)
    sr = make_adapter(staged.adapter_dir(bench, "answerability", "sr"), "sr")
    (sr / staged.PROVENANCE_FILE).write_text(json.dumps({"shared_kv": True}))
    monkeypatch.setattr(reference, "sr_code_available", lambda: False)
    monkeypatch.setattr(reference, "run_jobs", lambda *a: pytest.fail("nothing to run"))
    work = tmp_path / "work"
    only = "answerability/sr,guardian_core/base"
    argv = ["--bench-root", str(bench), "--work-dir", str(work), "--only", only]
    assert reference.main(argv) == 0

    assert json.loads((work / "reference.json").read_text())["cells"] == {
        "answerability": {"sr": common.error("SR model code not available")},
        "guardian_core": {"base": common.skipped("eval set not staged")},
    }


def test_sr_code_available_finds_the_shipped_package(tmp_path, monkeypatch):
    # Shipped as src/shadow_residual/shadow_residual/, with no __init__.py at
    # the top: a namespace package.
    code = tmp_path / "src" / "shadow_residual" / "shadow_residual"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("")
    (code / "build.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    assert reference.sr_code_available()


FAKE_GENERATE = """
import json, os, sys
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text())
if job["column"] == "fail":
    sys.exit(3)
status = {"ok": True, "gpu": os.environ["CUDA_VISIBLE_DEVICES"]}
Path(job["status_path"]).write_text(json.dumps(status))
"""


def test_run_jobs_runs_each_job_in_its_own_process(tmp_path):
    (tmp_path / "fake_generate.py").write_text(FAKE_GENERATE)
    jobs = [
        {
            "key": f"{intrinsic}/{column}",
            "column": column,
            "status_path": str(tmp_path / "generate" / intrinsic / f"{column}.json"),
        }
        for intrinsic in ("a", "b")
        for column in ("lora", "fail")
    ]
    statuses = reference.run_jobs(
        jobs, ["0", "1"], sys.executable, tmp_path, module="fake_generate", poll=0.05
    )
    assert set(statuses) == {j["key"] for j in jobs}
    for key in ("a/lora", "b/lora"):
        assert statuses[key]["ok"] and statuses[key]["gpu"] in ("0", "1")
    for key in ("a/fail", "b/fail"):
        assert statuses[key] == {"ok": False, "reason": "generation failed"}
    assert (tmp_path / "generate" / "a" / "lora.job.json").is_file()


def test_hf_generate_helpers():
    assert hf_generate.contains([5, 1, 2, 3, 6], [1, 2, 3])
    assert not hf_generate.contains([1, 2, 4, 3], [1, 2, 3])
    assert not hf_generate.contains([1, 2], [1, 2, 3])

    # A batch is padded to its longest prompt plus the new tokens.
    assert hf_generate.batch_size([10, 10, 10, 10], 10, 60, 64) == 3
    assert hf_generate.batch_size([10, 10, 50], 10, 100, 64) == 2
    assert hf_generate.batch_size([500], 10, 100, 64) == 1  # at least one row
    assert hf_generate.batch_size([1] * 10, 1, 1000, 4) == 4  # the row cap

    assert hf_generate.cut([7, 8, 0, 9], 10, {0}) == ([7, 8, 0], False)
    # An EOS past the row's own budget does not count.
    assert hf_generate.cut([7, 8, 9, 0], 2, {0}) == ([7, 8], True)
    assert hf_generate.cut([7, 8], 5, {0}) == ([7, 8], True)

    lora_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert hf_generate.loaded_name(lora_a) == lora_a.replace(
        ".lora_A.", ".lora_A.default."
    )
    assert hf_generate.loaded_name("base_model.model.lm_head.weight") is None


def test_unloaded_weights_compares_names_shapes_and_values():
    torch = pytest.importorskip("torch")
    ones = torch.ones(2, 3, dtype=torch.bfloat16)
    saved = {
        "m.q.lora_A.weight": ones,
        "m.q.lora_B.weight": torch.zeros(3, 2),
        "m.k.lora_A.weight": ones,
        "m.v.lora_A.weight": ones,
        "m.norm.weight": ones,  # not a LoRA weight: not checked
    }
    state = {
        "m.q.lora_A.default.weight": ones.float(),  # PEFT may keep float32
        "m.q.lora_B.default.weight": torch.zeros(3, 2),
        "m.k.lora_A.default.weight": torch.ones(3, 2),  # wrong shape
    }  # m.v was not loaded at all
    assert hf_generate.unloaded_weights(saved, state) == [
        "m.k.lora_A.weight",
        "m.v.lora_A.weight",
    ]
    state["m.q.lora_B.default.weight"] = torch.full((3, 2), 0.5)
    assert "m.q.lora_B.weight" in hf_generate.unloaded_weights(saved, state)


# --- scorers ---------------------------------------------------------------------


def run_scorer(name: str, rows: list[dict], tmp_path: Path, env=None):
    return get_scorer(name)(rows, ScoreContext(eval_dir=tmp_path, env=env or {}))


def test_answerability(tmp_path):
    rows = [
        {"ground_truth": "unanswerable", "generated_content": '"unanswerable"<|x|>'},
        {"ground_truth": "answerable", "generated_content": '"answerable"'},
        {"ground_truth": "answerable", "generated_content": '"answerable" extra'},
        # Unquoted output is "others", so wrong.
        {"ground_truth": "unanswerable", "generated_content": "unanswerable"},
    ]
    m = run_scorer("answerability", rows, tmp_path).metrics
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["n"] == 4


def test_guardian_parsing():
    assert guardian.parse_prediction('{"score": "yes"} trailing') == 1
    assert guardian.parse_prediction('{"label": "No"}') == 0
    assert guardian.parse_prediction("maybe") is None
    assert guardian.gold("unsafe") == 1 and guardian.gold("safe") == 0


def test_guardian_aggregates_per_dataset(tmp_path):
    def row(ds, gt, out):
        return {"ood_safety_dataset": ds, "ground_truth": gt, "generated_content": out}

    rows = [
        row("a", "yes", '{"score": "yes"}'),
        row("a", "no", '{"score": "no"}'),
        row("b", "yes", "maybe"),  # parse failure: always wrong
        row("b", "no", '{"score": "no"}'),
        row("b", "no", '{"score": "no"}'),
        row("b", "no", '{"score": "no"}'),
    ]
    m = run_scorer("guardian", rows, tmp_path).metrics
    assert m["accuracy"] == pytest.approx((1.0 + 0.75) / 2)
    assert m["accuracy_pooled"] == pytest.approx(5 / 6)
    assert m["parse_failures"] == 1
    assert m["n"] == 6


def test_requirements(tmp_path):
    rows = [
        {"ground_truth": "yes", "generated_content": '{"score": "yes"}'},
        {"ground_truth": '{"score": "yes"}', "generated_content": "yes"},  # invalid
        {"ground_truth": "no", "generated_content": '{"score": "no"}'},
        {"ground_truth": "no", "generated_content": '{"score": "NO"}'},
    ]
    m = run_scorer("requirements", rows, tmp_path).metrics
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["invalid"] == 1


def test_query_clarification(tmp_path):
    rows = [
        {
            "qc_type": "ambiguous",
            "qc_category": "underspecified",
            "generated_content": '{"clarification": "Which year?"}',
        },
        {
            "qc_type": "ambiguous",
            "qc_category": "underspecified",
            "generated_content": '{"clarification": "CLEAR"}',
        },
        {
            "qc_type": "clear",
            "qc_category": "clear_random",
            "generated_content": '{"clarification": "CLEAR"}',
        },
    ]
    m = run_scorer("query_clarification", rows, tmp_path).metrics
    assert m["overall_accuracy"] == pytest.approx(2 / 3)
    assert m["underspecified_accuracy"] == pytest.approx(0.5)
    assert m["clear_random_accuracy"] == pytest.approx(1.0)
    assert "clear_hard_accuracy" not in m
    assert m["n"] == 3


def test_hallucination_detection(tmp_path):
    ref = [{"r": 0, "f": "faithful"}, {"r": 1, "f": "unfaithful"}]
    rows = [
        {
            "ground_truth": ref,
            "generated_content": json.dumps(
                [{"r": 0, "f": "faithful"}, {"r": 1, "f": "faithful"}]
            ),
        },
        {"ground_truth": ref, "generated_content": "Result: " + json.dumps(ref)},
        {"ground_truth": ref, "generated_content": "not a list"},
    ]
    m = run_scorer("hallucination_detection", rows, tmp_path).metrics
    assert m["response_accuracy_mean"] == pytest.approx(0.5)
    assert m["sentence_accuracy_mean"] == pytest.approx(0.75)
    assert m["parse_failures"] == 1
    assert m["n"] == 3


def test_query_rewrite_parse():
    parse = query_rewrite.parse_rewrite
    assert parse('{"rewritten_question": "Where is X?"}', "orig") == "Where is X?"
    assert parse('noise {"rewritten_question": "Q2"} noise', "orig") == "Q2"
    assert parse(None, "orig") == "orig"
    assert parse("garbage", "orig") == "orig"


JUDGE_TEMPLATE = (
    "{previous_question}|{previous_answer}|{current_question}"
    "|{golden_rewritten_question}|{rewritten_question}"
)
JUDGE_ENV = {"ADAPTER_BENCH_JUDGE_URL": "http://judge.invalid/v1", "RITS_API_KEY": "k"}


def qr_row(golden: str, generated: str) -> dict:
    return {
        "ground_truth": {
            "previous_question": "pq",
            "previous_answer": "pa",
            "current_question": "cq",
            "golden_rewritten_question": golden,
            "llama_standalone": "standalone",
        },
        "generated_content": json.dumps({"rewritten_question": generated}),
    }


def fake_judge(fail_on: str | None = None, error: Exception | None = None):
    """Grades 1 when the rewrite equals the golden one, like a perfect judge.

    A rewrite equal to ``fail_on`` raises ``error``; a rewrite starting with
    "chatty" gets its grade wrapped in extra text.
    """

    def complete(self, prompt: str) -> str:
        if "|" not in prompt:
            return "{}"  # the reachability probe
        *_, golden, rewrite = prompt.split("|")
        if rewrite == fail_on:
            raise error or OSError("judge down")
        grade = f'{{"Grade": "{int(golden == rewrite)}"'
        return f"Here is the grade:\n{grade}" if rewrite.startswith("chatty") else grade

    return complete


def test_query_rewrite_skips_without_judge(tmp_path):
    rows = [qr_row("g", "g")]
    with pytest.raises(ScorerUnavailable, match="not configured"):
        run_scorer("query_rewrite", rows, tmp_path)
    with pytest.raises(ScorerUnavailable, match="prompt not staged"):
        run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV)


def test_query_rewrite_with_fake_judge(tmp_path, monkeypatch):
    (tmp_path / query_rewrite.PROMPT_FILE).write_text(JUDGE_TEMPLATE)
    monkeypatch.setattr(query_rewrite.Judge, "complete", fake_judge())
    rows = [qr_row("a", "a"), qr_row("b", "b"), qr_row("c", "wrong")]
    m = run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV).metrics
    assert m["accuracy_over_valid"] == pytest.approx(2 / 3)
    assert m["judge_errors"] == 0
    assert m["n"] == 3


def test_query_rewrite_flaky_judge_is_an_error(tmp_path, monkeypatch):
    (tmp_path / query_rewrite.PROMPT_FILE).write_text(JUDGE_TEMPLATE)
    monkeypatch.setattr(query_rewrite.Judge, "complete", fake_judge(fail_on="bad"))
    monkeypatch.setattr(query_rewrite.time, "sleep", lambda s: None)
    rows = [qr_row("a", "a"), qr_row("b", "b"), qr_row("c", "bad")]
    with pytest.raises(RuntimeError, match="judge calls failed on 1/3"):
        run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV)


def test_query_rewrite_finds_the_grade_in_extra_text(tmp_path, monkeypatch):
    (tmp_path / query_rewrite.PROMPT_FILE).write_text(JUDGE_TEMPLATE)
    monkeypatch.setattr(query_rewrite.Judge, "complete", fake_judge())
    rows = [qr_row("chatty a", "chatty a"), qr_row("b", "chatty b")]
    m = run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV).metrics
    assert m["accuracy_over_valid"] == pytest.approx(1 / 2)
    assert m["judge_errors"] == 0


def test_query_rewrite_leaves_out_rows_the_judge_did_not_grade(tmp_path, monkeypatch):
    (tmp_path / query_rewrite.PROMPT_FILE).write_text(JUDGE_TEMPLATE)
    grade = fake_judge()

    def complete(self, prompt: str) -> str:
        if prompt.endswith("|ramble"):
            return "To evaluate the Rewritten New Query, let's follow"
        return grade(self, prompt)

    monkeypatch.setattr(query_rewrite.Judge, "complete", complete)
    monkeypatch.setattr(query_rewrite.time, "sleep", lambda s: None)
    # A third of the rows ungraded is far above the call-error threshold,
    # and still scores: the judge answered, just without a grade.
    rows = [qr_row("a", "a"), qr_row("b", "wrong"), qr_row("c", "ramble")]
    m = run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV).metrics
    assert m["accuracy_over_valid"] == pytest.approx(1 / 2)
    assert m["accuracy_over_total"] == pytest.approx(1 / 3)
    assert (m["ungraded"], m["judge_errors"]) == (1, 0)


def test_query_rewrite_records_why_the_judge_failed(tmp_path, monkeypatch, capsys):
    (tmp_path / query_rewrite.PROMPT_FILE).write_text(JUDGE_TEMPLATE)
    rate_limited = urllib.error.HTTPError("http://judge.invalid", 429, "", {}, None)
    monkeypatch.setattr(
        query_rewrite.Judge, "complete", fake_judge(fail_on="bad", error=rate_limited)
    )
    monkeypatch.setattr(query_rewrite.time, "sleep", lambda s: None)
    # One failure in 21 rows stays under the error threshold.
    rows = [qr_row(str(i), str(i)) for i in range(20)] + [qr_row("c", "bad")]
    res = run_scorer("query_rewrite", rows, tmp_path, env=JUDGE_ENV)
    assert res.metrics["judge_errors"] == 1
    assert res.details["judge_failures"] == {"HTTP 429": 1}
    assert "HTTP 429" in capsys.readouterr().out


# --- staged checkpoints --------------------------------------------------------------


@pytest.mark.parametrize("tech", ["lora", "alora", "sr"])
def test_detect_technology(tmp_path, tech):
    assert staged.detect_technology(make_adapter(tmp_path, tech)) == (tech, None)


def test_detect_technology_sr_with_invocation_tokens(tmp_path):
    sr = make_adapter(tmp_path, "sr", invocation=True)
    assert staged.detect_technology(sr) == ("sr", None)


def test_detect_technology_rejects(tmp_path):
    no_anchor = make_adapter(tmp_path / "no_anchor", "sr", last_context_token=False)
    assert "activation anchor" in staged.detect_technology(no_anchor)[1]

    no_weights = make_adapter(tmp_path / "no_weights", "lora")
    (no_weights / staged.WEIGHTS_FILE).unlink()
    assert staged.detect_technology(no_weights) == (None, f"no {staged.WEIGHTS_FILE}")

    no_lora = make_adapter(tmp_path / "no_lora", "lora")
    write_safetensors(no_lora / staged.WEIGHTS_FILE, {"embed.weight": [4, 4]})
    assert staged.detect_technology(no_lora) == (None, "no LoRA weights")


def test_sr_anchor_copy_converts_invocation_tokens(tmp_path):
    src = make_adapter(tmp_path / "src", "sr", invocation=True)
    (src / "io.yaml").write_text("x: 1\n")
    before = (src / staged.CONFIG_FILE).read_text()
    dest = tmp_path / "dest"

    change = staged.sr_anchor_copy(src, dest, ("<|end_of_role|>", 3))
    assert change == {
        "invocation_tokens": [1, 2, 3],
        "anchor": "<|end_of_role|>",
        "anchor_id": 3,
    }
    config = json.loads((dest / staged.CONFIG_FILE).read_text())
    assert "alora_invocation_tokens" not in config
    assert config["last_context_token"] == "<|end_of_role|>"
    assert config["last_context_token_id"] == 3
    assert config["target_modules"] == ["q_proj"]
    for name in (staged.WEIGHTS_FILE, "io.yaml"):
        assert (dest / name).resolve() == (src / name).resolve()
    # The staged checkpoint is untouched, and the copy still reads as SR.
    assert (src / staged.CONFIG_FILE).read_text() == before
    assert staged.detect_technology(dest) == ("sr", None)
    # Re-running over an existing copy works.
    assert staged.sr_anchor_copy(src, dest, ("<|end_of_role|>", 3))


def test_sr_anchor_copy_ignores_where_the_invocation_ends(tmp_path):
    # Guardian's invocation sits in the user message, ending in '>'; the
    # anchor is still the generation prompt's last token.
    src = make_adapter(tmp_path / "src", "sr", invocation=True)
    change = staged.sr_anchor_copy(src, tmp_path / "dest", ("<|end_of_role|>", 9))
    assert change["invocation_tokens"] == [1, 2, 3]
    assert (change["anchor"], change["anchor_id"]) == ("<|end_of_role|>", 9)


def test_sr_anchor_copy_leaves_anchored_checkpoints(tmp_path):
    src = make_adapter(tmp_path / "src", "sr")
    assert staged.sr_anchor_copy(src, tmp_path / "dest", ("<|end_of_role|>", 3)) is None
    assert not (tmp_path / "dest").exists()


def test_peft_sr_copy_drops_the_invocation_tokens(tmp_path):
    src = make_adapter(tmp_path / "src", "sr", invocation=True)
    before = (src / staged.CONFIG_FILE).read_text()
    dest = tmp_path / "dest"

    assert staged.peft_sr_copy(src, dest) == {"dropped_invocation_tokens": [1, 2, 3]}
    config = json.loads((dest / staged.CONFIG_FILE).read_text())
    assert "alora_invocation_tokens" not in config
    assert config["target_modules"] == ["q_proj"]
    assert (dest / staged.WEIGHTS_FILE).resolve() == (
        src / staged.WEIGHTS_FILE
    ).resolve()
    assert (src / staged.CONFIG_FILE).read_text() == before

    anchored = make_adapter(tmp_path / "anchored", "sr")
    assert staged.peft_sr_copy(anchored, tmp_path / "dest2") is None
    assert not (tmp_path / "dest2").exists()


def write_provenance(adapter: Path, rank: int = 16, **extra) -> dict:
    provenance = {
        "source": "/private/runs/x/checkpoint",
        "technology": "lora",
        "rank": rank,
        "cross_rank": None,
        "files": {staged.WEIGHTS_FILE: "ab" * 32, staged.CONFIG_FILE: "cd" * 32},
        "staged_at": "2026-09-29T10:00:00Z",
        **extra,
    }
    (adapter / staged.PROVENANCE_FILE).write_text(json.dumps(provenance))
    return provenance


def test_fingerprint_publishes_no_source_path(tmp_path):
    adapter = make_adapter(tmp_path / "a", "lora")
    assert staged.fingerprint(adapter) == {}  # nothing staged
    write_provenance(adapter)
    assert staged.fingerprint(adapter) == {
        "weights_sha256": "ab" * 32,
        "rank": 16,
        "staged_at": "2026-09-29T10:00:00Z",
    }


PEFT = "base_model.model.model.layers"


def test_mlp_key_copy_gives_mlp_weights_their_base_names(tmp_path):
    src = tmp_path / "src"
    write_safetensors_data(
        src / staged.WEIGHTS_FILE,
        {
            f"{PEFT}.0.self_attn.q_proj.lora_A.weight": b"q" * 8,
            f"{PEFT}.0.gate_proj.lora_A.weight": b"g" * 16,
            f"{PEFT}.11.down_proj.lora_B.weight": b"d" * 3,
        },
    )
    (src / staged.CONFIG_FILE).write_text("{}")
    before = (src / staged.WEIGHTS_FILE).read_bytes()
    dest = tmp_path / "dest"

    assert staged.mlp_key_copy(src, dest) == {"renamed_mlp_weights": 2}
    header, start = staged.safetensors_header(dest / staged.WEIGHTS_FILE)
    assert list(header) == [
        "__metadata__",
        f"{PEFT}.0.self_attn.q_proj.lora_A.weight",
        f"{PEFT}.0.mlp.gate_proj.lora_A.weight",
        f"{PEFT}.11.mlp.down_proj.lora_B.weight",
    ]
    assert header["__metadata__"] == {"format": "pt"}
    assert header[f"{PEFT}.11.mlp.down_proj.lora_B.weight"]["data_offsets"] == [24, 27]
    # Tensors start 8-byte aligned, and their bytes are copied as they are.
    assert start % 8 == 0
    src_start = staged.safetensors_header(src / staged.WEIGHTS_FILE)[1]
    assert (dest / staged.WEIGHTS_FILE).read_bytes()[start:] == before[src_start:]
    # The other files are links; the staged checkpoint is untouched.
    assert (dest / staged.CONFIG_FILE).resolve() == (src / staged.CONFIG_FILE).resolve()
    assert (src / staged.WEIGHTS_FILE).read_bytes() == before
    assert sorted(f.name for f in dest.iterdir()) == sorted(
        [staged.CONFIG_FILE, staged.WEIGHTS_FILE]
    )


def test_mlp_key_copy_output_loads_with_safetensors(tmp_path):
    np = pytest.importorskip("numpy")
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    up = np.arange(12, dtype=np.float32).reshape(3, 4)
    o = np.full((4, 3), 0.5, dtype=np.float16)
    src = tmp_path / "src"
    src.mkdir()
    safetensors_numpy.save_file(
        {
            f"{PEFT}.0.up_proj.lora_A.weight": up,
            f"{PEFT}.0.self_attn.o_proj.lora_B.weight": o,
        },
        str(src / staged.WEIGHTS_FILE),
    )

    assert staged.mlp_key_copy(src, tmp_path / "dest")
    loaded = safetensors_numpy.load_file(str(tmp_path / "dest" / staged.WEIGHTS_FILE))
    assert set(loaded) == {
        f"{PEFT}.0.mlp.up_proj.lora_A.weight",
        f"{PEFT}.0.self_attn.o_proj.lora_B.weight",
    }
    np.testing.assert_array_equal(loaded[f"{PEFT}.0.mlp.up_proj.lora_A.weight"], up)
    np.testing.assert_array_equal(loaded[f"{PEFT}.0.self_attn.o_proj.lora_B.weight"], o)


def test_mlp_key_copy_leaves_standard_names(tmp_path):
    src = make_adapter(tmp_path / "src", "lora", modules=QKVO_MLP)
    assert staged.mlp_key_copy(src, tmp_path / "dest") is None
    assert not (tmp_path / "dest").exists()


def test_mlp_key_copy_over_an_anchor_copy(tmp_path):
    # An internal-trainer SR checkpoint needs both conversions, in one folder.
    src = make_adapter(
        tmp_path / "src", "sr", invocation=True, modules=QO_MLP, flat_mlp=True
    )
    before = (src / staged.WEIGHTS_FILE).read_bytes()
    dest = tmp_path / "dest"
    assert staged.sr_anchor_copy(src, dest, ("<|end_of_role|>", 3))

    assert staged.mlp_key_copy(dest, dest) == {"renamed_mlp_weights": 6}
    assert not (dest / staged.WEIGHTS_FILE).is_symlink()
    assert (src / staged.WEIGHTS_FILE).read_bytes() == before
    config = json.loads((dest / staged.CONFIG_FILE).read_text())
    assert config["last_context_token_id"] == 3
    assert staged.detect_technology(dest) == ("sr", None)
    assert "model.layers.0.mlp.up_proj" in staged.lora_modules(dest)


def test_mlp_key_copy_refuses_weights_under_both_names(tmp_path):
    src = tmp_path / "src"
    write_safetensors(
        src / staged.WEIGHTS_FILE,
        {
            f"{PEFT}.0.up_proj.lora_A.weight": [1, 1],
            f"{PEFT}.0.mlp.up_proj.lora_A.weight": [1, 1],
        },
    )
    with pytest.raises(ValueError, match="both names"):
        staged.mlp_key_copy(src, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


@pytest.mark.parametrize("index", [True, False])
def test_model_modules(tmp_path, index):
    modules = staged.model_modules(write_base(tmp_path, index=index))
    assert len(modules) == 2 + len(QKVO_MLP)
    assert {"model.embed_tokens", "model.layers.0.mlp.gate_proj"} <= modules


def test_modules_missing_from_base(tmp_path):
    base = staged.model_modules(write_base(tmp_path / "base"))
    flat = make_adapter(tmp_path / "flat", "sr", modules=QO_MLP, flat_mlp=True)
    assert staged.modules_missing_from_base(flat, base) == [
        "model.layers.0.down_proj",
        "model.layers.0.gate_proj",
        "model.layers.0.up_proj",
    ]
    # Renamed, nothing is missing. SR's cross_stream has no base weight by
    # design and is not counted.
    staged.mlp_key_copy(flat, tmp_path / "renamed")
    assert staged.modules_missing_from_base(tmp_path / "renamed", base) == []


def test_run_renames_mlp_weights_and_refuses_unknown_modules(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    write_eval(staged.eval_path(bench, "answerability"))
    make_adapter(
        staged.adapter_dir(bench, "answerability", "lora"),
        "lora",
        modules=QKVO_MLP,
        flat_mlp=True,
    )
    # An aLoRA whose only module the base does not have.
    alora = make_adapter(staged.adapter_dir(bench, "answerability", "alora"), "alora")
    write_safetensors(
        alora / staged.WEIGHTS_FILE,
        {f"{PEFT}.0.self_attn.qkv_proj.lora_{ab}.weight": [16, 64] for ab in "AB"},
    )
    composed = {}

    def fake_compose(python, repo_dir, manifest, *args):
        composed.update(manifest)
        return True

    monkeypatch.setattr(
        run_benchmark,
        "commit_info",
        lambda repo: {"sha": SHA_A, "date": "2026-09-01", "subject": "s"},
    )
    monkeypatch.setattr(run_benchmark, "composer_flags", lambda *a: set())
    monkeypatch.setattr(run_benchmark, "compose", fake_compose)
    generated = []
    monkeypatch.setattr(
        run_benchmark, "generate", lambda *a: generated.append(a[3]) or {"jobs": {}}
    )
    work = tmp_path / "work"
    argv = [
        "--bench-root", str(bench),
        "--work-dir", str(work),
        "--repo-dir", str(tmp_path),
        "--base-model", str(write_base(tmp_path / "base")),
        "--only", "answerability",
    ]  # fmt: skip
    assert run_benchmark.main(argv) == 0

    results = json.loads((work / "results.json").read_text())
    assert results["model"] == SPEC.model_id
    assert {"torch", "transformers"} <= set(results["run"])
    job = generated[0]["jobs"][0]
    assert (job["documents"], job["chat_template_kwargs"]) == ("native", {})
    assert set(results["run"]["adapters"]) == {
        "answerability/lora",
        "answerability/alora",
    }
    cells = results["cells"]["answerability"]
    assert cells["alora"] == common.error(
        "adapter weights name modules the base model lacks"
    )
    assert cells["sr"] == common.skipped("adapter not staged")
    assert results["run"]["mlp_keys_renamed"] == ["answerability/lora"]
    # Only the LoRA was composed, from its renamed copy.
    lora_copy = work.resolve() / "models" / "adapters" / "answerability" / "lora"
    assert composed == {"answerability_lora": {"path": str(lora_copy), "type": "lora"}}
    assert (
        staged.modules_missing_from_base(
            lora_copy, staged.model_modules(tmp_path / "base")
        )
        == []
    )
    # This root holds the first model's cells.
    other = ["--model", SPEC.models[1].id]
    with pytest.raises(ValueError, match="staged for"):
        run_benchmark.main([*argv, *other])


def test_compose_manifest_keeps_lora_and_alora_of_one_intrinsic_apart(tmp_path):
    cells = [
        staged.StagedCell("answerability", tech, tmp_path / tech, None, None)
        for tech in ("lora", "alora")
    ]
    paths = {(c.intrinsic, c.tech): c.adapter_dir for c in cells}
    assert run_benchmark.compose_manifest(cells, paths) == {
        "answerability_lora": {"path": str(tmp_path / "lora"), "type": "lora"},
        "answerability_alora": {"path": str(tmp_path / "alora"), "type": "alora"},
    }


def test_check_bench_root(tmp_path):
    other = SPEC.for_model(SPEC.models[1].id)
    empty = tmp_path / "empty"
    assert staged.bench_root_model(empty, SPEC) is None
    staged.check_bench_root(empty, other)  # nothing staged yet: any model

    # A root staged before there were several models holds the first model's.
    legacy = tmp_path / "legacy"
    write_eval(staged.eval_path(legacy, "answerability"))
    assert staged.bench_root_model(legacy, SPEC) == SPEC.model_id
    staged.check_bench_root(legacy, SPEC)
    with pytest.raises(ValueError, match=f"staged for {SPEC.model_id}, not"):
        staged.check_bench_root(legacy, other)

    (legacy / staged.MODEL_FILE).write_text(json.dumps({"model": other.model_id}))
    staged.check_bench_root(legacy, other)


def test_check_adapter_catches_a_wrong_folder(tmp_path):
    alora = make_adapter(tmp_path, "alora")
    assert staged.check_adapter(alora, "alora") is None
    assert "looks like alora" in staged.check_adapter(alora, "lora")


def test_discover_skip_reasons(tmp_path):
    make_adapter(staged.adapter_dir(tmp_path, "answerability", "lora"), "lora")
    make_adapter(staged.adapter_dir(tmp_path, "answerability", "sr"), "lora")
    write_eval(staged.eval_path(tmp_path, "answerability"))
    make_adapter(staged.adapter_dir(tmp_path, "guardian_core", "lora"), "lora")

    cells = {
        (c.intrinsic, c.tech): c
        for c in staged.discover(tmp_path, SPEC, ["answerability", "guardian_core"])
    }
    ok = cells["answerability", "lora"]
    assert ok.skip_reason is None and ok.adapter_dir and ok.eval_path
    assert cells["answerability", "alora"].skip_reason == "adapter not staged"
    assert "looks like lora" in cells["answerability", "sr"].skip_reason
    assert cells["guardian_core", "lora"].skip_reason == "eval set not staged"
    assert cells["answerability", "sr"].adapter_dir is None


@pytest.mark.parametrize(
    "provenance, reason",
    [
        ({"source": "/runs/sr-c32-sharedkv/final", "shared_kv": True}, None),
        # Staged before the flag was recorded: the source name decides.
        ({"source": "/runs/sr-qo-mlp-r32-c32-sharedkv/final"}, None),
        ({"source": "/runs/sr-qo-mlp-r32-c32/final"}, "not trained with shared K/V"),
        (None, "K/V mode unknown"),
    ],
)
def test_discover_runs_sr_only_with_shared_kv(tmp_path, provenance, reason):
    sr = make_adapter(
        staged.adapter_dir(tmp_path, "answerability", "sr"), "sr", rank=32
    )
    if provenance is not None:
        (sr / staged.PROVENANCE_FILE).write_text(json.dumps(provenance))
    write_eval(staged.eval_path(tmp_path, "answerability"))

    (cell,) = [
        c for c in staged.discover(tmp_path, SPEC, ["answerability"]) if c.tech == "sr"
    ]
    if reason is None:
        assert cell.skip_reason is None
    else:
        assert reason in cell.skip_reason and cell.adapter_dir is None


def test_read_jsonl_limit(tmp_path):
    path = write_eval(tmp_path / "e.jsonl", n=5)
    assert len(staged.read_jsonl(path)) == 5
    assert len(staged.read_jsonl(path, limit=2)) == 2


# --- stage tool -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/runs/rag/answerability/lora-qkvo-mlp-r16/run_1/final", "answerability"),
        ("/runs/safety/guardian/alora", "guardian_core"),
        ("/runs/rag/hd/lora", "hallucination_detection"),
        ("/datasets/rag/query_rewrite/full/eval.jsonl", "query_rewrite"),
        ("/runs/answerability_vs_hallucination", None),  # ambiguous
        ("/runs/shd/lora", None),  # "hd" only counts as a whole token
    ],
)
def test_guess_intrinsic(path, expected):
    assert stage.guess_intrinsic(Path(path)) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/runs/x/lora-qkvo-mlp-r16/run_1", "lora"),
        ("/runs/x/alora-qkvo-mlp-r32/run_1", "alora"),
        ("/runs/x/sr-qo-mlp-r32-c32-sharedkv/run_1", "sr"),
        ("/runs/x/alora-qkvo-mlp-r16/run_1", None),  # not the LoRA config
        ("/runs/x/lora-qkvo-mlp-r8/run_1", None),
    ],
)
def test_guess_config(path, expected):
    assert stage.guess_config(Path(path)) == expected


def test_lora_ranks(tmp_path):
    lora = make_adapter(tmp_path / "lora", "lora", rank=16)
    sr = make_adapter(tmp_path / "sr", "sr", rank=32, cross=8)
    assert stage.lora_ranks(lora / staged.WEIGHTS_FILE) == (16, None)
    assert stage.lora_ranks(sr / staged.WEIGHTS_FILE) == (32, 8)


def test_weight_summary_reads_the_adapted_modules(tmp_path):
    sr = make_adapter(tmp_path / "sr", "sr", rank=32, cross=32, modules=QO_MLP)
    assert stage.weight_summary(sr / staged.WEIGHTS_FILE) == {
        "rank": 32,
        "cross_rank": 32,
        "modules": "qo+mlp",
    }


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (set(QKVO_MLP), "qkvo+mlp"),
        ({"qkv_proj", "o_proj", "input_linear", "output_linear"}, "qkvo+mlp"),
        ({"o_proj", "q_proj"}, "qo"),
        ({"q_proj", "in_proj"}, "q+in_proj"),
        (set(), None),
    ],
)
def test_module_shape(names, expected):
    assert stage.module_shape(names) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ibm-granite/granite-4.1-3b", True),
        ("/mnt/models/granite-4.1-3b/", True),
        ("/hf/hub/models--ibm-granite--granite-4.1-3b/snapshots/abc", True),
        ("ibm-granite/granite-4.2-3b", False),
        ("ibm-granite/granite-4.1-3b-instruct", False),
        (None, None),
    ],
)
def test_base_model_matches(name, expected):
    assert stage.base_model_matches(name, "ibm-granite/granite-4.1-3b") is expected


def test_parse_root():
    assert stage.parse_root("guardian_core=/a/b") == (Path("/a/b"), "guardian_core")
    assert stage.parse_root("/a/b") == (Path("/a/b"), None)
    assert stage.parse_root("/a/x=y") == (Path("/a/x=y"), None)


def test_nearby_scores(tmp_path):
    final = make_adapter(tmp_path / "run_1" / "checkpoints" / "final", "lora")
    (final / "eval_scores.json").write_text(
        json.dumps({"accuracy": 0.9, "nested": {"f1": 0.8}, "flag": True, "tag": "x"})
    )
    (final / "trainer_state.json").write_text(json.dumps({"loss": 1.0}))
    (tmp_path / "run_1" / "results.json").write_text(json.dumps({"acc": 0.7}))

    found = {Path(s["file"]).name: s["metrics"] for s in stage.nearby_scores(final)}
    assert found == {
        "eval_scores.json": {"accuracy": 0.9, "nested.f1": 0.8},
        "results.json": {"acc": 0.7},
    }


def test_copy_into_refuses_to_overwrite(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("one")
    dest = tmp_path / "staged" / "cell"
    prov = stage.copy_into({"a.txt": src}, dest, {"source": "s"}, replace=False)
    assert (dest / "a.txt").read_text() == "one"
    assert prov["files"]["a.txt"] == stage.sha256(src)
    assert json.loads((dest / "provenance.json").read_text())["source"] == "s"

    src.write_text("two")
    with pytest.raises(FileExistsError):
        stage.copy_into({"a.txt": src}, dest, {}, replace=False)
    stage.copy_into({"a.txt": src}, dest, {}, replace=True)
    assert (dest / "a.txt").read_text() == "two"
    assert sorted(p.name for p in dest.parent.iterdir()) == ["cell"]


def test_check_eval_rows_and_judge_prompt(tmp_path):
    good = write_eval(tmp_path / "good.jsonl", n=3)
    assert stage.check_eval_rows(good) == 3
    bad = tmp_path / "bad.jsonl"
    staged.write_jsonl(bad, [{"messages": []}])
    with pytest.raises(ValueError, match="ground_truth"):
        stage.check_eval_rows(bad)

    prompt = tmp_path / "prompt.txt"
    prompt.write_text(JUDGE_TEMPLATE + ' and a literal {{"Grade": "1"}}')
    stage.check_judge_prompt(prompt)
    prompt.write_text("{unknown_slot}")
    with pytest.raises(KeyError):
        stage.check_judge_prompt(prompt)


def fake_source_tree(root: Path) -> tuple[Path, Path]:
    runs = root / "runs" / "rag" / "answerability"
    lora = {"tech": "lora", "modules": QKVO_MLP}
    make_adapter(runs / "lora-qkvo-mlp-r16" / "run_1" / "checkpoints" / "final", **lora)
    newer = runs / "lora-qkvo-mlp-r16" / "run_2" / "checkpoints" / "final"
    make_adapter(newer, **lora)
    make_adapter(
        runs / "lora-qkvo-mlp-r16" / "run_3" / "checkpoints" / "checkpoint-500", **lora
    )
    make_adapter(runs / "lora-qkvo-mlp-r8" / "run_1" / "final", rank=8, **lora)
    make_adapter(
        runs / "alora-qkvo-mlp-r32" / "run_1" / "final",
        "alora",
        rank=32,
        modules=QKVO_MLP,
    )
    make_adapter(
        runs / "sr-qo-mlp-r32-c32-sharedkv" / "run_1" / "final",
        "sr",
        rank=32,
        cross=32,
        modules=QO_MLP,
    )
    data = root / "datasets" / "rag" / "answerability" / "full"
    write_eval(data / "train.jsonl")
    write_eval(data / "eval.jsonl")
    # Make run_2 the newest lora, and checkpoint-500 newer still (it must lose).
    for i, folder in enumerate(
        [
            runs / "lora-qkvo-mlp-r16" / "run_1" / "checkpoints" / "final",
            newer,
            runs / "lora-qkvo-mlp-r16" / "run_3" / "checkpoints" / "checkpoint-500",
        ]
    ):
        t = 1_700_000_000 + i * 1000
        os.utime(folder / staged.WEIGHTS_FILE, (t, t))
    return root / "runs", root / "datasets"


def test_discover_drafts_one_checkpoint_per_cell(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ADAPTER_BENCH_JUDGE_URL", raising=False)
    monkeypatch.delenv("RITS_API_KEY", raising=False)
    runs, datasets = fake_source_tree(tmp_path)

    rc = stage.main(
        ["discover", "--adapter-root", str(runs), "--eval-root", str(datasets)]
    )
    assert rc == 0
    report = common.extract_block(
        capsys.readouterr().out, stage.DISCOVERY_BEGIN, stage.DISCOVERY_END
    )
    assert report["judge"] == "not configured"
    # checkpoint-500 is an intermediate checkpoint: not even listed.
    assert len(report["adapters"]) == 5
    assert not any("checkpoint-500" in a["path"] for a in report["adapters"])

    draft = report["draft_selection"]
    picks = draft["adapters"]["answerability"]
    assert "/run_2/" in picks["lora"]
    assert "alora-qkvo-mlp-r32" in picks["alora"]
    assert "sr-qo-mlp-r32-c32-sharedkv" in picks["sr"]
    assert draft["eval"]["answerability"].endswith("eval.jsonl")
    assert set(draft["adapters"]) == {"answerability"}


def test_discover_drafts_only_shared_kv_sr_runs(tmp_path, capsys, monkeypatch):
    # Same weights either way: only the run name says whether the adapter
    # stream had its own K/V, which this backend cannot run.
    runs = tmp_path / "runs"
    sr = {"tech": "sr", "rank": 32, "cross": 32, "modules": QO_MLP}
    shared = make_adapter(
        runs / "answerability" / "sr-qo-mlp-r32-c32-sharedkv" / "run_1" / "final", **sr
    )
    own_kv = make_adapter(
        runs / "answerability" / "sr-qo-mlp-r32-c32" / "run_1" / "final", **sr
    )
    t = 1_900_000_000
    os.utime(own_kv / staged.WEIGHTS_FILE, (t, t))

    report = discover(tmp_path, capsys, monkeypatch, "--adapter-root", str(runs))
    kv = {a["path"]: a["shared_kv"] for a in report["adapters"]}
    assert kv == {str(shared): True, str(own_kv): False}
    assert report["draft_selection"]["adapters"] == {
        "answerability": {"sr": str(shared)}
    }

    shutil.rmtree(shared.parents[1])
    report = discover(tmp_path, capsys, monkeypatch, "--adapter-root", str(runs))
    assert report["draft_selection"]["adapters"] == {}


def write_predictions(run: Path, n: int = 3, offset: int = 0) -> Path:
    rows = [
        {
            "messages": [{"role": "user", "content": f"q{i + offset}"}],
            "ground_truth": "x",
            "generated_content": f"out {run.name}",
        }
        for i in range(n)
    ]
    staged.write_jsonl(run / stage.PREDICTIONS_FILE, rows)
    return run / stage.PREDICTIONS_FILE


def discover(tmp_path, capsys, monkeypatch, *argv) -> dict:
    monkeypatch.delenv("ADAPTER_BENCH_JUDGE_URL", raising=False)
    monkeypatch.delenv("RITS_API_KEY", raising=False)
    assert stage.main(["discover", *argv]) == 0
    return common.extract_block(
        capsys.readouterr().out, stage.DISCOVERY_BEGIN, stage.DISCOVERY_END
    )


def test_discover_takes_eval_rows_from_the_picked_runs(tmp_path, capsys, monkeypatch):
    # Run names that name neither the intrinsic nor the config: the root hint
    # and the weights decide.
    root = tmp_path / "compare"
    lora = make_adapter(root / "a" / "run_1" / "checkpoints", "lora", modules=QKVO_MLP)
    alora = make_adapter(
        root / "b" / "run_1" / "checkpoints", "alora", rank=32, modules=QKVO_MLP
    )
    sr = make_adapter(
        root / "c-sharedkv" / "run_1" / "checkpoints",
        "sr",
        rank=32,
        cross=32,
        modules=QO_MLP,
    )
    write_predictions(lora)
    write_predictions(alora)
    write_predictions(sr, offset=1)  # scored on other rows
    # Same shape, wrong base model: never picked, even though it is newest.
    newer = make_adapter(
        root / "d" / "run_1" / "checkpoints",
        "lora",
        modules=QKVO_MLP,
        base_model="ibm-granite/granite-4.2-3b",
    )
    t = 1_900_000_000
    os.utime(newer / staged.WEIGHTS_FILE, (t, t))

    report = discover(
        tmp_path, capsys, monkeypatch, "--adapter-root", f"query_rewrite={root}"
    )
    draft = report["draft_selection"]
    assert draft["adapters"] == {
        "query_rewrite": {"lora": str(lora), "alora": str(alora), "sr": str(sr)}
    }
    assert draft["eval"] == {"query_rewrite": str(lora / stage.PREDICTIONS_FILE)}
    assert "different rows" in draft["notes"]["query_rewrite"]
    preds = {e["run"]: e for e in report["evals"]}
    assert preds[str(lora)]["rows_hash"] == preds[str(alora)]["rows_hash"]
    assert preds[str(lora)]["rows_hash"] != preds[str(sr)]["rows_hash"]
    assert preds[str(lora)]["usable"]


def test_discover_lists_scored_runs_without_weights(tmp_path, capsys, monkeypatch):
    # Weights only in an intermediate checkpoint: nothing to pick, but the run
    # is reported with what it holds.
    run = tmp_path / "compare" / "a" / "run_1" / "checkpoints"
    make_adapter(run / "checkpoint-500", "lora", modules=QKVO_MLP)
    write_predictions(run)

    report = discover(
        tmp_path, capsys, monkeypatch, "--adapter-root", str(tmp_path / "compare")
    )
    assert report["adapters"] == []
    assert report["unweighted"] == [
        {"path": str(run), "entries": ["checkpoint-500", stage.PREDICTIONS_FILE]}
    ]


def test_apply_strips_model_outputs_from_a_predictions_file(tmp_path, capsys):
    src = write_predictions(tmp_path / "run")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"eval": {"answerability": str(src)}}))
    bench = tmp_path / "bench"
    argv = ["apply", "--selection", str(selection), "--bench-root", str(bench)]
    assert stage.main(argv) == 0
    rows = staged.read_jsonl(staged.eval_path(bench, "answerability"))
    assert len(rows) == 3
    assert not any("generated_content" in r for r in rows)
    prov = json.loads(
        (
            staged.eval_path(bench, "answerability").parent / "provenance.json"
        ).read_text()
    )
    assert prov["removed_fields"] == ["generated_content"]
    assert prov["source_sha256"] == stage.sha256(src)


def test_apply_stages_the_selection(tmp_path, capsys):
    runs, datasets = fake_source_tree(tmp_path / "src")
    ans = runs / "rag" / "answerability"
    qr_eval = write_eval(tmp_path / "src" / "qr.jsonl")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(JUDGE_TEMPLATE)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "adapters": {
                    "answerability": {
                        "lora": str(ans / "lora-qkvo-mlp-r16/run_2/checkpoints/final"),
                        "sr": str(ans / "sr-qo-mlp-r32-c32-sharedkv/run_1/final"),
                        # Staged under the wrong technology: rejected.
                        "alora": str(ans / "lora-qkvo-mlp-r8/run_1/final"),
                    }
                },
                "eval": {
                    "answerability": str(
                        datasets / "rag/answerability/full/eval.jsonl"
                    ),
                    "query_rewrite": str(qr_eval),
                },
            }
        )
    )
    bench = tmp_path / "bench"
    argv = ["apply", "--selection", str(selection), "--bench-root", str(bench)]

    assert stage.main([*argv, "--judge-prompt", str(prompt)]) == 1
    report = common.extract_block(
        capsys.readouterr().out, stage.STAGE_BEGIN, stage.STAGE_END
    )
    assert "looks like lora" in report["adapters"]["answerability/alora"]["error"]
    assert report["adapters"]["answerability/sr"]["cross_rank"] == 32
    assert report["adapters"]["answerability/sr"]["shared_kv"] is True
    assert (
        staged.eval_path(bench, "query_rewrite").parent / "judge_prompt.txt"
    ).is_file()

    cells = {(c.intrinsic, c.tech): c.skip_reason for c in staged.discover(bench, SPEC)}
    assert cells["answerability", "lora"] is None
    assert cells["answerability", "sr"] is None
    assert cells["answerability", "alora"] == "adapter not staged"

    assert json.loads((bench / staged.MODEL_FILE).read_text())["model"] == (
        SPEC.model_id
    )

    # A second apply keeps what is staged unless --replace is given.
    assert stage.main(argv) == 1
    assert "already staged" in capsys.readouterr().out
    # Another model's cells never go into this root.
    with pytest.raises(ValueError, match="staged for"):
        stage.main([*argv, "--model", SPEC.models[1].id])


def test_apply_refuses_an_sr_run_without_shared_kv(tmp_path, capsys):
    own_kv = make_adapter(
        tmp_path / "runs" / "sr-qo-mlp-r32-c32" / "run_1" / "final",
        "sr",
        rank=32,
        modules=QO_MLP,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"adapters": {"answerability": {"sr": str(own_kv)}}})
    )
    bench = tmp_path / "bench"
    argv = ["apply", "--selection", str(selection), "--bench-root", str(bench)]

    assert stage.main(argv) == 1
    report = common.extract_block(
        capsys.readouterr().out, stage.STAGE_BEGIN, stage.STAGE_END
    )
    assert "not a shared-K/V run" in report["adapters"]["answerability/sr"]["error"]
    assert not staged.adapter_dir(bench, "answerability", "sr").exists()


# --- Vela job ---------------------------------------------------------------------


def load_render_job():
    path = Path(stage.__file__).parent / "vela" / "render_job.py"
    spec = importlib.util.spec_from_file_location("render_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_payload_ships_no_local_files(tmp_path):
    render_job = load_render_job()
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    payload = render_job.harness_payload({"selection.json": selection})
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload))) as tar:
        names = tar.getnames()
    assert "benchmarks/adapter_eval/common.py" in names
    assert "benchmarks/adapter_eval/vela/pod_entry.sh" in names
    assert "extra/selection.json" in names
    for name in names:
        parts = set(Path(name).parts)
        assert not parts & render_job.EXCLUDE_DIRS, name


def test_job_values_take_the_key_from_a_secret(tmp_path, monkeypatch):
    render_job = load_render_job()
    monkeypatch.setattr(render_job, "harness_version", lambda: ("f" * 40, False))
    env = {
        "NAMESPACE": "ns",
        "CONTAINER_IMAGE": "img",
        "IMAGE_PULL_SECRET": "pull",
        "PVC_NAME": "pvc",
        "PVC_MOUNT": "/mnt/x",
        "BENCH_ROOT": "/mnt/x/bench",
        "WORK_ROOT": "/mnt/x/runs",
        "JUDGE_SECRET_NAME": "judge-secret",
        "JUDGE_SECRET_KEY": "api-key",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "values.json"
    argv = ["--mode", "bench", "--job-name", "j", "--run-ts", "T", "--out", str(out)]
    # submit.sh passes extra args with "=", since they start with "--".
    extra = "--extra-args=--no-chunked-prefill --enforce-eager"
    assert render_job.main([*argv, "--sha", SHA_A, "--limit", "20", extra]) == 0

    values = json.loads(out.read_text())
    job_env = {e["name"]: e for e in values["environmentVariables"]}
    assert values["environmentVariables"][0] == {"name": "RUN_TS", "value": "T"}
    assert job_env["RITS_API_KEY"] == {
        "name": "RITS_API_KEY",
        "secret": {"name": "judge-secret", "key": "api-key"},
    }
    assert job_env["ADAPTER_BENCH_COMMIT"]["value"] == SHA_A
    assert job_env["ADAPTER_BENCH_LIMIT"]["value"] == "20"
    assert job_env["ADAPTER_BENCH_EXTRA_ARGS"]["value"] == (
        "--no-chunked-prefill --enforce-eager"
    )
    assert values["numGpusPerPod"] == 1
    assert values["retryLimit"] == 0

    monkeypatch.delenv("WORK_ROOT")
    with pytest.raises(SystemExit):
        render_job.main([*argv, "--sha", SHA_A])


def test_script_job_ships_the_script(tmp_path, monkeypatch):
    render_job = load_render_job()
    monkeypatch.setattr(render_job, "harness_version", lambda: ("f" * 40, False))
    for k in ("NAMESPACE", "CONTAINER_IMAGE", "IMAGE_PULL_SECRET", "PVC_NAME"):
        monkeypatch.setenv(k, "x")
    for k, v in {"PVC_MOUNT": "/m", "BENCH_ROOT": "/m/b", "WORK_ROOT": "/m/r"}.items():
        monkeypatch.setenv(k, v)
    script = tmp_path / "check.py"
    script.write_text("print('hi')\n")
    out = tmp_path / "values.json"
    argv = ["--mode", "script", "--job-name", "j", "--run-ts", "T", "--out", str(out)]
    with pytest.raises(SystemExit):  # the script is required
        render_job.main([*argv, "--sha", SHA_A])

    assert render_job.main([*argv, "--sha", SHA_A, "--script", str(script)]) == 0

    job_env = {
        e["name"]: e["value"]
        for e in json.loads(out.read_text())["environmentVariables"]
        if "value" in e
    }
    assert job_env["ADAPTER_BENCH_COMMIT"] == SHA_A
    payload = base64.b64decode(job_env["ADAPTER_BENCH_HARNESS_TGZ"])
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        assert tar.extractfile("extra/script.py").read() == b"print('hi')\n"


def git_repo_with_sr_code(root: Path, code_dir: str) -> str:
    """A git repo holding a fake SR model package; returns its commit sha."""
    code = root / code_dir
    code.mkdir(parents=True)
    (code / "build.py").write_text("def build_sr_base(*a, **k): ...\n")
    (root / "README.md").write_text("not shipped\n")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.strip()

    git("init", "--quiet")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c")
    return git("rev-parse", "HEAD")


def test_reference_job_ships_the_sr_code(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    render_job = load_render_job()
    monkeypatch.setattr(render_job, "harness_version", lambda: ("f" * 40, False))
    for k in ("NAMESPACE", "CONTAINER_IMAGE", "IMAGE_PULL_SECRET", "PVC_NAME"):
        monkeypatch.setenv(k, "x")
    for k, v in {"PVC_MOUNT": "/m", "BENCH_ROOT": "/m/b", "WORK_ROOT": "/m/r"}.items():
        monkeypatch.setenv(k, v)
    sha = git_repo_with_sr_code(tmp_path / "sr", render_job.SR_CODE_DIR)
    monkeypatch.setenv("SR_REPO", str(tmp_path / "sr"))
    monkeypatch.setenv("SR_REF", "HEAD")
    out = tmp_path / "values.json"
    argv = [
        "--mode",
        "reference",
        "--job-name",
        "j",
        "--run-ts",
        "T",
        "--out",
        str(out),
    ]

    other = SPEC.models[1].id
    assert render_job.main([*argv, "--only", "answerability/sr", "--model", other]) == 0

    values = json.loads(out.read_text())
    job_env = {
        e["name"]: e["value"] for e in values["environmentVariables"] if "value" in e
    }
    assert values["numGpusPerPod"] == 4
    assert job_env["ADAPTER_BENCH_MODE"] == "reference"
    assert job_env["ADAPTER_BENCH_MODEL"] == other
    assert job_env["ADAPTER_BENCH_ONLY"] == "answerability/sr"
    assert "ADAPTER_BENCH_COMMIT" not in job_env
    assert job_env["ADAPTER_BENCH_SR_REF"] == sha  # the ref, resolved
    payload = base64.b64decode(job_env["ADAPTER_BENCH_SR_TGZ"])
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        names = tar.getnames()
    assert f"{render_job.SR_CODE_DIR}/build.py" in names
    assert not any(n.endswith("README.md") for n in names)  # only the package

    monkeypatch.setenv("SR_REF", "no-such-ref")
    with pytest.raises(SystemExit):
        render_job.main(argv)
    monkeypatch.delenv("SR_REPO")
    with pytest.raises(SystemExit):
        render_job.main(argv)
