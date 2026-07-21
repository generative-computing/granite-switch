# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BUILD.md rendering."""

from types import SimpleNamespace

from granite_switch.composer.reporting.model_card import (
    render_model_card,
    write_model_card,
)


def _fake_base_config(**overrides):
    defaults = dict(
        model_type="granitemoehybrid",
        architectures=["GraniteMoeHybridForCausalLM"],
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=32,
        vocab_size=49152,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_adapter_index():
    return {
        "model_info": {
            "num_adapters": 3,
            "base_model": "granite-4.0-micro",
        },
        "adapters": [
            {
                "adapter_index": 1,
                "adapter_name": "rag",
                "technology": "alora",
                "control_token": {"token": "<|rag|>", "id": 100},
                "io_config": "io_configs/rag/io.yaml",
            },
            {
                "adapter_index": 2,
                "adapter_name": "citations",
                "technology": "lora",
                "control_token": {"token": "<|citations|>", "id": 101},
                "io_config": "io_configs/citations/io.yaml",
            },
            {
                "adapter_index": 3,
                "adapter_name": "base",
                "technology": "lora",
                "control_token": {"token": "<|base|>", "id": 102},
                "built_in": True,
            },
        ],
    }


class TestRenderModelCard:
    # ---- Title / layout ----
    def test_starts_with_title(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
        )
        # No front matter; card starts with the H1 directly
        assert md.lstrip().startswith("# Granite Switch Composed Model")
        assert not md.startswith("---")

    def test_section_order(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            base_param_count=1_000_000,
            composed_param_count=1_100_000,
            compose_settings={"technology_filter": "alora"},
        )
        idx_title = md.index("# Granite Switch Composed Model")
        idx_base = md.index("## Base Model")
        idx_adapters = md.index("## Embedded Adapters")
        idx_details = md.index("## Composition Details")
        assert idx_title < idx_base < idx_adapters < idx_details

    # ---- Base Model section ----
    def test_base_model_identifier_and_arch(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
        )
        assert "ibm-granite/granite-4.0-micro" in md
        assert "granitemoehybrid" in md
        assert "GraniteMoeHybridForCausalLM" in md
        assert "2048" in md

    def test_param_counts_and_delta_in_composition_details(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            base_param_count=3_402_836_480,
            composed_param_count=3_776_507_411,
        )
        # Composition-specific info stays out of Base Model.
        base_start = md.index("## Base Model")
        adapters_start = md.index("## Embedded Adapters")
        base_section = md[base_start:adapters_start]
        assert "Param delta" not in base_section
        assert "base_param_count" not in base_section

        details = md[md.index("## Composition Details") :]
        assert "base_param_count: 3,402,836,480" in details
        assert "composed_param_count: 3,776,507,411" in details
        assert "Param delta: +10.98%" in details

    def test_params_absent_when_not_provided(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
        )
        assert "Param delta" not in md
        assert "base_param_count" not in md

    def test_missing_optional_config_field(self):
        cfg = _fake_base_config()
        del cfg.num_attention_heads
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=cfg,
            adapter_index=_fake_adapter_index(),
        )
        assert "Attention heads" not in md
        assert "Hidden size" in md

    # ---- Adapter table (7 columns) ----
    def test_adapter_rows_and_columns(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 16, 8],
            adapter_alphas=[8.0, 32.0, 8.0],
            adapter_targets=[
                {"q_proj", "v_proj"},
                {"q_proj", "v_proj", "o_proj"},
                None,
            ],
        )
        assert (
            "| # | Name | Technology | Control Token | Token ID | Rank | Alpha | Target Modules | Source |"
            in md
        )
        # Adapter names
        assert "rag" in md
        assert "citations" in md
        # Technologies
        assert "alora" in md
        assert "lora" in md
        # Ranks
        assert "16" in md
        # Alpha + Target Modules columns present
        assert "| Alpha |" in md
        assert "| Target Modules |" in md
        # Target modules rendered sorted + comma-joined
        assert "o_proj, q_proj, v_proj" in md
        assert "q_proj, v_proj" in md

    def test_control_token_pipes_escaped(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
        )
        # Pipes inside a cell must be escaped so they don't break columns
        assert r"<\|rag\|>" in md
        assert r"<\|citations\|>" in md
        for line in md.splitlines():
            if not line.startswith("| "):
                continue
            cell_separators = line.replace(r"\|", "").count("|")
            # Header is 10 pipes (9 cols + 2 edges = 10 separators).
            assert cell_separators == 10, (
                f"Row has wrong column count (likely unescaped |): {line}"
            )

    def test_source_column_built_in_fallback(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            adapter_sources=[
                "ibm-granite/granite-lib-rag-r1.0",
                "ibm-granite/granite-lib-rag-r1.0",
                None,  # built-in
            ],
        )
        assert "ibm-granite/granite-lib-rag-r1.0" in md
        assert "built-in" in md

    def test_no_adapters(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index={"model_info": {}, "adapters": []},
        )
        assert "No adapters embedded" in md

    # ---- Composition Details (bottom section, plain YAML-style text) ----
    def test_composition_details_contains_param_counts(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            base_param_count=3_402_836_480,
            composed_param_count=3_776_507_411,
        )
        details = md[md.index("## Composition Details") :]
        # Counts are thousands-separated for readability. Strict YAML parsers
        # would need to strip commas, but the section is plain markdown text
        # (not fenced) and the same values appear on the bulleted Params line.
        assert "base_param_count: 3,402,836,480" in details
        assert "composed_param_count: 3,776,507,411" in details

    def test_composition_details_contains_compose_settings(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            compose_settings={"technology_filter": "alora"},
        )
        details = md[md.index("## Composition Details") :]
        assert "compose_settings:" in details
        assert "technology_filter:" in details
        assert "alora" in details

    def test_composition_details_contains_adapter_sources(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            adapter_commits_by_source={
                "ibm-granite/granite-lib-rag-r1.0": "6e4a75e35f1cb272e8d15b4615fb0a123398d1cf",
            },
        )
        details = md[md.index("## Composition Details") :]
        assert "adapter_sources:" in details
        assert "6e4a75e35f1cb272e8d15b4615fb0a123398d1cf" in details

    def test_composition_details_list_values(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
            compose_settings={"include_adapters": ["query_*", "answerability"]},
        )
        # Lists render as YAML sequences
        assert "include_adapters:" in md
        assert '- "query_*"' in md
        assert '- "answerability"' in md

    def test_composition_details_omitted_when_empty(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
        )
        assert "## Composition Details" not in md

    def test_composition_details_not_fenced(self):
        md = render_model_card(
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            base_param_count=1_000_000,
            composed_param_count=1_100_000,
        )
        # Plain text, not a ```yaml fence
        assert "```yaml" not in md
        assert "```" not in md


class TestWriteModelCard:
    def test_writes_file_to_output_dir(self, tmp_path):
        out = tmp_path / "composed"
        out.mkdir()
        path = write_model_card(
            output_path=str(out),
            base_model_name="ibm-granite/granite-4.0-micro",
            base_config=_fake_base_config(),
            adapter_index=_fake_adapter_index(),
            adapter_ranks=[8, 8, 8],
        )
        assert path.name == "BUILD.md"
        assert path.exists()
        content = path.read_text()
        assert "ibm-granite/granite-4.0-micro" in content
        assert "rag" in content
