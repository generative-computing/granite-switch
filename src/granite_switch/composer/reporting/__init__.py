# SPDX-License-Identifier: Apache-2.0
"""Compose reporting utilities for Granite Switch."""

from .adapter_analysis import print_source_adapter_analysis
from .compose_report import generate_compose_report
from .model_card import render_model_card, write_build_doc, write_model_card
from .population_table import (
    generate_adapter_population_table,
    print_adapter_population_table,
)

__all__ = [
    "generate_adapter_population_table",
    "generate_compose_report",
    "print_adapter_population_table",
    "print_source_adapter_analysis",
    "render_model_card",
    "write_build_doc",
    "write_model_card",
]
