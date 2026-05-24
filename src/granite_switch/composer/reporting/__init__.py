# SPDX-License-Identifier: Apache-2.0
"""Compose reporting utilities for Granite Switch."""

from .population_table import generate_adapter_population_table, print_adapter_population_table
from .compose_report import generate_compose_report
from .adapter_analysis import print_source_adapter_analysis
from .model_card import render_model_card, write_model_card, write_build_doc

__all__ = [
    'generate_adapter_population_table',
    'print_adapter_population_table',
    'generate_compose_report',
    'print_source_adapter_analysis',
    'render_model_card',
    'write_model_card',
    'write_build_doc',
]
