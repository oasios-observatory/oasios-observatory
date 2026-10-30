#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/abbreviator_s.py

"""Generate compact identifiers for speculative ASI scenarios."""

from typing import Any


def short_code(value: str) -> str:
    """Convert a categorical value into a short uppercase code."""

    if not isinstance(value, str) or not value.strip():
        return "UNK"

    words = value.replace("-", " ").replace("_", " ").split()

    return "".join(
        word[0].upper()
        for word in words
    )[:3] or "UNK"


def abbreviate(core: dict[str, Any]) -> str:
    """
    Generate a deterministic scenario title.

    The title is independent of database state.
    """

    if not isinstance(core, dict):
        raise TypeError("Scenario parameters must be a dictionary.")

    fields = (
        "initial_origin",
        "development_dynamics",
        "architecture",
        "deployment_topology",
        "oversight_type",
        "oversight_effectiveness",
        "substrate",
    )

    parts = [
        short_code(core.get(field, "UNK"))
        for field in fields
    ]

    return "-".join(parts)
