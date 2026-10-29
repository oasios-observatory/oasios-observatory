#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/timeline.py

"""
Signal-influenced timeline generation for ASI scenarios.

Generates chronologically ordered timeline phases while preserving
schema-compatible fields and applying parameter-driven time shifts.
"""

from datetime import datetime
from typing import Any

import numpy as np

from oasios.logger import log


# Maximum parameter-driven timeline shift, in years.
MAX_SHIFT = 3

# Earliest permitted pivot year.
START_SCALING_ERA = 2021

# Numeric influence of oversight effectiveness.
OVERSIGHT_MAPPING = {
    "none": 0.0,
    "ineffective": 0.0,
    "partial": 0.5,
    "effective": 1.0,
    "unknown": 0.0,
}


def _numeric_oversight(value: Any) -> float:
    """Convert oversight effectiveness into a numeric control factor."""

    if not isinstance(value, str):
        raise ValueError(
            "oversight_effectiveness must be a string."
        )

    normalized = value.strip().lower()

    if normalized not in OVERSIGHT_MAPPING:
        raise ValueError(
            f"Unsupported oversight_effectiveness: {value!r}"
        )

    return OVERSIGHT_MAPPING[normalized]


def generate_scenario_timeline(
    params: dict[str, Any],
    precursor_density: float = 0.0,
    dominant_tags: list[str] | None = None,
    reference_year: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a parameter-influenced, chronologically ordered timeline.

    Args:
        params: Scenario parameters.
        precursor_density: Precursor signal density, expected in [0, 1].
        dominant_tags: Dominant precursor tags.
        reference_year: Optional year used for reproducible tests.

    Returns:
        A list of timeline phases with phase, years, and description.

    Raises:
        TypeError: If the parameters or tags have invalid types.
        ValueError: If numeric inputs are invalid.
    """

    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    if dominant_tags is None:
        dominant_tags = []

    if not isinstance(dominant_tags, list):
        raise TypeError("dominant_tags must be a list.")

    if not all(isinstance(tag, str) for tag in dominant_tags):
        raise TypeError("Every dominant tag must be a string.")

    if not isinstance(precursor_density, (int, float)):
        raise TypeError("precursor_density must be numeric.")

    if not np.isfinite(precursor_density):
        raise ValueError("precursor_density must be finite.")

    if not 0.0 <= precursor_density <= 1.0:
        raise ValueError(
            "precursor_density must be between 0 and 1."
        )

    if reference_year is None:
        current_year = datetime.now().year
    else:
        if (
            isinstance(reference_year, bool)
            or not isinstance(reference_year, int)
            or reference_year < START_SCALING_ERA
        ):
            raise ValueError(
                f"reference_year must be an integer >= "
                f"{START_SCALING_ERA}."
            )

        current_year = reference_year

    # -------------------------------------------------------------
    # 1. Read scenario parameters
    # -------------------------------------------------------------

    agency = params.get("agency_level", 0.5)
    opacity = params.get("opacity", 0.5)
    alignment = params.get("alignment_score", 0.5)

    for name, value in (
        ("agency_level", agency),
        ("opacity", opacity),
        ("alignment_score", alignment),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"{name} must be a finite number between 0 and 1."
            )

    oversight_value = params.get(
        "oversight_effectiveness",
        "unknown",
    )

    oversight_numeric = _numeric_oversight(oversight_value)

    # -------------------------------------------------------------
    # 2. Calculate parameter-driven shift
    # -------------------------------------------------------------

    risk_factor = (agency + opacity) / 2.0

    control_factor = (
        alignment + oversight_numeric
    ) / 2.0

    net_shift = (
        risk_factor - control_factor
    ) * MAX_SHIFT

    # -------------------------------------------------------------
    # 3. Apply signal-driven adjustments
    # -------------------------------------------------------------

    signal_density_shift = (
        precursor_density * -2.0
        if precursor_density > 0.7
        else 0.0
    )

    tag_shift = (
        -1
        if "asi_direct" in dominant_tags
        else 0
    )

    total_shift = int(
        np.round(
            net_shift
            + signal_density_shift
            + tag_shift
        )
    )

    # -------------------------------------------------------------
    # 4. Calculate timeline years
    # -------------------------------------------------------------

    pivot_year = max(
        START_SCALING_ERA,
        current_year + total_shift,
    )

    # Ensure the scaling era is never an invalid year range.
    scaling_end = max(
        START_SCALING_ERA,
        pivot_year - 1,
    )

    window_start = pivot_year + 1
    window_end = pivot_year + 5

    log.info(
        "timeline.shift_applied",
        shift=total_shift,
        pivot=pivot_year,
    )

    # -------------------------------------------------------------
    # 5. Construct schema-compatible timeline phases
    # -------------------------------------------------------------

    timeline = [
        {
            "phase": "Precursors & Foundations",
            "years": "2000-2020",
            "description": (
                "Early internet development, deep learning origins, "
                "and foundations of modern AI."
            ),
        },
        {
            "phase": "Scaling Era",
            "years": (
                f"{START_SCALING_ERA}-{scaling_end}"
            ),
            "description": (
                "Large language models, agents, and multimodal "
                "systems accelerate capability development."
            ),
        },
        {
            "phase": "Pivot Year",
            "years": str(pivot_year),
            "description": (
                "A key inflection point in capability development, "
                "deployment, or oversight."
            ),
        },
        {
            "phase": "Emergence Window",
            "years": (
                f"{window_start}-{window_end}"
            ),
            "description": (
                "A projected emergence window influenced by "
                "capability accumulation and oversight conditions."
            ),
        },
        {
            "phase": "Long-Term Equilibrium",
            "years": f"{window_end + 1}+",
            "description": (
                "A longer-term period in which the consequences "
                "of advanced AI deployment become more apparent."
            ),
        },
    ]

    return timeline
