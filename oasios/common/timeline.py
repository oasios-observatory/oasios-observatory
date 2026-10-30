#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Signal-informed timeline generation for ASI scenarios."""

import math

from datetime import datetime
from typing import Any

from oasios.logger import log


START_SCALING_ERA = 2021
MIN_PIVOT_YEAR = 2022
MAX_SHIFT = 3

OVERSIGHT_MAPPING = {
    "none": 0.0,
    "ineffective": 0.0,
    "partial": 0.5,
    "effective": 1.0,
    "unknown": 0.0,
}


def _validate_score(
    name: str,
    value: Any,
) -> float:
    """Validate a normalized numerical score."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(
            f"{name} must be between 0 and 1."
        )

    return float(value)


def _numeric_oversight(value: Any) -> float:
    """Convert oversight effectiveness to a numeric factor."""

    if not isinstance(value, str):
        raise ValueError(
            "oversight_effectiveness must be a string."
        )

    normalized = value.strip().lower()

    if normalized not in OVERSIGHT_MAPPING:
        raise ValueError(
            f"Unsupported oversight effectiveness: {value}"
        )

    return OVERSIGHT_MAPPING[normalized]


def generate_scenario_timeline(
    params: dict[str, Any],
    precursor_density: float = 0.0,
    dominant_tags: list[str] | None = None,
    reference_year: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate a chronologically ordered scenario timeline.

    Precursor density and dominant tags influence the
    projected pivot year.
    """

    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    if dominant_tags is None:
        dominant_tags = []

    if not isinstance(dominant_tags, list):
        raise TypeError("dominant_tags must be a list.")

    if not all(
        isinstance(tag, str)
        for tag in dominant_tags
    ):
        raise TypeError(
            "Every dominant tag must be a string."
        )

    precursor_density = _validate_score(
        "precursor_density",
        precursor_density,
    )

    agency = _validate_score(
        "agency_level",
        params.get("agency_level", 0.5),
    )

    opacity = _validate_score(
        "opacity",
        params.get("opacity", 0.5),
    )

    alignment = _validate_score(
        "alignment_score",
        params.get("alignment_score", 0.5),
    )

    oversight = _numeric_oversight(
        params.get(
            "oversight_effectiveness",
            "unknown",
        )
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
                f"reference_year must be >= {START_SCALING_ERA}."
            )

        current_year = reference_year

    # ---------------------------------------------------------
    # PARAMETER-DRIVEN ADJUSTMENT
    # ---------------------------------------------------------

    risk_factor = (agency + opacity) / 2.0

    control_factor = (alignment + oversight) / 2.0

    parameter_shift = (
        risk_factor - control_factor
    ) * MAX_SHIFT

    # ---------------------------------------------------------
    # PRECURSOR-DRIVEN ADJUSTMENT
    # ---------------------------------------------------------

    density_shift = (
        -2.0
        if precursor_density > 0.7
        else 0.0
    )

    tag_shift = (
        -1.0
        if "asi_direct" in dominant_tags
        else 0.0
    )

    total_shift = int(
        round(
            parameter_shift
            + density_shift
            + tag_shift
        )
    )

    # ---------------------------------------------------------
    # TIMELINE BOUNDARIES
    # ---------------------------------------------------------

    pivot_year = max(
        MIN_PIVOT_YEAR,
        current_year + total_shift,
    )

    scaling_start = START_SCALING_ERA
    scaling_end = pivot_year - 1

    emergence_start = pivot_year + 1
    emergence_end = pivot_year + 5

    equilibrium_start = emergence_end + 1

    timeline = [
        {
            "phase": "Precursors & Foundations",
            "years": "2000-2020",
            "description": (
                "Early internet development, deep learning origins, "
                "and foundations of modern artificial intelligence."
            ),
        },
        {
            "phase": "Scaling Era",
            "years": (
                f"{scaling_start}-{scaling_end}"
            ),
            "description": (
                "Large language models, autonomous agents, and "
                "multimodal systems accelerate capability development."
            ),
        },
        {
            "phase": "Pivot Year",
            "years": str(pivot_year),
            "description": (
                "A major inflection point in capability development, "
                "deployment, or oversight."
            ),
        },
        {
            "phase": "Emergence Window",
            "years": (
                f"{emergence_start}-{emergence_end}"
            ),
            "description": (
                "A projected emergence window influenced by "
                "capability accumulation, precursor signals, "
                "and oversight conditions."
            ),
        },
        {
            "phase": "Long-Term Equilibrium",
            "years": f"{equilibrium_start}+",
            "description": (
                "A longer-term period in which the consequences "
                "of advanced AI deployment become apparent."
            ),
        },
    ]

    log.info(
        "timeline.generated",
        pivot_year=pivot_year,
        parameter_shift=parameter_shift,
        precursor_density=precursor_density,
        dominant_tags=dominant_tags,
    )

    return timeline
