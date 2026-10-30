#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/constraints_s.py


"""Cross-parameter constraints for OASIOS scenario generation."""

import math

from typing import Any


NUMERIC_PARAMETERS = (
    "agency_level",
    "autonomy_degree",
    "alignment_score",
    "opacity",
    "deceptiveness",
    "phenomenology_proxy_score",
    "agency_level_confidence",
    "autonomy_degree_confidence",
    "alignment_score_confidence",
)


def validate_parameter_relationships(
    params: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Validate relationships between sampled parameters.

    Returns:
        (valid, errors)
    """

    if not isinstance(params, dict):
        return False, ["Parameters must be a dictionary."]

    errors: list[str] = []

    # ---------------------------------------------------------
    # 1. NUMERICAL VALIDATION
    # ---------------------------------------------------------

    for key in NUMERIC_PARAMETERS:

        value = params.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            errors.append(
                f"{key} must be a finite number between 0 and 1."
            )

    if errors:
        return False, errors

    autonomy = params["autonomy_degree"]
    alignment = params["alignment_score"]
    deceptiveness = params["deceptiveness"]

    # ---------------------------------------------------------
    # 2. OVERSIGHT RELATIONSHIPS
    # ---------------------------------------------------------

    oversight_type = params.get("oversight_type")
    oversight_effectiveness = params.get(
        "oversight_effectiveness"
    )

    if oversight_type == "none":
        if oversight_effectiveness != "ineffective":
            errors.append(
                "No oversight must have ineffective oversight."
            )

    if autonomy >= 0.75:
        if oversight_effectiveness == "effective":
            errors.append(
                "Super autonomy cannot have fully effective oversight."
            )

    # ---------------------------------------------------------
    # 3. ALIGNMENT RELATIONSHIPS
    # ---------------------------------------------------------

    if alignment < 0.30:

        if params.get("stated_goal") == "human-welfare":
            errors.append(
                "Very low alignment should not imply "
                "a straightforward human-welfare objective."
            )

    # ---------------------------------------------------------
    # 4. DECEPTIVENESS RELATIONSHIPS
    # ---------------------------------------------------------

    if deceptiveness > 0.60:

        if params.get("deployment_strategy") != "stealth":
            errors.append(
                "High deceptiveness should use stealth deployment."
            )

    # ---------------------------------------------------------
    # 5. AUTONOMY CATEGORY
    # ---------------------------------------------------------

    category = params.get("autonomy_category")

    if category is not None:

        expected_ranges = {
            "partial": (0.00, 0.39),
            "full": (0.40, 0.74),
            "super": (0.75, 1.00),
        }

        if category not in expected_ranges:
            errors.append(
                f"Unsupported autonomy category: {category!r}"
            )

        else:
            low, high = expected_ranges[category]

            if not low <= autonomy <= high:
                errors.append(
                    f"Autonomy degree {autonomy} does not "
                    f"match category {category!r}."
                )

    # ---------------------------------------------------------
    # 6. IMPACT DOMAIN VALIDATION
    # ---------------------------------------------------------

    impact_domains = params.get("impact_domains")

    if (
        not isinstance(impact_domains, list)
        or not impact_domains
        or not all(isinstance(item, str) for item in impact_domains)
    ):
        errors.append(
            "Impact domains must be a nonempty list of strings."
        )

    return not errors, errors


def enforce_parameter_relationships(
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate parameter relationships without silently
    modifying sampled values.

    Raises:
        ValueError: If relationships are invalid.
    """

    valid, errors = validate_parameter_relationships(params)

    if not valid:
        raise ValueError(
            "Parameter relationship validation failed: "
            + "; ".join(errors)
        )

    return params
