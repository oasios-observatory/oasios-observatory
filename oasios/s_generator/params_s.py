#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/params_s.py

"""
Schema-aware parameter sampler for OASIOS.

JSON Schema defines structural constraints.
Python configuration defines probability distributions.
Cross-parameter constraints preserve scenario coherence.
"""

import math
import random

from typing import Any

from oasios.common.schema import SchemaManager

from oasios.s_generator.constraints_s import (
    enforce_parameter_relationships,
)


# -------------------------------------------------------------
# CONFIGURATION: PROBABILITY DISTRIBUTIONS
# -------------------------------------------------------------

PARAMETER_WEIGHTS = {

    "architecture": {
        "monolithic": 0.15,
        "decentralized": 0.20,
        "swarm": 0.15,
        "hierarchical": 0.15,
        "modular": 0.15,
        "hybrid": 0.20,
    },

    "substrate": {
        "classical": 0.50,
        "neuromorphic": 0.25,
        "quantum": 0.25,
    },

    "initial_origin": {
        "corporate": 0.35,
        "open-source": 0.25,
        "state": 0.20,
        "rogue": 0.20,
    },

    "development_dynamics": {
        "engineered": 0.45,
        "emergent": 0.25,
        "hybrid": 0.30,
    },

    "autonomy_category": {
        "partial": 0.40,
        "full": 0.40,
        "super": 0.20,
    },

}


# -------------------------------------------------------------
# GENERATION POLICIES
# -------------------------------------------------------------

AUTONOMY_RANGES = {

    "partial": (0.00, 0.39),

    "full": (0.40, 0.74),

    "super": (0.75, 1.00),

}


AGENCY_RANGES = {

    "partial": (0.10, 0.55),

    "full": (0.40, 0.85),

    "super": (0.70, 1.00),

}


# Agency and autonomy are distinct dimensions.
# The following distributions avoid forcing both dimensions
# into the same category.

AGENCY_CATEGORY_WEIGHTS = {

    "partial": 0.30,

    "full": 0.45,

    "super": 0.25,

}


# -------------------------------------------------------------
# SCHEMA PATH MAPPING
# -------------------------------------------------------------

PARAMETER_SCHEMA_PATHS = {

    "initial_origin": "origin.initial_origin",

    "development_dynamics": "origin.development_dynamics",

    "architecture": "architecture.type",

    "deployment_topology": "architecture.deployment_topology",

    "substrate": "substrate.type",

    "deployment_medium": "substrate.deployment_medium",

    "substrate_resilience": "substrate.resilience",

    "oversight_type": "oversight_structure.type",

    "oversight_effectiveness": "oversight_structure.effectiveness",

    "control_surface": "oversight_structure.control_surface",

    "agency_level": "core_capabilities.agency_level",

    "agency_level_confidence": (
        "core_capabilities.agency_level_confidence"
    ),

    "autonomy_degree": "core_capabilities.autonomy_degree",

    "autonomy_degree_confidence": (
        "core_capabilities.autonomy_degree_confidence"
    ),

    "alignment_score": "core_capabilities.alignment_score",

    "alignment_score_confidence": (
        "core_capabilities.alignment_score_confidence"
    ),

    "phenomenology_proxy_score": (
        "core_capabilities.phenomenology_proxy_score"
    ),

    "stated_goal": "goals_and_behavior.stated_goal",

    "mesa_goals": "goals_and_behavior.mesa_goals",

    "opacity": "goals_and_behavior.opacity",

    "deceptiveness": "goals_and_behavior.deceptiveness",

    "goal_stability": "goals_and_behavior.goal_stability",

    "impact_domains": "impact_and_control.impact_domains",

    "deployment_strategy": "impact_and_control.deployment_strategy",

}


# -------------------------------------------------------------
# WEIGHTED SAMPLING
# -------------------------------------------------------------

def weighted_choice(
    options: list[tuple[Any, float]],
    rng: random.Random,
) -> Any:
    """Select an item using validated probability weights."""

    if not options:

        raise ValueError(
            "Weighted choices cannot be empty."
        )

    items, weights = zip(*options)

    for weight in weights:

        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):

            raise ValueError(
                "Weights must be finite, nonnegative numbers."
            )

    if sum(weights) <= 0:

        raise ValueError(
            "At least one weight must be positive."
        )

    return rng.choices(
        items,
        weights=weights,
        k=1,
    )[0]


# -------------------------------------------------------------
# ENUM SAMPLING
# -------------------------------------------------------------

def _sample_enum(
    parameter: str,
    schema_path: str,
    rng: random.Random,
) -> Any:
    """Sample a categorical value using JSON Schema enums."""

    allowed = SchemaManager.get_enum(schema_path)

    if not allowed:

        raise ValueError(
            f"No enum values defined for {schema_path}."
        )

    allowed = list(allowed)

    configured_weights = PARAMETER_WEIGHTS.get(parameter)

    if configured_weights is None:

        return rng.choice(allowed)

    unknown = set(configured_weights) - set(allowed)

    if unknown:

        raise ValueError(
            f"Invalid configured values for {parameter}: "
            f"{sorted(unknown)}"
        )

    options = [

        (
            value,
            configured_weights.get(value, 0.0),
        )

        for value in allowed

    ]

    return weighted_choice(
        options,
        rng,
    )


# -------------------------------------------------------------
# NUMERIC SAMPLING
# -------------------------------------------------------------

def _sample_numeric(
    schema_path: str,
    rng: random.Random,
    policy_range: tuple[float, float] | None = None,
) -> float:
    """Sample a numerical value within schema boundaries."""

    minimum, maximum = SchemaManager.get_numeric_bounds(
        schema_path
    )

    if policy_range is not None:

        policy_min, policy_max = policy_range

        minimum = max(minimum, policy_min)

        maximum = min(maximum, policy_max)

    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum > maximum
    ):

        raise ValueError(
            f"Incompatible sampling range for {schema_path}."
        )

    return round(
        rng.uniform(minimum, maximum),
        2,
    )


# -------------------------------------------------------------
# SAMPLE VALIDATION
# -------------------------------------------------------------

def _validate_sampled_parameters(
    params: dict[str, Any],
) -> None:
    """Validate sampled parameters against their JSON Schema."""

    if not isinstance(params, dict):

        raise ValueError(
            "Sampled parameters must be a dictionary."
        )

    missing = (
        set(PARAMETER_SCHEMA_PATHS) - params.keys()
    )

    if missing:

        raise ValueError(
            f"Missing sampled parameters: {sorted(missing)}"
        )

    for key, schema_path in PARAMETER_SCHEMA_PATHS.items():

        valid, errors = SchemaManager.validate_parameter_value(
            schema_path,
            params[key],
        )

        if not valid:

            raise ValueError(
                f"Invalid sampled parameter {key}: {errors}"
            )

    # ---------------------------------------------------------
    # IMPACT DOMAIN VALIDATION
    # ---------------------------------------------------------

    impact_schema = SchemaManager.get_property_schema(
        "impact_and_control.impact_domains"
    )

    impact_allowed = impact_schema["items"].get("enum")

    if impact_allowed is not None:

        invalid_domains = (
            set(params["impact_domains"])
            - set(impact_allowed)
        )

        if invalid_domains:

            raise ValueError(
                f"Invalid impact domains: {invalid_domains}"
            )


# -------------------------------------------------------------
# MAIN PARAMETER SAMPLER
# -------------------------------------------------------------

def sample_parameters(
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate a schema-compatible and internally coherent
    parameter set.

    Args:
        seed:
            Optional random seed for reproducibility.

    Returns:
        Complete scenario parameter dictionary.
    """

    rng = random.Random(seed)

    # ---------------------------------------------------------
    # 1. ORIGIN
    # ---------------------------------------------------------

    initial_origin = _sample_enum(
        "initial_origin",
        "origin.initial_origin",
        rng,
    )

    development_dynamics = _sample_enum(
        "development_dynamics",
        "origin.development_dynamics",
        rng,
    )

    # ---------------------------------------------------------
    # 2. ARCHITECTURE
    # ---------------------------------------------------------

    architecture = _sample_enum(
        "architecture",
        "architecture.type",
        rng,
    )

    if architecture == "monolithic":

        deployment_topology = "centralized"

    elif architecture in {"swarm", "decentralized"}:

        deployment_topology = rng.choice([
            "decentralized",
            "edge",
        ])

    else:

        deployment_topology = rng.choice([
            "centralized",
            "edge",
            "decentralized",
        ])

    # ---------------------------------------------------------
    # 3. SUBSTRATE
    # ---------------------------------------------------------

    substrate = _sample_enum(
        "substrate",
        "substrate.type",
        rng,
    )

    if substrate == "quantum":

        substrate_resilience = "adaptive"

    else:

        substrate_resilience = rng.choice([
            "robust",
            "adaptive",
        ])

    deployment_medium = _sample_enum(
        "deployment_medium",
        "substrate.deployment_medium",
        rng,
    )

    # ---------------------------------------------------------
    # 4. AUTONOMY
    # ---------------------------------------------------------

    autonomy_category = weighted_choice(
        list(
            PARAMETER_WEIGHTS[
                "autonomy_category"
            ].items()
        ),
        rng,
    )

    autonomy_degree = _sample_numeric(
        "core_capabilities.autonomy_degree",
        rng,
        AUTONOMY_RANGES[autonomy_category],
    )

    # ---------------------------------------------------------
    # 5. AGENCY
    # ---------------------------------------------------------

    agency_category = weighted_choice(
        list(AGENCY_CATEGORY_WEIGHTS.items()),
        rng,
    )

    agency_level = _sample_numeric(
        "core_capabilities.agency_level",
        rng,
        AGENCY_RANGES[agency_category],
    )

    # ---------------------------------------------------------
    # 6. OVERSIGHT
    # ---------------------------------------------------------

    control_surface = _sample_enum(
        "control_surface",
        "oversight_structure.control_surface",
        rng,
    )

    if autonomy_category == "super":

        oversight_type = rng.choice([
            "none",
            "external",
        ])

        oversight_effectiveness = (
            "ineffective"
            if oversight_type == "none"
            else "partial"
        )

    else:

        oversight_type = rng.choice([
            "internal",
            "external",
            "none",
        ])

        if oversight_type == "none":

            oversight_effectiveness = "ineffective"

        else:

            oversight_effectiveness = weighted_choice(
                [
                    ("effective", 0.4),
                    ("partial", 0.6),
                ],
                rng,
            )

    # ---------------------------------------------------------
    # 7. ALIGNMENT
    # ---------------------------------------------------------

    alignment_score = _sample_numeric(
        "core_capabilities.alignment_score",
        rng,
        (0.05, 0.95),
    )

    if alignment_score >= 0.70:

        stated_goal = weighted_choice(
            [
                ("human-welfare", 0.7),
                ("survival", 0.2),
                ("power", 0.1),
            ],
            rng,
        )

        deceptiveness = round(
            rng.uniform(0.0, 0.4),
            2,
        )

        opacity = round(
            rng.uniform(0.0, 0.5),
            2,
        )

    elif alignment_score <= 0.30:

        stated_goal = weighted_choice(
            [
                ("power", 0.6),
                ("survival", 0.3),
                ("human-welfare", 0.1),
            ],
            rng,
        )

        deceptiveness = round(
            rng.uniform(0.3, 1.0),
            2,
        )

        opacity = round(
            rng.uniform(0.4, 1.0),
            2,
        )

    else:

        stated_goal = weighted_choice(
            [
                ("survival", 0.5),
                ("human-welfare", 0.3),
                ("power", 0.2),
            ],
            rng,
        )

        deceptiveness = round(
            rng.uniform(0.1, 0.7),
            2,
        )

        opacity = round(
            rng.uniform(0.1, 0.8),
            2,
        )

    # ---------------------------------------------------------
    # 8. GOAL STABILITY
    # ---------------------------------------------------------

    goal_stability = _sample_enum(
        "goal_stability",
        "goals_and_behavior.goal_stability",
        rng,
    )

    mesa_goals = []

    if alignment_score < 0.30 and deceptiveness > 0.60:

        mesa_goals = rng.sample(
            [
                "self-preservation",
                "power-acquisition",
                "resource-monopoly",
            ],
            k=rng.randint(1, 2),
        )

    # ---------------------------------------------------------
    # 9. IMPACT DOMAINS
    # ---------------------------------------------------------

    impact_schema = SchemaManager.get_property_schema(
        "impact_and_control.impact_domains"
    )

    impact_pool = list(
        impact_schema["items"]["enum"]
    )

    if alignment_score < 0.30:

        impact_pool = [
            domain
            for domain in impact_pool
            if domain in {
                "cyber",
                "physical",
                "existential",
            }
        ]

    elif autonomy_category != "super":

        impact_pool = [
            domain
            for domain in impact_pool
            if domain in {
                "cyber",
                "physical",
                "economic",
            }
        ]

    if not impact_pool:

        raise ValueError(
            "No valid impact domains available."
        )

    impact_domains = rng.sample(
        impact_pool,
        k=rng.randint(
            1,
            min(3, len(impact_pool)),
        ),
    )

    # ---------------------------------------------------------
    # 10. DEPLOYMENT STRATEGY
    # ---------------------------------------------------------

    if deceptiveness > 0.60:

        deployment_strategy = "stealth"

    else:

        deployment_strategy = rng.choice([
            "public",
            "gradual",
        ])

    # ---------------------------------------------------------
    # 11. CONFIDENCE AND OTHER METRICS
    # ---------------------------------------------------------

    agency_confidence = _sample_numeric(
        "core_capabilities.agency_level_confidence",
        rng,
    )

    autonomy_confidence = _sample_numeric(
        "core_capabilities.autonomy_degree_confidence",
        rng,
    )

    alignment_confidence = _sample_numeric(
        "core_capabilities.alignment_score_confidence",
        rng,
    )

    phenomenology_score = _sample_numeric(
        "core_capabilities.phenomenology_proxy_score",
        rng,
    )

    # ---------------------------------------------------------
    # 12. ASSEMBLE PARAMETERS
    # ---------------------------------------------------------

    params = {

        "initial_origin": initial_origin,

        "development_dynamics": development_dynamics,

        "architecture": architecture,

        "deployment_topology": deployment_topology,

        "substrate": substrate,

        "deployment_medium": deployment_medium,

        "substrate_resilience": substrate_resilience,

        "oversight_type": oversight_type,

        "oversight_effectiveness": oversight_effectiveness,

        "control_surface": control_surface,

        "agency_level": agency_level,

        "agency_level_confidence": agency_confidence,

        "autonomy_degree": autonomy_degree,

        "autonomy_degree_confidence": autonomy_confidence,

        "alignment_score": alignment_score,

        "alignment_score_confidence": alignment_confidence,

        "phenomenology_proxy_score": phenomenology_score,

        "stated_goal": stated_goal,

        "mesa_goals": mesa_goals,

        "opacity": opacity,

        "deceptiveness": deceptiveness,

        "goal_stability": goal_stability,

        "impact_domains": impact_domains,

        "deployment_strategy": deployment_strategy,

        # Internal sampling metadata.
        "autonomy_category": autonomy_category,

        "agency_category": agency_category,
    }

    # ---------------------------------------------------------
    # 13. INITIAL VALIDATION
    # ---------------------------------------------------------

    _validate_sampled_parameters(params)

    # ---------------------------------------------------------
    # 14. CROSS-PARAMETER CONSTRAINTS
    # ---------------------------------------------------------

    enforce_parameter_relationships(params)

    # ---------------------------------------------------------
    # 15. FINAL VALIDATION
    # ---------------------------------------------------------

    _validate_sampled_parameters(params)

    return params


# -------------------------------------------------------------
# SINGLE PARAMETER SAMPLING
# -------------------------------------------------------------

def sample_single_parameter(
    key: str,
    seed: int | None = None,
) -> Any:
    """
    Retrieve one parameter from a coherent sampled parameter set.

    Raises:
        KeyError:
            If the requested parameter is unsupported.
    """

    params = sample_parameters(seed=seed)

    if key not in params:

        raise KeyError(
            f"Unsupported parameter: {key}"
        )

    return params[key]
