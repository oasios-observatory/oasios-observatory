# oasios/s_generator/params_s.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained parameter sampler for ASI scenario generation.

Generates schema-compatible parameters while maintaining
basic relationships between autonomy, agency, alignment,
oversight, opacity, and deceptiveness.
"""

import random
from typing import Any


def weighted_choice(options):
    """Return an item selected according to the supplied weights."""
    items, weights = zip(*options)
    return random.choices(items, weights=weights, k=1)[0]


def sample_parameters() -> dict[str, Any]:
    """Generate a coherent set of scenario parameters."""

    # ---------------------------------------------------------
    # ORIGIN
    # ---------------------------------------------------------

    initial_origin = random.choice([
            "corporate",
            "open-source",
            "state",
            "military",
            "academic",
            "rogue",
            "individual",
            "accidental",
        ]
    )

    development_dynamics = random.choice([
        "engineered",
        "emergent",
        "hybrid",
    ])

    # ---------------------------------------------------------
    # ARCHITECTURE
    # ---------------------------------------------------------

    architecture = random.choice([
        "monolithic",
        "decentralized",
        "swarm",
        "hierarchical",
        "modular",
        "hybrid",
    ])

    if architecture == "monolithic":
        deployment_topology = "centralized"

    elif architecture in {"swarm", "decentralized"}:
        deployment_topology = random.choice(
            [
                "decentralized",
                "distributed",
                "edge",
            ]
        )

    else:
        deployment_topology = random.choice(
            [
                "centralized",
                "edge",
                "decentralized",
                "distributed",
                "hybrid",
            ]
        )

    # ---------------------------------------------------------
    # SUBSTRATE
    # ---------------------------------------------------------

    substrate = random.choice(
        [
            "classical",
            "neuromorphic",
            "quantum",
            "biological",
            "hybrid",
            "unknown",
        ]
    )

    substrate_resilience = random.choice(
        [
            "robust",
            "fragile",
            "adaptive",
            "unknown",
        ]
    )

    deployment_medium = random.choice([
        "cloud",
        "edge",
        "embedded",
    ])

    # ---------------------------------------------------------
    # AUTONOMY AND AGENCY
    # ---------------------------------------------------------

    autonomy_category = weighted_choice([
        ("partial", 0.4),
        ("full", 0.4),
        ("super", 0.2),
    ])

    autonomy_ranges = {
        "partial": (0.00, 0.39),
        "full": (0.40, 0.74),
        "super": (0.75, 1.00),
    }

    autonomy_low, autonomy_high = autonomy_ranges[
        autonomy_category
    ]

    autonomy_degree = round(
        random.uniform(autonomy_low, autonomy_high),
        2,
    )

    agency_ranges = {
        "partial": (0.10, 0.55),
        "full": (0.40, 0.85),
        "super": (0.70, 1.00),
    }

    agency_low, agency_high = agency_ranges[
        autonomy_category
    ]

    agency_level = round(
        random.uniform(agency_low, agency_high),
        2,
    )

    # ---------------------------------------------------------
    # OVERSIGHT
    # ---------------------------------------------------------

    control_surface = random.choice([
        "technical",
        "legal",
        "social",
    ])

    if autonomy_category == "super":
        oversight_type = random.choice([
            "none",
            "external",
        ])

        oversight_effectiveness = (
            "ineffective"
            if oversight_type == "none"
            else "partial"
        )

    else:
        oversight_type = random.choice([
            "internal",
            "external",
            "none",
        ])

        if oversight_type == "none":
            oversight_effectiveness = "ineffective"
        else:
            oversight_effectiveness = weighted_choice([
                ("effective", 0.4),
                ("partial", 0.6),
            ])

    # ---------------------------------------------------------
    # ALIGNMENT, GOALS, AND DECEPTION
    # ---------------------------------------------------------

    alignment_score = round(
        random.uniform(0.05, 0.95),
        2,
    )

    if alignment_score >= 0.7:
        stated_goal = weighted_choice([
            ("human-welfare", 0.7),
            ("survival", 0.2),
            ("power", 0.1),
        ])

        deceptiveness = round(
            random.uniform(0.0, 0.4),
            2,
        )

        opacity = round(
            random.uniform(0.0, 0.5),
            2,
        )

    elif alignment_score <= 0.3:
        stated_goal = weighted_choice([
            ("power", 0.6),
            ("survival", 0.3),
            ("human-welfare", 0.1),
        ])

        deceptiveness = round(
            random.uniform(0.3, 1.0),
            2,
        )

        opacity = round(
            random.uniform(0.4, 1.0),
            2,
        )

    else:
        stated_goal = weighted_choice([
            ("survival", 0.5),
            ("human-welfare", 0.3),
            ("power", 0.2),
        ])

        deceptiveness = round(
            random.uniform(0.1, 0.7),
            2,
        )

        opacity = round(
            random.uniform(0.1, 0.8),
            2,
        )

    goal_stability = random.choice([
        "fixed",
        "fluid",
    ])

    mesa_goals = []

    if alignment_score < 0.3 and deceptiveness > 0.6:
        mesa_goals = random.sample(
            [
                "self-preservation",
                "power-acquisition",
                "resource-monopoly",
            ],
            k=random.randint(1, 2),
        )

    # ---------------------------------------------------------
    # IMPACT DOMAINS
    # ---------------------------------------------------------

    if alignment_score < 0.3:
        impact_pool = [
            "cyber",
            "physical",
            "existential",
        ]

    elif autonomy_category == "super":
        impact_pool = [
            "cyber",
            "physical",
            "economic",
            "existential",
        ]

    else:
        impact_pool = [
            "cyber",
            "physical",
            "economic",
        ]

    impact_domains = random.sample(
        impact_pool,
        k=random.randint(
            1,
            min(3, len(impact_pool)),
        ),
    )

    # ---------------------------------------------------------
    # DEPLOYMENT STRATEGY
    # ---------------------------------------------------------

    if deceptiveness > 0.6:
        deployment_strategy = random.choices(
            ["stealth", "gradual", "rapid"],
            weights=[0.7, 0.2, 0.1],
            k=1,
        )[0]

    else:
        deployment_strategy = random.choice(
            [
                "public",
                "gradual",
                "rapid",
                "containment",
            ]
        )

    # ---------------------------------------------------------
    # RETURN PARAMETERS
    # ---------------------------------------------------------

    return {
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
        "autonomy_degree": autonomy_degree,
        "autonomy_category": autonomy_category,
        "alignment_score": alignment_score,
        "agency_level_confidence": 0.7,
        "autonomy_degree_confidence": 0.7,
        "alignment_score_confidence": 0.6,
        "phenomenology_proxy_score": round(
            random.uniform(0.0, 1.0),
            2,
        ),
        "stated_goal": stated_goal,
        "mesa_goals": mesa_goals,
        "opacity": opacity,
        "deceptiveness": deceptiveness,
        "goal_stability": goal_stability,
        "impact_domains": impact_domains,
        "deployment_strategy": deployment_strategy,
    }


def sample_single_parameter(key: str) -> Any:
    """
    Sample one parameter for genetic algorithm mutation.

    Returns None for unsupported keys.
    """

    # Sample a complete coherent parameter set, then retrieve
    # the requested value. This ensures the returned value has
    # the correct type and valid range.
    params = sample_parameters()

    if key in params:
        return params[key]

    # Retain support for the categorical autonomy label.
    if key == "autonomy_category":
        return params["autonomy_category"]

    return None