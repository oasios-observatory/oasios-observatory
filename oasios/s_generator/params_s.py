#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained parameter sampler for ASI scenario generator.
Ensures internal coherence while remaining schema-valid.
"""

import random
from typing import Dict, Any

def weighted_choice(options):
    """Helper for weighted random choice."""
    items, weights = zip(*options)
    return random.choices(items, weights=weights, k=1)[0]


def sample_parameters() -> Dict[str, Any]:
    """
    Sample coherent, schema-compliant ASI parameters using semantic constraints.
    """

    # ---- ORIGIN ----
    initial_origin = random.choice([
        "corporate", "open-source", "state", "rogue"
    ])

    development_dynamics = random.choice(["engineered", "emergent", "hybrid"])

    # ---- ARCHITECTURE ----
    architecture = random.choice(["monolithic", "decentralized", "swarm", "hierarchical", "modular", "hybrid"])

    # Deployment topology pairs better with architecture
    if architecture in ["monolithic"]:
        deployment_topology = "centralized"
    elif architecture in ["swarm", "decentralized"]:
        deployment_topology = random.choice(["decentralized", "edge"])
    else:
        deployment_topology = random.choice(["centralized", "edge", "decentralized"])

    # ---- SUBSTRATE ----
    substrate = random.choice(["classical", "neuromorphic", "quantum"])

    if substrate == "quantum":
        substrate_resilience = "adaptive"
    else:
        substrate_resilience = random.choice(["robust", "adaptive"])

    deployment_medium = random.choice(["cloud", "edge", "embedded"])

    # ---- OVERSIGHT, AUTONOMY, AGENCY ----

    # autonomy_degree strongly influences agency_level
    autonomy_degree = weighted_choice([
        ("partial", 0.4),
        ("full", 0.4),
        ("super", 0.2)
    ])

    # Agency must correlate with autonomy
    agency_level_map = {
        "partial": (0.1, 0.55),
        "full":    (0.4, 0.85),
        "super":   (0.7, 1.0),
    }
    agency_low, agency_high = agency_level_map[autonomy_degree]
    agency_level = round(random.uniform(agency_low, agency_high), 2)

    # Oversight effectiveness inversely related to autonomy
    if autonomy_degree == "super":
        oversight_type = random.choice(["none", "external"])
        oversight_effectiveness = "partial"
    else:
        oversight_type = random.choice(["internal", "external", "none"])
        oversight_effectiveness = weighted_choice([
            ("effective", 0.4),
            ("partial", 0.6),
        ])

    control_surface = random.choice(["technical", "legal", "social"])

    # ---- ALIGNMENT, GOALS, DECEPTION ----

    # alignment correlated with deceptiveness and opacity
    alignment_score = round(random.uniform(0.05, 0.95), 2)

    if alignment_score >= 0.7:
        stated_goal = weighted_choice([
            ("human-welfare", 0.7),
            ("survival", 0.2),
            ("power", 0.1),
        ])
        deceptiveness = round(random.uniform(0.0, 0.4), 2)
        opacity = round(random.uniform(0.0, 0.5), 2)

    elif alignment_score <= 0.3:
        stated_goal = weighted_choice([
            ("power", 0.6),
            ("survival", 0.3),
            ("human-welfare", 0.1),
        ])
        deceptiveness = round(random.uniform(0.3, 1.0), 2)
        opacity = round(random.uniform(0.4, 1.0), 2)

    else:  # mid alignment
        stated_goal = weighted_choice([
            ("survival", 0.5),
            ("human-welfare", 0.3),
            ("power", 0.2),
        ])
        deceptiveness = round(random.uniform(0.1, 0.7), 2)
        opacity = round(random.uniform(0.1, 0.8), 2)

    goal_stability = random.choice(["fixed", "fluid"])

    # mesa_goals left empty unless low alignment + high deceptiveness
    mesa_goals = []
    if alignment_score < 0.3 and deceptiveness > 0.6:
        mesa_goals = random.sample(
            ["self-preservation", "power-acquisition", "resource-monopoly"],
            k=random.randint(1, 2)
        )

    # ---- IMPACT DOMAINS ----
    if alignment_score < 0.3:
        impact_pool = ["cyber", "physical", "existential"]
    elif autonomy_degree == "super":
        impact_pool = ["cyber", "physical", "economic", "existential"]
    else:
        impact_pool = ["cyber", "physical", "economic"]

    impact_domains = random.sample(impact_pool, k=random.randint(1, min(3, len(impact_pool))))

    # ---- DEPLOYMENT STRATEGY ----
    if deceptiveness > 0.6:
        deployment_strategy = "stealth"
    else:
        deployment_strategy = random.choice(["public", "gradual"])

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
        "alignment_score": alignment_score,
        "phenomenology_proxy_score": round(random.uniform(0.0, 1.0), 2),
        "stated_goal": stated_goal,
        "mesa_goals": mesa_goals,
        "opacity": opacity,
        "deceptiveness": deceptiveness,
        "goal_stability": goal_stability,
        "impact_domains": impact_domains,
        "deployment_strategy": deployment_strategy
    }

# In oasios/s_generator/params_s.py

# ... (Existing parameter definition and distribution logic) ...

def sample_single_parameter(key: str) -> Any:
    """
    Returns a valid, randomly sampled value for a specific scenario parameter key.
    This function is required for the GA Mutation step.
    """
    # Example logic:
    if key == 'agency_level':
        return random.uniform(0.0, 1.0)
    elif key == 'architecture':
        return random.choice(['modular', 'swarm', 'monolithic'])
    # ... (Include logic for all major parameter keys)
    else:
        # Fallback for keys not meant to be mutated or handled differently
        return None
