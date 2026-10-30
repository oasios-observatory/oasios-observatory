# oasios/tracker/scenario_interface_t.py

import structlog

from typing import Dict, Any, List

from .database_t import DatabaseManager

log = structlog.get_logger()


class ScenarioInterface:

    def __init__(self):

        self.db = DatabaseManager()

        log.info("scenario_interface_initialized")

    def get_top_k_active_groups(
        self,
        k: int = 5,
    ) -> List[Dict[str, Any]]:

        if k < 1:
            raise ValueError("k must be greater than zero")

        k = min(k, 100)

        groups = self.db.get_top_groups_by_emergence(k)

        log.info(
            "top_groups_retrieved",
            count=len(groups),
        )

        return groups

    def generate_scenario_seed(
        self,
        group_id: str,
    ) -> Dict[str, Any]:

        group = self.db.get_group_details(group_id)

        if not group:

            log.warning(
                "scenario_group_not_found",
                group_id=group_id,
            )

            return {}

        epsilon = float(
            group.get("emergence_index_epsilon") or 0
        )

        kappa = float(
            group.get("coherence_kappa") or 0
        )

        xi = float(
            group.get("cross_domain_span_xi") or 0
        )

        primary_type = group.get(
            "primary_type",
            "unknown",
        )

        # Validate stored metrics.

        for name, value in [
            ("epsilon", epsilon),
            ("kappa", kappa),
            ("xi", xi),
        ]:

            if not 0 <= value <= 1:

                raise ValueError(
                    f"Invalid group metric: {name}={value}"
                )

        # Scenario variables.

        trajectory_magnitude = (
            epsilon * 0.7
            + xi * 0.3
        )

        trajectory_speed = epsilon * 10

        systemic_risk = xi * (1 - kappa)

        transition_type = (
            "Continuous_Growth"
            if kappa > 0.7
            else "Discontinuity_Potential"
        )

        seed = {

            "seed_id": f"SCENARIO-{group_id[:8]}",

            "source_group_id": group_id,

            "core_stressor": primary_type,

            "scenario_variables": {

                "V1_Transition_Magnitude": round(
                    trajectory_magnitude, 3
                ),

                "V2_Transition_Speed": round(
                    trajectory_speed, 2
                ),

                "V3_Systemic_Risk_Index": round(
                    systemic_risk, 3
                ),

                "V4_Transition_Type": transition_type,

                "V5_Dominant_Capability_Theme": (
                    self._get_capability_theme(primary_type)
                ),
            },

            "source_metrics": {

                "emergence_index_epsilon": epsilon,

                "coherence_kappa": kappa,

                "cross_domain_span_xi": xi,
            },

            "governance_note": (
                "Hypothetical scenario initialization. "
                "Metrics represent heuristic research indicators, "
                "not predictions of future events."
            ),
        }

        self.db.log_access(
            "anomaly_groups",
            group_id,
            "Read",
            "Scenario Generation",
            "SIL_Interface",
        )

        log.info(
            "scenario_seed_generated",
            seed_id=seed["seed_id"],
        )

        return seed

    @staticmethod
    def _get_capability_theme(primary_type: str) -> str:

        mapping = {

            "structural":
                "C2_Manipulation_and_Tooling_Control",

            "behavioral":
                "C4_Autonomy_and_Sustained_Pursuit",

            "temporal":
                "D1_Innovation_Acceleration_Pacing",

            "unknown":
                "C5_Social_and_Ecological_Interaction",
        }

        return mapping.get(
            primary_type,
            "Conceptual_Shift",
        )

    def generate_policy_report(
        self,
        group_id: str,
    ) -> str:

        seed = self.generate_scenario_seed(group_id)

        if not seed:

            return "Error: Group not found."

        variables = seed["scenario_variables"]

        return f"""
# OASIOS Policy Foresight Report

## Scenario Identification

- Scenario ID: {seed['seed_id']}
- Source Group: {group_id}
- Primary Pattern: {seed['core_stressor']}

## Capability Domain

{variables['V5_Dominant_Capability_Theme']}

## Transition Metrics

| Metric | Value |
|---|---:|
| Transition Magnitude | {variables['V1_Transition_Magnitude']} |
| Transition Speed | {variables['V2_Transition_Speed']}/10 |
| Systemic Risk Index | {variables['V3_Systemic_Risk_Index']} |
| Emergence Index | {seed['source_metrics']['emergence_index_epsilon']} |
| Coherence | {seed['source_metrics']['coherence_kappa']} |
| Cross-Domain Span | {seed['source_metrics']['cross_domain_span_xi']} |

## Transition Classification

{variables['V4_Transition_Type']}

## Policy Considerations

Policy simulations should examine:

1. Coordination and governance mechanisms.
2. Infrastructure resilience.
3. Human oversight and accountability.
4. Cross-domain dependencies.
5. Potential capability-transition pathways.

## Governance Qualification

{seed['governance_note']}

This report is a research-support artifact.
It must not be interpreted as a validated forecast,
probability estimate, or definitive risk assessment.
"""
