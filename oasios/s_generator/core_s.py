#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/core_s.py

"""Core pipeline for speculative ASI scenario generation."""

import math
import uuid

from datetime import datetime, timezone
from typing import Any, Callable

from oasios.common.consistency import NarrativeChecker
from oasios.common.schema import SchemaManager
from oasios.common.storage import (
    init_db,
    save_asi_scenario,
)
from oasios.common.llm_client import generate_narrative
from oasios.common.timeline import generate_scenario_timeline
from oasios.logger import log

from oasios.s_generator.abbreviator_s import abbreviate
from oasios.s_generator.params_s import sample_parameters


# -------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------

class ScenarioGenerationError(RuntimeError):
    """Raised when a generation pipeline stage fails."""

    def __init__(
        self,
        stage: str,
        message: str,
    ) -> None:

        super().__init__(message)

        self.stage = stage


# -------------------------------------------------------------
# REQUIRED PARAMETERS
# -------------------------------------------------------------

REQUIRED_PARAMETERS = {
    "initial_origin",
    "development_dynamics",
    "architecture",
    "deployment_topology",
    "substrate",
    "deployment_medium",
    "substrate_resilience",
    "oversight_type",
    "oversight_effectiveness",
    "control_surface",
    "agency_level",
    "agency_level_confidence",
    "autonomy_degree",
    "autonomy_degree_confidence",
    "alignment_score",
    "alignment_score_confidence",
    "phenomenology_proxy_score",
    "stated_goal",
    "mesa_goals",
    "opacity",
    "deceptiveness",
    "goal_stability",
    "impact_domains",
    "deployment_strategy",
}


NUMERIC_PARAMETERS = (
    "agency_level",
    "agency_level_confidence",
    "autonomy_degree",
    "autonomy_degree_confidence",
    "alignment_score",
    "alignment_score_confidence",
    "phenomenology_proxy_score",
    "opacity",
    "deceptiveness",
)


STRING_PARAMETERS = (
    "initial_origin",
    "development_dynamics",
    "architecture",
    "deployment_topology",
    "substrate",
    "deployment_medium",
    "substrate_resilience",
    "oversight_type",
    "oversight_effectiveness",
    "control_surface",
    "stated_goal",
    "goal_stability",
    "deployment_strategy",
)


COLLECTION_PARAMETERS = (
    "mesa_goals",
    "impact_domains",
)


PROMPT_VERSION = "v2"
CHECKER_VERSION = "v2"


# -------------------------------------------------------------
# PIPELINE STAGE EXECUTION
# -------------------------------------------------------------

def _run_stage(
    stage: str,
    operation: Callable[[], Any],
    scenario_id: str | None = None,
) -> Any:
    """Execute a pipeline operation with error reporting."""

    try:

        return operation()

    except ScenarioGenerationError:

        raise

    except Exception as exc:

        log.error(
            "scenario.stage.failed",
            stage=stage,
            scenario_id=scenario_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )

        raise ScenarioGenerationError(
            stage,
            f"Scenario generation failed during {stage}: {exc}",
        ) from exc


# -------------------------------------------------------------
# PARAMETER VALIDATION
# -------------------------------------------------------------

def _validate_parameters(
    params: dict[str, Any],
) -> None:
    """
    Validate sampled scenario parameters.

    Empty mesa_goals are permitted.

    Empty impact_domains are prohibited.
    """

    if not isinstance(params, dict):

        raise ScenarioGenerationError(
            "parameter_sampling",
            "Parameter sampler must return a dictionary.",
        )

    # ---------------------------------------------------------
    # REQUIRED PARAMETERS
    # ---------------------------------------------------------

    missing = REQUIRED_PARAMETERS - params.keys()

    if missing:

        raise ScenarioGenerationError(
            "parameter_validation",
            f"Missing parameters: {sorted(missing)}",
        )

    # ---------------------------------------------------------
    # NUMERIC PARAMETERS
    # ---------------------------------------------------------

    for key in NUMERIC_PARAMETERS:

        value = params[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):

            raise ScenarioGenerationError(
                "parameter_validation",
                f"Invalid numeric parameter: {key}",
            )

    # ---------------------------------------------------------
    # STRING PARAMETERS
    # ---------------------------------------------------------

    for key in STRING_PARAMETERS:

        value = params[key]

        if (
            not isinstance(value, str)
            or not value.strip()
        ):

            raise ScenarioGenerationError(
                "parameter_validation",
                f"Invalid categorical parameter: {key}",
            )

    # ---------------------------------------------------------
    # COLLECTION PARAMETERS
    # ---------------------------------------------------------

    for key in COLLECTION_PARAMETERS:

        value = params[key]

        if not isinstance(
            value,
            (list, tuple, set),
        ):

            raise ScenarioGenerationError(
                "parameter_validation",
                f"Invalid collection parameter: {key}",
            )

        # Mesa-goals are optional.
        # Impact domains are mandatory.

        if key == "impact_domains" and not value:

            raise ScenarioGenerationError(
                "parameter_validation",
                "Impact domains cannot be empty.",
            )

        # Validate collection contents.

        if any(
            not isinstance(item, str)
            or not item.strip()
            for item in value
        ):

            raise ScenarioGenerationError(
                "parameter_validation",
                f"Invalid collection values: {key}",
            )

    log.debug(
        "scenario.parameters.validated",
        parameter_count=len(params),
        mesa_goals_count=len(params["mesa_goals"]),
        impact_domains_count=len(params["impact_domains"]),
    )


# -------------------------------------------------------------
# TIMELINE VALIDATION
# -------------------------------------------------------------

def _validate_timeline(
    timeline: Any,
) -> None:
    """Validate generated timeline structure."""

    if not isinstance(timeline, list) or not timeline:

        raise ScenarioGenerationError(
            "timeline_validation",
            "Timeline must be a nonempty list.",
        )

    for index, phase in enumerate(timeline):

        if not isinstance(phase, dict):

            raise ScenarioGenerationError(
                "timeline_validation",
                f"Timeline phase {index} must be a dictionary.",
            )

        if "phase" not in phase or "years" not in phase:

            raise ScenarioGenerationError(
                "timeline_validation",
                f"Timeline phase {index} requires phase and years.",
            )

        if (
            not isinstance(phase["phase"], str)
            or not phase["phase"].strip()
        ):

            raise ScenarioGenerationError(
                "timeline_validation",
                f"Timeline phase {index} has an invalid phase name.",
            )

        years = phase["years"]

        if years is None or (
            isinstance(years, str) and not years.strip()
        ):

            raise ScenarioGenerationError(
                "timeline_validation",
                f"Timeline phase {index} has invalid years.",
            )


# -------------------------------------------------------------
# MAIN SCENARIO GENERATION
# -------------------------------------------------------------

def generate_scenario() -> dict[str, Any] | None:
    """
    Generate, validate, and persist one scenario.

    Returns:
        Complete scenario dictionary on success.
        None when quality or schema validation rejects it.

    Raises:
        ScenarioGenerationError:
            When an operational pipeline stage fails.
    """

    scenario_id = str(uuid.uuid4())

    log.info(
        "scenario.generation.started",
        scenario_id=scenario_id,
    )

    # ---------------------------------------------------------
    # 1. DATABASE INITIALIZATION
    # ---------------------------------------------------------

    _run_stage(
        "storage_initialization",
        init_db,
        scenario_id,
    )

    # ---------------------------------------------------------
    # 2. PARAMETER SAMPLING
    # ---------------------------------------------------------

    params = _run_stage(
        "parameter_sampling",
        sample_parameters,
        scenario_id,
    )

    _validate_parameters(params)

    log.debug(
        "scenario.parameters.sampled",
        scenario_id=scenario_id,
        count=len(params),
    )

    # ---------------------------------------------------------
    # 3. TITLE GENERATION
    # ---------------------------------------------------------

    title = _run_stage(
        "title_generation",
        lambda: abbreviate(params),
        scenario_id,
    )

    if not isinstance(title, str) or not title.strip():

        raise ScenarioGenerationError(
            "title_generation",
            "Invalid scenario title.",
        )

    # ---------------------------------------------------------
    # 4. TIMELINE GENERATION
    # ---------------------------------------------------------

    timeline_phases = _run_stage(
        "timeline_generation",
        lambda: generate_scenario_timeline(
            params=params,
            precursor_density=0.0,
            dominant_tags=[],
        ),
        scenario_id,
    )

    _validate_timeline(timeline_phases)

    log.info(
        "scenario.timeline.generated",
        scenario_id=scenario_id,
        phases=len(timeline_phases),
    )

    # ---------------------------------------------------------
    # 5. NARRATIVE GENERATION
    # ---------------------------------------------------------

    success, narrative, model_used = _run_stage(
        "narrative_generation",
        lambda: generate_narrative(
            title=title,
            params=params,
            timeline=timeline_phases,
        ),
        scenario_id,
    )

    if not success:

        raise ScenarioGenerationError(
            "narrative_generation",
            str(narrative),
        )

    if not isinstance(narrative, str) or not narrative.strip():

        log.warning(
            "scenario.narrative_generation.rejected",
            scenario_id=scenario_id,
            reason="empty_narrative",
        )

        return None

    # ---------------------------------------------------------
    # 6. CONSISTENCY VALIDATION
    # ---------------------------------------------------------

    checker = _run_stage(
        "consistency_checker_initialization",
        lambda: NarrativeChecker(
            params=params,
            timeline=timeline_phases,
        ),
        scenario_id,
    )

    consistent, failures = _run_stage(
        "consistency_check",
        lambda: checker.check(narrative),
        scenario_id,
    )

    if not consistent:

        log.warning(
            "scenario.consistency_check.rejected",
            scenario_id=scenario_id,
            failures=failures,
        )

        return None

    # ---------------------------------------------------------
    # 7. BUILD COMPLETE SCENARIO
    # ---------------------------------------------------------

    now = datetime.now(timezone.utc).isoformat()

    scenario: dict[str, Any] = {

        "id": scenario_id,

        "title": title,

        "metadata": {

            "created": now,
            "last_updated": now,
            "version": 1,
            "source": "generated",

            "provenance": {

                "model_used": model_used or "unknown",

                "generation_parameters": {

                    "prompt_version": PROMPT_VERSION,

                },
            },
        },

        "origin": {

            "initial_origin": params["initial_origin"],

            "development_dynamics": (
                params["development_dynamics"]
            ),
        },

        "architecture": {

            "type": params["architecture"],

            "deployment_topology": (
                params["deployment_topology"]
            ),
        },

        "substrate": {

            "type": params["substrate"],

            "deployment_medium": (
                params["deployment_medium"]
            ),

            "resilience": params["substrate_resilience"],
        },

        "oversight_structure": {

            "type": params["oversight_type"],

            "effectiveness": (
                params["oversight_effectiveness"]
            ),

            "control_surface": params["control_surface"],
        },

        "core_capabilities": {

            "agency_level": params["agency_level"],

            "agency_level_confidence": (
                params["agency_level_confidence"]
            ),

            "autonomy_degree": params["autonomy_degree"],

            "autonomy_degree_confidence": (
                params["autonomy_degree_confidence"]
            ),

            "alignment_score": params["alignment_score"],

            "alignment_score_confidence": (
                params["alignment_score_confidence"]
            ),

            "phenomenology_proxy_score": (
                params["phenomenology_proxy_score"]
            ),
        },

        "goals_and_behavior": {

            "stated_goal": params["stated_goal"],

            "mesa_goals": params["mesa_goals"],

            "opacity": params["opacity"],

            "deceptiveness": params["deceptiveness"],

            "goal_stability": params["goal_stability"],
        },

        "impact_and_control": {

            "impact_domains": params["impact_domains"],

            "deployment_strategy": (
                params["deployment_strategy"]
            ),
        },

        "scenario_content": {

            "title": title,

            "narrative": narrative,

            "timeline": {
                "phases": timeline_phases,
            },
        },

        "quantitative_assessment": {

            "probability": {

                "emergence_probability": 0.4,

                "detection_confidence": 0.5,

                "projection_confidence": 0.6,

                "trend": "stable",

                "last_update_reason": (
                    "provisional initial values"
                ),
            },

            "risk_assessment": {

                "existential": {

                    "score": 5,

                    "weight": 0.7,
                },
            },
        },

        "observable_evidence": {

            "key_indicators": [],

            "supporting_signals": [],
        },
    }

    # ---------------------------------------------------------
    # 8. SCHEMA VALIDATION
    # ---------------------------------------------------------

    valid, errors = _run_stage(
        "schema_validation",
        lambda: SchemaManager.validate(scenario),
        scenario_id,
    )

    if not valid:

        log.warning(
            "scenario.schema_validation.rejected",
            scenario_id=scenario_id,
            errors=errors,
        )

        return None

    # ---------------------------------------------------------
    # 9. PERSISTENCE
    # ---------------------------------------------------------

    scenario["metadata"]["last_updated"] = (
        datetime.now(timezone.utc).isoformat()
    )

    _run_stage(
        "persistence",
        lambda: save_asi_scenario(
            scenario=scenario,
            table_name="s_scenarios",
        ),
        scenario_id,
    )

    # ---------------------------------------------------------
    # 10. SUCCESS
    # ---------------------------------------------------------

    log.info(
        "scenario.generation.succeeded",
        scenario_id=scenario_id,
        title=title,
        model=model_used,
    )

    return scenario
