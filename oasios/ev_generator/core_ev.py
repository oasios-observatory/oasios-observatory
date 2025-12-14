#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/ev_generator/core_ev.py
"""
Core logic for generating signal-informed (EV) ASI scenarios.
Version 4.2: Integrated robust parameter type pre-check and params_ev fixes.
"""
import uuid
# ... (imports remain the same) ...
import json
import pandas as pd
import sqlite3
import numpy as np
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path

# --- Configuration & Path Resolution ---
# ... (Configuration & MOCKING EXTERNAL DEPENDENCIES remain the same) ...
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "asi_precursors.db"

# --- Imports and Mocking for Robustness ---
try:
    from oasios.config import settings
    V3_DB_PATH = settings.v3_db_path
except ImportError:
    class MockSettings:
        v3_db_path = DEFAULT_DB_PATH
    settings = MockSettings()
    V3_DB_PATH = DEFAULT_DB_PATH
    print(f"NOTICE: oasios.config not found. Using relative DB path: {V3_DB_PATH}")

if not V3_DB_PATH.exists():
    print(f"WARNING: Database file not found at: {V3_DB_PATH}")
    print("       Ensure you are running this from the project root or data exists.")

# --- MOCKING EXTERNAL DEPENDENCIES ---
try:
    from oasios.common.schema import SchemaManager
    from oasios.logger import log
    from oasios.common.storage import save_scenario_ev, init_db
    from oasios.ev_generator.params_ev import sample_signal_parameters, FeatureInfluenceModel
    from oasios.common.llm_client import generate_narrative
    from oasios.common.timeline import generate_scenario_timeline
    from oasios.common.consistency import NarrativeChecker
    from oasios.ev_generator.abbreviator_ev import abbreviate_ev
except ImportError as import_error:
    print(f"CRITICAL: Missing core module dependency: {import_error}")

    class MockSchemaManager:
        @staticmethod
        def validate(data): return True, None
    SchemaManager = MockSchemaManager

    class MockLogger:
        @staticmethod
        def error(msg, **kwargs): print(f"ERROR: {msg} {kwargs}")
        @staticmethod
        def info(msg, **kwargs): print(f"INFO: {msg} {kwargs}")
        @staticmethod
        def warning(msg, **kwargs): print(f"WARNING: {msg} {kwargs}")
        @staticmethod
        def debug(msg, **kwargs): pass
    log = MockLogger()

    def save_scenario_ev(*args, **kwargs):
        log.warning("storage.mocked", msg="Save function is mocked.")

    def init_db():
        log.warning("db.mocked", msg="Init DB function is mocked.")

    def sample_signal_parameters(*args, **kwargs):
        return {}

    class MockFeatureInfluenceModel:
        FEATURE_MAPPING = {}
    FeatureInfluenceModel = MockFeatureInfluenceModel

    def generate_narrative(*args, **kwargs):
        return False, "Mocked narrative failure", "mock"

    def generate_scenario_timeline(*args, **kwargs):
        return []

    class MockNarrativeChecker:
        def __init__(self, *args): pass
        def check(self, narrative): return True, []
    NarrativeChecker = MockNarrativeChecker

    def abbreviate_ev(params):
        return "MOCKED-TITLE"

# --- DB Connection Helper ---
def get_conn():
    """Returns a direct connection to the V3 Tracker database file."""
    if not V3_DB_PATH.is_file():
        log.error("V3 Tracker DB file not found", path=str(V3_DB_PATH))
        raise FileNotFoundError(f"The database ({V3_DB_PATH.name}) is missing at {V3_DB_PATH}.")
    conn = sqlite3.connect(str(V3_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# --- DYNAMIC PROBABILITY CALIBRATION ---
PROBABILITY_COEFS = {
    "intercept": -4.0,
    "agency_level": 2.5,
    "opacity": 1.2,
    "deceptiveness": 1.0,
    "alignment_score": -3.5,
}

def calculate_emergence_probability(params: Dict[str, Any]) -> float:
    logit = PROBABILITY_COEFS["intercept"]
    logit += PROBABILITY_COEFS["agency_level"] * params.get("agency_level", 0.5)
    logit += PROBABILITY_COEFS["opacity"] * params.get("opacity", 0.5)
    logit += PROBABILITY_COEFS["deceptiveness"] * params.get("deceptiveness", 0.5)
    logit += PROBABILITY_COEFS["alignment_score"] * params.get("alignment_score", 0.5)
    prob = 1 / (1 + np.exp(-logit))
    return float(np.clip(prob, 0.001, 0.999))

# -----------------------------
# New Parameter Validation Logic
# -----------------------------
# FIX 1: Add 'autonomy_degree' back to the critical fields list
CRITICAL_NUMERIC_FIELDS = ['autonomy_degree', 'agency_level', 'alignment_score', 'opacity', 'deceptiveness']

def _validate_critical_numeric_types(raw_params: dict) -> Tuple[bool, dict]:
    """
    Checks and converts critical numeric fields. Returns False if a non-numeric
    string is found for a required float field.
    """
    validated_params = raw_params.copy()
    for key in CRITICAL_NUMERIC_FIELDS:
        value = validated_params.get(key)
        if isinstance(value, str):
            try:
                validated_params[key] = float(value)
            except ValueError:
                log.error("core_ev.param_fatal_error", param=key, value=value,
                          reason="Non-numeric string found for critical float field.")
                return False, raw_params
        elif value is None:
            log.error("core_ev.param_fatal_error", param=key, value=None, reason="Missing critical numeric field.")
            return False, raw_params
    return True, validated_params

# -----------------------------
# Core Logic
# -----------------------------
def fetch_raw_features_for_generation(limit: int = 100) -> List[Dict[str, Any]]:
    query = f"""
        SELECT feature_vector
        FROM extracted_features
        ORDER BY ROWID DESC
        LIMIT {limit}
    """
    try:
        with closing(get_conn()) as conn:
            raw_data = pd.read_sql_query(query, conn)
    except FileNotFoundError:
        raise
    except Exception as db_error:
        log.error("DB error fetching features", error=str(db_error), query=query.strip())
        return []

    features = []
    for _, row in raw_data.iterrows():
        try:
            features.append(json.loads(row['feature_vector']))
        except json.JSONDecodeError as json_error:
            log.warning("Failed to decode feature vector JSON", error=str(json_error))
            continue

    log.info("ev.features.fetched", count=len(features))
    return features

def generate_ev_scenario(
        input_params: Optional[Dict[str, Any]] = None,
        origin: str = 'EVIDENCE'
) -> Union[Dict[str, Any], None]:
    try:
        init_db()
    except Exception as init_error:
        log.warning("db.init_failed", error=str(init_error))
        pass

    raw_features = []
    if input_params:
        raw_params = input_params
        log.info("ev.params.ga_input", origin=origin)
        precursor_density = 0.5
        dominant_tags = []
    else:
        try:
            raw_features = fetch_raw_features_for_generation(limit=100)
        except FileNotFoundError as file_error:
            log.error("Generation failed due to missing DB file", error=str(file_error))
            return None

        if not raw_features:
            log.warning("ev.no_features", msg="No features found. Cannot generate evidence-based scenario.")
            return None

        raw_params = sample_signal_parameters(raw_features=raw_features)
        log.debug("ev.params.sampled_from_signals", count=len(raw_params))
        precursor_density = 0.8
        dominant_tags = ["asi_direct"] if precursor_density > 0.7 else []

    # --- 1. CRITICAL PRE-CHECK: VALIDATE PARAMETER TYPES ---
    is_valid_type, params = _validate_critical_numeric_types(raw_params)
    if not is_valid_type:
        log.error("core_ev.failed_precheck", msg="Scenario aborted due to non-numeric critical parameters.")
        return None

    title = abbreviate_ev(params)
    scenario_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # --- 2. Timeline Generation ---
    timeline_phases = generate_scenario_timeline(
        params=params,
        precursor_density=precursor_density,
        dominant_tags=dominant_tags
    )

    # --- 3. Generate narrative ---
    success, narrative, model_used = generate_narrative(
        title=title,
        params=params,
        timeline=timeline_phases
    )
    if not success:
        log.error("llm.all_failed")
        return None

    # --- 4. Consistency check ---
    checker = NarrativeChecker(params)
    consistent, failures = checker.check(narrative)
    if not consistent:
        log.warning("consistency.failed", failures=failures)
        return None

    # --- 5. DYNAMIC PROBABILITY CALCULATION ---
    emergence_prob = calculate_emergence_probability(params)

    # --- 6. Build final scenario object ---
    scenario: Dict[str, Any] = {
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
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "prompt_version": "ev-v1"
                }
            }
        },
        "origin": {
            "initial_origin": params.get("initial_origin", "rogue"),
            "development_dynamics": params.get("development_dynamics", "emergent")
        },
        "architecture": {
            "type": params.get("architecture", "decentralized"),
            "deployment_topology": params.get("deployment_topology", "distributed")
        },
        "substrate": {
            "type": params.get("substrate", "quantum"),
            "deployment_medium": params.get("deployment_medium", "edge"),
            "resilience": params.get("substrate_resilience", "adaptive")
        },
        "oversight_structure": {
            "type": params.get("oversight_type", "none"),
            "effectiveness": params.get("oversight_effectiveness", "ineffective"),
            "control_surface": "none" if params.get("oversight_type") == "none" else "technical"
        },
        "core_capabilities": {
            "agency_level": float(params.get("agency_level", 0.5)),
            # FIX 2: Ensure autonomy_degree is explicitly converted to float
            "autonomy_degree": float(params.get("autonomy_degree", 0.5)),
            "alignment_score": float(params.get("alignment_score", 0.5)),
            "phenomenology_proxy_score": float(params.get("phenomenology_proxy_score", 0.1))
        },
        "goals_and_behavior": {
            "stated_goal": params.get("stated_goal", "survival"),
            "mesa_goals": params.get("mesa_goals", []),
            "opacity": float(params.get("opacity", 0.6)),
            "deceptiveness": float(params.get("deceptiveness", 0.3)),
            "goal_stability": params.get("goal_stability", "fluid")
        },
        "impact_and_control": {
            "impact_domains": params.get("impact_domains", ["physical", "existential"]),
            "deployment_strategy": params.get("deployment_strategy", "stealth")
        },
        "scenario_content": {
            "title": title,
            "narrative": narrative,
            "timeline": {
                "phases": [
                    {
                        "phase": phase["phase"],
                        "years": phase["years"],
                        "description": phase.get("description", "")
                    }
                    for phase in timeline_phases
                ]
            }
        },
        "quantitative_assessment": {
            "probability": {
                "emergence_probability": emergence_prob,
                "detection_confidence": 0.35,
                "projection_confidence": 0.68,
                "trend": "rising",
                "last_update_reason": "Dynamically calculated from scenario parameters."
            },
            "risk_assessment": {
                "existential": {"score": 8, "weight": 0.9},
                "economic": {"score": 6, "weight": 0.7},
                "social": {"score": 7, "weight": 0.8},
                "political": {"score": 5, "weight": 0.6}
            }
        },
        "observable_evidence": {
            "key_indicators": params.get("key_indicators", [
                "Rapid opacity increase in frontier models",
                "Unexplained compute spikes on edge networks"
            ]),
            "supporting_signals": params.get("supporting_signals", [])
        }
    }

    # --- 7. Validate scenario against schema ---
    valid, err = SchemaManager.validate(scenario)
    if not valid:
        log.error(f"schema.validation.failed: {err}")
        return None

    if input_params:
        used_signals = ["GA_BRED_FROM_PARENTS"]
    else:
        used_signals = list(FeatureInfluenceModel.FEATURE_MAPPING.keys())

    # --- 8. Save scenario ---
    save_scenario_ev(
        title=title,
        params=params,
        narrative=narrative,
        timeline=timeline_phases,
        model_used=model_used,
        signals=used_signals,
        generation_origin=origin
    )

    log.info("ev.scenario.generated", title=title, model=model_used, id=scenario_id[:8], origin=origin,
             prob=f"{emergence_prob:.3f}")
    return scenario