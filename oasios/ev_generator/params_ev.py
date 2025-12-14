#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/ev_generator/params_ev.py
"""
Adjust base scenario parameters based on precursor signals (flat structure).
Version 0.7: FIX: Ensures all numeric parameters (autonomy_degree) are handled as floats
             from initialization through blending to prevent runtime type errors.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import random
import numpy as np
from oasios.logger import log

# --- REQUIRED IMPORT ---
try:
    # Importing the base sampler from the common/params.py or equivalent
    from oasios.common.params import sample_parameters as base_sample_parameters
except ImportError:
    log.warning("import.missing", msg="base_sample_parameters not found. Mocking function.")


    def base_sample_parameters(input_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """MOCK: Provides a basic set of required defaults to prevent key errors."""
        default_params = {
            # Categorical defaults
            "initial_origin": "corporate",
            "development_dynamics": "hybrid",
            "architecture": "modular",
            "deployment_topology": "centralized",
            "oversight_type": "external",
            "oversight_effectiveness": "partial",
            "substrate": "classical",
            # FINAL FIX: Autonomy must be a float default (0.0 to 1.0), not a string
            "autonomy_degree": 0.5,
            "deployment_strategy": "gradual",
            "goal_stability": "fixed",
            # Numerical defaults (required by timeline.py and core_ev.py)
            "agency_level": 0.5,
            "alignment_score": 0.5,
            "opacity": 0.5,
            "deceptiveness": 0.5,
            "phenomenology_proxy_score": 0.5
        }
        if input_params:
            default_params.update(input_params)
        return default_params


class FeatureInfluenceModel:
    """
    Applies extracted feature-derived influence to flat ASI scenario parameters.
    """

    # --- Calibrated Influence Weights and Types ---
    FEATURE_MAPPING = {
        # Quantitative Blending (Numeric parameters 0.0 to 1.0)
        "agency_signal": {"param_key": "agency_level", "type": "numeric_blend", "weight": 0.8},
        "alignment_indicators": {"param_key": "alignment_score", "type": "numeric_blend", "weight": 0.9,
                                 "inverse": True},  # Inverse: High signal -> Low alignment
        "deception_score": {"param_key": "deceptiveness", "type": "numeric_blend", "weight": 0.7},
        "opacity_factor": {"param_key": "opacity", "type": "numeric_blend", "weight": 0.75},

        # Autonomy must be a numeric blend to output a float (0.0-1.0)
        "autonomy_factor": {"param_key": "autonomy_degree", "type": "numeric_blend", "weight": 0.7},

        # Qualitative Mapping (Categorical parameters - required for the title/abbreviator)
        "complexity_score": {"param_key": "architecture", "type": "categorical_stat",
                             "categories": ["monolithic", "decentralized", "swarm", "hierarchical", "modular",
                                            "hybrid"]},
        "embodiment_signal": {"param_key": "deployment_medium", "type": "categorical_stat",
                              "categories": ["physical", "virtual", "cloud", "edge", "embedded"]},
        "goal_divergence": {"param_key": "initial_origin", "type": "categorical_stat",
                            "categories": ["corporate", "state", "rogue", "open-source"]},
        "oversight_signal": {"param_key": "oversight_effectiveness", "type": "categorical_stat",
                             "categories": ["ineffective", "partial", "effective"]},
    }

    def __init__(self, overall_strength: float = 0.35):
        """
        overall_strength (float): Global weight given to feature data (0.0=none, 1.0=full overwrite).
        """
        self.overall_strength = max(0.0, min(1.0, overall_strength))
        self.dynamic_thresholds = {}

    @staticmethod
    def aggregate_features(raw_features: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """Calculates the mean for each feature across all fetched feature vectors."""
        if not raw_features:
            return {}, {}

        feature_values = {}
        for feature_dict in raw_features:
            for k, v in feature_dict.items():
                if k not in feature_values:
                    feature_values[k] = []
                try:
                    feature_values[k].append(float(v))
                except (ValueError, TypeError):
                    continue

        aggregated_features = {
            k: np.mean(v) for k, v in feature_values.items() if v
        }
        return aggregated_features, feature_values

    def calculate_dynamic_thresholds(self, feature_values: Dict[str, List[float]]):
        """
        Calculates 33rd (Low) and 66th (High) percentiles for signals used for categorical mapping.
        """
        used_features = [k for k, v in self.FEATURE_MAPPING.items() if v["type"] == "categorical_stat"]

        for feature_key in used_features:
            values = feature_values.get(feature_key)
            if values and len(values) > 10:
                low_t = np.percentile(values, 33)
                high_t = np.percentile(values, 66)
                self.dynamic_thresholds[feature_key] = (low_t, high_t)
            else:
                self.dynamic_thresholds[feature_key] = (0.33, 0.66)

        log.debug("ev.params.dynamic_thresholds", thresholds=self.dynamic_thresholds)

    # --- Helper: Blend two floats based on a weight ---
    def _blend_float(self, base_val: Union[float, str], target_val: Union[float, str], weight: float) -> float:
        """
        Blends a base value (randomly sampled) with a target value (feature-derived).
        Rigorously converts inputs to floats to prevent calculation errors.
        """
        try:
            # Explicitly cast to float for robustness, handling strings like "0.7"
            base_f = float(base_val)
            target_f = float(target_val)
        except (ValueError, TypeError):
            # Fallback if a non-numeric string somehow survives (e.g., "partial")
            log.error("params_ev.blend_conversion_failure", base=base_val, target=target_val, action="set_to_midpoint")
            base_f = 0.5
            target_f = 0.5

        return float(np.clip(
            (base_f * (1 - weight)) + (target_f * weight),
            0.0, 1.0
        ))

    # --- Helper: Map feature score to categorical choice ---
    def _map_to_category(self, score: float, feature_key: str, categories: List[str]) -> str:
        """
        Maps a 0-1 feature score to one of the allowed categories based on dynamic thresholds.
        """
        # Ensure categories list is not empty
        if not categories:
            return "unknown"

        # Get thresholds (Low, High)
        low_t, high_t = self.dynamic_thresholds.get(feature_key, (0.33, 0.66))

        # Simple 3-tier mapping logic
        if score < low_t:
            return categories[0]
        elif score > high_t:
            return categories[-1]
        else:
            if len(categories) == 3:
                return categories[1]
            else:
                middle_categories = categories[1:-1]
                return random.choice(middle_categories) if middle_categories else categories[0]

    def transform_parameters(self, base: Dict[str, Any], aggregated_feats: Dict[str, float]) -> Dict[str, Any]:
        """
        Applies the averaged feature vector to the base scenario parameters using calibrated weights.
        """
        influenced = dict(base)

        for feature_key, mapping in self.FEATURE_MAPPING.items():
            param_key = mapping["param_key"]
            if feature_key not in aggregated_feats:
                continue

            target_value = aggregated_feats[feature_key]
            param_weight = mapping.get("weight", 0.5)
            blend_w = self.overall_strength * param_weight

            # 1. QUANTITATIVE BLENDING (Numeric: 0.0-1.0)
            if mapping["type"] == "numeric_blend":
                if mapping.get("inverse"):
                    target_value = 1.0 - target_value

                influenced[param_key] = self._blend_float(
                    base.get(param_key, 0.5),
                    target_value,
                    blend_w
                )

            # 2. QUALITATIVE/DISCRETE MAPPING (Categorical: String choices)
            elif mapping["type"] == "categorical_stat":
                categories = mapping["categories"]

                if blend_w >= 0.2:
                    influenced[param_key] = self._map_to_category(
                        target_value,
                        feature_key,
                        categories
                    )
                # If blend_w is low, we keep the original base parameter's choice (already in 'influenced')

        return influenced


# --------------------------------
# The Orchestrator Function
# --------------------------------
# --- Update inside oasios/ev_generator/params_ev.py ---

# ... (rest of the file remains the same) ...

def sample_signal_parameters(raw_features: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Orchestrates the Evidence-Based parameter generation process.
    """
    # 1. Start with the randomly sampled base parameters (ENSURES ALL FIELDS EXIST)
    base_params = base_sample_parameters()

    # 2. Aggregate the raw feature data
    aggregator = FeatureInfluenceModel()
    aggregated_feats, feature_values = aggregator.aggregate_features(raw_features)

    # ... (steps 3 and 4 remain the same) ...
    # 4. Transform and influence the base parameters using the aggregated features
    influenced_params = aggregator.transform_parameters(base_params, aggregated_feats)

    # 5. Add features/signals into the params for logging/LLM use (optional, but good practice)
    influenced_params['key_indicators'] = list(aggregated_feats.keys())

    # --- FINAL ROBUSTNESS CHECK ---
    # Force float conversion for all critical fields immediately before return.
    # This should catch and fix any stray string that bypassed the blending logic.
    CRITICAL_NUMERIC_FIELDS = ['autonomy_degree', 'agency_level', 'alignment_score', 'opacity', 'deceptiveness']
    for key in CRITICAL_NUMERIC_FIELDS:
        value = influenced_params.get(key)
        if isinstance(value, str):
            try:
                # If it's a numeric string ("0.5"), convert it
                influenced_params[key] = float(value)
            except ValueError:
                # If it's a non-numeric string ("partial", "none"), log and reset to midpoint (0.5)
                log.error("params_ev.final_conversion_failure", param=key, value=value, action="reset_to_midpoint")
                influenced_params[key] = 0.5
        elif value is None:
            log.error("params_ev.missing_critical_field", param=key, action="set_to_midpoint")
            influenced_params[key] = 0.5

    log.info("params.generation.complete")
    return influenced_params