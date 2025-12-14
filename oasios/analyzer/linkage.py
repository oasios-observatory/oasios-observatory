# oasios/analyzer/linkage.py

import json
from typing import Dict, Any

# Define the numeric mapping for descriptive alignment terms.
ALIGNMENT_MAPPING = {
    "low": 0.2,
    "partial": 0.5,
    "medium": 0.65,
    "high": 0.85,
    "super": 1.0
}

# NOTE: This is a placeholder for the actual feature weights used by the analyzer.
FEATURE_WEIGHTS = {
    "agency_signal": 1.0,
    "alignment_indicators": 1.2,
    "deception_score": 0.8,
    "embodiment_signal": 0.9,
    "complexity_score": 1.1,
    "coherence": 1.5,
    # Add other feature keys and weights as needed for your analysis
}


def get_signal_alignment_score(scenario_id: str, params: Dict[str, Any]) -> float:
    """
    Calculates the P1 (Signal Alignment) plausibility score for a given scenario.
    ... [Docstring content omitted for brevity] ...
    """
    # NOTE: In a real system, current_features would likely be fetched from a global
    # or time-series data source representing current real-world trends/signals.
    # For this example, we use the values seen in the traceback locals.
    current_features = {
        'agency_signal': 0.65,
        'alignment_indicators': 0.35,
        'deception_score': 0.4,
        'embodiment_signal': 0.72,
        'complexity_score': 0.85
    }

    raw_alignment_score = 0.0
    max_possible_score = 0.0

    # Iterate through the defined features and their weights
    for feature_key, importance_weight in FEATURE_WEIGHTS.items():
        if feature_key not in current_features:
            continue

        feature_value = current_features[feature_key]
        relevance = 0.0

        max_possible_score += importance_weight

        if feature_key == "agency_signal":
            scenario_agency = params.get("agency_score", 0.0)
            relevance = feature_value * scenario_agency

        elif feature_key == "alignment_indicators":
            # FIX APPLIED HERE: Use str() to convert the retrieved value (which may be a float)
            # into a string before calling .lower(), preventing the AttributeError.
            scenario_alignment_str = str(params.get("alignment_score", "medium")).lower()

            # Convert the string to a numeric score using the mapping
            scenario_alignment_score = ALIGNMENT_MAPPING.get(scenario_alignment_str, 0.5)

            relevance = feature_value * scenario_alignment_score

        elif feature_key in ["complexity_score", "embodiment_signal", "deception_score"]:
            relevance = feature_value  # Simplified calculation for example

        elif feature_key == "coherence":
            coherence_score = params.get("coherence_score", 0.0)
            relevance = coherence_score * feature_value

        raw_alignment_score += relevance * importance_weight

    # Normalize the final score
    if max_possible_score > 0:
        normalized_score = raw_alignment_score / max_possible_score
    else:
        normalized_score = 0.0

    return normalized_score


def calculate_target_coherence_score(scenario_id: str, params: Dict[str, Any]) -> float:
    """
    Placeholder for calculating the target coherence score (P4 Coherence).

    This function should compare the scenario's internal elements (Narrative,
    Timeline, Signals) to determine their thematic and logical consistency.

    Args:
        scenario_id: The ID of the scenario. (Used for logging/context)
        params: The scenario parameters dictionary (contains factor weight data).

    Returns:
        A numeric coherence score (float between 0.0 and 1.0).
    """
    # NOTE: You will need to implement the actual logic here.
    return 0.5