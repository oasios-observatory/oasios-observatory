# oasios/tracker/classifier_t_v3.py
# Implements the Feature & Signal Analysis Layer (FSAL) logic using ECO Ontology.

import re
import structlog
from typing import Dict, List, Any, Tuple

log = structlog.get_logger()

# --- Placeholder Configuration (Assumes config_t_v3.py holds these) ---
# NOTE: In a real environment, you would import this from a dedicated config file.
WEIGHTS = {
    "asi_direct": 5.0,
    "full_autonomy": 3.0,
    "alignment": -1.5,  # Negative or low weight often used for safety research signals
    "open_source": 0.5,
    "autonomy_full": 1.5,
    "autonomy_partial": 0.5,
    "architecture_modular": 0.7,
    "stars_100_multiplier": 0.01,
    "stars_1000_flat_bonus": 2.0,
}


# --- End Placeholder ---


def classify_architecture(description: str) -> str:
    """
    Ontological Classification (Layer B): Classify AI architecture type.
    Returns one of: 'swarm', 'modular', 'federated', 'layered', or 'monolithic'.
    """
    desc_lower = description.lower()
    if "swarm" in desc_lower:
        return "swarm"
    if "modular" in desc_lower:
        return "modular"
    if "federated" in desc_lower:
        return "federated"
    if any(word in desc_lower for word in ["layers", "stack", "hierarchical"]):
        return "layered"
    return "monolithic"


def classify_autonomy(description: str) -> str:
    """
    Ontological Classification (Layer C4): Classify autonomy level.
    Returns one of: 'full', 'partial', or 'controlled'.
    """
    desc_lower = description.lower()
    # Look for terms indicating self-management or self-direction
    if re.search(r"\bself[- ]?(tasking|govern|replicate|direct)\b", desc_lower):
        return "full"
    elif "autonomous" in desc_lower or "agentic" in desc_lower:
        return "partial"
    return "controlled"


def classify_signal_description(description: str) -> List[str]:
    """
    Keyword-based signal tagging (Layer C & D).
    """
    tags = []
    desc_lower = description.lower()

    # Core ASI/AGI concepts
    if "superintelligence" in desc_lower or "asi" in desc_lower or "agi" in desc_lower:
        tags.append("asi_direct")

    # Autonomy/Agency features
    if "autonomy" in desc_lower or "agent" in desc_lower:
        tags.append("full_autonomy")

    # Control/Safety concepts (Layer D: Emergent Patterns)
    if "alignment" in desc_lower or "safety" in desc_lower or "control problem" in desc_lower:
        tags.append("alignment")

    # Open-source availability (Layer A3: Tooling)
    if "open-source" in desc_lower or "github" in desc_lower or "huggingface" in desc_lower:
        tags.append("open_source")

    # Novel Paradigm Tags (Layer B6)
    if "recombinant" in desc_lower or "meta-learning" in desc_lower:
        tags.append("novel_paradigm")

    return tags


def calculate_feature_vector(metadata: Dict[str, Any], tags: List[str], architecture: str, autonomy: str) -> Dict[
    str, float]:
    """
    Calculates the normalized ECO feature vector (0.0 to 1.0) based on the ontology.
    (This vector is essential for the downstream Anomaly & Pattern Synthesis Layer).
    """
    desc_lower = metadata.get("description", "").lower()
    score = metadata.get("score", 0.0)

    # Initialize the five core ECO Feature Keys
    vector = {
        "modularity": 0.0,
        "decentralization": 0.0,
        "agentic_behavior": 0.0,
        "alignment_indicators": 0.0,
        "risk_factors": 0.0,
    }

    # --- 1. Ontological Mapping (Architecture/Autonomy) ---
    # Layer B -> Layer D (Modularity, Decentralization)
    if architecture in ("modular", "layered"):
        vector["modularity"] = 0.8
    if architecture in ("swarm", "federated"):
        vector["decentralization"] = 1.0

    # Layer C4 -> Layer D (Agentic Behavior)
    if autonomy == "full":
        vector["agentic_behavior"] = 1.0
    elif autonomy == "partial":
        vector["agentic_behavior"] = 0.5

    # --- 2. Keyword Mapping (Alignment/Risk) ---

    # Alignment/Safety
    if "alignment" in tags or re.search(r"\b(safety|oversight|control)\b", desc_lower):
        vector["alignment_indicators"] = 1.0

    # Risk/Existential Threats
    if "x-risk" in desc_lower or re.search(r"\b(unaligned|catastrophic|existential risk)\b", desc_lower):
        vector["risk_factors"] = 1.0

    # --- 3. Score-to-Risk Scaling ---
    # Scale risk based on final score magnitude (capped at 10.0)
    score_relevance = score / 10.0
    vector["risk_factors"] = max(vector["risk_factors"], score_relevance)

    # Ensure all vectors are capped at 1.0
    for key in vector:
        vector[key] = min(vector[key], 1.0)

    return vector


def classify_and_score(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    FSAL Main Function: Orchestrates classification, weighted scoring, and
    feature vector calculation.
    """
    title = metadata.get("title", "")
    desc = metadata.get("description", "")
    source = metadata.get("source", "unknown")

    initial_score = 0.0

    # --- 1. Ontological Classification ---
    architecture = classify_architecture(desc)
    autonomy = classify_autonomy(desc)
    tags = classify_signal_description(desc)

    # Add source to tags for detailed tracking
    tags.append(source)

    # --- 2. Weighted Scoring (Score is a composite index) ---

    # Score from Tags (Keywords)
    if "asi_direct" in tags:
        initial_score += WEIGHTS["asi_direct"]
    if "full_autonomy" in tags:
        initial_score += WEIGHTS["full_autonomy"]
    if "alignment" in tags:
        initial_score += WEIGHTS["alignment"]  # Can be negative/positive depending on policy
    if "open_source" in tags:
        initial_score += WEIGHTS["open_source"]

    # Score from Architecture/Autonomy
    if autonomy == "full":
        initial_score += WEIGHTS["autonomy_full"]
    elif autonomy == "partial":
        initial_score += WEIGHTS["autonomy_partial"]

    if architecture == "modular":
        initial_score += WEIGHTS["architecture_modular"]

    # Score from Magnitude (GitHub Stars)
    stars = metadata.get("stars", 0)
    if source == "github":
        if stars > 100:
            initial_score += (stars - 100) * WEIGHTS["stars_100_multiplier"]
        if stars > 1000:
            initial_score += WEIGHTS["stars_1000_flat_bonus"]

    final_score = min(max(initial_score, 0.0), 10.0)  # Cap score between 0.0 and 10.0

    # --- 3. Calculate Feature Vector ---
    # Pass calculated score and current classifications to the feature engine
    metadata["score"] = final_score
    feature_vector = calculate_feature_vector(metadata, tags, architecture, autonomy)

    # --- 4. Final Structured Output ---

    # The CoreTracker expects to unpack score, tags, and features from the result dictionary
    return {
        "signal_type": "precursor" if final_score > 3 else "noise",
        "score": final_score,
        "tags": sorted(list(set(tags))),  # Unique and sorted list of tags
        "architecture": architecture,
        "autonomy": autonomy,
        "features": feature_vector
    }