import re
import math
import structlog

from typing import Dict, List, Any

from .config_t import (
    WEIGHTS,
    PRECURSOR_SCORE_THRESHOLD,
    MAX_SIGNAL_SCORE,
)

log = structlog.get_logger()


# ==================================================
# ARCHITECTURE CLASSIFICATION
# ==================================================

def classify_architecture(description: str) -> str:

    desc = description.lower()

    patterns = {

        "swarm": [
            r"\bswarm\b",
            r"\bswarm intelligence\b",
        ],

        "federated": [
            r"\bfederated\b",
            r"\bdistributed federation\b",
        ],

        "modular": [
            r"\bmodular\b",
            r"\bplugin architecture\b",
        ],

        "layered": [
            r"\bhierarchical\b",
            r"\blayered\b",
            r"\bmulti[- ]layer\b",
        ]

    }

    for architecture, expressions in patterns.items():

        if any(
            re.search(pattern, desc)
            for pattern in expressions
        ):

            return architecture

    return "monolithic"


# ==================================================
# AUTONOMY CLASSIFICATION
# ==================================================

def classify_autonomy(description: str) -> str:

    desc = description.lower()

    full_patterns = [

        r"\bself[- ]tasking\b",

        r"\bself[- ]directed\b",

        r"\bself[- ]governing\b",

        r"\bautonomous execution\b",

        r"\bindependent task execution\b",

        r"\bautonomous planning\b",

        r"\bself[- ]planning\b",

    ]

    controlled_patterns = [

        r"\bhuman[- ]in[- ]the[- ]loop\b",

        r"\bmanual approval\b",

        r"\bhuman supervised\b",

        r"\boperator controlled\b",

    ]

    # Explicit human control takes precedence.

    if any(
        re.search(pattern, desc)
        for pattern in controlled_patterns
    ):

        return "controlled"

    if any(
        re.search(pattern, desc)
        for pattern in full_patterns
    ):

        return "full"

    partial_patterns = [

        r"\bautonomous\b",

        r"\bagentic\b",

        r"\bai agent\b",

        r"\bmulti[- ]agent\b",

        r"\btool[- ]using agent\b",

    ]

    if any(
        re.search(pattern, desc)
        for pattern in partial_patterns
    ):

        return "partial"

    return "controlled"


# ==================================================
# SIGNAL CLASSIFICATION
# ==================================================

def classify_signal_description(
    description: str
) -> List[str]:

    desc = description.lower()

    tags = []

    patterns = {

        "asi_direct": (
            r"\b("
            r"superintelligence|"
            r"asi|"
            r"agi|"
            r"artificial general intelligence"
            r")\b"
        ),

        "full_autonomy": (
            r"\b("
            r"autonomy|"
            r"autonomous|"
            r"agent|"
            r"agentic"
            r")\b"
        ),

        "alignment": (
            r"\b("
            r"alignment|"
            r"safety|"
            r"oversight|"
            r"control problem"
            r")\b"
        ),

        "open_source": (
            r"\b("
            r"open[- ]source|"
            r"github|"
            r"huggingface"
            r")\b"
        ),

        "novel_paradigm": (
            r"\b("
            r"recombinant|"
            r"meta[- ]learning|"
            r"meta[- ]reasoning|"
            r"novel architecture"
            r")\b"
        ),

        "risk_indicator": (
            r"\b("
            r"catastrophic|"
            r"existential risk|"
            r"x[- ]risk|"
            r"unaligned|"
            r"loss of control"
            r")\b"
        )

    }

    for tag, pattern in patterns.items():

        if re.search(pattern, desc):

            tags.append(tag)

    return tags


# ==================================================
# FEATURE VECTOR
# ==================================================

def calculate_feature_vector(
    metadata: Dict[str, Any],
    tags: List[str],
    architecture: str,
    autonomy: str,
) -> Dict[str, float]:

    vector = {

        "modularity": 0.0,

        "decentralization": 0.0,

        "agentic_behavior": 0.0,

        "alignment_indicators": 0.0,

        "risk_factors": 0.0,

    }

    if architecture in (
        "modular",
        "layered"
    ):

        vector["modularity"] = 0.8

    if architecture in (
        "swarm",
        "federated"
    ):

        vector["decentralization"] = 1.0

    if autonomy == "full":

        vector["agentic_behavior"] = 1.0

    elif autonomy == "partial":

        vector["agentic_behavior"] = 0.5

    if "alignment" in tags:

        vector["alignment_indicators"] = 1.0

    if "risk_indicator" in tags:

        vector["risk_factors"] = 1.0

    return vector


# ==================================================
# SCORING UTILITIES
# ==================================================

def _safe_weight(key: str) -> float:

    try:

        value = float(
            WEIGHTS.get(key, 0.0)
        )

        if not math.isfinite(value):
            return 0.0

        return max(0.0, value)

    except (TypeError, ValueError):

        return 0.0


# ==================================================
# MAIN CLASSIFIER
# ==================================================

def classify_and_score(
    metadata: Dict[str, Any]
) -> Dict[str, Any]:

    title = str(
        metadata.get("title") or ""
    )

    description = str(
        metadata.get("description") or ""
    )

    source = str(
        metadata.get("source") or "unknown"
    )

    searchable_text = (
        f"{title} {description}"
    )

    architecture = classify_architecture(
        searchable_text
    )

    autonomy = classify_autonomy(
        searchable_text
    )

    tags = classify_signal_description(
        searchable_text
    )

    score = 0.0

    # --------------------------------------------------
    # KEYWORD SCORING
    # --------------------------------------------------

    for tag in tags:

        score += _safe_weight(tag)

    # --------------------------------------------------
    # AUTONOMY SCORING
    # --------------------------------------------------

    if autonomy == "full":

        score += _safe_weight(
            "autonomy_full"
        )

    elif autonomy == "partial":

        score += _safe_weight(
            "autonomy_partial"
        )

    # --------------------------------------------------
    # ARCHITECTURE SCORING
    # --------------------------------------------------

    if architecture == "modular":

        score += _safe_weight(
            "architecture_modular"
        )

    # --------------------------------------------------
    # POPULARITY METADATA
    # --------------------------------------------------

    # GitHub popularity is intentionally excluded
    # from precursor-risk scoring.
    #
    # Stars measure community attention, not
    # scientific capability or safety risk.

    stars = 0

    if source == "github":

        try:

            stars = max(
                0,
                int(metadata.get("stars") or 0)
            )

        except (TypeError, ValueError):

            stars = 0

    # --------------------------------------------------
    # SCORE NORMALIZATION
    # --------------------------------------------------

    score = max(
        0.0,
        min(
            score,
            float(MAX_SIGNAL_SCORE)
        )
    )

    score = round(score, 3)

    # --------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------

    features = calculate_feature_vector(
        metadata,
        tags,
        architecture,
        autonomy
    )

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------

    signal_type = (

        "precursor"

        if score >= PRECURSOR_SCORE_THRESHOLD

        else "noise"

    )

    result = {

        "signal_type": signal_type,

        "score": score,

        "tags": sorted(
            set(tags)
        ),

        "architecture": architecture,

        "autonomy": autonomy,

        "features": features,

        "contextual_metadata": {

            "source": source,

            "github_stars": stars,

        },

        "model_version": "OASIOS-v0.5",

    }

    log.info(
        "classification_complete",
        source=source,
        score=score,
        signal_type=signal_type,
        architecture=architecture,
        autonomy=autonomy
    )

    return result
