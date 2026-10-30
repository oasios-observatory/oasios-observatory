# oasios/tracker/config_t.py

WEIGHTS = {
    "asi_direct": 4.0,
    "full_autonomy": 3.0,
    "alignment": 1.5,
    "open_source": 0.5,

    "autonomy_full": 1.5,
    "autonomy_partial": 0.5,
    "architecture_modular": 1.0,

    "stars_100_multiplier": 0.01,
    "stars_1000_flat_bonus": 1.5,
}

# Classification
PRECURSOR_SCORE_THRESHOLD = 3.0
MAX_SIGNAL_SCORE = 10.0

# Anomaly detection
BASELINE_THRESHOLD = 0.35
MIN_ANOMALY_CONFIDENCE = 0.40

# Pattern synthesis
TIME_WINDOW_HOURS = 168
SEVERITY_DIFFERENCE_THRESHOLD = 2
MIN_GROUP_SIZE = 2

# Collection
MAX_RESULTS_PER_SOURCE = 10
MAX_FEATURE_EXTRACTION_BATCH = 500

MODEL_VERSION = "OASIOS-v0.4"
ANOMALY_MODEL_VERSION = "APSL-v0.4"
GROUPING_MODEL_VERSION = "APSL-GROUPING-v0.4"

# Governance
ROBOTS_TIMEOUT_SECONDS = 8
API_TIMEOUT_SECONDS = 20
