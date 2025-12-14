# oasios/tracker/config_t_v3.py (NEW FILE)

# --- Signal Scoring Weights ---
# The weights should sum up to a desirable maximum value for a perfect signal.
# These values multiply the score contributed by the presence of a tag or feature.

WEIGHTS = {
    # Direct Keywords (High Impact)
    "asi_direct": 4.0,  # e.g., "superintelligence" or "ASI" mentioned
    "full_autonomy": 3.0,  # Highly relevant signal
    "alignment": 2.5,  # Key safety/control feature

    # Architecture/Autonomy Features (Moderate Impact)
    "autonomy_full": 1.5,
    "autonomy_partial": 0.5,
    "architecture_modular": 1.0,  # Modular systems are often considered a precursor stage

    # Contextual Factors (Low Impact - Can be multiplied by data magnitude)
    "open_source": 0.5,  # Visibility/reproducibility factor

    # Magnitude Multipliers (for GitHub)
    "stars_100_multiplier": 0.01,  # Multiplier applied to the actual star count for signals > 100
    "stars_1000_flat_bonus": 1.5,  # Fixed bonus for very popular projects
}