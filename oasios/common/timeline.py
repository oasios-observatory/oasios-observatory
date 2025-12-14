#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/timeline.py
"""
Version 0.2: Implements signal-influenced, continuous timeline shifting for realism.
Removes static dynamic_timeline() for consistency.
"""
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from oasios.logger import log  # Assuming logger is available

# Define the maximum time-shift that can be applied by a parameter
MAX_SHIFT = 3  # Max years to shift the window

# Define the mapping (This is a good place to put it, or import it from a config/constants file)
OVERSIGHT_MAPPING = {
    'none': 0.0,
    'partial': 0.5,
    'full': 1.0,
}


def generate_scenario_timeline(params: Dict[str, Any], precursor_density: float, dominant_tags: List[str]) -> List[
    Dict[str, Any]]:
    """
    Generate a signal-influenced timeline using parameter values for acceleration/deceleration.
    This replaces the old, static 'dynamic_timeline' function.
    """
    current_year = datetime.now().year

    # 1. CALCULATE ACCELERATION/BRAKING FACTOR (Continuous Influence)

    # Positive Accelerators (speed up the timeline)
    agency = params.get("agency_level", 0.5)
    opacity = params.get("opacity", 0.5)

    # Negative Decelerators (slow down the timeline)
    alignment = params.get("alignment_score", 0.5)

    # 🐛 FIX: Correctly retrieve and map the string parameter to a numeric value
    # 1a. Get the string value for oversight effectiveness
    oversight_eff_str = params.get("oversight_effectiveness", "none")

    # 1b. Map the string to its numerical counterpart using the defined dictionary.
    #    We use .lower() for case-insensitivity and default to 0.0 ("none") if not found.
    oversight_eff_numeric = OVERSIGHT_MAPPING.get(oversight_eff_str.lower(), 0.0)

    # Net Shift Calculation: (Risks) - (Controls)
    risk_factor = (agency + opacity) / 2
    # 1c. Use the numerical value in the arithmetic calculation, resolving the UFuncTypeError
    control_factor = (alignment + oversight_eff_numeric) / 2

    # Map the net difference to a shift in years (e.g., -MAX_SHIFT to +MAX_SHIFT)
    net_shift = (risk_factor - control_factor) * MAX_SHIFT

    # 2. APPLY SIGNAL/TAG-DRIVEN OFFSET
    # Shift based on observed signal data density and high-risk tags
    signal_density_shift = precursor_density * (-2) if precursor_density > 0.7 else 0
    tag_shift = -1 if "asi_direct" in dominant_tags else 0

    total_shift = int(np.round(net_shift + signal_density_shift + tag_shift))

    # 3. CONSTRUCT THE TIMELINE

    # Pivot Year moves based on the total shift (earlier for risk, later for control)
    pivot_year = current_year + total_shift

    # Emergence Window
    window_start = pivot_year + 1
    window_end = pivot_year + 5

    # Check for negative years which indicates an overly aggressive shift for a pre-2000 start
    start_scaling_era = 2021
    if pivot_year < start_scaling_era:
        log.warning("timeline.shift_cap", msg=f"Pivot year {pivot_year} capped to {start_scaling_era}.",
                    shift=total_shift)
        pivot_year = start_scaling_era
        window_start = pivot_year + 1
        window_end = pivot_year + 5

    log.info("timeline.shift_applied", shift=total_shift, pivot=pivot_year)

    return [
        {
            "phase": "Precursors & Foundations",
            # Starting at 2000 is much more relevant for modern ASI risk
            "years": "2000-2020",
            "description": "Early internet, deep learning origins, pre-transformer scale."
        },
        {
            "phase": "Scaling Era",
            "years": f"{start_scaling_era}-{pivot_year - 1}",
            "description": "LLMs, agents, multi-modal systems accelerate, leading to the pivot."
        },
        {
            "phase": "Pivot Year",
            "years": str(pivot_year),
            "description": "Key inflection point: a hidden final leap or public capability breakthrough."
        },
        {
            "phase": "Emergence Window",
            "years": f"{window_start}-{window_end}",
            "description": "Evidence-weighted ASI emergence window. Timing influenced by speed of capability accumulation."
        },
        {
            "phase": "Long-Term Equilibrium",
            "years": f"{window_end + 1}+",
            "description": "Post-ASI outcome space."
        },
    ]

# The simple, static function is now removed.
# You must update core_ev.py to call generate_scenario_timeline(...) instead of dynamic_timeline()