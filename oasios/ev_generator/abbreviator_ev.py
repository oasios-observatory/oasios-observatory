#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/ev_generator/abbreviator_ev.py
"""
Dedicated Abbreviator for Evidence-Based (EV) scenarios.
Uses the 'ev_scenarios' table count for naming consistency.
"""
from typing import Dict, Any
# Import the specific counter for EV scenarios
from oasios.common.storage import get_next_ev_scenario_number


def short_code(value: str) -> str:
    """Generates a 3-letter uppercase abbreviation from a string value."""
    # Split by space or dash, take first letter of each word, join, then slice to 3
    return "".join(word[0].upper() for word in value.replace("-", " ").split())[:3] or "UNK"

def abbreviate_ev(core: Dict[str, Any]) -> str:
    """
    Generates a consistent scenario ID for EV scenarios based on key parameters
    and the next sequential number from the 'ev_scenarios' table.
    Format: ORIGIN-DYN-ARCH-TOPO-OVERSIGHT-EFF-SUBSTRATE-XXX
    """
    # These parts are derived directly from the scenario parameters
    parts = [
        short_code(core.get("initial_origin", "UNK")),
        short_code(core.get("development_dynamics", "UNK")),
        short_code(core.get("architecture", "UNK")),
        short_code(core.get("deployment_topology", "UNK")),
        short_code(core.get("oversight_type", "UNK")),
        short_code(core.get("oversight_effectiveness", "UNK")),
        short_code(core.get("substrate", "UNK")),
    ]
    # Use the dedicated EV scenario counter for consistent, sequential numbering
    num = get_next_ev_scenario_number()
    return f"{'-'.join(parts)}-{num:03d}"