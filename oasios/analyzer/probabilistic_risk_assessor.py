#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# probabilistic_risk_assessor.py

import sqlite3
import json
import random
from typing import Dict, Any, List

# --- CONFIGURATION (Adjust these paths/settings) ---
SCENARIO_DB_PATH = '//data/asi_scenarios.db'  # Your existing DB
PRECURSOR_DB_PATH = '/data/asi_precursors.db'  # The new DB
TABLE_NAME = 'ev_scenarios'  # Table holding your generated scenarios


# --- 1. PROBABILISTIC MODEL CORE LOGIC ---

def _encode_development_dynamics(dynamics: str) -> float:
    """Converts the categorical development_dynamics to a numerical influence score."""
    # Emergent and Hybrid are riskier than Engineered in terms of control/speed
    mapping = {
        'engineered': 0.2,
        'hybrid': 0.6,
        'emergent': 1.0,
    }
    return mapping.get(dynamics.lower(), 0.5)


def calculate_emergence_probability(
        params: Dict[str, Any],
        precursor_intensity: float
) -> float:
    """
    Calculates the scenario's emergence probability using a weighted quantitative model.
    This simulates a simple Bayesian Network dependency flow.
    """

    # 1. Input Parameters (Ensured to be numeric by the previous fixes)
    agency = params.get('agency_level', 0.5)
    autonomy = params.get('autonomy_degree', 0.5)
    alignment = params.get('alignment_score', 0.5)
    deceptiveness = params.get('deceptiveness', 0.5)

    # 2. Encoded Categorical Input
    dynamics_cat = params.get('development_dynamics', 'hybrid')
    dynamics_score = _encode_development_dynamics(dynamics_cat)

    # 3. Define Weights (Expert Judgment)
    W_CAPABILITY = 0.35  # Weight for AI's internal power (Agency/Autonomy)
    W_MISALIGNMENT = 0.30  # Weight for AI's risk profile (Alignment/Deceptiveness)
    W_DYNAMICS = 0.15  # Weight for emergence speed
    W_PRECURSOR = 0.20  # Weight for external, real-world signals (The new data)

    # 4. Calculate Combined Risk Factors

    # Risk Factor 1: Capability (High is bad)
    capability_risk = (agency + autonomy) / 2

    # Risk Factor 2: Misalignment (Deceptiveness amplifies low Alignment)
    # Note: 1 - alignment gives us a score where 1.0 is full misalignment
    misalignment_risk = (1 - alignment) * (1 + 0.5 * deceptiveness) / 1.5  # Normalized

    # 5. Calculate Raw Probability Score
    raw_score = (
            W_CAPABILITY * capability_risk +
            W_MISALIGNMENT * misalignment_risk +
            W_DYNAMICS * dynamics_score +
            W_PRECURSOR * precursor_intensity
    )

    # 6. Final Normalization and Clamping (0.0 to 1.0)
    # We use a tanh-like function (or simple clamp) to ensure the output is a valid probability

    # Assuming weights sum to 1.0, the max raw_score is 1.0. We clamp to be safe.
    final_prob = max(0.01, min(0.99, raw_score))

    # Apply a soft scaling function (optional: to bias toward low/high extremes)
    # Here, we will just use the clamped score.

    # Probabilities should be small for ASI emergence
    return final_prob * 0.2  # Scale down so the max probability is 0.2 (20%)


# --- 2. DATABASE INTERFACE FUNCTIONS ---

def get_average_precursor_intensity(db_path: str) -> float:
    """
    Fetches the average 'intensity' score from the new precursor database.
    (MOCK: Returns a random value if the DB access fails).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ASSUMPTION: Your new precursor DB has a table 'precursors' with a column 'intensity'
        cursor.execute("SELECT AVG(intensity) FROM precursors;")
        avg_intensity = cursor.fetchone()[0]

        conn.close()
        return float(avg_intensity) if avg_intensity is not None else 0.5

    except (sqlite3.Error, FileNotFoundError, IndexError):
        # Fallback to random value to simulate dynamic precursor data if DB access fails
        print(f"WARNING: Could not read precursor DB at {db_path}. Using random intensity.")
        return random.uniform(0.1, 0.9)


def get_all_scenarios(db_path: str, table: str) -> List[Dict[str, Any]]:
    """Retrieves scenario ID and required parameter data from the scenario database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # The JSON schema is complex, so we retrieve the whole JSON object if stored as one,
    # or rely on pre-parsed column data. Assuming the full JSON is stored in a 'data' column.
    # We will retrieve 'id' and the full scenario data.
    cursor.execute(f"SELECT id, parameters FROM {table};")

    scenarios = []
    for row_id, json_data in cursor.fetchall():
        try:
            scenario = json.loads(json_data)
            # Flatten the required parameters for the calculation function
            params = {
                'id': row_id,
                'development_dynamics': scenario['origin']['development_dynamics'],
                # Retrieve core capabilities directly
                'agency_level': scenario['core_capabilities']['agency_level'],
                'autonomy_degree': scenario['core_capabilities']['autonomy_degree'],
                'alignment_score': scenario['core_capabilities']['alignment_score'],
                # Retrieve goals and behavior parameters
                'deceptiveness': scenario['goals_and_behavior']['deceptiveness'],
                # Pass the full data for later update
                'full_data': scenario
            }
            scenarios.append(params)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Skipping scenario {row_id} due to missing field or corrupt JSON: {e}")

    conn.close()
    return scenarios


def update_scenario_probability(db_path: str, table: str, scenario_id: str, new_prob: float, full_data: Dict[str, Any]):
    """Updates the scenario's emergence_probability and saves the modified JSON back."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Update the JSON structure with the new probability
    # Ensure the required structure exists
    if 'quantitative_assessment' not in full_data:
        full_data['quantitative_assessment'] = {'probability': {}, 'risk_assessment': {}}
    if 'probability' not in full_data['quantitative_assessment']:
        full_data['quantitative_assessment']['probability'] = {}

    full_data['quantitative_assessment']['probability']['emergence_probability'] = round(new_prob, 4)

    # 2. Update the database record
    updated_json = json.dumps(full_data)

    # ASSUMPTION: The primary key is 'id' and the data column is 'data'
    cursor.execute(f"UPDATE {table} SET data = ? WHERE id = ?;", (updated_json, scenario_id))

    conn.commit()
    conn.close()


# --- 3. MAIN EXECUTION ---

def run_probabilistic_assessment():
    """Orchestrates the data fetching, calculation, and database update."""

    print(f"--- Starting Probabilistic Risk Assessment ---")

    # Step 1: Get the new, dynamic precursor data
    precursor_intensity = get_average_precursor_intensity(PRECURSOR_DB_PATH)
    print(f"-> Current average precursor intensity: {precursor_intensity:.2f}")

    # Step 2: Get scenarios for calculation
    scenarios_to_process = get_all_scenarios(SCENARIO_DB_PATH, TABLE_NAME)
    print(f"-> Found {len(scenarios_to_process)} scenarios to assess.")

    if not scenarios_to_process:
        print("No scenarios found. Aborting.")
        return

    # Step 3: Loop through and calculate/update
    processed_count = 0
    for params in scenarios_to_process:
        # Calculate the new probability based on scenario parameters and precursor data
        new_prob = calculate_emergence_probability(params, precursor_intensity)

        # Update the scenario in the database
        update_scenario_probability(
            SCENARIO_DB_PATH,
            TABLE_NAME,
            params['id'],
            new_prob,
            params['full_data']
        )
        processed_count += 1
        print(f"   [UPDATED] Scenario ID: {params['id']} | New P(E): {new_prob:.4f}")

    print(f"--- Assessment Complete. {processed_count} scenarios updated. ---")


if __name__ == '__main__':
    run_probabilistic_assessment()