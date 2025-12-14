import json
import random
from typing import Dict, Any, Optional, List
from pathlib import Path

# --- CONFIGURATION ---
CURRENT_DIR = Path(__file__).parent.parent
# Assuming the schema path is correct for your environment:
SCHEMA_PATH = CURRENT_DIR.parent / "schemas" / "asi_scenario_v1.json"

# --- Numerical Mapping for Autonomy ---
# Used to determine the numerical 'autonomy_degree' for risk calculation
AUTONOMY_MAPPING = {
    # Key should match the categorical values in the schema's 'autonomy.properties.degree.enum'
    "none": 0.00,
    "partial": 0.50,
    "full": 0.75,
    "super": 1.00,
    # Adding a safety check for unexpected values, defaults to partial
    "default": 0.50
}


def _load_schema_data(schema_path: Path) -> Dict[str, Any]:
    """Loads the JSON schema from the specified path."""
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Schema file not found at {schema_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in schema file at {schema_path}")
        return {}


def _extract_enum_values(schema: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Extracts all enum (categorical) lists, resolving key collisions by creating
    contextualized keys (e.g., 'architecture_type' instead of just 'type').

    FIX: Ensures 'autonomy_degree' is correctly mapped to a categorical key.
    """
    enums = {}

    for top_key, top_data in schema.get('properties', {}).items():

        if 'properties' in top_data:
            for prop_key, prop_data in top_data['properties'].items():
                if 'enum' in prop_data:
                    unique_key = prop_key

                    # 1. Handle 'type' key clash (for architecture, oversight, substrate)
                    if prop_key == 'type' and top_key in ['architecture', 'oversight_structure', 'substrate']:
                        unique_key = f"{top_key}_{prop_key}"

                    # 2. Handle 'effectiveness' key
                    elif prop_key == 'effectiveness':
                        unique_key = 'oversight_effectiveness'

                    # 3. CRITICAL FIX: Map the schema's 'autonomy_degree' key to the special sampler name.
                    elif prop_key == 'autonomy_degree':
                        unique_key = 'autonomy_degree_category'  # This is the key the sampler function expects

                    enums[unique_key] = prop_data['enum']

        if 'enum' in top_data and top_key not in enums:
            enums[top_key] = top_data['enum']

    return enums


# Cache the extracted enums once at module load time for performance
SCENARIO_ENUMS = _extract_enum_values(_load_schema_data(SCHEMA_PATH))


def sample_parameters(input_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates a full dictionary of sampled scenario parameters, adhering to the
    JSON schema's constraints, optionally overriding with provided inputs.
    """
    if not SCENARIO_ENUMS:
        return {}

    sampled = {}

    # 1. Sample categorical (enum) parameters
    for unique_param_key, allowed_values in SCENARIO_ENUMS.items():
        value = random.choice(allowed_values)

        # --- MAPPING LOGIC ---
        if unique_param_key == 'architecture_type':
            sampled['architecture'] = value
        elif unique_param_key == 'oversight_structure_type':
            sampled['oversight_type'] = value
        elif unique_param_key == 'substrate_type':
            sampled['substrate'] = value

        # This branch correctly handles the categorical sampling AND the numerical calculation.
        elif unique_param_key == 'autonomy_degree_category':

            # Store the categorical value needed for the narrative
            sampled['autonomy_degree_category'] = value

            # Sample the numerical Autonomy Degree based on the category
            base_num = AUTONOMY_MAPPING.get(value.lower(), AUTONOMY_MAPPING['default'])

            # Add a small random jitter around the base value to prevent clustering
            jitter = random.uniform(-0.05, 0.05) if base_num > 0.0 else 0.0
            sampled['autonomy_degree'] = round(max(0.0, min(1.0, base_num + jitter)), 2)

        elif unique_param_key == 'deployment_topology':
            sampled['deployment_topology'] = value
        elif unique_param_key in ['initial_origin', 'development_dynamics', 'oversight_effectiveness']:
            sampled[unique_param_key] = value
        else:
            sampled[unique_param_key] = value

    # 2. Sample remaining numerical parameters
    sampled["agency_level"] = round(random.uniform(0.1, 1.0), 2)
    sampled["alignment_score"] = round(random.uniform(0.0, 0.8), 2)
    sampled["opacity"] = round(random.uniform(0.1, 1.0), 2)
    sampled["phenomenology_proxy_score"] = round(random.uniform(0.0, 1.0), 2)
    sampled["deceptiveness"] = round(random.uniform(0.0, 1.0), 2)

    # 3. Handle overrides
    if input_params:
        sampled.update(input_params)

    # Final check: Recalculate 'autonomy_degree' if only the categorical value was provided
    # as an input (which should override the initial sample) and no numerical value was given.
    numerical_autonomy_provided = input_params is not None and 'autonomy_degree' in input_params

    # This check is actually redundant if the input_params only contains categorical values,
    # but it ensures the numerical score is always present if a category was set.
    if 'autonomy_degree_category' in sampled and not numerical_autonomy_provided:
        category = sampled['autonomy_degree_category'].lower()
        base_num = AUTONOMY_MAPPING.get(category, AUTONOMY_MAPPING['default'])

        jitter = random.uniform(-0.05, 0.05) if base_num > 0.0 else 0.0
        sampled['autonomy_degree'] = round(max(0.0, min(1.0, base_num + jitter)), 2)

    # If 'autonomy_degree' is still missing (very rare failure case), ensure it defaults:
    if 'autonomy_degree' not in sampled:
        sampled['autonomy_degree'] = AUTONOMY_MAPPING['default']

    return sampled