# oasios/analyzer/generator_ga.py
"""
Handles the Reproduction phase of the Genetic Algorithm:
Crossover and Mutation of scenario parameters to generate the next generation.
"""
import random
from typing import Dict, Any, List, Union

from oasios.logger import log

# --- Dependency Assumption ---
# This function is assumed to be the core sampling logic from s_generator/params_s.py
# It takes a parameter key (e.g., 'architecture') and returns a valid random value.
try:
    # Use the real sampler if available
    from oasios.s_generator.params_s import sample_single_parameter
except ImportError:
    # Define a mock sampler for development if the real one isn't imported yet
    def sample_single_parameter(key: str) -> Union[str, float, int, List[str], None]:
        """MOCK: Returns a random value for a given parameter key."""
        if key == 'agency_level': return round(random.uniform(0.0, 1.0), 2)
        if key == 'alignment_score': return round(random.uniform(0.0, 1.0), 4)
        if key == 'architecture': return random.choice(['modular', 'swarm', 'monolithic'])
        if key == 'stated_goal': return random.choice(['survival', 'power', 'optimization'])
        # Return None for unknown keys or complex types like mesa_goals
        return None

    # --- Breeding Functions ---


def perform_crossover(parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines the parameters of two parent scenarios using Uniform Crossover.
    The child randomly inherits each parameter's value from Parent 1 or Parent 2.

    Args:
        parent1_params: Parameters dict of the first parent.
        parent2_params: Parameters dict of the second parent.

    Returns:
        A new parameter dictionary (the child's raw genes).
    """
    child_params = {}

    # Use the keys from the first parent as the template
    all_keys = parent1_params.keys()

    for key in all_keys:
        val1 = parent1_params.get(key)
        val2 = parent2_params.get(key)

        # Decide which parent's value to inherit (50/50 chance)
        # Prioritize non-None values if one is missing, otherwise random choice
        if val1 is None and val2 is not None:
            child_params[key] = val2
        elif val2 is None and val1 is not None:
            child_params[key] = val1
        elif val1 is not None and val2 is not None:
            child_params[key] = val1 if random.random() < 0.5 else val2
        else:
            child_params[key] = None  # Both are None

    return child_params


def apply_mutation(params: Dict[str, Any], mutation_rate: float = 0.08) -> Dict[str, Any]:
    """
    Applies mutation to the child parameters. Each parameter has a chance (mutation_rate)
    to be randomly re-sampled using the base speculative sampler.

    Args:
        params: The parameter dictionary after crossover.
        mutation_rate: The probability (0.0 to 1.0) of any single parameter mutating.

    Returns:
        The parameter dictionary after potential mutations (the child's final genes).
    """
    mutated_params = params.copy()

    for key in mutated_params.keys():
        # Check if this specific parameter should be mutated
        if random.random() < mutation_rate:
            # Resample the value using the base generator's logic
            new_value = sample_single_parameter(key)

            # Mutation only occurs if a valid new value is returned by the sampler
            if new_value is not None and new_value != mutated_params[key]:
                log.debug(
                    "breeder.mutation.applied",
                    key=key,
                    old_val=mutated_params[key],
                    new_val=new_value
                )
                mutated_params[key] = new_value

    return mutated_params


def breed_new_generation(parents: List[Dict[str, Any]], mutation_rate: float = 0.08) -> List[Dict[str, Any]]:
    """
    Orchestrates the breeding process from a list of selected parents.
    Assumes the parents list has an even number of scenarios for pairing.

    Args:
        parents: List of selected parent scenarios (must contain 'params' key).
        mutation_rate: The mutation rate for each parameter.

    Returns:
        A list of new child parameter dictionaries ready for the LLM.
    """
    log.info("breeder.start", num_parents=len(parents))

    # Ensure parents list has an even length for pairing
    num_parents = len(parents)
    if num_parents < 2 or num_parents % 2 != 0:
        log.error("breeder.invalid_parents", msg="Need at least two, and an even number, of parents for pairing.")
        return []

    new_children_params = []

    # Iterate over parents, pairing them up (0 and 1, 2 and 3, etc.)
    for i in range(0, num_parents, 2):
        p1_params = parents[i]['params']
        p2_params = parents[i + 1]['params']

        # 1. Crossover
        child_raw_params = perform_crossover(p1_params, p2_params)

        # 2. Mutation
        child_final_params = apply_mutation(child_raw_params, mutation_rate)

        new_children_params.append(child_final_params)

    log.info("breeder.complete", num_children=len(new_children_params))
    return new_children_params