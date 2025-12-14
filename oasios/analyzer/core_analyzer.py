# oasios/analyzer/core_analyzer.py

import logging
import random # <--- Make sure this line exists
from typing import Dict, Any, List

# --- FIX: Ensure 'calculate_target_coherence_score' is imported ---
from .linkage import get_signal_alignment_score, calculate_target_coherence_score

# Setup logging configuration (omitted for brevity)
logger = logging.getLogger(__name__)


class ScenarioAnalyzer:
    """
    Core class for evaluating a scenario's plausibility index (PI).
    """

    # Inside the ScenarioAnalyzer class in oasios/analyzer/core_analyzer.py

    class ScenarioAnalyzer:
        # ... (existing methods like __init__, evaluate_scenario, etc.)

        def select_parents(self, population: List[Dict[str, Any]], num_parents_needed: int) -> List[Dict[str, Any]]:
            """
            Selects candidate scenarios from the population to be parents for the next generation.

            This implementation uses a simplified Tournament Selection process:
            1. Randomly selects a small 'tournament_size' number of individuals.
            2. Picks the individual with the highest 'plausibility_index' (fittest) from that group.
            3. Repeats until 'num_parents_needed' individuals are selected.

            Args:
                population: The list of scenario dictionaries.
                num_parents_needed: The total number of parents required (e.g., 2x num_children).

            Returns:
                A list of selected parent scenario dictionaries.
            """
            if not population:
                self.log.warning("SELECTION_EMPTY_POPULATION", reason="Cannot select parents from an empty population.")
                return []

            # Determine the tournament size. A common ratio is 2-5, or sqrt(population_size)
            tournament_size = max(2, int(len(population) ** 0.5))

            selected_parents: List[Dict[str, Any]] = []

            while len(selected_parents) < num_parents_needed:
                # 1. Randomly select candidates for the tournament
                # It's important to use k=min(tournament_size, len(population)) to avoid errors
                # if the population size is unexpectedly small.
                try:
                    tournament_candidates = random.sample(population, k=min(tournament_size, len(population)))
                except ValueError as e:
                    # This should ideally not happen if k is handled correctly, but good for robustness
                    self.log.error("SELECTION_ERROR", error=str(e), k=min(tournament_size, len(population)),
                                   pop_size=len(population))
                    break

                # 2. Find the winner: the scenario with the highest plausibility_index
                winner = max(tournament_candidates, key=lambda s: s.get('plausibility_index', 0.0))

                # 3. Add the winner to the selected parents list
                selected_parents.append(winner)

            self.log.info("SELECTION_COMPLETE", selected=len(selected_parents), needed=num_parents_needed)
            return selected_parents

        # ... (other existing methods)

    def __init__(self, factor_weights: Dict[str, float] = None):
        # Default factor weights for the Plausibility Index (PI) calculation:
        self.FACTOR_WEIGHTS = {
            "p1_signal_alignment": 0.25,
            "p2_consistency": 0.35,
            "p3_complexity_penalty": 0.2,
            "p4_coherence": 0.2
        }

        if factor_weights:
            self.FACTOR_WEIGHTS.update(factor_weights)

        logger.info({"event": "Analyzer initialized with custom factor weights, including P4 coherence."})

    def get_plausibility_index(self, scenario_id: str, params: Dict[str, Any]) -> float:
        """
        Calculates the weighted Plausibility Index (PI) for a single scenario.

        The PI is a composite score of four factors (P1-P4).

        Args:
            scenario_id: Unique ID of the scenario.
            params: Dictionary containing all scenario parameters.

        Returns:
            The final normalized Plausibility Index (0.0 to 1.0).
        """

        # --- P1: Signal Alignment Score ---
        # How well the scenario's fixed parameters align with current signals/features.
        p1_alignment = get_signal_alignment_score(scenario_id, params)

        # --- P2: Consistency Score (High scores mean high consistency) ---
        # NOTE: This is a placeholder for the actual consistency check.
        # Consistency checks usually compare parameters against defined rulesets.
        p2_consistency = 1.0  # Placeholder: Assume maximum consistency for now.

        # --- P3: Complexity Penalty (Penalty for high-complexity, low-resilience scenarios) ---
        # The complex calculation here is a placeholder for a dedicated function.
        architecture_complexity = {"hybrid": 0.4, "monolithic": 0.2, "modular": 0.6}.get(params.get("architecture"),
                                                                                         0.5)
        substrate_resilience = {"robust": 0.9, "medium": 0.5, "fragile": 0.1}.get(params.get("substrate_resilience"),
                                                                                  0.5)

        # The complexity score is calculated here (e.g., complexity * (1 - resilience))
        p3_complexity = architecture_complexity * (1.0 - substrate_resilience)
        # We invert it for the "penalty" factor, keeping it low for a high PI.
        p3_complexity_penalty = p3_complexity

        # --- P4: Target Coherence Score (Thematic and Narrative Logic) ---
        # Coherence depends heavily on the narrative/timeline, but uses calculated factors.

        # Create a parameters dictionary for P4 analysis
        params_for_p4 = params.copy()
        params_for_p4["p1_alignment_score"] = p1_alignment
        params_for_p4["p2_consistency_score"] = p2_consistency
        params_for_p4["p3_complexity_penalty"] = p3_complexity_penalty

        # Crude inversion of penalty for oversight effectiveness (used in the locals)
        p2_oversight_effectiveness_raw = {"partial": 0.5, "high": 0.8, "none": 0.0}.get(
            params.get("oversight_effectiveness"), 0.5)
        params_for_p4["oversight_effectiveness_P2"] = 1.0 - (1.0 - p2_oversight_effectiveness_raw)

        # FIX APPLIED HERE: Pass the required two arguments: scenario_id and params_for_p4
        p4_coherence = calculate_target_coherence_score(scenario_id, params_for_p4)

        # --- Composite Index Calculation ---
        # The Plausibility Index is the weighted sum of four factors.

        plausibility_index = (
                (p1_alignment * self.FACTOR_WEIGHTS["p1_signal_alignment"]) +
                (p2_consistency * self.FACTOR_WEIGHTS["p2_consistency"]) +
                ((1.0 - p3_complexity_penalty) * self.FACTOR_WEIGHTS[
                    "p3_complexity_penalty"]) +  # (1-penalty) to make low penalty = high score
                (p4_coherence * self.FACTOR_WEIGHTS["p4_coherence"])
        )

        # Note: The sum of weights is 1.0, so the result is already normalized (0.0 to 1.0).

        # You would typically have a database update function here,
        # but for CLI/GA purposes, we just return the calculated index.
        # self.db_handler.update_scenario_plausibility(scenario_id, plausibility_index)

        return plausibility_index

# --- Placeholder for database handler or other methods if they were in core_analyzer.py ---