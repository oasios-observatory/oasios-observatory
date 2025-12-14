# oasios/analyzer/cli_analyzer.py
"""
CLI entrypoint for the Scenario Analyzer, managing the Genetic Algorithm (GA) evolution cycle.
"""
import typer
import json

from typing import Dict, Any, List
from oasios.logger import log
# Dependencies for Analysis, Breeding, and Generation
from oasios.analyzer.core_analyzer import ScenarioAnalyzer
from oasios.analyzer.generator_ga import breed_new_generation
from oasios.ev_generator.core_ev import generate_ev_scenario  # We'll call this to generate narratives
from oasios.common.storage import get_scenarios_from_db, save_scenario_ev, update_scenario_index

app = typer.Typer()


@app.command()
def evolve(
        generations: int = typer.Option(1, "--generations", "-g", help="Number of evolutionary cycles to run."),
        num_children: int = typer.Option(10, "--children", "-c",
                                         help="Number of new scenarios to generate in each cycle."),
        mutation_rate: float = typer.Option(0.08, "--mutation", "-m",
                                            help="Probability of a parameter mutating (0.0 to 1.0)."),
        population_size: int = typer.Option(50, "--population", "-p",
                                            help="Maximum size of the scenario population to maintain.")
):
    """
    Runs the automated Genetic Algorithm (GA) cycle to evolve the scenario population
    towards higher Plausibility Index scores.
    """
    analyzer = ScenarioAnalyzer()

    log.info("GA_EVOLUTION_START", generations=generations, children_per_cycle=num_children)

    for gen in range(1, generations + 1):
        log.info("GENERATION_START", generation=gen)

        # 1. EVALUATION (Score the entire population)
        # Fetch all EV scenarios (which contain the 'params' and 'id')
        population = get_scenarios_from_db(table_name="ev_scenarios")

        if not population:
            log.error("GA_FAIL", msg="No existing EV scenarios found to start the evolution.")
            return

        # Calculate/Update Plausibility Index for the entire population
        # NOTE: This updates the database with the new 'plausibility_index' column
        for scenario in population:
            params = json.loads(scenario['params'])
            index = analyzer.get_plausibility_index(scenario['id'], params)

            # Store the index back into the scenario dictionary
            scenario['plausibility_index'] = index
            update_scenario_index(scenario['id'], index)  # Assuming this function updates the DB

        # 2. SELECTION (Choose Parents)
        # We need pairs of parents to create 'num_children' children
        num_parents_needed = num_children * 2
        selected_parents = analyzer.select_parents(population, num_parents=num_parents_needed)

        if len(selected_parents) < num_parents_needed:
            log.warning("SELECTION_INSUFFICIENT", needed=num_parents_needed, selected=len(selected_parents))
            # Continue with what we have, or stop
            if len(selected_parents) < 2:
                log.error("GA_FAIL", msg="Insufficient parents selected to continue breeding.")
                return

        # 3. REPRODUCTION (Crossover and Mutation)
        # The breed function returns a list of new parameter dictionaries
        new_children_params = breed_new_generation(selected_parents, mutation_rate)

        # 4. GENERATION (Create Narratives)
        log.info("GENERATION_NARRATIVES_START", count=len(new_children_params))

        for child_params in new_children_params:
            # Call the EV generator's core pipeline, but pass the pre-generated genes.
            # We must modify generate_ev_scenario() to accept input parameters and
            # set the generation_origin flag to 'GA_CROSSOVER'.

            # MOCK CALL: Assuming generate_ev_scenario is modified to accept input_params
            scenario_data = generate_ev_scenario(input_params=child_params, origin="GA_CROSSOVER")

            # The scenario_data (id, params, narrative, etc.) is saved within generate_ev_scenario()

        # 5. POPULATION MANAGEMENT (Replacement/Culling)
        # Sort all scenarios by Plausibility Index and cull the weakest to maintain size.

        # Refetch the current total population including the new children
        current_population = get_scenarios_from_db(table_name="ev_scenarios")

        if len(current_population) > population_size:
            log.info("POPULATION_CULL_START", current_size=len(current_population), target_size=population_size)

            # Sort by Plausibility Index (lowest first)
            current_population.sort(key=lambda s: s.get('plausibility_index', 0.0))

            # Identify scenarios to delete
            num_to_cull = len(current_population) - population_size
            scenarios_to_delete = current_population[:num_to_cull]

            for scenario in scenarios_to_delete:
                # MOCK CALL: Assuming a function exists to delete scenarios by ID
                # delete_scenario_by_id(scenario['id'])
                log.debug("POPULATION_CULLED", id=scenario['id'][:8], index=scenario.get('plausibility_index'))

            log.info("POPULATION_CULL_COMPLETE", culled_count=num_to_cull)

        log.info("GENERATION_END", generation=gen)

    log.info("GA_EVOLUTION_COMPLETE")


if __name__ == "__main__":
    app()