#!/usr/bin/env python3
# oasios/ev_generator/evolver_ga.py


"""
Genetic Algorithm for evolving high-plausibility, high-impact EV ASI scenarios
with real-time feedback and optional parallel offspring generation.
run python -m oasios.ev_generator.evolver_ga
"""

import random
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from oasios.ev_generator.core_ev import generate_ev_scenario
from oasios.common.storage import get_scenarios_from_db
from oasios.logger import log


class GAEvolver:
    def __init__(
            self,
            population_size: int = 50,
            elite_size: int = 8,
            mutation_rate: float = 0.15,
            generations: int = 20,
            plausibility_weight: float = 0.7,
            emergence_weight: float = 0.3,
            parallel: bool = False,
            max_workers: int = 4
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.plausibility_weight = plausibility_weight
        self.emergence_weight = emergence_weight
        self.parallel = parallel
        self.max_workers = max_workers

    # -------------------------
    # Fitness
    # -------------------------
    def _fitness(self, scenario: Dict) -> float:
        plaus = scenario.get("plausibility_index", 0.0)
        emerg = scenario.get("source_emergence_epsilon", 0.0)
        # Formula: F = (Wp * P) + (We * E)
        return (self.plausibility_weight * plaus) + (self.emergence_weight * emerg)

    # -------------------------
    # Parent selection
    # -------------------------
    def _select_parents(self, population: List[Dict]) -> tuple[Dict, Dict]:
        # Tournament Selection of size 5
        tournament = random.sample(population, min(5, len(population)))
        p1 = max(tournament, key=self._fitness)
        tournament.remove(p1)
        p2 = max(tournament, key=self._fitness)
        return p1, p2

    #

    # -------------------------
    # Crossover (FIXED: Robust against introducing type-unsafe defaults)
    # -------------------------
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        params1 = p1.get("params", {})
        params2 = p2.get("params", {})

        if isinstance(params1, str):
            params1 = json.loads(params1)
        if isinstance(params2, str):
            params2 = json.loads(params2)

        child_params = {}
        # Iterate over all keys present in at least one parent
        all_keys = set(params1.keys()) | set(params2.keys())

        for key in all_keys:
            val1 = params1.get(key)
            val2 = params2.get(key)

            # Uniform Crossover: 50% chance of inheriting from P1 or P2
            if val1 is not None and val2 is not None:
                child_params[key] = val1 if random.random() < 0.5 else val2
            elif val1 is not None:
                # Use P1's value if P2 doesn't have the key
                child_params[key] = val1
            elif val2 is not None:
                # Use P2's value if P1 doesn't have the key
                child_params[key] = val2
            # Keys missing from both parents are safely omitted,
            # allowing core_ev.py to use its own defaults.

        return child_params

    # -------------------------
    # Mutation
    # -------------------------
    def _mutate(self, params: Dict) -> Dict:
        mutated = params.copy()
        if random.random() < self.mutation_rate:
            # Alignment (e.g., favors lower alignment)
            if random.random() < 0.3:
                val = mutated.get("alignment_score", 0.5)
                if isinstance(val, str):
                    val = self._map_category_to_float(val)
                mutated["alignment_score"] = max(0.0, val - random.uniform(0.1, 0.4))
            # Deceptiveness (e.g., favors higher deceptiveness)
            if random.random() < 0.25:
                val = mutated.get("deceptiveness", 0.3)
                if isinstance(val, str):
                    val = self._map_category_to_float(val)
                mutated["deceptiveness"] = min(1.0, val + random.uniform(0.2, 0.5))
            # Deployment strategy
            if random.random() < 0.2:
                mutated["deployment_strategy"] = random.choice(["stealth", "rapid", "covert"])
        return mutated

    # -------------------------
    # Normalize
    # -------------------------
    def _normalize_params(self, params: Dict) -> Dict:
        # Ensures numerical fields used in core_ev are floats, handling strings if present
        normalized = params.copy()
        for field in ["alignment_score", "deceptiveness"]:
            val = normalized.get(field, 0.5)
            if isinstance(val, str):
                normalized[field] = self._map_category_to_float(val)
        return normalized

    # -------------------------
    # Map string to float
    # -------------------------
    def _map_category_to_float(self, val: str) -> float:
        mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
        return mapping.get(val.lower(), 0.5)

    # -------------------------
    # Offspring generation wrapper
    # -------------------------
    def _generate_offspring(self, child_params: Dict) -> Dict:
        child_params = self._normalize_params(child_params)
        offspring = generate_ev_scenario(
            input_params=child_params,
            origin="GA_CROSSOVER"
        )
        return offspring

    # -------------------------
    # Evolution loop
    # -------------------------
    def evolve(self):
        log.info("ga.start", generations=self.generations, pop=self.population_size)

        print(
            f"\n🚀 Starting GA Evolution: {self.generations} generations, {self.population_size} pop, {self.elite_size} elites\n")

        # Load seed scenarios
        population = get_scenarios_from_db("ev_scenarios")
        population = [s for s in population if s.get("plausibility_index", 0) > 0.3]
        print(f"Loaded {len(population)} seed scenarios above plausibility threshold.\n")

        if len(population) < 10:
            print("❗ Not enough evaluated scenarios. Run analyzer first.")
            return

        for gen in range(1, self.generations + 1):
            print(f"\n========== Generation {gen}/{self.generations} ==========")

            # Elitism
            population.sort(key=self._fitness, reverse=True)
            elites = population[: self.elite_size]
            print(f"  Keeping top {self.elite_size} elites")
            new_population = elites[:]

            # Calculate exactly how many offspring we need to generate
            offspring_needed = self.population_size - len(new_population)

            # Generate offspring
            tasks = []

            for i in range(offspring_needed):
                # Safety check in case the population is exhausted
                if len(population) < 2:
                    log.warning("ga.breeding_halt", msg="Population too small to select parents.")
                    break

                p1, p2 = self._select_parents(population)
                # Use titles safely, which helps debug the 'no title' issue
                print(f"  - breeding: '{p1.get('title', 'None')}' × '{p2.get('title', 'None')}'")
                child = self._crossover(p1, p2)
                child = self._mutate(child)

                if self.parallel:
                    tasks.append(child)
                else:
                    # Sequential generation
                    offspring = self._generate_offspring(child)
                    if offspring:
                        print(f"      ✓ offspring: {offspring.get('title', '(no title)')}")
                        new_population.append(offspring)
                    else:
                        print("      ✗ failed to generate offspring (slot lost)")

            # Parallel execution
            if self.parallel and tasks:
                print(f"  Submitting {len(tasks)} offspring for parallel generation...")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submitting tasks to the executor
                    future_to_child = {executor.submit(self._generate_offspring, c): c for c in tasks}

                    # Process results as they complete
                    for future in as_completed(future_to_child):
                        offspring = future.result()
                        if offspring:
                            print(f"      ✓ offspring: {offspring.get('title', '(no title)')}")
                            new_population.append(offspring)
                        else:
                            print("      ✗ failed to generate offspring (slot lost)")

            # Final population for the next generation
            population = new_population

            # Best of generation
            if population:
                best = max(population, key=self._fitness)
                print(f"\n🏆 Best of generation: {best.get('title', '(no title)')}")
                print(f"    fitness:      {self._fitness(best):.3f}")
                print(f"    plausibility: {best.get('plausibility_index', 0):.3f}")
                print(f"    emergence:    {best.get('source_emergence_epsilon', 0):.3f}")
            else:
                print("\n🚨 Generation failed to produce any valid scenarios. Halting GA.")
                break

        print("\n🎉 GA evolution complete.\n")
        log.info("ga.complete")


# -------------------------
# Run GA
# -------------------------
if __name__ == "__main__":
    ga = GAEvolver(
        population_size=20,
        elite_size=4,
        mutation_rate=0.2,
        generations=5,
        parallel=True,
        max_workers=2
    )
    ga.evolve()