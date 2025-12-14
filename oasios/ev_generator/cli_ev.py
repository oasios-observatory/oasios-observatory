#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/ev_generator/cli_ev.py
"""
CLI entrypoint for signal-informed EV scenario generation and evolution.
"""

import typer
from oasios.ev_generator.core_ev import generate_ev_scenario
from oasios.logger import log
from oasios.ev_generator.evolver_ga import GAEvolver


# Initialize the main Typer application
app = typer.Typer(help="Generate and evolve signal-informed EV ASI scenarios.")

# ----------------------------------------------------
# 1. COMMAND: generate (Evidence-Based Scenarios)
# ----------------------------------------------------

@app.command()
def generate(n: int = typer.Option(
    default=None,
    help="Number of EV scenarios to generate. If not provided, you will be prompted."
)):
    """Generate N EV ASI scenarios."""

    # Interactive prompt if n is not provided
    if n is None:
        try:
            n = int(typer.prompt("Enter number of EV scenarios to generate"))
        except ValueError:
            typer.echo("Invalid input. Defaulting to 1.")
            n = 1

    log.info("ev.starting_generation", total=n)
    typer.echo(f"\n🧠 Generating {n} EV scenario{'s' if n != 1 else ''}...\n")

    for i in range(1, n + 1):
        log.info("ev.generating", i=i)
        typer.echo(f"⚙️  Generating scenario {i} of {n}...\n")
        scenario = generate_ev_scenario()
        if scenario:
            typer.echo(f"✅ Generated: {scenario['title']}\n")
        else:
            typer.echo(f"❌ Scenario {i} generation failed.\n")

    typer.echo("\n✨ Done.\n")


if __name__ == "__main__":
    app()


# ----------------------------------------------------
# 2. COMMAND: evolve (Genetic Algorithm Process)
# ----------------------------------------------------

@app.command(name="evolve")
def evolve_scenarios(
    generations: int = typer.Option(5, "--generations", "-g", help="Number of GA generations to run."),
    population: int = typer.Option(20, "--population", "-p", help="Population size for each generation."),
    workers: int = typer.Option(4, "--workers", "-w", help="Max parallel workers for offspring generation."),
    # Boolean option: Typer handles the --no-parallel flag automatically
    parallel: bool = typer.Option(True, help="Enable parallel processing for offspring generation. Use --no-parallel to disable.")
):
    """Run the Genetic Algorithm to evolve high-fitness scenarios from the existing population."""

    typer.echo(f"\n🧬 Starting GA Evolution for {generations} generations...")
    typer.echo(f"   Pop Size: {population}, Parallel: {parallel}, Workers: {workers}\n")

    ga = GAEvolver(
        population_size=population,
        generations=generations,
        parallel=parallel,
        max_workers=workers
    )
    ga.evolve()

    typer.echo("\n--- GA Evolution complete ---")


if __name__ == "__main__":
    app()