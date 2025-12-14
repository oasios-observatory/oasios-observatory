# oasios/s_generator/cli_s.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entrypoint for precursor-informed scenario generation
"""

import typer
from oasios.s_generator.core_s import generate_scenario
from oasios.logger import log

app = typer.Typer(help="Generate speculative ASI scenarios using core_s pipeline.")


@app.command()
def generate(n: int = typer.Option(
    default=None,
    help="Number of scenarios to generate. If not provided, you will be prompted."
)):
    """Generate N ASI scenarios."""

    # If not provided via CLI, prompt the user interactively
    if n is None:
        try:
            n = int(typer.prompt("Enter number of scenarios to generate"))
        except ValueError:
            typer.echo("Invalid input. Defaulting to 1.")
            n = 1

    log.info("starting_generation", total=n)
    typer.echo(f"\n🧠 Generating {n} scenario{'s' if n != 1 else ''}...\n")

    for i in range(1, n + 1):
        log.info("generating", i=i)
        scenario = generate_scenario()
        if scenario:
            typer.echo(f"✅ Generated: {scenario['title']}")
        else:
            typer.echo(f"❌ Scenario {i} generation failed.")
    typer.echo("\n✨ Done.\n")


if __name__ == "__main__":
    app()