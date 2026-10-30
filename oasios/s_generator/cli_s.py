#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/cli_s.py

"""CLI entrypoint for speculative ASI scenario generation."""

from typing import Annotated

import typer

from oasios.logger import log
from oasios.s_generator.core_s import (
    ScenarioGenerationError,
    generate_scenario,
)

app = typer.Typer(
    help="Generate speculative ASI scenarios using the OASIOS pipeline.",
    no_args_is_help=False,
)


@app.command()
def generate(
    n: Annotated[
        int | None,
        typer.Option(
            "--n",
            "-n",
            min=1,
            help="Number of scenarios to generate. Prompts if omitted.",
        ),
    ] = None,
) -> None:
    """Generate one or more ASI scenarios."""

    # ---------------------------------------------------------
    # 1. INPUT
    # ---------------------------------------------------------

    if n is None:
        try:
            n = typer.prompt(
                "Enter number of scenarios to generate",
                default=1,
                type=int,
            )

        except (typer.Abort, KeyboardInterrupt):
            typer.echo("\nGeneration cancelled.", err=True)
            raise typer.Exit(code=130)

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        typer.echo(
            "Error: The number of scenarios must be at least 1.",
            err=True,
        )
        raise typer.Exit(code=2)

    # ---------------------------------------------------------
    # 2. BATCH INITIALIZATION
    # ---------------------------------------------------------

    succeeded = 0
    rejected = 0
    failed = 0

    log.info(
        "generation.batch.started",
        total=n,
    )

    typer.echo(
        f"\n🧠 Generating {n} scenario{'s' if n != 1 else ''}...\n"
    )

    # ---------------------------------------------------------
    # 3. GENERATION
    # ---------------------------------------------------------

    for index in range(1, n + 1):

        typer.echo(f"Generating scenario {index}/{n}...")

        try:
            scenario = generate_scenario()

        except KeyboardInterrupt:

            log.warning(
                "generation.batch.interrupted",
                index=index,
                succeeded=succeeded,
                rejected=rejected,
                failed=failed,
            )

            typer.echo(
                "\nGeneration interrupted.",
                err=True,
            )

            raise typer.Exit(code=130)

        except ScenarioGenerationError as exc:

            failed += 1

            log.error(
                "generation.scenario.failed",
                index=index,
                stage=exc.stage,
                error=str(exc),
            )

            typer.echo(
                f"❌ Scenario {index} failed during {exc.stage}.",
                err=True,
            )

            continue

        except Exception as exc:

            failed += 1

            log.exception(
                "generation.scenario.unexpected_error",
                extra={
                    "index": index,
                    "error_type": type(exc).__name__,
                },
            )

            typer.echo(
                f"❌ Scenario {index} failed unexpectedly.",
                err=True,
            )

            continue

        # -----------------------------------------------------
        # 4. RESULT CLASSIFICATION
        # -----------------------------------------------------

        if scenario is None:

            rejected += 1

            typer.echo(
                f"⚠️ Scenario {index} rejected by validation."
            )

            continue

        succeeded += 1

        typer.echo(
            f"✅ Generated: {scenario['title']}"
        )

    # ---------------------------------------------------------
    # 5. BATCH SUMMARY
    # ---------------------------------------------------------

    typer.echo("\n✨ Generation finished.")

    typer.echo(f"   Successful: {succeeded}")
    typer.echo(f"   Rejected:   {rejected}")
    typer.echo(f"   Failed:     {failed}")
    typer.echo(f"   Total:      {n}")

    log.info(
        "generation.batch.completed",
        total=n,
        succeeded=succeeded,
        rejected=rejected,
        failed=failed,
    )

    # ------------------------------
    #---------------------------
    # 6. EXIT STATUS
    # ---------------------------------------------------------

    # Rejections are expected quality-control outcomes.
    # Operational failures produce a nonzero exit code.

    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
