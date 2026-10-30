# oasios/tracker/cli_t.py

import typer
import structlog
import json

from oasios.tracker.core_t import CoreTracker
from oasios.tracker.scenario_interface_t import ScenarioInterface

log = structlog.get_logger()

app = typer.Typer(
    help="OASIOS — AI Capability Foresight Engine",
    no_args_is_help=False,
)

tracker_instance = CoreTracker()

scenario_interface = ScenarioInterface()


def safe_execute(func, *args, **kwargs):

    try:

        return func(*args, **kwargs)

    except Exception:

        log.exception(
            "cli_execution_failed",
            function=func.__name__,
        )

        typer.echo(
            f"Operation failed: {func.__name__}",
            err=True,
        )

        raise typer.Exit(code=1)


@app.command(name="sweep")
def full_sweep():

    typer.echo(
        "\nStarting OASIOS Full Sweep..."
    )

    result = safe_execute(
        tracker_instance.run_full_sweep
    )

    typer.echo(
        json.dumps(result, indent=4)
    )

    if not result["successful"]:

        typer.echo(
            "\nPipeline failed or was incomplete.",
            err=True,
        )

        raise typer.Exit(code=1)

    typer.echo(
        "\nPipeline completed successfully."
    )


@app.command(name="list-patterns")
def list_patterns(

    k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        min=1,
        max=100,
        help="Number of patterns to display.",
    )

):

    typer.echo(
        f"\nQuerying top {k} anomaly groups..."
    )

    groups = safe_execute(
        scenario_interface.get_top_k_active_groups,
        k,
    )

    if not groups:

        typer.echo(
            "No anomaly groups found. Run 'sweep' first."
        )

        return

    typer.echo(
        "\nSYSTEMIC EMERGENCE PATTERNS"
    )

    for index, group in enumerate(groups, 1):

        typer.echo(
            f"\n--- Pattern #{index} ---"
        )

        typer.echo(
            f"ID: {group['group_id']}"
        )

        typer.echo(
            f"Type: {group['primary_type']}"
        )

        typer.echo(
            f"Emergence Index: "
            f"{group['emergence_index_epsilon']:.3f}"
        )

        typer.echo(
            f"Coherence: "
            f"{group['coherence_kappa']:.3f}"
        )

        typer.echo(
            f"Cross-Domain Span: "
            f"{group['cross_domain_span_xi']:.3f}"
        )


@app.command(name="scenario")
def generate_scenario(

    group_id: str = typer.Argument(
        ...,
        help="An existing anomaly group ID.",
    )

):

    typer.echo(
        f"\nGenerating scenario for {group_id}..."
    )

    seed = safe_execute(
        scenario_interface.generate_scenario_seed,
        group_id,
    )

    if not seed:

        typer.echo(
            "Group not found.",
            err=True,
        )

        raise typer.Exit(code=1)

    typer.echo(
        json.dumps(
            seed,
            indent=4,
        )
    )


@app.command(name="report")
def generate_policy_report(

    group_id: str = typer.Argument(
        ...,
        help="An existing anomaly group ID.",
    )

):

    report = safe_execute(
        scenario_interface.generate_policy_report,
        group_id,
    )

    if report.startswith("Error:"):

        typer.echo(report, err=True)

        raise typer.Exit(code=1)

    typer.echo(report)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):

    if ctx.invoked_subcommand is None:

        ctx.invoke(full_sweep)


if __name__ == "__main__":

    app()
