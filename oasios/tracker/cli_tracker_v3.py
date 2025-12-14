# oasios/tracker/cli_tracker_v3.py
# ECO Orchestrator CLI — Exposing ERL, FSAL, APSL, and SIL commands.

import typer
import requests
from typing import Optional
import structlog
import json

# 💡 V3 Imports
from oasios.tracker.core_t_v3 import CoreTracker
from oasios.tracker.database_t_v3 import DatabaseManager
from oasios.tracker.scenario_interface_v3 import ScenarioInterface  # NEW: SIL module

log = structlog.get_logger()
app = typer.Typer(help="ECO Orchestrator — High-Assurance AI Capability Foresight Engine")

# 💡 Initialize V3 instances
tracker_instance = CoreTracker()
db_manager = DatabaseManager()
scenario_interface = ScenarioInterface()


def safe_execute(func, *args, **kwargs):
    """Safely executes a CoreTracker or Engine method."""
    func_name = func.__name__
    try:
        result = func(*args, **kwargs)
        return result
    except requests.exceptions.RequestException as e:
        typer.echo(f"\n🚨 Critical API Error while executing {func_name}: {e}", err=True)
        log.error("cli_api_failed", function=func_name, error=str(e))
        return 0
    except Exception as e:
        typer.echo(f"\n❌ An unexpected error occurred in {func_name}: {e}", err=True)
        log.error("cli_unexpected_error", function=func_name, error=str(e))
        return 0


# --- Core Orchestration Commands ---

@app.command(name="sweep")
def full_sweep():
    """
    Runs the complete 4-stage pipeline: ERL -> FSAL -> APSL.
    This triggers ingestion, feature extraction, anomaly inference, and pattern synthesis.
    """
    typer.echo("🚀 Starting ECO Full Sweep (Ingestion, Feature, Anomaly Synthesis)...")
    # Delegate the entire process to the Orchestrator
    safe_execute(tracker_instance.run_full_sweep)
    typer.echo("\n✅ ECO Pipeline complete. Results are stored in anomaly_groups.")


# --- APSL/SIL Reporting Commands (The new value-add) ---
# Separated from the old 'analyze' command to reflect the new pipeline

@app.command(name="list-patterns")
def list_patterns(k: int = typer.Option(5, "--top-k", "-k", help="Top K patterns ranked by Emergence Index")):
    """
    Lists the top K systemic pattern groups identified by the APSL.
    These patterns are candidates for scenario modeling.
    """
    typer.echo(f"🔍 Querying top {k} Anomaly Pattern Groups (ranked by Emergence Index)...")

    try:
        # 💡 NEW: Use the SIL interface to query the APSL output
        top_groups = scenario_interface.get_top_k_active_groups(k=k)

        if not top_groups:
            typer.echo("No anomaly groups found. Run 'eco sweep' first.")
            return

        typer.echo(f"\n✨ Top {k} Systemic Emergence Patterns:")
        for i, group in enumerate(top_groups):
            typer.echo(f"--- Pattern #{i + 1} ({group['primary_type'].upper()}) ---")
            typer.echo(f"  ID: {group['group_id']}")
            typer.echo(f"  Emergence Index (Epsilon): {group['emergence_index_epsilon']:.3f}")
            typer.echo(f"  Coherence (Kappa): {group['coherence_kappa']:.3f}")
            typer.echo(f"  Cross-Domain Span (Xi): {group['cross_domain_span_xi']:.3f}")
            typer.echo("------------------------------------")

    except Exception as e:
        typer.echo(f"Pattern query failed: {e}", err=True)


@app.command(name="scenario")
def generate_scenario(group_id: str = typer.Argument(..., help="The ID of the anomaly group to simulate.")):
    """
    Generates a structured Scenario Seed (S) from a pattern group (G) for external simulation.
    """
    typer.echo(f"🧠 Generating Scenario Seed for Group ID: {group_id}...")

    try:
        # 💡 NEW: Generate the full Scenario Seed (S)
        seed = scenario_interface.generate_scenario_seed(group_id)

        if not seed:
            typer.echo(f"Group ID {group_id} not found or seed generation failed.", err=True)
            return

        # Display the output clearly
        typer.echo("\n--- Scenario Seed (Conceptual Initial Conditions) ---")
        typer.echo(json.dumps(seed, indent=4))
        typer.echo("-----------------------------------------------------")
        typer.echo("\nUse V1/V2/V3 variables as input for foresight models.")

    except Exception as e:
        typer.echo(f"Scenario generation failed: {e}", err=True)


@app.command(name="report")
def generate_policy_report(group_id: str = typer.Argument(..., help="The ID of the anomaly group for the report.")):
    """
    Generates a human-readable Policy Foresight Report based on a pattern group.
    """
    typer.echo(f"📰 Generating Policy Report for Group ID: {group_id}...")

    try:
        # 💡 NEW: Use the helper report function
        report = scenario_interface.generate_policy_report(group_id)
        typer.echo(report)
        typer.echo("\n--- Governance Log Entry Recorded (eco.governance.log_access) ---")
    except Exception as e:
        typer.echo(f"Report generation failed: {e}", err=True)


# --- Callback/Main Logic ---

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    ECO Orchestrator CLI (V3)
    (Emergent Capability Observatory)
    """
    if ctx.invoked_subcommand is None:
        # Default action is to run the full sweep
        ctx.invoke(full_sweep)


if __name__ == "__main__":
    app()