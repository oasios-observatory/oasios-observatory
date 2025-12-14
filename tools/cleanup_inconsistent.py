#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/tools/cleanup_inconsistent.py
"""
CLI script to check all previously generated scenarios for consistency
against their numerical parameters and delete any inconsistent entries
from the database.
"""
import typer
from typing import Dict, Any, List, Tuple, Optional
import json  # Ensure json is imported for parsing params
from oasios.logger import log
# Assuming Storage is now correctly imported and implemented in storage.py
from oasios.common.storage import Storage
from oasios.common.consistency import NarrativeChecker

# Initialize the Typer application
app = typer.Typer(help="Tools for managing and cleaning the OASIOS scenario database.")


@app.command(name="check-and-clean")
def check_and_clean_database(
        table_name: str = typer.Option(
            "ev_scenarios", "--table", "-t",
            help="The name of the database table to check (e.g., s_scenarios or ev_scenarios)."
        )
):
    """
    Retrieves all scenarios from the specified table, checks their narrative
    consistency, and deletes any inconsistent ones.
    """
    log.info("cleanup.starting", table=table_name)
    typer.echo(f"🔍 Starting consistency check and cleanup for table: **{table_name}**")

    # 1. Initialize Storage and Database Connection
    storage: Optional[Storage] = None
    try:
        storage = Storage()
        storage.initialize()  # Ensure connection and tables exist
        # REMOVED: db = storage.db  <-- THIS CAUSED THE ERROR
    except Exception as e:
        log.error("cleanup.db_init_failed", error=str(e))
        typer.echo(f"❌ Error initializing database: {e}")
        return

    # 2. Fetch all scenarios
    try:
        # We now use the storage object's method directly.
        # Select 'id' not 'scenario_id' as per the table schema (id is the primary key)
        raw_scenarios = storage.get_rows_by_condition(table_name, columns=["id", "params", "narrative"])
        total_count = len(raw_scenarios)
        typer.echo(f"📄 Found {total_count} scenarios in **{table_name}** to check.")
        log.info("cleanup.scenarios_fetched", count=total_count)
    except Exception as e:
        log.error("cleanup.fetch_failed", table=table_name, error=str(e))
        typer.echo(f"❌ Error fetching scenarios from table {table_name}: {e}")
        return

    # 3. Perform Consistency Checks and Prepare Deletions
    inconsistent_ids: List[str] = []

    with typer.progressbar(raw_scenarios, label="Checking scenarios for inconsistencies...") as progress:
        for scenario in progress:
            # The primary key is 'id' in the database, not 'scenario_id'
            scenario_id = scenario['id']

            # Note: DB row values are retrieved by column name (lowercase)
            params = scenario['params']
            narrative = scenario['narrative']

            # Instantiate the checker with the parameters for the current scenario
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    log.error("cleanup.json_error", scenario_id=scenario_id,
                              msg="Params column is string but not valid JSON. Skipping.")
                    # Skip this scenario if parameters are corrupted
                    continue

            checker = NarrativeChecker(params)

            # Run the check
            is_consistent, failures = checker.check(narrative)

            if not is_consistent:
                inconsistent_ids.append(scenario_id)
                # Log the failure, including a specific check result for later review
                log.warning("cleanup.inconsistent_found", scenario_id=scenario_id, failures=failures)

    inconsistent_count = len(inconsistent_ids)

    # 4. Execute Deletions
    if inconsistent_count > 0:
        typer.echo(f"\n🗑️ Found **{inconsistent_count} inconsistent scenarios**. Deleting them now...")

        # Build the WHERE clause to delete all inconsistent IDs at once
        # WHERE id IN ('id1', 'id2', ...)
        id_list_str = ", ".join(f"'{id}'" for id in inconsistent_ids)
        condition = f"id IN ({id_list_str})"

        try:
            # Execute the deletion using the storage object's method
            # We already checked for None, but for type safety check again
            if storage:
                deleted_count = storage.delete_rows(table_name, condition=condition)
                typer.echo(f"✅ Successfully deleted {deleted_count} rows from **{table_name}**.")
                log.info("cleanup.deletion_complete", table=table_name, deleted_count=deleted_count)

        except Exception as e:
            log.error("cleanup.deletion_failed", table=table_name, error=str(e))
            typer.echo(f"❌ Error during deletion: {e}")

    else:
        typer.echo("\n🎉 No inconsistencies found. Database is clean!")
        log.info("cleanup.no_inconsistencies")

    # 5. Final Output
    final_consistent_count = total_count - inconsistent_count
    typer.echo(f"\n--- Cleanup Summary ---")
    typer.echo(f"Total Scenarios Checked: {total_count}")
    typer.echo(f"Scenarios Found Inconsistent: {inconsistent_count}")
    typer.echo(f"Scenarios Remaining Consistent: {final_consistent_count}")
    typer.echo(f"-----------------------")
    log.info("cleanup.summary", total=total_count, inconsistent=inconsistent_count, consistent=final_consistent_count)


if __name__ == "__main__":
    app()