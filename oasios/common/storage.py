#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/storage.py

import sqlite3
import json
import uuid
from contextlib import closing
from pathlib import Path
from typing import Dict, Any, List, Optional
from oasios.config import settings
from oasios.logger import log


# ------------------------------------------------------------
# DB Setup
# ------------------------------------------------------------

def _ensure_db_path():
    """Ensure database directory + file exist."""
    # NOTE: Assuming settings.db_path holds the path to asi_scenarios.db
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        db_path.touch()
        log.info("db.created", path=str(db_path))


def get_conn():
    """Return sqlite3 connection with row dicts enabled."""
    _ensure_db_path()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_table(table_name: str):
    """Create the standard scenario table schema, including GA fields."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            title TEXT,
            params TEXT,
            narrative TEXT,
            timeline TEXT,
            model_used TEXT,
            signals TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            plausibility_index REAL DEFAULT 0.0,
            generation_origin TEXT DEFAULT 'SPECULATIVE'
        );
        """
    )

    conn.commit()
    conn.close()


def init_db():
    """
    Create both tables used by generators.
    Always safe to call.
    """
    _ensure_db_path()

    init_table("s_scenarios")  # s-generator storage
    init_table("ev_scenarios")  # ev-generator storage

    log.info("storage.initialized", tables=["s_scenarios", "ev_scenarios"])
    print("[storage] Database initialized and tables ensured.")


# --- EV Scenario Counter ---
def get_next_ev_scenario_number() -> int:
    """
    Returns the next sequential number for an Evidence-Based (EV) scenario
    by counting existing entries in the 'ev_scenarios' table.
    """
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ev_scenarios")
    count = cursor.fetchone()[0]
    conn.close()
    return count + 1


# --------------------------------
# V2.0: Core DB Operations (Required for Cleanup Script)
# --------------------------------

class Storage:
    """
    A lightweight wrapper for core database operations, providing
    access to the low-level functions. Used by the checker/cleanup tools.
    """

    def __init__(self):
        # We don't need a persistent connection object here, just the methods
        pass

    def initialize(self):
        """Ensures the DB file and tables exist."""
        init_db()

    def get_rows_by_condition(self, table_name: str, columns: Optional[List[str]] = None, condition: str = "1=1") -> \
    List[Dict[str, Any]]:
        """
        Fetches rows from a table, optionally restricting columns and using a WHERE clause.

        Args:
            table_name: The name of the table to query.
            columns: A list of columns to select. If None, selects all columns (*).
            condition: The WHERE clause (default "1=1" means select all).

        Returns:
            A list of dictionaries representing the selected rows.
        """
        column_list = ", ".join(columns) if columns else "*"
        sql = f"SELECT {column_list} FROM {table_name} WHERE {condition};"

        with closing(get_conn()) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            # Fetch all results as a list of dictionaries. sqlite3.Row allows this via dict()
            results = [dict(row) for row in cur.fetchall()]
            return results

    def delete_rows(self, table_name: str, condition: str = "0=1") -> int:
        """
        Deletes rows from a table based on a WHERE clause.

        Args:
            table_name: The name of the table to delete from.
            condition: The WHERE clause (e.g., "id IN ('id1', 'id2')"). Default "0=1" is a safe no-op.

        Returns:
            The number of rows deleted.
        """
        if condition == "1=1":
            log.warning("db.delete_all_warning", table=table_name,
                        msg="Attempted to delete all rows with condition '1=1'. Preventing accidental mass deletion.")
            return 0  # Prevent accidental truncation without an explicit condition

        sql = f"DELETE FROM {table_name} WHERE {condition};"

        with closing(get_conn()) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            deleted_count = cur.rowcount
            conn.commit()
            return deleted_count


# ------------------------------------------------------------
# Saving Logic (Modified for GA tracking and title storage)
# ------------------------------------------------------------

# ... (Rest of the file remains the same) ...

def save_scenario(
        table_name: str,
        *,
        title: str,  # <--- CHANGE 2a: NEW ARGUMENT
        params,
        narrative,
        timeline,
        model_used,
        signals=None,
        generation_origin: str = 'SPECULATIVE'
):
    """
    Generic save function used by all generators.
    """
    scenario_id = str(uuid.uuid4())
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        f"""
        INSERT INTO {table_name}
        (id, title, params, narrative, timeline, model_used, signals, generation_origin)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            scenario_id,
            title,  # <--- CHANGE 2d: NEW VALUE
            json.dumps(params),
            narrative,
            json.dumps(timeline),
            model_used,
            json.dumps(signals or []),
            generation_origin,
        ]
    )

    conn.commit()
    conn.close()

    return scenario_id


# ------------------------------------------------------------
# Wrappers for S & EV Generators
# ------------------------------------------------------------

def save_scenario_s(*, title, params, narrative, timeline, model_used):
    """Wrapper for s-generator (no signals field, origin defaults to SPECULATIVE)."""
    return save_scenario(
        "s_scenarios",
        title=title,
        params=params,
        narrative=narrative,
        timeline=timeline,
        model_used=model_used,
        signals=[],
        generation_origin='SPECULATIVE'
    )


def save_scenario_ev(*, title: str, params, narrative, timeline, model_used, signals,
                     generation_origin: str = 'EVIDENCE'):
    """
    Wrapper for ev-generator, now including the generation_origin and title.
    Default origin is 'EVIDENCE'. GA loop will pass 'GA_CROSSOVER'.
    """
    return save_scenario(
        "ev_scenarios",
        title=title,  # <--- CHANGE 3: PASS TITLE
        params=params,
        narrative=narrative,
        timeline=timeline,
        model_used=model_used,
        signals=signals,
        generation_origin=generation_origin
    )


# ------------------------------------------------------------
# GA/Analyzer Functions
# ------------------------------------------------------------

def get_scenarios_from_db(table_name: str) -> List[Dict[str, Any]]:
    """
    Fetches all scenarios from the specified table. Crucial for Analyzer evaluation.
    (Kept for compatibility, though Storage.get_rows_by_condition is better)
    """
    # Simply delegate to the new Storage method
    storage = Storage()
    return storage.get_rows_by_condition(table_name)


def update_scenario_index(scenario_id: str, plausibility_index: float):
    """
    Updates the plausibility_index column for a specific scenario ID in the ev_scenarios table.
    """
    db_path = settings.db_path
    sql = """
          UPDATE ev_scenarios
          SET plausibility_index = ?
          WHERE id = ?; \
          """

    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(sql, (plausibility_index, scenario_id))
            conn.commit()
            log.debug("db.update_index.success", id=scenario_id[:8], index=plausibility_index)
    except sqlite3.OperationalError as e:
        log.error("db.update_index.failed", error=str(e), id=scenario_id[:8])
        # Inform the user about the likely cause
        print("\n--- DATABASE ERROR ---\n"
              "Operation failed! This likely means the 'plausibility_index' column "
              "was NOT added to the 'ev_scenarios' table via migration/init_db.\n"
              f"Error: {e}\n"
              "----------------------\n")


# --- Other Utilities ---
def save_asi_scenario(scenario: dict):
    """Save full v1-compliant scenario as JSON"""
    # Placeholder for future implementation
    pass