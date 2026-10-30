#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLite persistence for OASIOS scenarios."""

import json
import math
import sqlite3
import uuid

from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from oasios.config import settings
from oasios.logger import log


ALLOWED_TABLES = frozenset({
    "s_scenarios",
    "ev_scenarios",
})


class StorageError(RuntimeError):
    """Raised when a scenario persistence operation fails."""


def _validate_table_name(table_name: str) -> str:
    """Validate table names against the application whitelist."""

    if table_name not in ALLOWED_TABLES:
        raise ValueError(
            f"Unsupported database table: {table_name!r}"
        )

    return table_name


def _ensure_db_path() -> Path:
    """Create the database directory if necessary."""

    db_path = Path(settings.db_path)

    try:
        db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as exc:
        raise StorageError(
            f"Unable to prepare database path: {exc}"
        ) from exc

    return db_path


def get_conn() -> sqlite3.Connection:
    """Create a configured SQLite connection."""

    db_path = _ensure_db_path()

    try:
        conn = sqlite3.connect(
            str(db_path),
            timeout=30,
        )

        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA foreign_keys = ON")

        return conn

    except sqlite3.Error as exc:
        raise StorageError(
            f"Unable to connect to database: {exc}"
        ) from exc


def _table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    """Check whether a table exists."""

    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def _get_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    """Return existing table column names."""

    rows = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return {
        row["name"]
        for row in rows
    }


def init_table(table_name: str) -> None:
    """Create or migrate a scenario table."""

    table_name = _validate_table_name(table_name)

    try:
        with closing(get_conn()) as conn:
            with conn:

                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT '',
                        params TEXT NOT NULL DEFAULT '{{}}',
                        narrative TEXT NOT NULL DEFAULT '',
                        timeline TEXT NOT NULL DEFAULT '[]',
                        model_used TEXT,
                        signals TEXT NOT NULL DEFAULT '[]',
                        created_at TIMESTAMP
                            DEFAULT CURRENT_TIMESTAMP,
                        plausibility_index REAL DEFAULT 0.0,
                        generation_origin TEXT
                            DEFAULT 'SPECULATIVE',
                        scenario_data TEXT
                    )
                    """
                )

                existing = _get_columns(
                    conn,
                    table_name,
                )

                migrations = {
                    "title": "TEXT NOT NULL DEFAULT ''",
                    "params": "TEXT NOT NULL DEFAULT '{}'",
                    "narrative": "TEXT NOT NULL DEFAULT ''",
                    "timeline": "TEXT NOT NULL DEFAULT '[]'",
                    "model_used": "TEXT",
                    "signals": "TEXT NOT NULL DEFAULT '[]'",
                    "created_at": "TIMESTAMP",
                    "plausibility_index": "REAL DEFAULT 0.0",
                    "generation_origin": (
                        "TEXT DEFAULT 'SPECULATIVE'"
                    ),
                    "scenario_data": "TEXT",
                }

                for column, definition in migrations.items():

                    if column not in existing:

                        conn.execute(
                            f"""
                            ALTER TABLE {table_name}
                            ADD COLUMN {column} {definition}
                            """
                        )

                        log.info(
                            "db.column.migrated",
                            table=table_name,
                            column=column,
                        )

    except sqlite3.Error as exc:

        log.error(
            "db.table_init.failed",
            table=table_name,
            error=str(exc),
        )

        raise StorageError(
            f"Unable to initialize table {table_name}: {exc}"
        ) from exc


def init_db() -> None:
    """Initialize and migrate all application tables."""

    _ensure_db_path()

    init_table("s_scenarios")
    init_table("ev_scenarios")

    log.info(
        "storage.initialized",
        tables=sorted(ALLOWED_TABLES),
    )


def _serialize(value: Any) -> str:
    """Serialize data using strict JSON compliance."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to serialize scenario data: {exc}"
        ) from exc


def save_asi_scenario(
    scenario: dict[str, Any],
    table_name: str = "s_scenarios",
) -> str:
    """
    Persist a complete scenario.

    The complete JSON object is authoritative.
    Legacy columns remain available for compatibility.
    """

    table_name = _validate_table_name(table_name)

    if not isinstance(scenario, dict):
        raise TypeError(
            "Scenario must be a dictionary."
        )

    scenario_id = scenario.get("id")
    title = scenario.get("title")

    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Scenario ID is required.")

    try:
        uuid.UUID(scenario_id)
    except ValueError as exc:
        raise ValueError(
            "Scenario ID must be a valid UUID."
        ) from exc

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Scenario title is required.")

    metadata = scenario.get("metadata", {})
    provenance = metadata.get("provenance", {})

    params = {
        "origin": scenario.get("origin", {}),
        "architecture": scenario.get("architecture", {}),
        "substrate": scenario.get("substrate", {}),
        "oversight_structure": scenario.get(
            "oversight_structure", {}
        ),
        "core_capabilities": scenario.get(
            "core_capabilities", {}
        ),
        "goals_and_behavior": scenario.get(
            "goals_and_behavior", {}
        ),
        "impact_and_control": scenario.get(
            "impact_and_control", {}
        ),
    }

    content = scenario.get("scenario_content", {})

    narrative = content.get("narrative", "")

    timeline = content.get("timeline", {}).get(
        "phases",
        [],
    )

    signals = scenario.get(
        "observable_evidence",
        {},
    ).get(
        "supporting_signals",
        [],
    )

    model_used = provenance.get(
        "model_used",
        "unknown",
    )

    generation_origin = str(
        metadata.get("source", "generated")
    ).upper()

    scenario_json = _serialize(scenario)
    params_json = _serialize(params)
    timeline_json = _serialize(timeline)
    signals_json = _serialize(signals)

    init_table(table_name)

    sql = f"""
        INSERT INTO {table_name}
        (
            id,
            title,
            params,
            narrative,
            timeline,
            model_used,
            signals,
            generation_origin,
            scenario_data
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        scenario_id,
        title,
        params_json,
        narrative,
        timeline_json,
        model_used,
        signals_json,
        generation_origin,
        scenario_json,
    )

    try:

        with closing(get_conn()) as conn:
            with conn:
                conn.execute(sql, values)

    except sqlite3.Error as exc:

        log.error(
            "db.scenario_save.failed",
            table=table_name,
            scenario_id=scenario_id,
            error=str(exc),
        )

        raise StorageError(
            f"Unable to save scenario: {exc}"
        ) from exc

    log.info(
        "db.scenario_save.succeeded",
        table=table_name,
        scenario_id=scenario_id,
    )

    return scenario_id


def save_scenario(
    table_name: str,
    *,
    title: str,
    params: dict[str, Any],
    narrative: str,
    timeline: list[dict[str, Any]],
    model_used: Optional[str],
    signals: Optional[list[Any]] = None,
    generation_origin: str = "SPECULATIVE",
    scenario_id: Optional[str] = None,
) -> str:
    """Persist using the legacy storage interface."""

    table_name = _validate_table_name(table_name)

    init_table(table_name)

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Invalid scenario title.")

    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("Invalid scenario narrative.")

    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary.")

    if not isinstance(timeline, list):
        raise TypeError("timeline must be a list.")

    scenario_id = scenario_id or str(uuid.uuid4())

    values = (
        scenario_id,
        title,
        _serialize(params),
        narrative,
        _serialize(timeline),
        model_used,
        _serialize(signals or []),
        generation_origin,
    )

    try:

        with closing(get_conn()) as conn:
            with conn:

                conn.execute(
                    f"""
                    INSERT INTO {table_name}
                    (
                        id,
                        title,
                        params,
                        narrative,
                        timeline,
                        model_used,
                        signals,
                        generation_origin
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

    except sqlite3.Error as exc:

        raise StorageError(
            f"Unable to save scenario: {exc}"
        ) from exc

    return scenario_id


def save_scenario_s(
    *,
    title: str,
    params: dict[str, Any],
    narrative: str,
    timeline: list[dict[str, Any]],
    model_used: Optional[str],
    scenario_id: Optional[str] = None,
) -> str:

    return save_scenario(
        "s_scenarios",
        title=title,
        params=params,
        narrative=narrative,
        timeline=timeline,
        model_used=model_used,
        signals=[],
        generation_origin="SPECULATIVE",
        scenario_id=scenario_id,
    )


def save_scenario_ev(
    *,
    title: str,
    params: dict[str, Any],
    narrative: str,
    timeline: list[dict[str, Any]],
    model_used: Optional[str],
    signals: list[Any],
    generation_origin: str = "EVIDENCE",
    scenario_id: Optional[str] = None,
) -> str:

    return save_scenario(
        "ev_scenarios",
        title=title,
        params=params,
        narrative=narrative,
        timeline=timeline,
        model_used=model_used,
        signals=signals,
        generation_origin=generation_origin,
        scenario_id=scenario_id,
    )


class Storage:
    """Convenience wrapper for database operations."""

    def initialize(self) -> None:
        init_db()

    def get_rows_by_condition(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        condition: str = "1=1",
        parameters: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Retrieve rows using a trusted SQL condition."""

        table_name = _validate_table_name(table_name)

        if columns:
            if any(
                not isinstance(column, str)
                or not column.replace("_", "").isalnum()
                for column in columns
            ):
                raise ValueError("Invalid column identifier.")

            column_list = ", ".join(columns)

        else:
            column_list = "*"

        sql = (
            f"SELECT {column_list} FROM {table_name} "
            f"WHERE {condition}"
        )

        try:

            with closing(get_conn()) as conn:

                rows = conn.execute(
                    sql,
                    parameters,
                ).fetchall()

                return [
                    dict(row)
                    for row in rows
                ]

        except sqlite3.Error as exc:

            raise StorageError(
                f"Unable to retrieve scenarios: {exc}"
            ) from exc

    def delete_rows(
        self,
        table_name: str,
        condition: str = "0=1",
        parameters: tuple[Any, ...] = (),
    ) -> int:

        table_name = _validate_table_name(table_name)

        if condition.strip() == "1=1":
            log.warning(
                "db.delete_all_blocked",
                table=table_name,
            )
            return 0

        try:

            with closing(get_conn()) as conn:
                with conn:

                    cursor = conn.execute(
                        f"DELETE FROM {table_name} WHERE {condition}",
                        parameters,
                    )

                    return cursor.rowcount

        except sqlite3.Error as exc:

            raise StorageError(
                f"Unable to delete scenarios: {exc}"
            ) from exc


def get_scenarios_from_db(
    table_name: str,
) -> list[dict[str, Any]]:

    return Storage().get_rows_by_condition(table_name)


def get_next_ev_scenario_number() -> int:

    init_table("ev_scenarios")

    try:

        with closing(get_conn()) as conn:

            row = conn.execute(
                "SELECT COUNT(*) FROM ev_scenarios"
            ).fetchone()

            return int(row[0]) + 1

    except sqlite3.Error as exc:

        raise StorageError(
            f"Unable to count EV scenarios: {exc}"
        ) from exc


def update_scenario_index(
    scenario_id: str,
    plausibility_index: float,
) -> None:

    if (
        isinstance(plausibility_index, bool)
        or not isinstance(plausibility_index, (int, float))
        or not math.isfinite(plausibility_index)
        or not 0.0 <= plausibility_index <= 1.0
    ):
        raise ValueError(
            "Plausibility index must be between 0 and 1."
        )

    init_table("ev_scenarios")

    try:

        with closing(get_conn()) as conn:
            with conn:

                cursor = conn.execute(
                    """
                    UPDATE ev_scenarios
                    SET plausibility_index = ?
                    WHERE id = ?
                    """,
                    (plausibility_index, scenario_id),
                )

                if cursor.rowcount == 0:

                    log.warning(
                        "db.update_index.not_found",
                        scenario_id=scenario_id,
                    )

    except sqlite3.Error as exc:

        raise StorageError(
            f"Unable to update scenario index: {exc}"
        ) from exc
