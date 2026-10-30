#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#oasios/common/storage.py
"""Hardened SQLite persistence for OASIOS scenarios."""

import json
import math
import re
import sqlite3
import uuid

from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from oasios.config import settings
from oasios.logger import log


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_TABLES = frozenset({
    "s_scenarios",
    "ev_scenarios",
})

ALLOWED_COLUMNS = frozenset({
    "id",
    "title",
    "params",
    "narrative",
    "timeline",
    "model_used",
    "signals",
    "created_at",
    "plausibility_index",
    "generation_origin",
    "scenario_data",
})

ALLOWED_OPERATORS = frozenset({
    "=",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
})

COLUMN_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class StorageError(RuntimeError):
    """Raised when a scenario persistence operation fails."""


class DuplicateScenarioError(StorageError):
    """Raised when a scenario ID already exists."""


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_table_name(table_name: str) -> str:
    """Validate database table names against the whitelist."""

    if (
        not isinstance(table_name, str)
        or table_name not in ALLOWED_TABLES
    ):
        raise ValueError(
            f"Unsupported database table: {table_name!r}"
        )

    return table_name


def _validate_uuid(value: str) -> str:
    """Validate canonical UUID representation."""

    if not isinstance(value, str):
        raise ValueError(
            "Scenario ID must be a string."
        )

    try:
        parsed = uuid.UUID(value)

    except (ValueError, AttributeError) as exc:
        raise ValueError(
            "Scenario ID must be a valid UUID."
        ) from exc

    if str(parsed) != value.lower():
        raise ValueError(
            "Scenario ID must use canonical UUID format."
        )

    return str(parsed)


def _require_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    """Require a dictionary."""

    if not isinstance(value, dict):
        raise ValueError(
            f"{field} must be a dictionary."
        )

    return value


def _require_list(
    value: Any,
    field: str,
) -> list[Any]:
    """Require a list."""

    if not isinstance(value, list):
        raise ValueError(
            f"{field} must be a list."
        )

    return value


def _require_string(
    value: Any,
    field: str,
    allow_empty: bool = False,
) -> str:
    """Require a valid string."""

    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string."
        )

    if not allow_empty and not value.strip():
        raise ValueError(
            f"{field} must not be empty."
        )

    return value


def _validate_scenario_metadata(
    scenario: dict[str, Any],
) -> None:
    """Validate the complete scenario structure."""

    metadata = _require_dict(
        scenario.get("metadata", {}),
        "metadata",
    )

    _require_dict(
        metadata.get("provenance", {}),
        "metadata.provenance",
    )

    for field in (
        "origin",
        "architecture",
        "substrate",
        "oversight_structure",
        "core_capabilities",
        "goals_and_behavior",
        "impact_and_control",
    ):
        _require_dict(
            scenario.get(field, {}),
            field,
        )

    content = _require_dict(
        scenario.get("scenario_content", {}),
        "scenario_content",
    )

    narrative = content.get("narrative", "")

    _require_string(
        narrative,
        "scenario_content.narrative",
        allow_empty=True,
    )

    timeline_data = _require_dict(
        content.get("timeline", {}),
        "scenario_content.timeline",
    )

    phases = _require_list(
        timeline_data.get("phases", []),
        "scenario_content.timeline.phases",
    )

    for index, phase in enumerate(phases):

        if not isinstance(phase, dict):
            raise ValueError(
                f"Timeline phase {index} must be a dictionary."
            )

        if "phase" not in phase or "years" not in phase:
            raise ValueError(
                f"Timeline phase {index} requires phase and years."
            )

    evidence = _require_dict(
        scenario.get("observable_evidence", {}),
        "observable_evidence",
    )

    _require_list(
        evidence.get("supporting_signals", []),
        "observable_evidence.supporting_signals",
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def _ensure_db_path() -> Path:
    """Create the database directory when necessary."""

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

        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        conn.execute(
            "PRAGMA busy_timeout = 30000"
        )

        return conn

    except sqlite3.Error as exc:
        raise StorageError(
            f"Unable to connect to database: {exc}"
        ) from exc


# ============================================================
# DATABASE INTROSPECTION
# ============================================================

def _table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:

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

    table_name = _validate_table_name(table_name)

    rows = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return {
        row["name"]
        for row in rows
    }


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_table(table_name: str) -> None:
    """Create or migrate a scenario table transactionally."""

    table_name = _validate_table_name(table_name)

    migrations = {
        "title": "TEXT NOT NULL DEFAULT ''",
        "params": "TEXT NOT NULL DEFAULT '{}'",
        "narrative": "TEXT NOT NULL DEFAULT ''",
        "timeline": "TEXT NOT NULL DEFAULT '[]'",
        "model_used": "TEXT",
        "signals": "TEXT NOT NULL DEFAULT '[]'",
        "created_at": "TIMESTAMP",
        "plausibility_index": "REAL DEFAULT 0.0",
        "generation_origin": "TEXT DEFAULT 'SPECULATIVE'",
        "scenario_data": "TEXT",
    }

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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        plausibility_index REAL DEFAULT 0.0,
                        generation_origin TEXT DEFAULT 'SPECULATIVE',
                        scenario_data TEXT
                    )
                    """
                )

                existing = _get_columns(
                    conn,
                    table_name,
                )

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
    """Initialize all application tables."""

    _ensure_db_path()

    init_table("s_scenarios")
    init_table("ev_scenarios")

    log.info(
        "storage.initialized",
        tables=sorted(ALLOWED_TABLES),
    )


# ============================================================
# SERIALIZATION
# ============================================================

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


# ============================================================
# COMPLETE SCENARIO PERSISTENCE
# ============================================================

def save_asi_scenario(
    scenario: dict[str, Any],
    table_name: str = "s_scenarios",
) -> str:
    """
    Persist a complete scenario.

    scenario_data remains authoritative.
    Legacy columns remain available for compatibility.
    """

    table_name = _validate_table_name(table_name)

    scenario = _require_dict(
        scenario,
        "scenario",
    )

    scenario_id = _validate_uuid(
        scenario.get("id")
    )

    title = _require_string(
        scenario.get("title"),
        "title",
    )

    _validate_scenario_metadata(scenario)

    metadata = scenario["metadata"]
    provenance = metadata["provenance"]

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

    content = scenario["scenario_content"]

    narrative = content["narrative"]

    timeline = content["timeline"]["phases"]

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

    if model_used is not None:
        _require_string(
            model_used,
            "model_used",
            allow_empty=True,
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

    except sqlite3.IntegrityError as exc:

        raise DuplicateScenarioError(
            f"Scenario already exists: {scenario_id}"
        ) from exc

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


# ============================================================
# LEGACY PERSISTENCE INTERFACE
# ============================================================

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

    title = _require_string(title, "title")
    narrative = _require_string(narrative, "narrative")

    _require_dict(params, "params")
    _require_list(timeline, "timeline")

    if signals is not None:
        _require_list(signals, "signals")

    if model_used is not None:
        _require_string(
            model_used,
            "model_used",
            allow_empty=True,
        )

    generation_origin = _require_string(
        generation_origin,
        "generation_origin",
    )

    if scenario_id is None:
        scenario_id = str(uuid.uuid4())
    else:
        scenario_id = _validate_uuid(scenario_id)

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

    init_table(table_name)

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

    except sqlite3.IntegrityError as exc:

        raise DuplicateScenarioError(
            f"Scenario already exists: {scenario_id}"
        ) from exc

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


# ============================================================
# STRUCTURED QUERY HELPERS
# ============================================================

def _build_where(
    filters: dict[str, tuple[str, Any]] | None,
) -> tuple[str, tuple[Any, ...]]:
    """Build a safe parameterized WHERE clause."""

    if filters is None or not filters:
        return "1=1", ()

    if not isinstance(filters, dict):
        raise TypeError(
            "Filters must be a dictionary."
        )

    clauses = []
    values = []

    for column, condition in filters.items():

        if (
            not isinstance(column, str)
            or not COLUMN_PATTERN.fullmatch(column)
            or column not in ALLOWED_COLUMNS
        ):
            raise ValueError(
                f"Unsupported column: {column!r}"
            )

        if (
            not isinstance(condition, tuple)
            or len(condition) != 2
        ):
            raise ValueError(
                f"Invalid filter for {column}."
            )

        operator, value = condition

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Unsupported operator: {operator}"
            )

        if value is None:
            if operator == "=":
                clauses.append(f"{column} IS NULL")
                continue

            if operator == "!=":
                clauses.append(f"{column} IS NOT NULL")
                continue

            raise ValueError(
                "NULL supports only equality operators."
            )

        clauses.append(
            f"{column} {operator} ?"
        )

        values.append(value)

    return (
        " AND ".join(clauses),
        tuple(values),
    )


# ============================================================
# STORAGE WRAPPER
# ============================================================

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
        """
        Retrieve rows using a trusted internal SQL condition.

        IMPORTANT:
        Never pass user-controlled SQL through condition.
        Use get_rows_by_filters() for external input.
        """

        table_name = _validate_table_name(table_name)

        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(
                "SQL condition must be a nonempty string."
            )

        if not isinstance(parameters, tuple):
            raise TypeError(
                "SQL parameters must be a tuple."
            )

        if columns:

            if any(
                not isinstance(column, str)
                or not COLUMN_PATTERN.fullmatch(column)
                or column not in ALLOWED_COLUMNS
                for column in columns
            ):
                raise ValueError(
                    "Invalid column identifier."
                )

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

    def get_rows_by_filters(
        self,
        table_name: str,
        filters: dict[str, tuple[str, Any]] | None = None,
        columns: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve records using safe structured filters."""

        table_name = _validate_table_name(table_name)

        if columns:

            if any(
                not isinstance(column, str)
                or not COLUMN_PATTERN.fullmatch(column)
                or column not in ALLOWED_COLUMNS
                for column in columns
            ):
                raise ValueError(
                    "Invalid column identifier."
                )

            column_list = ", ".join(columns)

        else:
            column_list = "*"

        where, parameters = _build_where(filters)

        sql = (
            f"SELECT {column_list} FROM {table_name} "
            f"WHERE {where}"
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
        """
        Delete using a trusted internal SQL condition.

        Never pass user-controlled SQL through condition.
        """

        table_name = _validate_table_name(table_name)

        if not isinstance(condition, str) or not condition.strip():
            raise ValueError(
                "SQL condition must be a nonempty string."
            )

        if not isinstance(parameters, tuple):
            raise TypeError(
                "SQL parameters must be a tuple."
            )

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
                        f"DELETE FROM {table_name} "
                        f"WHERE {condition}",
                        parameters,
                    )

                    return cursor.rowcount

        except sqlite3.Error as exc:

            raise StorageError(
                f"Unable to delete scenarios: {exc}"
            ) from exc

    def delete_rows_by_filters(
        self,
        table_name: str,
        filters: dict[str, tuple[str, Any]],
    ) -> int:
        """Delete records using safe structured filters."""

        table_name = _validate_table_name(table_name)

        if not filters:
            raise ValueError(
                "Deletion requires at least one filter."
            )

        where, parameters = _build_where(filters)

        try:

            with closing(get_conn()) as conn:

                with conn:

                    cursor = conn.execute(
                        f"DELETE FROM {table_name} "
                        f"WHERE {where}",
                        parameters,
                    )

                    return cursor.rowcount

        except sqlite3.Error as exc:

            raise StorageError(
                f"Unable to delete scenarios: {exc}"
            ) from exc


# ============================================================
# PUBLIC DATABASE HELPERS
# ============================================================

def get_scenarios_from_db(
    table_name: str,
) -> list[dict[str, Any]]:

    return Storage().get_rows_by_condition(
        table_name
    )


def get_next_ev_scenario_number() -> int:
    """Return the next evidence scenario number."""

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
    """Update the plausibility index of an EV scenario."""

    scenario_id = _validate_uuid(scenario_id)

    if (
        isinstance(plausibility_index, bool)
        or not isinstance(plausibility_index, (int, float))
        or not math.isfinite(plausibility_index)
        or not 0.0 <= plausibility_index <= 1.0
    ):
        raise ValueError(
            "Plausibility index must be a finite number "
            "between 0 and 1."
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
                    (
                        float(plausibility_index),
                        scenario_id,
                    ),
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
