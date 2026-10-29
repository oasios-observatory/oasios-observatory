#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/storage.py

"""SQLite persistence for generated ASI scenarios."""

import json
import sqlite3
import uuid

from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from oasios.config import settings
from oasios.logger import log


ALLOWED_TABLES = frozenset({"s_scenarios", "ev_scenarios"})


class StorageError(RuntimeError):
    """Raised when a scenario persistence operation fails."""


def _ensure_db_path() -> Path:
    """Create the database directory and file if necessary."""
    db_path = Path(settings.db_path)

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if not db_path.exists():
            db_path.touch()
            log.info("db.created", path=str(db_path))

    except OSError as exc:
        raise StorageError(
            f"Unable to prepare database path {db_path}: {exc}"
        ) from exc

    return db_path


def _validate_table_name(table_name: str) -> str:
    """Restrict table access to known application tables."""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Unsupported database table: {table_name!r}")

    return table_name


def get_conn() -> sqlite3.Connection:
    """Create a SQLite connection with dictionary-like rows."""
    db_path = _ensure_db_path()

    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    except sqlite3.Error as exc:
        raise StorageError(
            f"Unable to connect to database {db_path}: {exc}"
        ) from exc


def init_table(table_name: str) -> None:
    """Create a scenario table if it does not already exist."""
    table_name = _validate_table_name(table_name)

    try:
        with closing(get_conn()) as conn:
            with conn:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        params TEXT NOT NULL,
                        narrative TEXT NOT NULL,
                        timeline TEXT NOT NULL,
                        model_used TEXT,
                        signals TEXT NOT NULL DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        plausibility_index REAL DEFAULT 0.0,
                        generation_origin TEXT DEFAULT 'SPECULATIVE'
                    )
                    """
                )

    except sqlite3.Error as exc:
        log.error(
            "db.table_init.failed",
            table=table_name,
            error=str(exc),
        )
        raise StorageError(
            f"Unable to initialize table {table_name!r}: {exc}"
        ) from exc


def init_db() -> None:
    """Initialize all scenario tables."""
    _ensure_db_path()

    init_table("s_scenarios")
    init_table("ev_scenarios")

    log.info(
        "storage.initialized",
        tables=["s_scenarios", "ev_scenarios"],
    )


def get_next_ev_scenario_number() -> int:
    """Return the next count-based EV scenario number."""
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


class Storage:
    """Convenience wrapper for database operations."""

    def initialize(self) -> None:
        init_db()

    def get_rows_by_condition(
        self,
        table_name: str,
        columns: Optional[list[str]] = None,
        condition: str = "1=1",
    ) -> list[dict[str, Any]]:
        """
        Retrieve rows from a known table.

        WARNING:
            `columns` and `condition` are SQL fragments. They must only
            contain trusted application-controlled input.
        """
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
            f"WHERE {condition};"
        )

        try:
            with closing(get_conn()) as conn:
                rows = conn.execute(sql).fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as exc:
            raise StorageError(
                f"Unable to retrieve rows from {table_name!r}: {exc}"
            ) from exc

    def delete_rows(
        self,
        table_name: str,
        condition: str = "0=1",
    ) -> int:
        """Delete rows matching a trusted SQL condition."""
        table_name = _validate_table_name(table_name)

        if condition.strip() == "1=1":
            log.warning(
                "db.delete_all_blocked",
                table=table_name,
            )
            return 0

        sql = f"DELETE FROM {table_name} WHERE {condition};"

        try:
            with closing(get_conn()) as conn:
                with conn:
                    cursor = conn.execute(sql)
                    return cursor.rowcount

        except sqlite3.Error as exc:
            raise StorageError(
                f"Unable to delete rows from {table_name!r}: {exc}"
            ) from exc


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
    """
    Persist a scenario and return its authoritative ID.

    JSON serialization occurs before the transaction so serialization
    failures cannot leave a partially written scenario.
    """
    table_name = _validate_table_name(table_name)

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Scenario title must be a nonempty string.")

    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("Scenario narrative must be a nonempty string.")

    if not isinstance(params, dict):
        raise TypeError("Scenario parameters must be a dictionary.")

    if not isinstance(timeline, list):
        raise TypeError("Scenario timeline must be a list.")

    if signals is not None and not isinstance(signals, list):
        raise TypeError("Scenario signals must be a list or None.")

    try:
        params_json = json.dumps(params, ensure_ascii=False)
        timeline_json = json.dumps(timeline, ensure_ascii=False)
        signals_json = json.dumps(signals or [], ensure_ascii=False)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Scenario contains non-serializable data: {exc}"
        ) from exc

    scenario_id = scenario_id or str(uuid.uuid4())

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
            generation_origin
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            f"Unable to save scenario {scenario_id}: {exc}"
        ) from exc

    log.info(
        "db.scenario_save.succeeded",
        table=table_name,
        scenario_id=scenario_id,
    )

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
    """Persist a speculative scenario."""
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
    """Persist an evidence-based scenario."""
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


def get_scenarios_from_db(
    table_name: str,
) -> list[dict[str, Any]]:
    """Retrieve scenarios from a known scenario table."""
    return Storage().get_rows_by_condition(table_name)


def update_scenario_index(
    scenario_id: str,
    plausibility_index: float,
) -> None:
    """Update an EV scenario's plausibility score."""
    if not isinstance(plausibility_index, (int, float)):
        raise TypeError("Plausibility index must be numeric.")

    if isinstance(plausibility_index, bool):
        raise TypeError("Plausibility index cannot be boolean.")

    if not 0.0 <= plausibility_index <= 1.0:
        raise ValueError(
            "Plausibility index must be between 0 and 1."
        )

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
        log.error(
            "db.update_index.failed",
            scenario_id=scenario_id,
            error=str(exc),
        )
        raise StorageError(
            f"Unable to update scenario index: {exc}"
        ) from exc


def save_asi_scenario(scenario: dict[str, Any]) -> None:
    """Placeholder for future complete-schema scenario persistence."""
    raise NotImplementedError(
        "Full-schema scenario persistence is not implemented."
    )
