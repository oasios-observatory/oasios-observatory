# oasios/common/db.py
"""
Centralized database paths and connection utilities.
Resolves paths relative to project root regardless of cwd.
"""
from pathlib import Path
from contextlib import contextmanager
import sqlite3
from typing import ContextManager

# Project root = two levels above this file
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

SCENARIO_DB_PATH = DATA_DIR / "asi_scenarios.db"         # Correct name
PRECURSOR_DB_PATH = DATA_DIR / "precursor_signals.db"   # Correct name


@contextmanager
def get_scenario_conn() -> ContextManager[sqlite3.Connection]:
    """Connection to generated ASI scenarios."""
    conn = sqlite3.connect(SCENARIO_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_precursor_conn() -> ContextManager[sqlite3.Connection]:
    """Connection to real-world precursor signals."""
    conn = sqlite3.connect(PRECURSOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()