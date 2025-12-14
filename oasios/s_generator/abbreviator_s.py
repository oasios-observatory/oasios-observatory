#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/abbreviator_s.py
"""
Created on Sat Nov  8 16:14:53 2025

@author: mike
"""
import sqlite3
from oasios.config import settings
from pathlib import Path

db_path = Path(settings.db_path)

def _ensure_table():
    """Create table if missing — runs every time, 100% safe."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS s_scenarios (
            id TEXT PRIMARY KEY,
            params TEXT,
            narrative TEXT,
            timeline TEXT,
            model_used TEXT,
            signals TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    conn.commit()
    conn.close()

def get_next_scenario_number() -> int:
    """Return next number (001, 002…) — auto-creates DB + table."""
    _ensure_table()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM s_scenarios")
    count = cursor.fetchone()[0]
    conn.close()
    return count + 1

def short_code(value: str) -> str:
    return "".join(word[0].upper() for word in value.replace("-", " ").split())[:3] or "UNK"

def abbreviate(core: dict) -> str:
    parts = [
        short_code(core.get("initial_origin", "UNK")),
        short_code(core.get("development_dynamics", "UNK")),
        short_code(core.get("architecture", "UNK")),
        short_code(core.get("deployment_topology", "UNK")),
        short_code(core.get("oversight_type", "UNK")),
        short_code(core.get("oversight_effectiveness", "UNK")),
        short_code(core.get("substrate", "UNK")),
    ]
    num = get_next_scenario_number()
    return f"{'-'.join(parts)}-{num:03d}"