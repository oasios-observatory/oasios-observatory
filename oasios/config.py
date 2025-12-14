#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 15:33:57 2025

@author: mike
"""

# oasios/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

ROOT = Path(__file__).parent.parent

# --- Configuration for OASIOS Precursor Tracker ---
# This path is used by the DatabaseManager in tracker_v2
TRACKER_DB_PATH = ROOT / "data" / "precursor_signals.db"

# Ensure the 'data' directory exists when the application starts
(ROOT / "data").mkdir(parents=True, exist_ok=True)


# --- Configuration for Ollama ---
class Settings(BaseSettings):
    ollama_timeout: int = 300
    ollama_preferred_model: str = "llama3.1:8b"
    db_path: Path = ROOT / "data" / "asi_scenarios.db"
    schema_path: Path = ROOT / "schemas" / "asi_scenario_v1.json"
    log_level: str = "INFO"

    # NEW: Path for the V3 Tracker DB (source of signal features)
    # This path is now relative to ROOT, assuming the file is in the data directory.
    v3_db_path: Path = ROOT / "data" / "asi_precursors.db"

    model_config = {"env_file": ".env"}


settings = Settings()