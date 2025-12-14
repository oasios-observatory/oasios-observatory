#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#oasios/m_generator/schema_m.py

import sqlite3
from oasios.config import settings

def init_multi_asi_table():
    """Create dedicated table for multi-ASI briefings."""
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS m_scenarios (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created TIMESTAMP NOT NULL,
            last_updated TIMESTAMP NOT NULL,
            asi_count INTEGER NOT NULL,
            source TEXT DEFAULT 'multi_asi_v3',
            data JSON NOT NULL,
            threat_index REAL DEFAULT 0.0
        )
    ''')

    # Index for fast queries
    cur.execute('CREATE INDEX IF NOT EXISTS idx_multi_asi_created ON multi_asi_scenarios(created DESC)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_multi_asi_source ON multi_asi_scenarios(source)')

    conn.commit()
    conn.close()
    print("Multi-ASI table initialized: multi_asi_scenarios")