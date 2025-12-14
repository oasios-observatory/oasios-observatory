# oasios/storage/storage_m.py
import sqlite3
import json
from datetime import datetime, timezone
from oasios.config import settings
from .schema_m import init_multi_asi_table


def save_multi_asi_briefing(scenario: dict):
    """Save to dedicated multi_asi_scenarios table."""
    init_multi_asi_table()

    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    scenario["metadata"]["last_updated"] = now

    cur.execute('''
        INSERT OR REPLACE INTO m_scenarios 
        (id, title, created, last_updated, asi_count, source, data, threat_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        scenario["id"],
        scenario["title"],
        scenario["metadata"]["created"],
        now,
        len(scenario.get("asis", [])),
        scenario["metadata"].get("source", "multi_asi_v3"),
        json.dumps(scenario),
        scenario.get("quantitative_assessment", {}).get("threat_index", 0.0)
    ))

    conn.commit()
    conn.close()
    print(f"SAVED MULTI-ASI BRIEFING: {scenario['title']} ({len(scenario.get('asis', []))} ASIs)")