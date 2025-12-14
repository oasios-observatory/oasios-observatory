# oasios/m_generator/renderer.py
from typing import List, Dict
from datetime import datetime
from oasios.m_generator.ollama_m import generate_multi_asi_narrative
from .interact import Event

def render_interaction(swarm: List[dict], events: List[Event]) -> Dict:
    # Auto-title
    first = swarm[0]["title"] if swarm else "UNKNOWN"
    title = f"{first.split('-')[0]}-SWARM-{datetime.now().strftime('%Y%m%d')}"

    # Generate professional narrative
    narrative = generate_multi_asi_narrative(title, swarm)

    # Add interaction summary
    if events:
        summary = "\n\nKEY INTERACTIONS DETECTED:\n"
        for e in events:
            summary += f"• [{e.year}] {e.pattern.value.upper()}: {e.a} + {e.b}\n"
        narrative += summary

    threat = len(events) * 0.7

    return {
        "title": title,
        "narrative": narrative,
        "events": [e.__dict__ for e in events],
        "threat_index": threat
    }