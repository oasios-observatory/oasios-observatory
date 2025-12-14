import json, sqlite3
from datetime import datetime
from pathlib import Path
from oasios.common.db import get_precursor_conn, get_scenario_conn
from oasios.common.db import DATA_DIR
import subprocess

DB_PATH = DATA_DIR / "precursor_signals.db"

def run_ollama(model: str, prompt: str, temperature: float = 0.3) -> str:
    """Call a local Ollama model via subprocess."""
    cmd = ["ollama", "run", model]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        return proc.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"[ERROR] {e}"

def build_prompt(signal, scenario):
    """Compact context and instruction prompt."""
    signal_txt = f"{signal['title']}\n{signal['description'][:500]}"
    scenario_txt = scenario['narrative'][:1200]
    prompt = f"""
You are an AI research analyst. Given:

Signal:
{signal_txt}

Scenario narrative:
{scenario_txt}

Explain briefly how this real-world precursor might relate to the scenario.
Return JSON with fields:
- "explanation": short text (≤100 words)
- "derived_tags": list of 2–3 keywords
- "plausibility": number from 0 to 1
"""
    return prompt.strip()

def analyze_links(limit=10, model="mistral:7b", min_conf=0.5):
    """Run LLM analysis on top links."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    links = conn.execute(f"""
        SELECT l.id, l.signal_id, l.scenario_id, l.confidence
        FROM signal_scenario_links l
        LEFT JOIN signal_scenario_analyses a ON a.link_id = l.id
        WHERE a.id IS NULL AND l.confidence >= ?
        ORDER BY l.confidence DESC
        LIMIT ?
    """, (min_conf, limit)).fetchall()

    if not links:
        print("No unreviewed links found.")
        return []

    with get_precursor_conn() as p_conn, get_scenario_conn() as s_conn:
        for link in links:
            sig = p_conn.execute(
                "SELECT id, title, description FROM precursor_signals WHERE id=?",
                (link["signal_id"],)
            ).fetchone()
            scen = s_conn.execute(
                "SELECT data FROM scenarios WHERE id=?",
                (link["scenario_id"],)
            ).fetchone()

            if not sig or not scen:
                continue

            try:
                scen_json = json.loads(scen["data"])
                narrative = scen_json.get("scenario_content", {}).get("narrative", "")
            except:
                narrative = ""

            scenario_obj = {"narrative": narrative}
            prompt = build_prompt(sig, scenario_obj)
            output = run_ollama(model, prompt)

            try:
                result = json.loads(output)
            except:
                result = {"explanation": output[:300], "derived_tags": [], "plausibility": None}

            conn.execute("""
                INSERT INTO signal_scenario_analyses
                (link_id, model, plausibility, derived_tags, explanation, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                link["id"],
                model,
                result.get("plausibility"),
                json.dumps(result.get("derived_tags")),
                result.get("explanation"),
                datetime.utcnow().isoformat()
            ))
            conn.commit()

            print(f"✅ Reviewed link {link['id']} ({sig['title'][:40]})")

    conn.close()
