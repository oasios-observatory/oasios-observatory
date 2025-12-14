# oasios/m_generator/cli_s.py
#!/usr/bin/env python3
"""
OASIOS Observatory Swarm CLI
Runs parameter prompt immediately when executed.
"""
import typer
import random
from typing import Optional
from oasios.m_generator.core_m import create_multi_asi_scenario

app = typer.Typer(
    name="ASI Scenario Generator v0.1.1",
    help="Multi-ASI Scenarios Generator",
    add_completion=False,
)

def interactive_prompt() -> dict:
    """Prompt user for all parameters."""
    typer.echo(typer.style("\nOASIOS Observatory multi-ASI s_generator", fg=typer.colors.CYAN, bold=True))
    typer.echo("Generate classified intelligence briefing\n")

    while True:
        n_str = typer.prompt("Number of ASIs", default="5")
        try:
            n = int(n_str)
            if 2 <= n <= 50:
                break
            typer.echo("Enter a number between 2 and 50.")
        except ValueError:
            typer.echo("Invalid number.")

    typer.echo(f"Selected {n} ASIs. Using random selection from database.")

    save = typer.confirm("Save briefing to database?", default=True)

    seed_input = typer.prompt("Random seed (blank = true random)", default="", show_default=False)
    seed = int(seed_input) if seed_input.strip().isdigit() else None
    if seed:
        random.seed(seed)
        typer.echo(f"Seed locked: {seed}")

    return {"n": n, "save": save, "seed": seed}

@app.command()
def compose(
    n: Optional[int] = None,
    save: Optional[bool] = None,
    seed: Optional[int] = None,
):
    """Generate briefing — interactive or CLI flags."""
    if n is None:
        config = interactive_prompt()
        n = config["n"]
        save = config["save"]
        seed = config["seed"]

    if seed is not None:
        random.seed(seed)

    typer.echo(typer.style(f"\nComposing briefing with {n} ASIs...", fg=typer.colors.GREEN, bold=True))
    create_multi_asi_scenario(num_asis=n)

@app.command()
def list():
    """List saved v3 briefings."""
    import sqlite3, json
    from oasios.config import settings
    conn = sqlite3.connect(settings.db_path)
    cur = conn.cursor()
    cur.execute("SELECT title, data FROM scenarios WHERE json_extract(data, '$.metadata.source') = 'multi_asi_v3' ORDER BY rowid DESC LIMIT 10")
    rows = cur.fetchall()
    if not rows:
        typer.echo("No v3 briefings saved yet.")
        return
    for title, data in rows:
        d = json.loads(data)
        count = d.get("metadata", {}).get("asi_count", 0)
        typer.echo(f"{title} — {count} ASIs")

# AUTO-INTERACTIVE WHEN RUN DIRECTLY
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["list", "compose"]):
        # No command → run interactive compose
        compose()
    else:
        app()