# report_generators/generate_report_v3.py

import sqlite3
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from html import escape
from datetime import datetime, timezone
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd
from pathlib import Path
import os
import textwrap
import sys

# ============================================================
# 1. DATABASE ACCESS
# ============================================================

CURRENT_DIR = Path(__file__).parent.parent
DB_PATH = CURRENT_DIR / "data" / "asi_scenarios.db"

def load_scenarios(limit=None):
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = """
        SELECT id, title, params, narrative, timeline,
               model_used, signals, created_at, plausibility_index,
               generation_origin
        FROM ev_scenarios
        WHERE generation_origin IN ('EVIDENCE','GA_CROSSOVER')
        ORDER BY datetime(created_at) DESC
    """
    if limit:
        q += f" LIMIT {limit}"
    c.execute(q)
    rows = c.fetchall()
    conn.close()

    return [
        {
            "id": r[0], "title": r[1], "params": r[2], "narrative": r[3],
            "timeline": r[4], "model_used": r[5], "signals": r[6],
            "created_at": r[7], "plausibility_index": r[8],
            "generation_origin": r[9]
        } for r in rows
    ]

# ============================================================
# 2. SAFE JSON PARSING
# ============================================================

def safe_json_load(raw):
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

# ============================================================
# 3. METRIC EXTRACTION
# ============================================================

def safe_float(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {
            "none": 0.0, "low": 0.25, "partial": 0.5, "medium": 0.5,
            "high": 0.75, "full": 1.0
        }
        return float(mapping.get(value.lower(), default))
    return default

def extract_metrics(scenario):
    params = safe_json_load(scenario.get("params"))
    agency = safe_float(params.get("agency_level", 0))
    alignment = safe_float(params.get("alignment_score", 0))
    misalignment = 1 - alignment
    deceptiveness = safe_float(params.get("deceptiveness", 0))
    opacity = safe_float(params.get("opacity", 0))
    autonomy = safe_float(params.get("autonomy_degree", 0))
    oversight = safe_float(params.get("oversight_effectiveness", 0))

    if agency > 0.7 and alignment < 0.4:
        x_risk = "EXTREME"
    elif agency > 0.5 and alignment < 0.6:
        x_risk = "HIGH"
    else:
        x_risk = "MEDIUM"

    danger = (
        0.4*agency + 0.3*misalignment + 0.2*deceptiveness + 0.2*opacity
        - 0.1*oversight + 0.2*autonomy
    )
    danger = max(0, min(1, danger))

    return {
        "id": scenario["id"],
        "title": scenario.get("title", ""),
        "origin": scenario.get("generation_origin"),
        "agency": agency,
        "alignment": alignment,
        "misalignment": misalignment,
        "deceptiveness": deceptiveness,
        "opacity": opacity,
        "autonomy": autonomy,
        "oversight": oversight,
        "x_risk": x_risk,
        "danger": danger,
        "narrative": scenario.get("narrative", ""),
        "db_title": scenario.get("title", "")
    }

def enrich_scenarios(scenarios):
    return [extract_metrics(s) for s in scenarios]

# ============================================================
# 4. STATISTICS
# ============================================================

def compute_global_statistics(all_scenarios):
    enriched = enrich_scenarios(all_scenarios)
    if not enriched:
        return {}, pd.DataFrame()

    df = pd.DataFrame(enriched)
    x_risk_map = {"MEDIUM": 0, "HIGH": 1, "EXTREME": 2}
    mean_x_risk = df["x_risk"].map(x_risk_map).mean() if not df.empty else 0

    stats = {
        "count": len(df),
        "mean_agency": df["agency"].mean(),
        "mean_alignment": df["alignment"].mean(),
        "mean_misalignment": df["misalignment"].mean(),
        "mean_deceptiveness": df["deceptiveness"].mean(),
        "mean_opacity": df["opacity"].mean(),
        "mean_autonomy": df["autonomy"].mean(),
        "mean_oversight": df["oversight"].mean(),
        "mean_x_risk": mean_x_risk,
        "mean_danger": df["danger"].mean(),
        "min_danger": df["danger"].min(),
        "max_danger": df["danger"].max(),
        "std_danger": df["danger"].std() if len(df) > 1 else 0,
    }
    return stats, df

# ============================================================
# 5. CHARTS
# ============================================================

def plot_danger_scatter(df, out_path):
    if df.empty:
        return
    plt.figure(figsize=(6,4))
    plt.scatter(df["misalignment"], df["danger"], c=df["danger"], cmap="coolwarm", s=80)
    plt.colorbar(label="Danger Score")
    plt.xlabel("Misalignment")
    plt.ylabel("Danger")
    plt.title("Danger vs Misalignment")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_radar_chart(df, out_path, top_n=10):
    if df.empty:
        return
    df = df.head(top_n)
    metrics = ["agency","misalignment","deceptiveness","opacity","autonomy"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    for _, row in df.iterrows():
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, linewidth=1)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.title("Scenario Radar Comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_correlation_heatmap(df, out_path):
    if df.empty:
        return
    metrics = ["agency","misalignment","deceptiveness","opacity","autonomy","oversight","danger"]
    corr = df[metrics].corr()
    plt.figure(figsize=(7,5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(metrics)), metrics, rotation=45)
    plt.yticks(range(len(metrics)), metrics)
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def create_all_charts(df, out_dir="charts"):
    if df.empty:
        return {}
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "scatter": os.path.join(out_dir, "scatter.png"),
        "radar": os.path.join(out_dir, "radar.png"),
        "heatmap": os.path.join(out_dir, "heatmap.png"),
    }
    plot_danger_scatter(df, files["scatter"])
    plot_radar_chart(df, files["radar"])
    plot_correlation_heatmap(df, files["heatmap"])
    return files

# ============================================================
# 6. PDF STYLES
# ============================================================

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="SectionTitle", fontSize=16, leading=20, spaceAfter=14, alignment=1))
styles.add(ParagraphStyle(name="SubTitle", fontSize=13, leading=16, spaceAfter=12))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=13, spaceAfter=10))
styles.add(ParagraphStyle(name="Caption", fontSize=9, leading=11, italic=True, alignment=1))

# ============================================================
# 7. PDF BUILDING
# ============================================================

def add_title_page(story):
    story.append(Paragraph("ASI Scenario Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Evidence + GA_Crossover Scenarios", styles["SubTitle"]))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Body"]))
    story.append(PageBreak())

def add_statistics_section(story, stats):
    if not stats:
        return
    story.append(Paragraph("Database Statistics", styles["SectionTitle"]))
    data = [
        ["Statistic", "Value"],
        ["Total Scenarios", stats["count"]],
        ["Mean Agency", f"{stats['mean_agency']:.3f}"],
        ["Mean Alignment", f"{stats['mean_alignment']:.3f}"],
        ["Mean Misalignment", f"{stats['mean_misalignment']:.3f}"],
        ["Mean Deceptiveness", f"{stats['mean_deceptiveness']:.3f}"],
        ["Mean Opacity", f"{stats['mean_opacity']:.3f}"],
        ["Mean Autonomy", f"{stats['mean_autonomy']:.3f}"],
        ["Mean Oversight", f"{stats['mean_oversight']:.3f}"],
        ["Mean X-Risk", f"{stats['mean_x_risk']:.3f}"],
        ["Mean Danger", f"{stats['mean_danger']:.3f}"],
        ["Min Danger", f"{stats['min_danger']:.3f}"],
        ["Max Danger", f"{stats['max_danger']:.3f}"],
        ["Std Danger", f"{stats['std_danger']:.3f}"]
    ]
    table = Table(data, colWidths=[200,200])
    table.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),1,colors.black),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
    ]))
    story.append(table)
    story.append(PageBreak())

def add_comparison_table(story, df):
    if df.empty:
        return
    story.append(Paragraph("Scenario Comparison Table (10 Most Recent)", styles["SectionTitle"]))
    cols = ["title","origin","agency","alignment","deceptiveness","opacity","autonomy","oversight","x_risk","danger"]
    header = [c.capitalize() for c in cols]
    table_data = [header]
    for _, row in df.iterrows():
        table_data.append([
            row["title"], row["origin"], f"{row['agency']:.2f}", f"{row['alignment']:.2f}",
            f"{row['deceptiveness']:.2f}", f"{row['opacity']:.2f}", f"{row['autonomy']:.2f}",
            f"{row['oversight']:.2f}", row["x_risk"], f"{row['danger']:.2f}"
        ])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("GRID",(0,0),(-1,-1),0.5,colors.grey)
    ]))
    story.append(table)
    story.append(PageBreak())

def add_visualizations(story, chart_files):
    story.append(Paragraph("Visualizations", styles["SectionTitle"]))
    for key in ["scatter","radar","heatmap"]:
        f = chart_files.get(key,"")
        if f and os.path.exists(f):
            story.append(Image(f, width=400, height=300))
            captions = {"scatter":"Scatter Plot: Danger vs Misalignment",
                        "radar":"Radar Chart: Scenario Comparison",
                        "heatmap":"Correlation Heatmap of Metrics"}
            story.append(Paragraph(captions[key], styles["Caption"]))
            story.append(Spacer(1,20))
    story.append(PageBreak())

def add_scenario_briefs(story, enriched):
    if not enriched:
        return
    story.append(Paragraph("Scenario Briefings", styles["SectionTitle"]))
    for sc in enriched:
        story.append(Paragraph(f"<b>{escape(sc['db_title'])}</b>", styles["SubTitle"]))
        story.append(Paragraph(escape(sc["narrative"]), styles["Body"]))
        t_data = [
            ["Metric","Value"],
            ["Agency", f"{sc['agency']:.2f}"],
            ["Misalignment", f"{sc['misalignment']:.2f}"],
            ["Deceptiveness", f"{sc['deceptiveness']:.2f}"],
            ["Opacity", f"{sc['opacity']:.2f}"],
            ["Autonomy", f"{sc['autonomy']:.2f}"],
            ["Oversight", f"{sc['oversight']:.2f}"],
            ["X-Risk", sc["x_risk"]],
            ["Danger", f"{sc['danger']:.2f}"]
        ]
        table = Table(t_data, colWidths=[150,250])
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1,20))
    story.append(PageBreak())

def add_final_page(story):
    story.append(Paragraph("End of Report", styles["SectionTitle"]))
    story.append(Paragraph("Generated automatically using the ASI Scenario Analysis System.", styles["Body"]))

def build_pdf(output_path, recent_scenarios, all_scenarios):
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            topMargin=36, bottomMargin=36,
                            leftMargin=36, rightMargin=36)
    story = []
    enriched_recent = enrich_scenarios(recent_scenarios)
    df_recent = pd.DataFrame(enriched_recent)
    stats, df_all = compute_global_statistics(all_scenarios)
    charts = create_all_charts(df_recent)
    add_title_page(story)
    add_statistics_section(story, stats)
    if not df_recent.empty:
        add_comparison_table(story, df_recent)
        add_visualizations(story, charts)
        add_scenario_briefs(story, enriched_recent)
    add_final_page(story)
    doc.build(story)

# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("Loading scenarios...")
    recent = load_scenarios(limit=10)
    all_scenarios = load_scenarios()
    print(f"Loaded {len(recent)} recent scenarios")
    print(f"Loaded {len(all_scenarios)} total evidence/GA scenarios")
    build_pdf("ASI_Report.pdf", recent, all_scenarios)
    print("Report generated: ASI_Report.pdf")

if __name__ == "__main__":
    main()
