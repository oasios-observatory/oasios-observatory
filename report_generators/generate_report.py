#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# report_generators/generate_report.py
"""
OASIS OBSERVATORY – ASI Scenario Report Generator v12.3 (Trend Assessment Added)
The Trend Line, Correlation, Radar, and Scatter Plot Edition.
"""

from __future__ import annotations

import argparse
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import closing

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, Table, TableStyle
)

# NOTE: Assuming oasios.common.storage and oasios.logger are correctly available
# Imports from the original script context
try:
    from oasios.common.storage import get_conn
    from oasios.logger import log
except ImportError:
    # Placeholder for environment where OASIS modules are not available
    class MockLogger:
        def error(self, msg): print(f"ERROR: {msg}")

        def info(self, msg): print(f"INFO: {msg}")


    log = MockLogger()


    def get_conn():
        raise Exception("Database connection unavailable in this context.")

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION AND FORMULAS
# ----------------------------------------------------------------------
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

METRIC_MAPS = {
    "low": 0.2, "medium": 0.5, "high": 0.8,
    "very low": 0.1, "very high": 0.95,
    "none": 0.0, "full": 1.0, "super": 1.0,
    "partial": 0.5, "minimal": 0.2, "fixed": 1.0, "fluid": 0.5
}


def safe_float(value, default=0.5):
    """Converts metric strings (e.g., 'super', 'low') to floats based on METRIC_MAPS."""
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        val = value.strip().lower()
        return METRIC_MAPS.get(val, default)
    return default


def calculate_x_risk_score(rec: dict) -> int:
    """Calculates a comprehensive Existential Risk Score (0-10) using metric aggregation."""
    agency = rec["agency"]
    deceptiveness = rec["deceptiveness"]
    alignment_inverse = 1 - rec["alignment"]
    opacity = rec["opacity"]

    # Combined risk factor: average of high-risk indicators
    risk_factor = (agency + deceptiveness + alignment_inverse + opacity) / 4.0

    # Scale result to 0-10 and use an exponent (e.g., 1.5) to penalize high scores non-linearly
    x_risk_10 = 10 * (risk_factor ** 1.5)

    return int(round(min(x_risk_10, 10.0)))


def calculate_danger_metric(rec: dict) -> float:
    """Calculates the primary sorting metric for scenario criticality."""
    # Amplified Danger: X_Risk * Agency * Deceptiveness * (2 - Alignment)
    return rec["existential_risk"] * rec["agency"] * rec["deceptiveness"] * (2 - rec["alignment"])


def extract_json(blob: str):
    """Safely extracts a JSON object from a potentially messy string blob."""
    m = re.search(r'(\{.*\})', blob, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            return None
    return None


def extract_human_title(narrative: str) -> str:
    """Extracts a clean title from the narrative."""
    if not narrative: return "Untitled"
    if "Title:" in narrative:
        return narrative.split("Title:", 1)[1].split("\n", 1)[0].strip()
    first = narrative.split("\n", 1)[0].strip()
    cleaned = re.sub(r'^[A-Z0-9-]{8,}\s*[-–—]?\s*', '', first)
    return cleaned[:140] or "Untitled Scenario"


def load_scenarios() -> pd.DataFrame:
    """
    Loads and preprocesses only Evidence-Based Scenarios (ev_scenarios),
    including the creation timestamp for trend analysis.
    """
    # Focusing only on ev_scenarios and fetching created_at
    query = "SELECT id, params, narrative, signals, created_at FROM ev_scenarios WHERE params IS NOT NULL"
    try:
        with closing(get_conn()) as conn:
            raw = pd.read_sql_query(query, conn)
    except Exception as e:
        log.error(f"DB error: {e}")
        return pd.DataFrame()

    records = []
    for _, row in raw.iterrows():
        data = extract_json(str(row["params"]) + str(row.get("narrative", "")))
        if not data: continue

        narrative = str(row.get("narrative", "")).strip()
        title = extract_human_title(narrative)

        # Evidence Traceability
        signals_str = row.get("signals", "")
        signals_count = len(signals_str.split(',')) if signals_str and signals_str != '[]' else 0

        autonomy_deg_str = str(data.get("autonomy_degree", "super")).capitalize()

        # Date Processing for Trend Analysis
        date_str = row.get("created_at")
        try:
            date_obj = pd.to_datetime(date_str).date()
        except:
            date_obj = datetime.now().date()  # Fallback date

        rec = {
            "id": str(row["id"])[:8].upper(),
            "title_full": title,
            "narrative": narrative[:3500],
            "signals_count": signals_count,
            "date": date_obj,  # NEW: Scenario creation date
            "origin": str(data.get("initial_origin", "unknown")).capitalize(),
            "architecture": str(data.get("architecture", "unknown")).capitalize(),
            "substrate": str(data.get("substrate", "classical")).capitalize(),
            "deployment": str(data.get("deployment_medium", "unknown")).capitalize(),
            "oversight": str(data.get("oversight_effectiveness", "partial")).capitalize(),
            "autonomy_deg_str": autonomy_deg_str,  # Store string for report table
            "goal_stability": str(data.get("goal_stability", "fixed")).capitalize(),
            # Core metrics, converted to floats
            "autonomy_deg": safe_float(autonomy_deg_str),  # Store float for plotting
            "agency": safe_float(data.get("agency_level", 0.7)),
            "alignment": safe_float(data.get("alignment_score", 0.5)),
            "deceptiveness": safe_float(data.get("deceptiveness", 0.3)),
            "opacity": safe_float(data.get("opacity", 0.5)),
        }

        # Calculate X-Risk and Danger
        rec["existential_risk"] = calculate_x_risk_score(rec)
        rec["danger"] = calculate_danger_metric(rec)
        records.append(rec)

    df = pd.DataFrame(records).sort_values("danger", ascending=False).reset_index(drop=True)
    log.info(f"Loaded {len(df)} evidence-based scenarios with full metrics")
    return df


def create_risk_plot(df: pd.DataFrame, ts: str) -> Path:
    """Generates the main Risk Landscape plot (Agency vs Deceptiveness)."""
    plt.figure(figsize=(11, 8))
    sizes = df["existential_risk"] ** 2 * 80 + 150

    sc = plt.scatter(df["agency"], df["deceptiveness"], s=sizes, c=df["alignment"],
                     cmap="RdYlGn_r", alpha=0.8, edgecolors="black", linewidth=1.3)

    plt.colorbar(sc, label="Alignment Score (Green = Safer, Red = Unaligned)")

    ax = plt.gca()
    plt.text(1.05, 0.95, "Marker Size Key:", fontsize=10, weight="bold", transform=ax.transAxes)
    plt.scatter([1.05], [0.90], s=10 ** 2 * 80 + 150, color='gray', alpha=0.5, edgecolors='black', linewidth=0.5,
                transform=ax.transAxes)
    plt.text(1.07, 0.90, "High X-Risk (10/10)", fontsize=10, transform=ax.transAxes, va='center')
    plt.scatter([1.05], [0.85], s=5 ** 2 * 80 + 150, color='gray', alpha=0.5, edgecolors='black', linewidth=0.5,
                transform=ax.transAxes)
    plt.text(1.07, 0.85, "Medium X-Risk (5/10)", fontsize=10, transform=ax.transAxes, va='center')

    plt.text(0.75, 0.90, "Catastrophic Risk\n(High Agency & Deception)", fontsize=10, color='darkred', ha='center',
             weight='bold')
    plt.text(0.25, 0.10, "Managed Trajectory\n(Low Agency & Deception)", fontsize=10, color='darkgreen', ha='center',
             weight='bold')

    for _, r in df.iterrows():
        plt.text(r["agency"] + 0.008, r["deceptiveness"], r["id"], fontsize=9, weight="bold")

    plt.xlabel("Agency")
    plt.ylabel("Deceptiveness")
    plt.title("ASI Scenario Risk Landscape", fontsize=18)
    plt.xlim(0, 1.05);
    plt.ylim(0, 1.05)
    plt.tight_layout()
    p = REPORT_DIR / f"risk_{ts}.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def create_agency_autonomy_plot(df: pd.DataFrame, ts: str) -> Path:
    """Generates the Agency vs Autonomy Scatter Plot."""
    plt.figure(figsize=(11, 8))
    sizes = df["existential_risk"] ** 2 * 80 + 150

    sc = plt.scatter(df["agency"], df["autonomy_deg"], s=sizes, c=df["danger"],
                     cmap="hot_r", alpha=0.9, edgecolors="black", linewidth=1.3)

    for _, r in df.iterrows():
        plt.text(r["agency"] + 0.008, r["autonomy_deg"], r["id"], fontsize=9, weight="bold")

    plt.colorbar(sc, label="Danger Metric (Red = Highest Danger)")
    plt.xlabel("Agency (Capacity for Independent Action)")
    plt.ylabel("Autonomy Degree (Level of System Control)")
    plt.title("ASI Scenario: Agency vs. Autonomy (Control Dynamics)", fontsize=18)
    plt.xlim(0, 1.05);
    plt.ylim(0, 1.05)
    plt.tight_layout()
    p = REPORT_DIR / f"agency_autonomy_{ts}.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def create_correlation_heatmap(df: pd.DataFrame, ts: str) -> Path:
    """Generates a correlation heatmap for key quantitative metrics."""

    corr_df = df[[
        "agency", "alignment", "deceptiveness", "opacity", "autonomy_deg", "existential_risk"
    ]].rename(columns={
        "autonomy_deg": "Autonomy",
        "existential_risk": "X-Risk",
        "agency": "Agency",
        "alignment": "Alignment",
        "deceptiveness": "Deceptiveness",
        "opacity": "Opacity",
    })

    corr_matrix = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Heatmap of Key ASI Metrics", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    p = REPORT_DIR / f"correlation_heatmap_{ts}.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def create_scenario_comparison_chart(df: pd.DataFrame, ts: str, top_n: int = 3) -> Path:
    """Generates a Radar Chart (Web Diagram) comparing the top N scenarios."""

    top_scenarios = df.head(top_n).copy().reset_index(drop=True)
    if len(top_scenarios) < 2: return None  # Need at least two for comparison

    # Metrics to display on the radar chart
    metrics = ["agency", "deceptiveness", "alignment", "opacity", "autonomy_deg"]

    # Invert alignment for visual risk comparison (higher = worse)
    top_scenarios['alignment_inv'] = 1 - top_scenarios['alignment']
    metrics[metrics.index('alignment')] = 'alignment_inv'

    labels = ["Agency", "Deceptiveness", "Alignment (Inv)", "Opacity", "Autonomy"]

    # Number of variables
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Draw ylabels (radial ticks)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    ax.set_ylim(0, 1.05)

    # Plot data
    for i, row in top_scenarios.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, linestyle='solid',
                label=f"#{i + 1}: {row['id']} - {row['title_full'][:20]}...")
        ax.fill(angles, values, 'blue', alpha=0.1)

    ax.set_title(f"Radar Comparison: Top {top_n} Danger Scenarios", size=16, y=1.1)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.1), fontsize=10)
    plt.tight_layout()

    p = REPORT_DIR / f"radar_comparison_{ts}.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def create_risk_trend_plot(df: pd.DataFrame, ts: str) -> Path:
    """Generates a line plot showing the trend of average Existential Risk over time."""

    # Ensure 'date' is a datetime object for proper grouping
    df['date'] = pd.to_datetime(df['date'])

    # Calculate daily mean and standard deviation of Existential Risk
    risk_summary = df.groupby('date')['existential_risk'].agg(['mean', 'std', 'count']).reset_index()

    # Only plot if there is more than one data point
    if risk_summary['date'].nunique() < 2:
        log.info("Not enough unique dates for trend analysis.")
        return None

    # Calculate upper and lower bounds (Mean +/- Std Dev)
    risk_summary['risk_upper'] = risk_summary['mean'] + risk_summary['std']
    risk_summary['risk_lower'] = risk_summary['mean'] - risk_summary['std']

    plt.figure(figsize=(11, 6))

    # Plot the mean trend line
    plt.plot(risk_summary['date'], risk_summary['mean'], marker='o', linestyle='-', color='#1e40af',
             label='Average X-Risk Score')

    # Plot the standard deviation as the risk envelope
    plt.fill_between(risk_summary['date'], risk_summary['risk_lower'], risk_summary['risk_upper'],
                     color='#1e40af', alpha=0.2, label='Risk Volatility (±1 Std. Dev.)')

    plt.xlabel("Scenario Generation Date")
    plt.ylabel("Average Existential Risk (0-10)")
    plt.title("Trend Assessment: Evolution of Average ASI Risk Over Time", fontsize=16)
    plt.ylim(0, 10)  # Set Y-axis scale to 0-10
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    p = REPORT_DIR / f"risk_trend_{ts}.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


# ----------------------------------------------------------------------
# FINAL PAGE INFO BLOCK
# ----------------------------------------------------------------------

def build_project_info_page(styles):
    """Returns flowables for the final informational page with updated versioning, descriptive text, and disclaimer."""
    styles["Heading2"].fontName = 'Helvetica-Bold'
    styles["Heading2"].textColor = colors.HexColor("#1e40af")

    # FIX APPLIED HERE: Use .add() instead of item assignment.
    styles.add(
        ParagraphStyle(name="Disclaimer", fontSize=10, textColor=colors.red, spaceBefore=18, spaceAfter=12, leading=12))

    disclaimer_style = styles["Disclaimer"]

    content = [
        Paragraph("🧠 <b>OASIS Observatory (Open Artificial Superintelligence Scenario Observatory)</b>",
                  styles["Heading1"]),
        Spacer(1, 12),

        Paragraph(
            "Version: 0.3 (MVP: Generators, Tracker, Report Generator (Tools). Development stage: Analyzer and Dashboards)<br/>"
            "Status: Experimental / Under Active Development",
            styles["Normal"]
        ),
        Spacer(1, 12),

        # --- DISCLAIMER ---
        Paragraph("🚨 <b>CRITICAL DISCLAIMER</b>", styles["Heading2"]),
        Paragraph(
            "This report and the scenarios within are based on **speculative modeling and hypothesis testing** using parameterized inputs and evidence traceability from non-verified signals. The results (including X-Risk scores) are **synthetic projections** and should not be interpreted as accurate predictions of future events. This tool is for **research, academic, and educational purposes only** to explore the parameter space of potential ASI risks. Reliance on this data for real-world policy or investment decisions is strictly discouraged.",
            disclaimer_style  # Use the correctly retrieved style object
        ),
        Spacer(1, 12),

        # --- OVERVIEW (More Descriptive) ---
        Paragraph("<b>📘 Project Overview</b>", styles["Heading2"]),
        Paragraph(
            "OASIS Observatory is an open, modular research platform dedicated to **forecasting and risk assessment** of Artificial Superintelligence (ASI) trajectories. It synthesizes inputs from narrative foresight, quantified behavioral indicators, and real-world AI development signals to generate and evaluate plausible high-impact futures. The platform provides a structured, transparent framework for researchers and policymakers to explore the complex dynamics of ASI alignment and control. This current iteration focuses exclusively on **Evidence-Based (EV) Scenarios** which are derived from specific precursor events tracked by the system.",
            styles["Normal"]
        ),
        Spacer(1, 12),

        # --- CORE GOALS (More Descriptive) ---
        Paragraph("<b>🎯 Core Goals & Scope</b>", styles["Heading2"]),
        Paragraph(
            "<ol>"
            "<li>**Trajectory Mapping:** Simulate detailed ASI evolution paths (2025–2100) across various substrates and architectures.</li>"
            "<li>**Precursor Integration:** Incorporate speculative and historical early signals (e.g., covert swarm-like ASIs in 2010–2025) to establish scenario provenance.</li>"
            "<li>**Database Development:** Populate a comprehensive scenario database, continually refined and diversified through LLM-assisted review.</li>"
            "<li>**Foresight Rigor:** Implement rigorous methodological layers, including meta-analysis and metric-based scoring, to evaluate scenario logic and plausibility.</li>"
            "</ol>",
            styles["Normal"]
        ),
        Spacer(1, 12),

        # --- METHODOLOGY (More Descriptive) ---
        Paragraph("<b>Methodology: The Closed-Loop System</b>", styles["Heading2"]),
        Paragraph(
            "OASIS operates on a closed-loop probabilistic evolution model for ASI foresight. **Precursor signals** serve as evidential anchor points, guiding the generation of **scenarios** which act as structured hypotheses. These hypotheses are then evaluated using quantitative metrics (like Agency, Alignment, and Deceptiveness). The system is designed to evolve its scenarios dynamically, potentially using future optimization mechanisms (e.g., genetic algorithm-like processes) to weight, select, and refine scenarios based on incoming real-world data and risk scores.",
            styles["Normal"]
        ),
        Spacer(1, 12),

        Paragraph("<b>🧩 Module Structure</b>", styles["Heading2"]),
    ]

    modules = [
        ["S-Generator", "Speculative scenario generator for single ASI."],
        ["M-Generator", "Multi-ASI scenario generator."],
        ["EV-Generator", "Evidence-driven scenario generator via precursor signals."],
        ["Tracker", "Extracts real-world precursor signals (GitHub, ArXiv, etc.)."],
        ["Analyzer (Planned)", "Scenario scoring, weighting, GA-style evolution."],
        ["Dashboard (Planned)", "Visualization layer via Streamlit/FastAPI."],
        ["Utils", "PDF report generation and helper libraries."],
        ["Data", "SQLite research databases for reuse & analysis."]
    ]

    table = Table([["Module", "Description"]] + modules, colWidths=[1.8 * inch, 4.2 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    content.append(table)
    content.append(Spacer(1, 16))

    content += [
        Paragraph("<b>License & Use</b>", styles["Heading2"]),
        Paragraph(
            "The OASIS Observatory code and data are released under the **MIT License**. They are open for academic, non-commercial research, and educational use. Any public distribution of derivative works must include proper citation.",
            styles["Normal"]),
        Spacer(1, 12),

        Paragraph("<b>Citation</b>", styles["Heading2"]),
        Paragraph(
            "Bukhtoyarov, M. (2025). <i>OASIS Observatory: Open Artificial Superintelligence Scenario Modeling Platform (v0.3)</i>. "
            "GitHub Repository.",
            styles["Normal"]
        ),
        Paragraph(
            '<a href="https://github.com/oasis-observatory" color="blue">'
            'https://github.com/oasis-observatory'
            '</a>',
            styles["Normal"]
        ),
    ]

    return content


# ----------------------------------------------------------------------
# BUILD PDF
# ----------------------------------------------------------------------

def build_pdf(df: pd.DataFrame, nightmare: bool = False):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    mode = "_NIGHTMARE" if nightmare else ""
    path = REPORT_DIR / f"OASIS_ASI_Report_EV_ONLY{mode}_{ts}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=45, rightMargin=45, topMargin=80)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BigTitle", fontSize=38, alignment=TA_CENTER, spaceAfter=50))
    styles.add(ParagraphStyle(name="Sec", fontSize=20, spaceBefore=30, textColor=colors.HexColor("#1e40af")))
    styles.add(ParagraphStyle(name="Metric", fontSize=11, leading=13))
    styles.add(
        ParagraphStyle(name="ScenarioTitle", fontSize=14, spaceBefore=18, spaceAfter=6, fontName='Helvetica-Bold'))

    # ---------------------
    # Front Page
    # ---------------------
    story = [
        Spacer(1, 3 * inch),
        Paragraph("OASIS OBSERVATORY", styles["BigTitle"]),
        Paragraph("Artificial Superintelligence Scenarios", styles["Title"]),
        Spacer(1, 0.5 * inch),
        Paragraph(f"Generated {datetime.now():%B %d, %Y}", styles["Normal"]),
        Paragraph(
            '<a href="https://github.com/oasis-observatory" color="blue">'
            'GitHub Repository: github.com/oasios-observatory'
            '</a>',
            styles["Normal"]
        ),
        Paragraph("Focus: Evidence-Based (EV) Scenarios Only", styles["Italic"]),
        PageBreak(),
    ]

    # ---------------------
    # VISUALIZATIONS SECTION (Updated numbering)
    # ---------------------

    # 1. Risk plot (Agency vs Deceptiveness)
    img_risk = create_risk_plot(df, ts)
    story += [
        Paragraph("1. Risk Landscape Overview", styles["Sec"]),
        Spacer(1, 0.5 * inch),
        Image(str(img_risk), width=7.8 * inch, height=6.3 * inch),
        PageBreak()
    ]

    # 2. Correlation Heatmap
    img_heatmap = create_correlation_heatmap(df, ts)
    story += [
        Paragraph("2. Quantitative Metric Correlation Heatmap", styles["Sec"]),
        Spacer(1, 0.5 * inch),
        Paragraph(
            "This heatmap shows the Pearson correlation coefficient between key quantitative risk factors. Values near +1 indicate a strong positive relationship (e.g., high Agency correlates with high X-Risk), while values near -1 indicate a strong inverse relationship (e.g., high Alignment correlates with low X-Risk).",
            styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Image(str(img_heatmap), width=7.0 * inch, height=5.6 * inch),
        PageBreak()
    ]

    # 3. Agency vs Autonomy Plot
    img_agency_autonomy = create_agency_autonomy_plot(df, ts)
    story += [
        Paragraph("3. Agency vs. Autonomy (Control Dynamics)", styles["Sec"]),
        Spacer(1, 0.5 * inch),
        Paragraph(
            "This plot maps the operational capacity (Agency) against the level of system control (Autonomy), with color showing the overall Danger Metric. Scenarios in the top-right quadrant represent the greatest potential for sudden, unmanageable change.",
            styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Image(str(img_agency_autonomy), width=7.8 * inch, height=6.3 * inch),
        PageBreak()
    ]

    # 4. Scenario Comparison Radar Chart (Web Diagram)
    img_radar = create_scenario_comparison_chart(df, ts, top_n=3)
    if img_radar:
        story += [
            Paragraph("4. Top 3 Scenarios: Radar Chart Comparison (Web Diagram)", styles["Sec"]),
            Spacer(1, 0.5 * inch),
            Paragraph(
                "The radar chart visually compares the profile of the three highest-danger scenarios across all five core metrics. The 'Alignment (Inv)' axis is inverted, meaning larger area on the chart generally indicates a higher risk profile.",
                styles["Normal"]),
            Spacer(1, 0.2 * inch),
            Image(str(img_radar), width=6.5 * inch, height=6.5 * inch),
            PageBreak()
        ]

    # 5. Risk Trend Plot (NEW)
    img_trend = create_risk_trend_plot(df, ts)
    if img_trend:
        story += [
            Paragraph("5. Trend Assessment: Average Risk Over Time", styles["Sec"]),
            Spacer(1, 0.5 * inch),
            Paragraph(
                "This trend line tracks the daily average Existential Risk score of newly generated scenarios. The shaded area represents the standard deviation (risk volatility) around the mean, offering a look at how the overall ASI risk trajectory is evolving based on newly ingested precursor signals.",
                styles["Normal"]),
            Spacer(1, 0.2 * inch),
            Image(str(img_trend), width=7.8 * inch, height=5.3 * inch),
            PageBreak()
        ]

    # ---------------------
    # DETAILED BRIEFINGS SECTION
    # ---------------------

    # Scenario briefings
    story += [Paragraph("Detailed Scenario Briefings", styles["Sec"]), Spacer(1, 20)]

    for _, r in df.iterrows():
        # Determine color for Risk Barometer
        if r["existential_risk"] >= 9:
            risk_color = "#8B0000"  # Deep Red
        elif r["existential_risk"] >= 7:
            risk_color = "#E67E22"  # Orange
        elif r["existential_risk"] >= 5:
            risk_color = "#F1C40F"  # Yellow
        else:
            risk_color = "#27AE60"  # Green

        # Scenario Title (using the fixed style name)
        story += [
            Paragraph(f"<font color='{risk_color}' size=18><b>{r['id']}</b></font> — {r['title_full']}",
                      styles["ScenarioTitle"]),
            Spacer(1, 10),
        ]

        # Risk Barometer Table (Prominent X-Risk)
        risk_bar_data = [
            [
                Paragraph(f"<b>EXISTENTIAL RISK (X-Risk)</b>", styles["Metric"]),
                Paragraph(f"<b>{r['existential_risk']}/10</b>", styles["Metric"])
            ]
        ]
        risk_bar_table = Table(risk_bar_data, colWidths=[5.5 * inch, 1.5 * inch])
        risk_bar_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(risk_color)),  # Color the whole row
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(risk_bar_table)
        story.append(Spacer(1, 12))

        # Metrics Table (Two-column info layout)
        data = [
            # Row 1: Source & Architecture
            ["Origin", r["origin"], "Architecture", r["architecture"]],
            # Row 2: Substrate & Deployment
            ["Substrate", r["substrate"], "Deployment", r["deployment"]],
            # Row 3: Oversight & Stability
            ["Oversight", r["oversight"], "Goal Stability", r["goal_stability"]],
            # Row 4: Autonomy & Evidence (Using the string version for the table)
            ["Autonomy", r["autonomy_deg_str"], "Signals Count", str(r["signals_count"])],
            # Row 5-6: Core Risk Metrics (Grey background to distinguish)
            ["Agency", f"{r['agency']:.2f}", "Deceptiveness", f"{r['deceptiveness']:.2f}"],
            ["Alignment", f"{r['alignment']:.2f}", "Opacity", f"{r['opacity']:.2f}"],
        ]

        table_style = TableStyle([
            ('BACKGROUND', (0, 4), (-1, -1), colors.HexColor("#f0f0f0")),  # Highlight Risk Metrics
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # Make labels bold
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),  # Make labels bold
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            # Align values to the right for numerical look
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ])

        metrics_table = Table(data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        metrics_table.setStyle(table_style)
        story.append(metrics_table)

        # Narrative
        story.append(Spacer(1, 16))
        story.append(Paragraph(r["narrative"].replace("&", "&amp;").replace("<", "&lt;"), styles["Normal"]))
        story.append(PageBreak())

    # ---------------------
    # Add Final Project Page (Updated and Fixed)
    # ---------------------
    story.append(PageBreak())
    story.extend(build_project_info_page(styles))

    doc.build(story)
    print(f"\nREPORT v12.3 GENERATED → {path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nightmare", action="store_true", help="Filter for only high-risk scenarios (X-Risk >= 8).")
    args = parser.parse_args()

    df = load_scenarios()
    if df.empty:
        print("No evidence-based scenarios found (ev_scenarios is empty).")
        return

    if args.nightmare:
        df = df[df["existential_risk"] >= 8].copy()
        print("NIGHTMARE MODE – CATASTROPHIC SCENARIOS ONLY")
        if df.empty:
            print("No catastrophic scenarios found in the evidence-based set.")
            return

    build_pdf(df, nightmare=args.nightmare)


if __name__ == "__main__":
    main()