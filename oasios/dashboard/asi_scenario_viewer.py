#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OASIOS Observatory - Scenario Viewer

Displays:
1. The 10 most recently generated scenarios.
2. Database statistics.
3. Most common categorical scenario parameters.
4. Average numerical scenario parameters.

Run:
    streamlit run oasios/dashboard/asi_scenarios_viewer.py
"""

import json
import sqlite3

from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from oasios.config import settings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OASIOS Observatory",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 OASIOS Observatory")
st.caption("ASI Scenario Database Explorer")


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = Path(settings.db_path)

TABLES = {
    "Speculative": "s_scenarios",
    "Evidence-based": "ev_scenarios",
}

LATEST_SCENARIOS_LIMIT = 10


# Categorical scenario parameters.

CATEGORICAL_PARAMETERS = {
    "Initial Origin": ("origin", "initial_origin"),
    "Development Dynamics": ("origin", "development_dynamics"),

    "Architecture": ("architecture", "type"),
    "Deployment Topology": ("architecture", "deployment_topology"),

    "Substrate": ("substrate", "type"),
    "Deployment Medium": ("substrate", "deployment_medium"),
    "Substrate Resilience": ("substrate", "resilience"),

    "Oversight Type": ("oversight_structure", "type"),
    "Oversight Effectiveness": (
        "oversight_structure",
        "effectiveness",
    ),
    "Control Surface": (
        "oversight_structure",
        "control_surface",
    ),

    "Stated Goal": ("goals_and_behavior", "stated_goal"),
    "Goal Stability": ("goals_and_behavior", "goal_stability"),

    "Deployment Strategy": (
        "impact_and_control",
        "deployment_strategy",
    ),
}


# Numerical scenario parameters.

NUMERIC_PARAMETERS = {
    "Agency": ("core_capabilities", "agency_level"),
    "Autonomy": ("core_capabilities", "autonomy_degree"),
    "Alignment": ("core_capabilities", "alignment_score"),
    "Opacity": ("goals_and_behavior", "opacity"),
    "Deceptiveness": ("goals_and_behavior", "deceptiveness"),
}


# ============================================================
# DATABASE
# ============================================================

@st.cache_data(ttl=10, show_spinner=False)
def load_database_data():
    """Load scenario records from the configured OASIOS database."""

    if not DB_PATH.exists():
        return {
            "error": f"Database not found: {DB_PATH}",
            "scenarios": [],
        }

    scenarios = []

    try:
        with closing(
            sqlite3.connect(str(DB_PATH), timeout=30)
        ) as conn:

            conn.row_factory = sqlite3.Row

            for category, table in TABLES.items():

                # Check whether the table exists.
                table_exists = conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = ?
                    """,
                    (table,),
                ).fetchone()

                if not table_exists:
                    continue

                cursor = conn.execute(
                    f"SELECT * FROM {table}"
                )

                for row in cursor.fetchall():

                    record = dict(row)

                    scenario_id = record.get("id")

                    raw_data = record.get("scenario_data")

                    try:
                        if isinstance(raw_data, dict):
                            data = raw_data

                        elif isinstance(raw_data, str) and raw_data.strip():
                            data = json.loads(raw_data)

                        else:
                            data = {}

                        if not isinstance(data, dict):
                            data = {}

                    except (json.JSONDecodeError, TypeError) as exc:

                        st.warning(
                            f"Could not parse scenario "
                            f"{scenario_id}: {exc}"
                        )

                        data = {}

                    scenarios.append({
                        "id": scenario_id,
                        "title": (
                            record.get("title")
                            or data.get("title")
                            or f"Scenario {scenario_id}"
                        ),
                        "category": category,
                        "created_at": record.get("created_at"),
                        "model_used": record.get("model_used"),
                        "data": data,
                        "legacy_params": record.get("params"),
                        "legacy_narrative": record.get("narrative"),
                        "legacy_timeline": record.get("timeline"),
                    })

        # Sort scenarios from newest to oldest.
        scenarios.sort(
            key=lambda item: (
                str(item["created_at"] or ""),
                str(item["id"] or ""),
            ),
            reverse=True,
        )

        return {
            "error": None,
            "scenarios": scenarios,
        }

    except sqlite3.Error as exc:

        return {
            "error": str(exc),
            "scenarios": [],
        }


# ============================================================
# DATA EXTRACTION
# ============================================================

def get_nested_value(
    data: dict,
    section: str,
    parameter: str,
    default: Any = None,
):
    """Safely retrieve nested scenario parameters."""

    if not isinstance(data, dict):
        return default

    section_data = data.get(section, {})

    if not isinstance(section_data, dict):
        return default

    return section_data.get(parameter, default)


def extract_parameter(
    scenario: dict,
    section: str,
    parameter: str,
):
    """Extract a parameter from a scenario."""

    return get_nested_value(
        scenario.get("data", {}),
        section,
        parameter,
    )


def get_narrative(scenario: dict) -> str:
    """Extract scenario narrative."""

    data = scenario.get("data", {})

    content = data.get("scenario_content", {})

    if isinstance(content, dict):

        narrative = content.get("narrative")

        if narrative:
            return str(narrative)

    return (
        scenario.get("legacy_narrative")
        or "No narrative available."
    )


def get_timeline(scenario: dict) -> list:
    """Extract timeline phases."""

    data = scenario.get("data", {})

    content = data.get("scenario_content", {})

    if isinstance(content, dict):

        timeline = content.get("timeline", {})

        if isinstance(timeline, list):
            return timeline

        if isinstance(timeline, dict):

            phases = timeline.get("phases", [])

            if isinstance(phases, list):
                return phases

    try:

        legacy = scenario.get("legacy_timeline")

        if isinstance(legacy, list):
            return legacy

        if isinstance(legacy, str) and legacy.strip():

            parsed = json.loads(legacy)

            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, dict):
                return parsed.get("phases", [])

    except (json.JSONDecodeError, TypeError):
        pass

    return []


def format_value(value: Any) -> str:
    """Format parameter values for display."""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(
            str(item) for item in value
        ) or "None"

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    if value is None:
        return "Unknown"

    return str(value)


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(scenarios: list) -> dict:
    """Calculate database statistics."""

    statistics = {
        "total": len(scenarios),
        "speculative": 0,
        "evidence": 0,
        "categorical": {},
        "numeric": {},
    }

    # Count scenario categories.

    for scenario in scenarios:

        if scenario["category"] == "Speculative":
            statistics["speculative"] += 1

        else:
            statistics["evidence"] += 1

    # Calculate categorical frequencies.

    for label, (section, parameter) in CATEGORICAL_PARAMETERS.items():

        values = []

        for scenario in scenarios:

            value = extract_parameter(
                scenario,
                section,
                parameter,
            )

            if value is None:
                continue

            if isinstance(value, (list, tuple, set)):

                values.extend(
                    str(item)
                    for item in value
                    if item is not None
                )

            else:
                values.append(str(value))

        statistics["categorical"][label] = Counter(values)

    # Calculate numerical statistics.

    for label, (section, parameter) in NUMERIC_PARAMETERS.items():

        values = []

        for scenario in scenarios:

            value = extract_parameter(
                scenario,
                section,
                parameter,
            )

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):

                values.append(float(value))

        statistics["numeric"][label] = values

    return statistics


# ============================================================
# SCENARIO PARAMETERS
# ============================================================

def render_scenario_parameters(scenario: dict):
    """Render expandable scenario parameters."""

    parameter_rows = []

    for label, (section, parameter) in CATEGORICAL_PARAMETERS.items():

        value = extract_parameter(
            scenario,
            section,
            parameter,
        )

        if value is not None:

            parameter_rows.append({
                "Parameter": label,
                "Value": format_value(value),
            })

    for label, (section, parameter) in NUMERIC_PARAMETERS.items():

        value = extract_parameter(
            scenario,
            section,
            parameter,
        )

        if value is not None:

            parameter_rows.append({
                "Parameter": label,
                "Value": format_value(value),
            })

    with st.expander("⚙️ Scenario Parameters", expanded=False):

        if parameter_rows:

            st.dataframe(
                pd.DataFrame(parameter_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No structured parameters available.")

        st.markdown("#### Complete Scenario Data")

        st.json(scenario.get("data", {}))


# ============================================================
# LATEST SCENARIOS
# ============================================================

def render_scenario(scenario: dict):
    """Render an individual scenario."""

    st.subheader(
        f"📊 {scenario['title']}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.caption("Scenario ID")
        st.code(str(scenario["id"]))

    with col2:

        st.caption("Category")
        st.write(scenario["category"])

    with col3:

        st.caption("Creation Date")
        st.write(
            scenario["created_at"] or "Unknown"
        )

    st.caption(
        f"Model: {scenario['model_used'] or 'Unknown'}"
    )

    st.markdown("---")

    # Narrative.

    st.markdown("#### 📖 Full Narrative")

    st.markdown(
        get_narrative(scenario)
    )

    # Timeline.

    timeline = get_timeline(scenario)

    if timeline:

        st.markdown("#### 📅 Timeline")

        if all(isinstance(item, dict) for item in timeline):

            st.dataframe(
                pd.DataFrame(timeline),
                use_container_width=True,
                hide_index=True,
            )

        else:

            for index, phase in enumerate(timeline, start=1):
                st.write(f"**Phase {index}:** {format_value(phase)}")

    else:

        st.info("No timeline available.")

    # Expandable parameters.

    render_scenario_parameters(scenario)

    # Individual JSON download.

    download_data = {
        "id": scenario["id"],
        "title": scenario["title"],
        "category": scenario["category"],
        "created_at": scenario["created_at"],
        "model_used": scenario["model_used"],
        "scenario_data": scenario["data"],
    }

    st.download_button(
        label="⬇️ Download Scenario JSON",
        data=json.dumps(
            download_data,
            indent=4,
            ensure_ascii=False,
            default=str,
        ),
        file_name=f"scenario_{scenario['id']}.json",
        mime="application/json",
        key=f"download_{scenario['category']}_{scenario['id']}",
    )


def render_latest_scenarios(scenarios: list):
    """Display the latest ten scenarios."""

    st.header("🕒 Latest 10 Scenarios")

    latest = scenarios[:LATEST_SCENARIOS_LIMIT]

    if not latest:

        st.info("No scenarios found.")

        return

    st.caption(
        f"Showing {len(latest)} most recently generated scenarios."
    )

    for index, scenario in enumerate(latest, start=1):

        with st.expander(
            f"{index}. {scenario['title']} "
            f"— {scenario['category']} "
            f"— {scenario['created_at'] or 'Unknown date'}",
            expanded=(index == 1),
        ):

            render_scenario(scenario)


# ============================================================
# STATISTICS VIEW
# ============================================================

def render_statistics(scenarios: list):
    """Display database statistics."""

    st.header("📈 Database Statistics")

    stats = calculate_statistics(scenarios)

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Scenarios",
        stats["total"],
    )

    col2.metric(
        "Speculative Scenarios",
        stats["speculative"],
    )

    col3.metric(
        "Evidence-based Scenarios",
        stats["evidence"],
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Numerical statistics
    # --------------------------------------------------------

    st.subheader("📊 Average Parameter Values")

    numeric_rows = []

    for label, values in stats["numeric"].items():

        numeric_rows.append({
            "Parameter": label,
            "Average": (
                round(sum(values) / len(values), 3)
                if values else "N/A"
            ),
            "Minimum": (
                round(min(values), 3)
                if values else "N/A"
            ),
            "Maximum": (
                round(max(values), 3)
                if values else "N/A"
            ),
            "Samples": len(values),
        })

    st.dataframe(
        pd.DataFrame(numeric_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Categorical statistics
    # --------------------------------------------------------

    st.subheader("🏆 Most Common Parameters")

    for label, counter in stats["categorical"].items():

        with st.expander(
            label,
            expanded=label in (
                "Deployment Strategy",
                "Architecture",
                "Substrate",
                "Oversight Type",
            ),
        ):

            if not counter:

                st.info("No data available.")

                continue

            total_values = sum(counter.values())

            frequency_rows = [
                {
                    "Value": value,
                    "Count": count,
                    "Percentage": round(
                        count / total_values * 100,
                        1,
                    ),
                }
                for value, count in counter.most_common()
            ]

            st.dataframe(
                pd.DataFrame(frequency_rows),
                use_container_width=True,
                hide_index=True,
            )

            top_value, top_count = counter.most_common(1)[0]

            st.success(
                f"Most common: {top_value} "
                f"({top_count} occurrences)"
            )

    # --------------------------------------------------------
    # Database distribution
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📚 Scenario Distribution")

    distribution = pd.DataFrame({
        "Category": [
            "Speculative",
            "Evidence-based",
        ],
        "Count": [
            stats["speculative"],
            stats["evidence"],
        ],
    })

    st.bar_chart(
        distribution.set_index("Category")
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        st.header("🔭 OASIOS")

        st.caption("Scenario Observatory")

        st.markdown("---")

        st.write("**Database:**")

        st.code(str(DB_PATH))

        if st.button(
            "🔄 Refresh Database",
            use_container_width=True,
        ):

            load_database_data.clear()
            st.rerun()

        st.markdown("---")

        st.caption(
            "Explore generated scenarios and "
            "database-wide statistical information."
        )

    # --------------------------------------------------------
    # Database validation
    # --------------------------------------------------------

    if not DB_PATH.exists():

        st.error(
            f"Database not found: {DB_PATH}"
        )

        st.info(
            "Generate scenarios first using the OASIOS generator."
        )

        st.stop()

    # --------------------------------------------------------
    # Load database
    # --------------------------------------------------------

    with st.spinner("Loading OASIOS scenarios..."):

        result = load_database_data()

    if result["error"]:

        st.error(
            f"Database error: {result['error']}"
        )

        st.stop()

    scenarios = result["scenarios"]

    if not scenarios:

        st.warning(
            "The database exists, but no scenarios were found."
        )

        st.stop()

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    tab_latest, tab_statistics = st.tabs([
        "🕒 Latest 10 Scenarios",
        "📈 Database Statistics",
    ])

    with tab_latest:

        render_latest_scenarios(scenarios)

    with tab_statistics:

        render_statistics(scenarios)

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        f"OASIOS Observatory | "
        f"{len(scenarios)} scenarios loaded | "
        f"Database: {DB_PATH.name}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
