# oasios/dashboard/asi_scenarios_viewer.py
# Run in terminal 'streamlit run oasios/dashboard/asi_scenario_viewer.py'

import streamlit as st
import sqlite3
import json
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="OASIOS Observatory Viewer",
    page_icon="🔭",
    layout="wide"
)

# Database path
DB_PATH = Path("data/asi_scenarios.db")


def get_connection():
    """Create database connection"""
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return None
    return sqlite3.connect(str(DB_PATH))


def load_scenarios(table_name):
    """Load scenarios from specified table"""
    conn = get_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, title, data FROM {table_name}")
        rows = cursor.fetchall()

        scenarios = []
        for row in rows:
            try:
                data = json.loads(row[2])
                scenarios.append({
                    'id': row[0],
                    'title': row[1],
                    'data': data
                })
            except json.JSONDecodeError:
                st.warning(f"Failed to parse scenario {row[0]}")

        return scenarios
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []
    finally:
        conn.close()


def render_single_asi_scenario(scenario):
    """Render single ASI scenario details"""
    data = scenario['data']

    # Header
    st.subheader(f"📊 {scenario['title']}")

    # Metadata in columns
    col1, col2, col3 = st.columns(3)

    def safe_display(value):
        """Convert complex values to display strings"""
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    with col1:
        st.metric("Scenario ID", str(scenario['id'])[:8] + "...")
        if 'origin' in data:
            st.write("**Origin:**", safe_display(data['origin']))

    with col2:
        if 'emergence_type' in data:
            st.write("**Emergence Type:**", safe_display(data['emergence_type']))
        if 'alignment' in data:
            st.write("**Alignment:**", safe_display(data['alignment']))

    with col3:
        if 'speed' in data:
            st.write("**Speed:**", safe_display(data['speed']))
        if 'capability_level' in data:
            st.write("**Capability:**", safe_display(data['capability_level']))

    # Timeline
    if 'timeline' in data and data['timeline']:
        st.markdown("### 📅 Timeline")
        timeline_df = pd.DataFrame(data['timeline'])
        if not timeline_df.empty:
            st.dataframe(timeline_df, width = "stretch", hide_index=True)

    # Narrative
    if 'narrative' in data:
        st.markdown("### 📖 Narrative")
        st.markdown(f"_{data['narrative']}_")

    # Full data (expandable)
    with st.expander("🔍 View Raw JSON Data"):
        st.json(data)


def render_multi_asi_scenario(scenario):
    """Render multi-ASI scenario details"""
    data = scenario['data']

    # Header
    st.subheader(f"🌐 {scenario['title']}")

    # Swarm metadata
    col1, col2, col3 = st.columns(3)

    def safe_display(value):
        """Convert complex values to display strings"""
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    with col1:
        st.write("**Scenario ID:**", str(scenario['id'])[:8] + "...")
        if 'num_asis' in data:
            st.metric("Number of ASIs", data['num_asis'])

    with col2:
        if 'simulation_years' in data:
            st.metric("Simulation Years", data['simulation_years'])
        if 'total_interactions' in data:
            st.metric("Total Interactions", data['total_interactions'])

    with col3:
        if 'dominant_pattern' in data:
            st.write("**Dominant Pattern:**", safe_display(data['dominant_pattern']))

    # ASI Entities
    if 'asis' in data and data['asis']:
        st.markdown("### 🤖 ASI Entities")
        asis_list = []
        for asi in data['asis']:
            asis_list.append({
                'ID': asi.get('id', 'N/A'),
                'Name': asi.get('name', 'N/A'),
                'Origin': asi.get('origin', 'N/A'),
                'Alignment': asi.get('alignment', 'N/A'),
                'Capability': asi.get('capability_level', 'N/A')
            })
        st.dataframe(pd.DataFrame(asis_list), use_container_width=True, hide_index=True)

    # Interactions
    if 'interactions' in data and data['interactions']:
        st.markdown("### 🔗 Key Interactions")
        interactions_df = pd.DataFrame(data['interactions'][:20])  # Show first 20
        if not interactions_df.empty:
            st.dataframe(interactions_df, use_container_width=True, hide_index=True)
            if len(data['interactions']) > 20:
                st.info(f"Showing first 20 of {len(data['interactions'])} interactions")

    # Narrative
    if 'narrative' in data:
        st.markdown("### 📖 Swarm Narrative")
        st.markdown(f"_{data['narrative']}_")

    # Full data (expandable)
    with st.expander("🔍 View Raw JSON Data"):
        st.json(data)


# Main UI
st.title("🔭 OASIOS Observatory Viewer")
st.markdown("*Explore ASI scenario simulations*")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    view_mode = st.radio(
        "Select View",
        ["Single ASI Scenarios", "Multi-ASI Scenarios", "Statistics"]
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("OASIOS Observatory simulates trajectories of Artificial Superintelligence (ASI).")
    st.markdown("[GitHub Repository](https://github.com/oasios-observatory/oasios-observatory)")

# Check database exists
if not DB_PATH.exists():
    st.error(f"⚠️ Database not found at `{DB_PATH}`")
    st.info("Generate scenarios first using: `oasios generate --count 10`")
    st.stop()

# Single ASI View
if view_mode == "Single ASI Scenarios":
    st.header("Single ASI Scenarios")

    scenarios = load_scenarios("scenarios")

    if not scenarios:
        st.warning("No single ASI scenarios found in database.")
        st.info("Generate scenarios using: `oasios generate --count 10`")
    else:
        st.success(f"Found {len(scenarios)} scenarios")

        # Scenario selector
        scenario_titles = [f"{s['title']} ({s['id'][:8]}...)" for s in scenarios]
        selected_idx = st.selectbox("Select Scenario", range(len(scenarios)),
                                    format_func=lambda i: scenario_titles[i])

        if selected_idx is not None:
            scenario = scenarios[selected_idx]

            st.markdown("---")
            render_single_asi_scenario(scenario)

# Multi-ASI View
elif view_mode == "Multi-ASI Scenarios":
    st.header("Multi-ASI Scenarios")

    scenarios = load_scenarios("multi_asi_scenarios")

    if not scenarios:
        st.warning("No multi-ASI scenarios found in database.")
        st.info("Generate swarm scenarios using: `oasios swarm --count 5`")
    else:
        st.success(f"Found {len(scenarios)} swarm scenarios")

        # Scenario selector
        scenario_titles = [f"{s['title']} ({s['id'][:8]}...)" for s in scenarios]
        selected_idx = st.selectbox("Select Scenario", range(len(scenarios)),
                                    format_func=lambda i: scenario_titles[i])

        if selected_idx is not None:
            scenario = scenarios[selected_idx]

            st.markdown("---")
            render_multi_asi_scenario(scenario)

# Statistics View
elif view_mode == "Statistics":
    st.header("Database Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single ASI Scenarios")
        single_scenarios = load_scenarios("scenarios")
        st.metric("Total Scenarios", len(single_scenarios))

        if single_scenarios:
            # Extract some stats
            origins = [s['data'].get('origin', 'Unknown') for s in single_scenarios]
            origin_counts = pd.Series(origins).value_counts()
            st.bar_chart(origin_counts)

    with col2:
        st.subheader("Multi-ASI Scenarios")
        multi_scenarios = load_scenarios("multi_asi_scenarios")
        st.metric("Total Swarm Scenarios", len(multi_scenarios))

        if multi_scenarios:
            # Extract ASI counts
            asi_counts = [s['data'].get('num_asis', 0) for s in multi_scenarios]
            if asi_counts:
                st.metric("Avg ASIs per Swarm", f"{sum(asi_counts) / len(asi_counts):.1f}")

    # Combined timeline
    st.markdown("---")
    st.subheader("Generation Timeline")

    all_scenarios = single_scenarios + multi_scenarios
    if all_scenarios:
        st.info(f"Total scenarios across both tables: {len(all_scenarios)}")
    else:
        st.warning("No scenarios found in database.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit | OASIOS Observatory v0.1.1*")