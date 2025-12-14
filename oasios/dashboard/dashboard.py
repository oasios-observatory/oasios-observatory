# dashboard/dashboard.py
# OASIS Observatory — Live ASI Scenario Dashboard

import streamlit as st
import sqlite3
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ───────────────────── ULTRA-DEFENSIVE IMPORTS ─────────────────────
PCA = None
TfidfVectorizer = None
px = None
SKLEARN_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    TfidfVectorizer = None
    st.warning("scikit-learn not installed → Clustering disabled. Run: pip install scikit-learn")

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    st.warning("plotly not installed → Interactive plots disabled. Run: pip install plotly")

CLUSTERING_AVAILABLE = SKLEARN_AVAILABLE and PLOTLY_AVAILABLE

# ───────────────────── Page config ─────────────────────
st.set_page_config(page_title="OASIS Observatory", layout="wide")
st.title("OASIS Observatory")
st.caption("Open Artificial Superintelligence Scenario Observatory — Live Evidence-Grounded Foresight")
st.markdown(f"**Last refresh:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ───────────────────── Database ─────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "asi_scenarios.db"
if not DB_PATH.exists():
    st.error(f"Database not found: `{DB_PATH}`\n\nRun `oasios generate` and `oasios analyze link` first.")
    st.stop()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# ───────────────────── Load scenarios ─────────────────────
@st.cache_data(ttl=60, show_spinner="Loading scenarios...")
def load_scenarios():
    rows = conn.execute("SELECT id, title, data FROM scenarios").fetchall()
    scenarios = []
    for r in rows:
        try:
            data = json.loads(r["data"])

            prob = data.get("quantitative_assessment", {})\
                       .get("probability", {})\
                       .get("emergence_probability", 0.04)

            trend = data.get("quantitative_assessment", {})\
                        .get("probability", {})\
                        .get("trend", "stable")

            title = data.get("title", "Untitled Scenario")
            narrative = data.get("scenario_content", {}).get("narrative", "")[:1000].strip()
            origin = data.get("origin", {}).get("initial_origin", "unknown").capitalize()
            autonomy = str(data.get("core_capabilities", {}).get("autonomy_degree", "unknown")).capitalize()
            risk = int(data.get("quantitative_assessment", {})\
                           .get("risk_assessment", {})\
                           .get("existential", {}).get("score", 0))

            scenarios.append({
                "id": r["id"],
                "title": title,
                "probability": float(prob),
                "trend": trend,
                "narrative": narrative or "No narrative.",
                "origin": origin,
                "autonomy": autonomy,
                "risk": risk,
            })
        except Exception:
            continue
    return pd.DataFrame(scenarios)

df = load_scenarios()
if df.empty:
    st.error("No scenarios found.")
    st.stop()

df = df.sort_values("probability", ascending=False).reset_index(drop=True)

# ───────────────────── Sidebar filters ─────────────────────
st.sidebar.header("Filters")
min_prob = st.sidebar.slider("Min Probability", 0.0, 1.0, 0.01, 0.01)
origin_filter = st.sidebar.multiselect("Origin", options=sorted(df["origin"].unique()), default=list(df["origin"].unique()))
trend_filter = st.sidebar.multiselect("Trend", options=["increasing","stable","decreasing"], default=["increasing","stable"])

filtered = df[
    (df["probability"] >= min_prob) &
    (df["origin"].isin(origin_filter)) &
    (df["trend"].isin(trend_filter))
].copy()

# ───────────────────── Main layout ─────────────────────
col1, col2 = st.columns([2.2, 1])

with col1:
    st.subheader(f"Top Scenarios — {len(filtered)} shown")
    for _, row in filtered.head(30).iterrows():
        if row["trend"] == "increasing":
            badge = "Increasing"
            color = "#ff4757"
        elif row["trend"] == "decreasing":
            badge = "Decreasing"
            color = "#2ed573"
        else:
            badge = "Stable"
            color = "#636e72"

        st.markdown(
            f"### {row['title']}\n"
            f"**{row['probability']:.1%}** • "
            f"<span style='color:{color};font-weight:bold'>{badge}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"Origin: {row['origin']} • Autonomy: {row['autonomy']} • X-Risk: {row['risk']}/10")
        with st.expander("Narrative", expanded=False):
            st.write(row["narrative"])
        st.divider()

with col2:
    st.subheader("Signal Activity")
    links = conn.execute("""
        SELECT scenario_id, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM signal_scenario_links
        WHERE confidence >= 0.45
        GROUP BY scenario_id
        ORDER BY count DESC LIMIT 12
    """).fetchall()

    if links:
        st.write("**Most evidenced scenarios:**")
        for l in links:
            title_row = conn.execute("SELECT data FROM scenarios WHERE id=?", (l["scenario_id"],)).fetchone()
            try:
                title = json.loads(title_row["data"]).get("title", "??")
            except:
                title = "??"
            st.write(f"• **{title[:50]}**")
            st.caption(f"   {l['count']} signals (avg conf {l['avg_conf']:.2f})")
    else:
        st.info("No strong links yet.\nRun: `oasios analyze link`")

# ───────────────────── Clustering map ─────────────────────
st.markdown("---")
st.subheader("Scenario Space — Semantic Similarity Map")

if not CLUSTERING_AVAILABLE:
    st.info("Install packages to enable this view:\n\n`pip install scikit-learn plotly pandas`")
else:
    @st.cache_data(ttl=300)
    def compute_clustering():
        texts = df["narrative"].fillna("").tolist()
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
        X = vectorizer.fit_transform(texts)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.toarray())

        return pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "title": df["title"],
            "probability": df["probability"],
            "trend": df["trend"],
            "origin": df["origin"],
            "risk": df["risk"],
        })

    cluster_df = compute_clustering()

    fig = px.scatter(
        cluster_df,
        x="x", y="y",
        size=np.clip(cluster_df["probability"] * 180, 15, 150),
        color="trend",
        hover_name="title",
        hover_data={"origin": True, "probability": ":.1%", "risk": True, "x": False, "y": False},
        color_discrete_map={"increasing": "#ff4757", "stable": "#8c8c8c", "decreasing": "#2ed573"},
        title="ASI Scenario Space<br><sup>Size = Probability • Color = Trend</sup>"
    )
    fig.update_traces(marker=dict(line=dict(width=1.2, color="black")))
    fig.update_layout(height=760)
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────── Footer ─────────────────────
st.success("Dashboard auto-refreshes when new signals are linked.")
st.caption("OASIS Observatory v0.5 — MIT License — 2025")