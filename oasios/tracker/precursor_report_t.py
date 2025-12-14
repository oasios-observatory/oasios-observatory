# oasios/tracker/precursor_report_t.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path
import sys

# ──────────────────────────────────────────────────────────────
# Robust paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "asi_precursors.db"

if not DB_PATH.exists():
    print(f"Database not found at: {DB_PATH}")
    print("Run: python -m oasios.tracker.cli_tracker_v3 sweep")
    sys.exit(1)

print(f"Using database: {DB_PATH}")


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def safe_float(x, default=0.0):
    return float(x) if x is not None and pd.notna(x) else default


def create_pdf_report():
    conn = connect_db()
    try:
        # 1. Check what actually exists
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )['name'].tolist()

        print(f"Found tables: {tables}")

        # 2. Overview stats with safe handling
        stats = {}

        # Raw events
        raw_count = pd.read_sql_query("SELECT COUNT(*) as c FROM raw_events", conn).iloc[0]['c']
        stats['raw_events'] = raw_count

        # Anomaly groups (may be 0)
        group_stats = pd.read_sql_query("""
            SELECT 
                COUNT(*) as count,
                AVG(emergence_index_epsilon) as avg_epsilon,
                AVG(coherence_kappa) as avg_kappa,
                AVG(cross_domain_span_xi) as avg_xi
            FROM anomaly_groups
        """, conn).iloc[0]

        stats['groups'] = {
            'count': int(group_stats['count']),
            'avg_epsilon': safe_float(group_stats['avg_epsilon'], 0.0),
            'avg_kappa': safe_float(group_stats['avg_kappa'], 0.0),
            'avg_xi': safe_float(group_stats['avg_xi'], 0.0),
        }

        # Anomalies
        anomaly_stats = pd.read_sql_query("""
            SELECT COUNT(*) as c, AVG(severity) as avg_sev
            FROM anomalies
        """, conn).iloc[0]
        stats['anomalies'] = {
            'count': int(anomaly_stats['c']),
            'avg_severity': safe_float(anomaly_stats['avg_sev'], 0.0)
        }

        # Sources breakdown
        sources_df = pd.read_sql_query("""
            SELECT source_system, COUNT(*) as count 
            FROM raw_events 
            GROUP BY source_system 
            ORDER BY count DESC
        """, conn)

        # 3. Top groups (if any)
        top_groups_df = pd.read_sql_query("""
            SELECT 
                group_id, primary_type, description,
                emergence_index_epsilon as ε,
                coherence_kappa as κ,
                cross_domain_span_xi as ξ,
                creation_time
            FROM anomaly_groups
            ORDER BY emergence_index_epsilon DESC
            LIMIT 15
        """, conn)

        # 4. Generate PDF
        output_pdf = BASE_DIR / "data/ASI_Precursors_Emergence_Report.pdf"
        with PdfPages(output_pdf) as pdf:
            # Page 1: Title + Overview
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

            title = "ECO Emergence Observatory Report"
            ax.text(0.5, 0.95, title, ha='center', va='center', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.90, f"Generated: {now}", ha='center', va='center', fontsize=10)

            overview = f"""
OVERVIEW
────────────────────────────────
Raw Events Collected          → {stats['raw_events']:,}
Anomalies Detected            → {stats['anomalies']['count']:,}
Systemic Pattern Groups (G)   → {stats['groups']['count']:,}

EMERGENCE METRICS (if groups exist)
────────────────────────────────
Avg Emergence Index (ε)       → {stats['groups']['avg_epsilon']:.3f}
Avg Coherence (κ)             → {stats['groups']['avg_kappa']:.3f}
Avg Cross-Domain Span (ξ)     → {stats['groups']['avg_xi']:.3f}
Avg Anomaly Severity          → {stats['anomalies']['avg_severity']:.2f}

SOURCES
────────────────────────────────
{sources_df.to_string(index=False, header=False)}
            """.strip()

            if stats['groups']['count'] == 0:
                overview += "\n\nNo anomaly groups synthesized yet.\nRun multiple sweeps to build up signal volume."

            ax.text(0.05, 0.80, overview, fontsize=10, va='top', fontfamily='monospace')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 2: Top Pattern Groups Table
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            ax.text(0.5, 0.95, "Top Systemic Emergence Patterns (ranked by ε)",
                    ha='center', fontsize=14, fontweight='bold')

            if len(top_groups_df) == 0:
                ax.text(0.5, 0.5, "No pattern groups synthesized yet.\n\n"
                                 "The APSL synthesis layer needs more high-scoring anomalies.\n"
                                 "Try running 'sweep' several times over a few days.",
                        ha='center', va='center', fontsize=10, alpha=0.7)
            else:
                # Shorten description for display
                display_df = top_groups_df.copy()
                display_df['description'] = display_df['description'].str[:80] + "..."
                table = ax.table(cellText=display_df.values,
                                 colLabels=display_df.columns,
                                 cellLoc='center',
                                 loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.1, 1.8)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 3: Simple timeline (even if no groups)
            fig, ax = plt.subplots(figsize=(8.5, 6))
            if len(top_groups_df) > 0:
                times = pd.to_datetime(top_groups_df['creation_time'])
                ax.plot(times, top_groups_df['ε'], 'o-', color='#d62728')
                ax.set_title('Emergence Index (ε) Over Time')
                ax.set_ylabel('ε Value')
                plt.xticks(rotation=45)
            else:
                ax.text(0.5, 0.5, "No groups yet → no timeline", ha='center', va='center', alpha=0.6)
                ax.set_title('Emergence Index Timeline')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"\nReport generated: {output_pdf}")
        print(f"   → {stats['raw_events']} raw events, {stats['groups']['count']} pattern groups")

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    create_pdf_report()