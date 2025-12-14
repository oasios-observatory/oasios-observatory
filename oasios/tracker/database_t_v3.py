import sqlite3
import structlog
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

log = structlog.get_logger()
DB_PATH = "data/asi_precursors.db"


class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_precursor_db()

    # --- Utility Methods ---

    @staticmethod
    def _current_timestamp() -> str:
        """Returns current UTC timestamp as ISO string."""
        return datetime.utcnow().isoformat()

    # --- Connection / Execution Helpers ---

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _execute_write(self, sql: str, params: Union[tuple, List[tuple]]) -> int:
        """Execute a write or bulk write with proper transaction handling."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if isinstance(params, list) and params and isinstance(params[0], tuple):
                    cursor.executemany(sql, params)
                else:
                    cursor.execute(sql, params)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                log.error("db_write_failed", sql=sql, params=params, error=str(e))
                raise

    def _execute_read(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a read query and return list of dicts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                results = [dict(row) for row in cursor.fetchall()]
                return results
            except sqlite3.Error as e:
                log.error("db_read_failed", sql=sql, params=params, error=str(e))
                raise

    # --- Database Initialization ---

    def _init_precursor_db(self):
        log.info("db_init_start", path=self.db_path)
        conn = self._get_connection()
        cursor = conn.cursor()

        # Raw Data Layer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_events (
                event_id TEXT PRIMARY KEY,
                collected_at TIMESTAMP NOT NULL,
                source_system TEXT NOT NULL,
                raw_payload TEXT NOT NULL,
                hash TEXT NOT NULL,
                collection_method TEXT,
                retention_class TEXT
            );
        """)

        # Feature Layer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_features (
                feature_id TEXT PRIMARY KEY,
                event_id TEXT REFERENCES raw_events(event_id) ON DELETE CASCADE,
                feature_type TEXT NOT NULL,
                feature_value REAL,
                feature_vector TEXT,
                model_version TEXT,
                extracted_at TIMESTAMP NOT NULL
            );
        """)

        # Anomaly Core
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                anomaly_id TEXT PRIMARY KEY,
                anomaly_type TEXT NOT NULL,
                severity INTEGER CHECK (severity BETWEEN 1 AND 5),
                confidence REAL,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                description TEXT,
                classification_status TEXT,
                reviewer_notes TEXT,
                coherence_kappa REAL,
                cross_domain_span_xi REAL
            );
        """)

        # Anomaly Features Link
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_features (
                anomaly_id TEXT REFERENCES anomalies(anomaly_id) ON DELETE CASCADE,
                feature_id TEXT REFERENCES extracted_features(feature_id) ON DELETE CASCADE,
                weight REAL,
                PRIMARY KEY (anomaly_id, feature_id)
            );
        """)

        # Anomaly Groups
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_groups (
                group_id TEXT PRIMARY KEY,
                primary_type TEXT,
                description TEXT,
                emergence_index_epsilon REAL,
                coherence_kappa REAL,
                cross_domain_span_xi REAL,
                creation_time TIMESTAMP NOT NULL
            );
        """)

        # Anomaly Group Members
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_group_members (
                group_id TEXT REFERENCES anomaly_groups(group_id) ON DELETE CASCADE,
                anomaly_id TEXT REFERENCES anomalies(anomaly_id) ON DELETE CASCADE,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (group_id, anomaly_id)
            );
        """)

        # Provenance Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance_records (
                prov_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                process_description TEXT,
                system_actor TEXT,
                timestamp TIMESTAMP NOT NULL,
                integrity_hash TEXT
            );
        """)

        # Governance Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS governance_log (
                log_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                purpose TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL
            );
        """)

        conn.commit()
        conn.close()
        log.info("db_init_success", tables=8)

    # --- Raw Event Methods (ERL) ---

    def insert_raw_event(self, event_id: str, source_system: str, raw_payload: Dict[str, Any],
                         collection_method: str, retention_class: str, event_hash: str) -> int:
        collected_at = self._current_timestamp()
        payload_str = json.dumps(raw_payload, sort_keys=True)
        sql = """
            INSERT INTO raw_events (event_id, collected_at, source_system, raw_payload, hash, collection_method, retention_class)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        return self._execute_write(sql,
                                   (event_id, collected_at, source_system, payload_str, event_hash,
                                    collection_method, retention_class))

    def get_raw_events_for_feature_extraction(self) -> List[Dict[str, Any]]:
        # FIX: Added 'source_system' to the SELECT clause for FSAL.
        sql = "SELECT event_id, raw_payload, source_system FROM raw_events ORDER BY collected_at DESC LIMIT 500"
        results = self._execute_read(sql)
        # Automatically parse JSON payloads 
        for r in results:
            r["raw_payload"] = json.loads(r["raw_payload"])
        return results

    # --- Feature Methods (FSAL) ---

    def insert_extracted_features(self, features: List[Tuple]) -> int:
        sql = """
            INSERT INTO extracted_features (feature_id, event_id, feature_type, feature_value, feature_vector, model_version, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        return self._execute_write(sql, features)

    # --- Anomaly Inference Methods (APSL-Inference Input) ---

    def get_features_for_anomaly_inference(self) -> List[Dict[str, Any]]:
        """Retrieves features that have not yet been linked to an anomaly (unprocessed)."""
        sql = """
            SELECT 
                f.feature_id, f.feature_vector, f.extracted_at
            FROM extracted_features f
            LEFT JOIN anomaly_features af ON f.feature_id = af.feature_id
            WHERE af.feature_id IS NULL
            ORDER BY f.extracted_at ASC
            LIMIT 1000
        """
        results = self._execute_read(sql)
        for r in results:
            if r.get("feature_vector"):
                r["feature_vector"] = json.loads(r["feature_vector"])
        return results

    # --- Anomaly Methods (APSL-Inference Output) ---

    def insert_anomaly_record(self, anomalies: List[Tuple]) -> int:
        """Inserts new anomaly records (A)."""
        sql = """
            INSERT INTO anomalies (anomaly_id, anomaly_type, severity, confidence, first_seen, last_seen, description, classification_status, coherence_kappa, cross_domain_span_xi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        return self._execute_write(sql, anomalies)

    def insert_anomaly_features_links(self, links: List[Tuple]) -> int:
        """Inserts links between anomalies and the features that caused them."""
        sql = """
            INSERT INTO anomaly_features (anomaly_id, feature_id, weight)
            VALUES (?, ?, ?)
        """
        return self._execute_write(sql, links)

    # --- Synthesis Helper Methods (APSL-Synthesis) ---

    def get_ungrouped_anomalies(self) -> List[Dict[str, Any]]:
        """Retrieves anomalies that have not yet been assigned to a group (G)."""
        sql = """
            SELECT a.*
            FROM anomalies a
            LEFT JOIN anomaly_group_members agm ON a.anomaly_id = agm.anomaly_id
            WHERE agm.anomaly_id IS NULL AND a.classification_status != 'Grouped'
        """
        return self._execute_read(sql)

    def insert_anomaly_group(self, group_data: Dict[str, Any], member_tuples: List[Tuple]) -> int:
        """Insert group + members safely in a single transaction."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                group_sql = """
                    INSERT INTO anomaly_groups (group_id, primary_type, description, emergence_index_epsilon, coherence_kappa, cross_domain_span_xi, creation_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(group_sql, (
                    group_data['group_id'],
                    group_data['primary_type'],
                    group_data['description'],
                    group_data['emergence_index_epsilon'],
                    group_data['coherence_kappa'],
                    group_data['cross_domain_span_xi'],
                    self._current_timestamp()
                ))

                member_sql = """
                    INSERT INTO anomaly_group_members (group_id, anomaly_id, weight)
                    VALUES (?, ?, ?)
                """
                cursor.executemany(member_sql, member_tuples)
                conn.commit()
                log.info("pattern_synthesized", group_id=group_data['group_id'], members=len(member_tuples))
                return len(member_tuples)
            except sqlite3.Error as e:
                conn.rollback()
                log.error("anomaly_group_insert_failed", group_id=group_data['group_id'], error=str(e))
                raise

    def update_anomaly_status(self, anomaly_ids: List[str], status: str) -> int:
        if not anomaly_ids:
            return 0
        placeholders = ','.join(['?' for _ in anomaly_ids])
        # Ensure status field is updated based on processing step
        sql = f"UPDATE anomalies SET classification_status = ? WHERE anomaly_id IN ({placeholders})"
        params = tuple([status] + anomaly_ids)
        return self._execute_write(sql, params)

    def get_sources_for_anomalies(self, anomaly_ids: List[str]) -> List[str]:
        """Retrieves the source_system for all events linked to the given anomalies."""
        if not anomaly_ids:
            return []
        placeholders = ','.join(['?' for _ in anomaly_ids])

        sql = f"""
            SELECT DISTINCT r.source_system
            FROM raw_events r
            JOIN extracted_features f ON r.event_id = f.event_id
            JOIN anomaly_features af ON f.feature_id = af.feature_id
            WHERE af.anomaly_id IN ({placeholders})
        """

        params = tuple(anomaly_ids)
        results = self._execute_read(sql, params)
        return [r['source_system'] for r in results]

    # --- Scenario / SIL Methods ---

    def get_top_groups_by_emergence(self, k: int = 5) -> List[Dict[str, Any]]:
        sql = """
            SELECT group_id, primary_type, emergence_index_epsilon, coherence_kappa, cross_domain_span_xi
            FROM anomaly_groups
            ORDER BY emergence_index_epsilon DESC
            LIMIT ?
        """
        return self._execute_read(sql, (k,))

    def get_group_details(self, group_id: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT group_id, primary_type, description, emergence_index_epsilon, coherence_kappa, cross_domain_span_xi
            FROM anomaly_groups
            WHERE group_id = ?
        """
        result = self._execute_read(sql, (group_id,))
        return result[0] if result else None

    # --- Governance / Provenance Methods ---

    def record_provenance(self, entity_type: str, entity_id: str,
                          process_description: str, system_actor: str = "ECO_System",
                          integrity_hash: Optional[str] = None) -> str:
        prov_id = str(uuid.uuid4())
        timestamp = self._current_timestamp()
        sql = """
            INSERT INTO provenance_records (prov_id, entity_type, entity_id, process_description, system_actor, timestamp, integrity_hash) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_write(sql, (prov_id, entity_type, entity_id, process_description, system_actor, timestamp,
                                  integrity_hash))
        return prov_id

    def log_access(self, entity_type: str, entity_id: str, action: str, purpose: str, actor: str) -> str:
        log_id = str(uuid.uuid4())
        timestamp = self._current_timestamp()
        sql = """
            INSERT INTO governance_log (log_id, entity_type, entity_id, actor, action, purpose, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_write(sql, (log_id, entity_type, entity_id, actor, action, purpose, timestamp))
        return log_id