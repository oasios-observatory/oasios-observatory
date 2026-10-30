import sqlite3
import structlog
import uuid
import json
import os

from datetime import datetime, timezone
from contextlib import contextmanager

log = structlog.get_logger()

DB_PATH = "data/asi_precursors.db"


class DatabaseManager:

    def __init__(self, db_path: str = DB_PATH):

        self.db_path = db_path

        directory = os.path.dirname(
            os.path.abspath(db_path)
        )

        os.makedirs(directory, exist_ok=True)

        self._init_precursor_db()

    @staticmethod
    def _current_timestamp():

        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _get_connection(self):

        conn = sqlite3.connect(
            self.db_path,
            timeout=30
        )

        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")

        try:
            yield conn
            conn.commit()

        except Exception:
            conn.rollback()
            raise

        finally:
            conn.close()

    def _execute_read(self, sql, params=()):

        with self._get_connection() as conn:

            cursor = conn.execute(sql, params)

            return [
                dict(row)
                for row in cursor.fetchall()
            ]

    def _execute_write(self, sql, params):

        with self._get_connection() as conn:

            if (
                isinstance(params, list)
                and params
                and isinstance(params[0], tuple)
            ):

                cursor = conn.executemany(
                    sql,
                    params
                )

            else:

                cursor = conn.execute(
                    sql,
                    params
                )

            return cursor.rowcount

    # ==================================================
    # INITIALIZATION AND MIGRATIONS
    # ==================================================

    def _init_precursor_db(self):

        with self._get_connection() as conn:

            conn.executescript("""

                CREATE TABLE IF NOT EXISTS raw_events (

                    event_id TEXT PRIMARY KEY,
                    collected_at TIMESTAMP NOT NULL,
                    source_system TEXT NOT NULL,
                    raw_payload TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    collection_method TEXT,
                    retention_class TEXT,
                    feature_processed_at TIMESTAMP

                );

                CREATE TABLE IF NOT EXISTS extracted_features (

                    feature_id TEXT PRIMARY KEY,

                    event_id TEXT REFERENCES raw_events(event_id)
                        ON DELETE CASCADE,

                    feature_type TEXT NOT NULL,

                    feature_value REAL,

                    feature_vector TEXT,

                    model_version TEXT,

                    extracted_at TIMESTAMP NOT NULL,

                    anomaly_evaluated_at TIMESTAMP

                );

                CREATE TABLE IF NOT EXISTS anomalies (

                    anomaly_id TEXT PRIMARY KEY,

                    anomaly_type TEXT NOT NULL,

                    severity INTEGER CHECK(
                        severity BETWEEN 1 AND 5
                    ),

                    confidence REAL,

                    first_seen TIMESTAMP,

                    last_seen TIMESTAMP,

                    description TEXT,

                    classification_status TEXT,

                    reviewer_notes TEXT,

                    coherence_kappa REAL,

                    cross_domain_span_xi REAL

                );

                CREATE TABLE IF NOT EXISTS anomaly_features (

                    anomaly_id TEXT REFERENCES anomalies(anomaly_id)
                        ON DELETE CASCADE,

                    feature_id TEXT REFERENCES extracted_features(feature_id)
                        ON DELETE CASCADE,

                    weight REAL,

                    PRIMARY KEY(anomaly_id, feature_id)

                );

                CREATE TABLE IF NOT EXISTS anomaly_groups (

                    group_id TEXT PRIMARY KEY,

                    primary_type TEXT,

                    description TEXT,

                    emergence_index_epsilon REAL,

                    coherence_kappa REAL,

                    cross_domain_span_xi REAL,

                    creation_time TIMESTAMP NOT NULL

                );

                CREATE TABLE IF NOT EXISTS anomaly_group_members (

                    group_id TEXT REFERENCES anomaly_groups(group_id)
                        ON DELETE CASCADE,

                    anomaly_id TEXT REFERENCES anomalies(anomaly_id)
                        ON DELETE CASCADE,

                    weight REAL DEFAULT 1.0,

                    PRIMARY KEY(group_id, anomaly_id)

                );

                CREATE TABLE IF NOT EXISTS provenance_records (

                    prov_id TEXT PRIMARY KEY,

                    entity_type TEXT NOT NULL,

                    entity_id TEXT NOT NULL,

                    process_description TEXT,

                    system_actor TEXT,

                    timestamp TIMESTAMP NOT NULL,

                    integrity_hash TEXT

                );

                CREATE TABLE IF NOT EXISTS governance_log (

                    log_id TEXT PRIMARY KEY,

                    entity_type TEXT NOT NULL,

                    entity_id TEXT NOT NULL,

                    actor TEXT NOT NULL,

                    action TEXT NOT NULL,

                    purpose TEXT NOT NULL,

                    timestamp TIMESTAMP NOT NULL

                );

                CREATE INDEX IF NOT EXISTS idx_raw_events_processed
                ON raw_events(feature_processed_at);

                CREATE INDEX IF NOT EXISTS idx_features_evaluated
                ON extracted_features(anomaly_evaluated_at);

                CREATE INDEX IF NOT EXISTS idx_groups_emergence
                ON anomaly_groups(emergence_index_epsilon);

                CREATE INDEX IF NOT EXISTS idx_anomaly_features_feature
                ON anomaly_features(feature_id);

                CREATE INDEX IF NOT EXISTS idx_group_members_anomaly
                ON anomaly_group_members(anomaly_id);

            """)

            migrations = {

                "raw_events": [
                    ("feature_processed_at", "TIMESTAMP")
                ],

                "extracted_features": [
                    ("anomaly_evaluated_at", "TIMESTAMP")
                ]

            }

            for table, columns in migrations.items():

                existing = {

                    row["name"]

                    for row in conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()

                }

                for column, column_type in columns:

                    if column not in existing:

                        conn.execute(
                            f"ALTER TABLE {table} "
                            f"ADD COLUMN {column} {column_type}"
                        )

        log.info(
            "database_initialized",
            path=self.db_path
        )

    # ==================================================
    # RAW EVENTS
    # ==================================================

    def insert_raw_event(
        self,
        event_id,
        source_system,
        raw_payload,
        collection_method,
        retention_class,
        event_hash,
    ):

        sql = """

            INSERT OR IGNORE INTO raw_events
            (
                event_id,
                collected_at,
                source_system,
                raw_payload,
                hash,
                collection_method,
                retention_class
            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """

        return self._execute_write(
            sql,
            (
                event_id,
                self._current_timestamp(),
                source_system,
                json.dumps(
                    raw_payload,
                    sort_keys=True,
                    ensure_ascii=False
                ),
                event_hash,
                collection_method,
                retention_class,
            )
        )

    def get_raw_events_for_feature_extraction(self):

        results = self._execute_read("""

            SELECT
                r.event_id,
                r.raw_payload,
                r.source_system,
                r.hash

            FROM raw_events r

            WHERE r.feature_processed_at IS NULL

            AND NOT EXISTS (

                SELECT 1

                FROM extracted_features f

                WHERE f.event_id = r.event_id

            )

            ORDER BY r.collected_at ASC

            LIMIT 500

        """)

        for row in results:

            try:

                row["raw_payload"] = json.loads(
                    row["raw_payload"]
                )

            except (TypeError, json.JSONDecodeError):

                row["raw_payload"] = {}

                log.warning(
                    "invalid_raw_payload",
                    event_id=row["event_id"]
                )

        return results

    def mark_events_processed(self, event_ids):

        if not event_ids:
            return 0

        placeholders = ",".join(
            "?" for _ in event_ids
        )

        sql = f"""

            UPDATE raw_events

            SET feature_processed_at = ?

            WHERE event_id IN ({placeholders})

        """

        return self._execute_write(
            sql,
            tuple(
                [self._current_timestamp()]
                + list(event_ids)
            )
        )

    # ==================================================
    # FEATURES
    # ==================================================

    def insert_extracted_features(self, features):

        if not features:
            return 0

        sql = """

            INSERT INTO extracted_features
            (
                feature_id,
                event_id,
                feature_type,
                feature_value,
                feature_vector,
                model_version,
                extracted_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?)

        """

        return self._execute_write(
            sql,
            features
        )

    def get_features_for_anomaly_inference(self):

        results = self._execute_read("""

            SELECT
                feature_id,
                feature_value,
                feature_vector,
                extracted_at

            FROM extracted_features

            WHERE anomaly_evaluated_at IS NULL

            ORDER BY extracted_at ASC

            LIMIT 1000

        """)

        for row in results:

            try:

                vector = row.get("feature_vector")

                if isinstance(vector, str):

                    row["feature_vector"] = json.loads(
                        vector
                    )

                elif not isinstance(vector, dict):

                    row["feature_vector"] = {}

            except (TypeError, json.JSONDecodeError):

                row["feature_vector"] = {}

                log.warning(
                    "invalid_feature_vector",
                    feature_id=row["feature_id"]
                )

        return results

    def mark_features_evaluated(self, feature_ids):

        if not feature_ids:
            return 0

        placeholders = ",".join(
            "?" for _ in feature_ids
        )

        sql = f"""

            UPDATE extracted_features

            SET anomaly_evaluated_at = ?

            WHERE feature_id IN ({placeholders})

        """

        return self._execute_write(
            sql,
            tuple(
                [self._current_timestamp()]
                + list(feature_ids)
            )
        )

    # ==================================================
    # ANOMALIES
    # ==================================================

    def insert_anomaly_record(self, anomalies):

        if not anomalies:
            return 0

        sql = """

            INSERT INTO anomalies
            (
                anomaly_id,
                anomaly_type,
                severity,
                confidence,
                first_seen,
                last_seen,
                description,
                classification_status,
                coherence_kappa,
                cross_domain_span_xi
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """

        return self._execute_write(
            sql,
            anomalies
        )

    def insert_anomaly_features_links(self, links):

        if not links:
            return 0

        sql = """

            INSERT INTO anomaly_features
            (
                anomaly_id,
                feature_id,
                weight
            )

            VALUES (?, ?, ?)

        """

        return self._execute_write(
            sql,
            links
        )

    def persist_anomalies(self, anomalies, links):

        """
        Persist anomaly records and their feature relationships
        atomically.
        """

        if not anomalies:
            return 0

        with self._get_connection() as conn:

            conn.executemany("""

                INSERT INTO anomalies
                (
                    anomaly_id,
                    anomaly_type,
                    severity,
                    confidence,
                    first_seen,
                    last_seen,
                    description,
                    classification_status,
                    coherence_kappa,
                    cross_domain_span_xi
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            """, anomalies)

            if links:

                conn.executemany("""

                    INSERT INTO anomaly_features
                    (
                        anomaly_id,
                        feature_id,
                        weight
                    )

                    VALUES (?, ?, ?)

                """, links)

        return len(anomalies)

    def get_ungrouped_anomalies(self):

        return self._execute_read("""

            SELECT a.*

            FROM anomalies a

            LEFT JOIN anomaly_group_members agm

                ON a.anomaly_id = agm.anomaly_id

            WHERE agm.anomaly_id IS NULL

            AND COALESCE(
                a.classification_status,
                ''
            ) != 'Grouped'

            ORDER BY a.first_seen ASC

        """)

    def update_anomaly_status(
        self,
        anomaly_ids,
        status
    ):

        if not anomaly_ids:
            return 0

        placeholders = ",".join(
            "?" for _ in anomaly_ids
        )

        sql = f"""

            UPDATE anomalies

            SET classification_status = ?

            WHERE anomaly_id IN ({placeholders})

        """

        return self._execute_write(
            sql,
            tuple(
                [status] + list(anomaly_ids)
            )
        )

    def get_sources_for_anomalies(self, anomaly_ids):

        if not anomaly_ids:
            return []

        placeholders = ",".join(
            "?" for _ in anomaly_ids
        )

        sql = f"""

            SELECT DISTINCT r.source_system

            FROM raw_events r

            JOIN extracted_features f

                ON r.event_id = f.event_id

            JOIN anomaly_features af

                ON f.feature_id = af.feature_id

            WHERE af.anomaly_id IN ({placeholders})

        """

        results = self._execute_read(
            sql,
            tuple(anomaly_ids)
        )

        return [
            row["source_system"]
            for row in results
        ]

    # ==================================================
    # GROUPS
    # ==================================================

    def insert_anomaly_group(
        self,
        group_data,
        member_tuples
    ):

        with self._get_connection() as conn:

            conn.execute("""

                INSERT INTO anomaly_groups
                (
                    group_id,
                    primary_type,
                    description,
                    emergence_index_epsilon,
                    coherence_kappa,
                    cross_domain_span_xi,
                    creation_time
                )

                VALUES (?, ?, ?, ?, ?, ?, ?)

            """, (
                group_data["group_id"],
                group_data["primary_type"],
                group_data["description"],
                group_data["emergence_index_epsilon"],
                group_data["coherence_kappa"],
                group_data["cross_domain_span_xi"],
                self._current_timestamp(),
            ))

            if member_tuples:

                conn.executemany("""

                    INSERT INTO anomaly_group_members
                    (
                        group_id,
                        anomaly_id,
                        weight
                    )

                    VALUES (?, ?, ?)

                """, member_tuples)

            log.info(
                "anomaly_group_inserted",
                group_id=group_data["group_id"],
                members=len(member_tuples)
            )

            return len(member_tuples)

    def get_top_groups_by_emergence(self, k=5):

        k = max(
            1,
            min(int(k), 100)
        )

        return self._execute_read("""

            SELECT *

            FROM anomaly_groups

            ORDER BY emergence_index_epsilon DESC

            LIMIT ?

        """, (k,))

    def get_group_details(self, group_id):

        results = self._execute_read("""

            SELECT *

            FROM anomaly_groups

            WHERE group_id = ?

        """, (group_id,))

        return results[0] if results else None

    # ==================================================
    # PROVENANCE
    # ==================================================

    def record_provenance(
        self,
        entity_type,
        entity_id,
        process_description,
        system_actor="OASIOS_System",
        integrity_hash=None,
    ):

        prov_id = str(uuid.uuid4())

        self._execute_write("""

            INSERT INTO provenance_records
            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (
            prov_id,
            entity_type,
            entity_id,
            process_description,
            system_actor,
            self._current_timestamp(),
            integrity_hash,
        ))

        return prov_id

    # ==================================================
    # GOVERNANCE
    # ==================================================

    def log_access(
        self,
        entity_type,
        entity_id,
        action,
        purpose,
        actor,
    ):

        log_id = str(uuid.uuid4())

        self._execute_write("""

            INSERT INTO governance_log
            VALUES (?, ?, ?, ?, ?, ?, ?)

        """, (
            log_id,
            entity_type,
            entity_id,
            actor,
            action,
            purpose,
            self._current_timestamp(),
        ))

        return log_id
