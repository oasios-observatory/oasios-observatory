# oasios/tracker/anomaly_engine_v3.py
# APSL: Anomaly Inference (phi) and Pattern Synthesis (C) Engine

import structlog
import uuid
import json
import time
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# V3 Dependencies
# Assuming DatabaseManager is available
from oasios.tracker.database_t_v3 import DatabaseManager

# Assuming API_QUERIES is available (imported from core_t_v3 if needed, or defined here)
API_QUERIES = {"github": "...", "arxiv": "...", "semantic_scholar": "...", "huggingface": "...", "openml": "..."}

log = structlog.get_logger()

# --- CONSTANTS ---
ANOMALY_MODEL_VERSION = "ap-inference-v1.0"
GROUPING_MODEL_VERSION = "ap-synthesis-v1.0"
BASELINE_THRESHOLD = 0.15  # Adjusted lower for initial testing

# Clustering hyper-parameters (Simulated)
TIME_WINDOW_HOURS = 24 * 7
SEVERITY_DIFFERENCE_THRESHOLD = 2


class AnomalyEngine:
    def __init__(self):
        self.db = DatabaseManager()
        log.info("anomaly_engine_initialized")

    # --- APSL PHASE 1: ANOMALY INFERENCE (A = phi(x, B)) ---

    def infer_anomalies(self) -> int:
        """Evaluates extracted features against baselines (B) to identify anomalies (A)."""
        log.info("inference_start")

        features_to_process = self._get_unprocessed_features()
        if not features_to_process:
            log.info("inference_skipped", reason="No new features")
            return 0

        new_anomaly_records: List[Tuple] = []
        feature_link_records: List[Tuple] = []

        for feature in features_to_process:
            # NOTE: We need to parse the vector from JSON string if the DB manager didn't do it.
            # Based on the DB manager implementation, the 'feature_vector' should be a dict here.
            feature_vector = feature['feature_vector']  # Already a Dict

            anomaly_data, links_data = self._apply_anomaly_function(feature['feature_id'], feature_vector)

            if anomaly_data:
                new_anomaly_records.append(anomaly_data)
                feature_link_records.extend(links_data)

        total_inferred = self._store_inferred_data(new_anomaly_records, feature_link_records)
        # Removed mark_features_as_inferred, as linking to anomaly_features is the marker.

        log.info("inference_complete", inferred_count=total_inferred, features_processed=len(features_to_process))
        return total_inferred

    # --- APSL PHASE 2: PATTERN SYNTHESIS (G = C(A)) ---

    def group_anomalies(self) -> int:
        """
        Clusters anomalies (A) into cohesive, cross-domain groups (G).
        Implements C: A^k -> G.
        """
        log.info("synthesis_start")

        # 1. Input: Get anomalies without an existing group
        # NOTE: This DB method is still missing and needs implementation in database_t_v3.py
        anomalies_to_group = self.db.get_ungrouped_anomalies()
        if not anomalies_to_group:
            log.info("synthesis_skipped", reason="No new anomalies to group")
            return 0

        grouped_anomalies_ids = []
        new_groups: List[Tuple[Dict, List[Tuple]]] = []  # Group record dict, list of member link tuples

        # Simple Greedy Clustering: Group anomalies that are close in time and similar in type/severity
        while anomalies_to_group:
            root_anomaly = anomalies_to_group.pop(0)
            candidate_group = [root_anomaly]

            root_time = datetime.fromisoformat(root_anomaly['first_seen'])

            # Find matching anomalies from the remaining list
            i = 0
            while i < len(anomalies_to_group):
                candidate = anomalies_to_group[i]
                candidate_time = datetime.fromisoformat(candidate['first_seen'])

                # Check 1: Time Proximity
                time_diff = abs(root_time - candidate_time).total_seconds() / 3600

                # Check 2: Type/Severity Metric
                severity_diff = abs(root_anomaly['severity'] - candidate['severity'])

                if (time_diff <= TIME_WINDOW_HOURS and
                        severity_diff <= SEVERITY_DIFFERENCE_THRESHOLD and
                        root_anomaly['anomaly_type'] == candidate['anomaly_type']):

                    candidate_group.append(anomalies_to_group.pop(i))
                else:
                    i += 1

            # 2. Synthesis: Calculate Group Descriptors
            group_record, members_links = self._synthesize_group_descriptors(candidate_group)
            new_groups.append((group_record, members_links))

            grouped_anomalies_ids.extend([a['anomaly_id'] for a in candidate_group])

        # 3. Output & Linkage: Store new groups and update anomaly status
        total_groups = self._store_synthesized_data(new_groups)
        # NOTE: This DB method is still missing and needs implementation in database_t_v3.py
        self.db.update_anomaly_status(grouped_anomalies_ids, status="Grouped")

        log.info("synthesis_complete", groups_created=total_groups, anomalies_grouped=len(grouped_anomalies_ids))
        return total_groups

    # --- Auxiliary Synthesis Methods ---

    def _synthesize_group_descriptors(self, anomaly_list: List[Dict[str, Any]]) -> Tuple[Dict, List[Tuple]]:
        """
        Calculates the cluster properties and prepares insertion data.
        """
        group_id = str(uuid.uuid4())

        # Get unique source systems from the underlying features/events
        # NOTE: This DB method is still missing and needs implementation in database_t_v3.py
        source_systems = self.db.get_sources_for_anomalies([a['anomaly_id'] for a in anomaly_list])

        # --- Group Descriptors ---

        # 1. Coherence (kappa)
        time_stamps = [datetime.fromisoformat(a['first_seen']).timestamp() for a in anomaly_list]
        avg_time_diff = (max(time_stamps) - min(time_stamps)) / len(time_stamps) if len(time_stamps) > 1 else 0
        MAX_TIME_WINDOW_SEC = TIME_WINDOW_HOURS * 3600
        coherence = 1.0 - (avg_time_diff / MAX_TIME_WINDOW_SEC) if avg_time_diff else 1.0
        coherence = max(0.0, min(1.0, coherence))

        # 2. Cross-Domain Span (xi)
        source_count = len(set(source_systems))
        # Ensure we don't divide by zero if API_QUERIES is small
        possible_sources = len(API_QUERIES) if len(API_QUERIES) > 1 else 1
        CROSS_DOMAIN_SPAN = (source_count - 1) / (possible_sources - 1) if possible_sources > 1 else 0

        # 3. Emergence Index (epsilon)
        avg_severity = sum(a['severity'] for a in anomaly_list) / len(anomaly_list)
        emergence_index = (avg_severity / 5.0 + CROSS_DOMAIN_SPAN + coherence) / 3.0
        emergence_index = min(1.0, emergence_index)

        # Determine Primary Group Type
        primary_type = max(set(a['anomaly_type'] for a in anomaly_list),
                           key=list(a['anomaly_type'] for a in anomaly_list).count)

        group_record = {
            'group_id': group_id,
            'primary_type': primary_type,
            'description': f"Cluster of {len(anomaly_list)} {primary_type} anomalies.",
            'emergence_index_epsilon': emergence_index,
            'coherence_kappa': coherence,
            'cross_domain_span_xi': CROSS_DOMAIN_SPAN,
            'model_version': GROUPING_MODEL_VERSION
        }

        # Linkage Data - Tuple format for DB insertion
        members_links = [
            (group_id, a['anomaly_id'], a['confidence'])  # (group_id, anomaly_id, weight)
            for a in anomaly_list
        ]

        return group_record, members_links

    def _store_synthesized_data(self, new_groups: List[Tuple[Dict, List[Tuple]]]) -> int:
        """Inserts new group records and their member linkages, then records provenance."""

        # Format group dicts into a list of tuples for exec_write
        group_tuples: List[Tuple] = []
        for g_dict, _ in new_groups:
            # We must match the order expected by the DB insert_anomaly_group method
            group_tuples.append((
                g_dict['group_id'],
                g_dict['primary_type'],
                g_dict['description'],
                g_dict['emergence_index_epsilon'],
                g_dict['coherence_kappa'],
                g_dict['cross_domain_span_xi'],
                # creation_time is handled by the DB method
            ))

        # Flatten member links (already in tuple format)
        member_links_tuples = [link for _, links in new_groups for link in links]

        # NOTE: insert_anomaly_group now inserts group and members in one transaction
        inserted_groups = 0
        inserted_members = 0

        # We must iterate and call the transactional insert_anomaly_group one by one
        for group_dict, member_links in new_groups:
            # member_links is List[Tuple], group_dict is Dict
            inserted_members += self.db.insert_anomaly_group(group_dict, member_links)
            inserted_groups += 1

        # Record provenance for the new groups
        for g_dict, _ in new_groups:
            self.db.record_provenance(
                entity_type="anomaly_groups",
                entity_id=g_dict['group_id'],
                process_description=f"Pattern synthesis via {GROUPING_MODEL_VERSION}",
                system_actor="APSL Engine"
            )

        log.info("group_data_stored", groups=inserted_groups, links=inserted_members)
        return inserted_groups

    # --- Private/Helper DB Access Methods (New implementations) ---

    def _get_unprocessed_features(self) -> List[Dict[str, Any]]:
        """Retrieves features that have not yet been evaluated for anomalies."""
        # This calls the new DB method we need to implement
        return self.db.get_features_for_anomaly_inference()

    def _apply_anomaly_function(self, feature_id: str, feature_vector: Dict[str, float]) -> Tuple[
        Optional[Tuple], List[Tuple]]:
        # See implementation above
        # ... actual implementation is in section A ...
        if not feature_vector:
            return None, []

        vector_elements = list(feature_vector.values())
        score = sum(vector_elements) / len(vector_elements) if vector_elements else 0.0

        if score > BASELINE_THRESHOLD:
            anomaly_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            # Simple heuristic mapping
            severity = int(min(5, max(1, round(score * 5))))
            confidence = min(1.0, score + 0.1)

            # Anomaly Tuple (must match DB insert_anomaly_record order)
            anomaly_record_tuple = (
                anomaly_id,
                "Emergent_Pattern_Candidate",
                severity,
                confidence,
                timestamp,
                timestamp,
                f"Feature score ({score:.2f}) exceeded threshold ({BASELINE_THRESHOLD:.2f}).",
                "New",
                random.uniform(0.1, 0.9),
                random.uniform(0.1, 0.9)
            )

            # Linkage Tuple (must match DB insert_anomaly_features_links order)
            link_record_tuple = (anomaly_id, feature_id, confidence)

            return anomaly_record_tuple, [link_record_tuple]

        return None, []

    def _store_inferred_data(self, anomalies: List[Tuple], links: List[Tuple]) -> int:
        """Stores new anomaly records and their links to features."""
        if not anomalies:
            return 0

        # 1. Insert anomaly records
        inserted_anomalies = self.db.insert_anomaly_record(anomalies)

        # 2. Insert feature links (anomaly_features)
        inserted_links = self.db.insert_anomaly_features_links(links)

        # 3. Record Provenance
        for a in anomalies:
            self.db.record_provenance(
                entity_type="anomalies",
                entity_id=a[0],
                process_description=f"Anomaly inference via {ANOMALY_MODEL_VERSION}",
                system_actor="APSL Engine"
            )

        log.info("inferred_data_stored", anomalies=inserted_anomalies, links=inserted_links)
        return inserted_anomalies