import uuid
import structlog
import math

from datetime import datetime, timezone
from typing import Dict, Any, List

from oasios.tracker.database_t import DatabaseManager

log = structlog.get_logger()


ANOMALY_MODEL_VERSION = "ap-inference-v1.2"
GROUPING_MODEL_VERSION = "ap-synthesis-v1.2"

BASELINE_THRESHOLD = 0.25

TIME_WINDOW_HOURS = 168

SEVERITY_DIFFERENCE_THRESHOLD = 2

MAX_FEATURE_VALUE = 10.0

API_QUERIES = {
    "github": "...",
    "arxiv": "...",
    "semantic_scholar": "...",
    "huggingface": "...",
    "openml": "..."
}


class AnomalyEngine:

    def __init__(self):

        self.db = DatabaseManager()

        log.info(
            "anomaly_engine_initialized",
            inference_version=ANOMALY_MODEL_VERSION,
            grouping_version=GROUPING_MODEL_VERSION
        )

    # ==================================================
    # UTILITIES
    # ==================================================

    @staticmethod
    def _timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _safe_float(value, default=0.0):

        try:

            result = float(value)

            if not math.isfinite(result):
                return default

            return result

        except (TypeError, ValueError):

            return default

    @staticmethod
    def _parse_timestamp(value):

        timestamp = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if timestamp.tzinfo is None:

            timestamp = timestamp.replace(
                tzinfo=timezone.utc
            )

        return timestamp

    # ==================================================
    # PHASE 1: ANOMALY INFERENCE
    # ==================================================

    def infer_anomalies(self):

        log.info("inference_start")

        features = self.db.get_features_for_anomaly_inference()

        if not features:

            log.info(
                "inference_skipped",
                reason="No new features"
            )

            return 0

        anomalies = []
        links = []
        evaluations = []

        for feature in features:

            feature_id = feature["feature_id"]

            vector = feature.get(
                "feature_vector",
                {}
            )

            feature_value = feature.get(
                "feature_value",
                0.0
            )

            try:

                anomaly, link, detected = (
                    self._apply_anomaly_function(
                        feature_id,
                        vector,
                        feature_value
                    )
                )

                if anomaly:

                    anomalies.append(anomaly)
                    links.append(link)

                evaluations.append(feature_id)

            except Exception:

                log.exception(
                    "feature_inference_failed",
                    feature_id=feature_id
                )

                # Failed features remain eligible for retry.

        # Persist anomalies and relationships atomically.

        if anomalies:

            self.db.persist_anomalies(
                anomalies,
                links
            )

            # Provenance is recorded after successful persistence.

            for anomaly in anomalies:

                self.db.record_provenance(
                    entity_type="anomalies",
                    entity_id=anomaly[0],
                    process_description=(
                        f"Anomaly inference via "
                        f"{ANOMALY_MODEL_VERSION}"
                    ),
                    system_actor="APSL Engine"
                )

        # Only successfully evaluated features are marked.

        self.db.mark_features_evaluated(
            evaluations
        )

        log.info(
            "inference_complete",
            inferred_count=len(anomalies),
            features_processed=len(evaluations),
            failed_count=len(features) - len(evaluations)
        )

        return len(anomalies)

    # ==================================================
    # ANOMALY FUNCTION
    # ==================================================

    def _apply_anomaly_function(
        self,
        feature_id: str,
        feature_vector: Dict[str, float],
        feature_value: float = 0.0
    ):

        if not isinstance(feature_vector, dict):

            return None, None, False

        vector_values = []

        for value in feature_vector.values():

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):

                numeric = self._safe_float(value)

                # Feature indicators are expected in [0, 1].

                vector_values.append(
                    max(0.0, min(1.0, numeric))
                )

        vector_score = (

            sum(vector_values) / len(vector_values)

            if vector_values

            else 0.0

        )

        metadata_value = self._safe_float(
            feature_value
        )

        metadata_score = max(
            0.0,
            min(
                MAX_FEATURE_VALUE,
                metadata_value
            )
        ) / MAX_FEATURE_VALUE

        score = max(
            vector_score,
            metadata_score
        )

        score = max(
            0.0,
            min(1.0, score)
        )

        detected = score > BASELINE_THRESHOLD

        if not detected:

            return None, None, False

        anomaly_id = str(uuid.uuid4())

        timestamp = self._timestamp()

        severity = int(
            min(
                5,
                max(
                    1,
                    math.ceil(score * 5)
                )
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                score + 0.1
            )
        )

        anomaly_record = (

            anomaly_id,

            "Emergent_Pattern_Candidate",

            severity,

            confidence,

            timestamp,

            timestamp,

            (
                f"Combined feature score {score:.3f} "
                f"exceeded threshold "
                f"{BASELINE_THRESHOLD:.3f}."
            ),

            "New",

            0.0,

            0.0

        )

        link_record = (
            anomaly_id,
            feature_id,
            confidence
        )

        return anomaly_record, link_record, True

    # ==================================================
    # PHASE 2: PATTERN SYNTHESIS
    # ==================================================

    def group_anomalies(self):

        log.info("synthesis_start")

        anomalies = self.db.get_ungrouped_anomalies()

        if not anomalies:

            log.info(
                "synthesis_skipped",
                reason="No ungrouped anomalies"
            )

            return 0

        # Connected-component clustering.

        components = self._build_connected_components(
            anomalies
        )

        groups = []

        for component in components:

            group, links = (
                self._synthesize_group_descriptors(
                    component
                )
            )

            groups.append(
                (group, links)
            )

        grouped_ids = []

        for group, links in groups:

            self.db.insert_anomaly_group(
                group,
                links
            )

            grouped_ids.extend(
                anomaly_id
                for _, anomaly_id, _ in links
            )

            self.db.record_provenance(
                entity_type="anomaly_groups",
                entity_id=group["group_id"],
                process_description=(
                    f"Pattern synthesis via "
                    f"{GROUPING_MODEL_VERSION}"
                ),
                system_actor="APSL Engine"
            )

        self.db.update_anomaly_status(
            grouped_ids,
            status="Grouped"
        )

        log.info(
            "synthesis_complete",
            groups_created=len(groups),
            anomalies_grouped=len(grouped_ids)
        )

        return len(groups)

    # ==================================================
    # CONNECTED COMPONENTS
    # ==================================================

    def _are_related(self, first, second):

        try:

            first_time = self._parse_timestamp(
                first["first_seen"]
            )

            second_time = self._parse_timestamp(
                second["first_seen"]
            )

            time_diff = abs(
                (
                    first_time - second_time
                ).total_seconds()
            ) / 3600

            severity_diff = abs(
                int(first["severity"])
                - int(second["severity"])
            )

            same_type = (
                first["anomaly_type"]
                == second["anomaly_type"]
            )

            return (

                time_diff <= TIME_WINDOW_HOURS

                and severity_diff
                <= SEVERITY_DIFFERENCE_THRESHOLD

                and same_type

            )

        except (
            ValueError,
            TypeError,
            KeyError
        ):

            return False

    def _build_connected_components(
        self,
        anomalies: List[Dict[str, Any]]
    ):

        """
        Builds connected components using pairwise
        anomaly relationships.

        Unlike greedy root clustering, this allows
        transitive relationships.

        Example:

        A <-> B
        B <-> C

        A, B and C become one component.
        """

        count = len(anomalies)

        visited = set()

        components = []

        for start in range(count):

            if start in visited:
                continue

            component = []

            stack = [start]

            visited.add(start)

            while stack:

                current = stack.pop()

                component.append(
                    anomalies[current]
                )

                for candidate in range(count):

                    if candidate in visited:
                        continue

                    if self._are_related(
                        anomalies[current],
                        anomalies[candidate]
                    ):

                        visited.add(candidate)

                        stack.append(candidate)

            components.append(component)

        return components

    # ==================================================
    # GROUP DESCRIPTORS
    # ==================================================

    def _synthesize_group_descriptors(
        self,
        anomaly_list: List[Dict[str, Any]]
    ):

        if not anomaly_list:

            raise ValueError(
                "Cannot synthesize an empty anomaly group"
            )

        group_id = str(uuid.uuid4())

        anomaly_ids = [
            anomaly["anomaly_id"]
            for anomaly in anomaly_list
        ]

        sources = self.db.get_sources_for_anomalies(
            anomaly_ids
        )

        timestamps = [

            self._parse_timestamp(
                anomaly["first_seen"]
            ).timestamp()

            for anomaly in anomaly_list

        ]

        if len(timestamps) > 1:

            avg_time_diff = (
                max(timestamps) - min(timestamps)
            ) / (len(timestamps) - 1)

        else:

            avg_time_diff = 0.0

        max_window = TIME_WINDOW_HOURS * 3600

        coherence = max(
            0.0,
            min(
                1.0,
                1.0 - avg_time_diff / max_window
            )
        )

        # Use actual source diversity rather than the number
        # of configured API integrations.

        source_count = len(set(sources))

        known_source_count = max(
            1,
            len(API_QUERIES)
        )

        cross_domain_span = (

            (source_count - 1)
            / (known_source_count - 1)

            if known_source_count > 1

            else 0.0

        )

        cross_domain_span = max(
            0.0,
            min(1.0, cross_domain_span)
        )

        avg_severity = sum(
            int(anomaly["severity"])
            for anomaly in anomaly_list
        ) / len(anomaly_list)

        emergence_index = (

            avg_severity / 5.0
            + cross_domain_span
            + coherence

        ) / 3.0

        emergence_index = max(
            0.0,
            min(1.0, emergence_index)
        )

        type_counts = {}

        for anomaly in anomaly_list:

            anomaly_type = anomaly["anomaly_type"]

            type_counts[anomaly_type] = (
                type_counts.get(anomaly_type, 0) + 1
            )

        primary_type = max(
            type_counts,
            key=type_counts.get
        )

        group_record = {

            "group_id": group_id,

            "primary_type": primary_type,

            "description": (
                f"Cluster of {len(anomaly_list)} "
                f"{primary_type} anomalies."
            ),

            "emergence_index_epsilon": emergence_index,

            "coherence_kappa": coherence,

            "cross_domain_span_xi": cross_domain_span,

            "model_version": GROUPING_MODEL_VERSION

        }

        member_links = [

            (
                group_id,
                anomaly["anomaly_id"],
                anomaly["confidence"]
            )

            for anomaly in anomaly_list

        ]

        return group_record, member_links
