import requests
import xmltodict
import time
import structlog
import hashlib
import uuid
import json

from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import urlencode

from .database_t import DatabaseManager
from .classifier_t import classify_and_score
from .rate_limiter_t import RequestBudgetManager
from .anomaly_engine_t import AnomalyEngine

from oasios.tracker.robot_policy_t import can_fetch

log = structlog.get_logger()


# ==================================================
# CONFIGURATION
# ==================================================

API_QUERIES = {

    "github": (
        "agentic OR multimodality OR "
        "alignment-research OR rlhf"
    ),

    "arxiv": (
        "cat:cs.AI OR cat:cs.LG"
    ),

    "semantic_scholar": (
        "AGI OR catastrophic risk OR meta-learning"
    ),

    "huggingface": (
        "agentic OR AGI OR rlhf OR safety"
    ),

    "openml": (
        "multimodal,agentic,meta-learning"
    )

}

MAX_RESULTS_PER_SOURCE = 10

MODEL_VERSION = "OASIOS-v0.5"

REQUEST_TIMEOUT = 20

RATE_MANAGER = RequestBudgetManager()


class CoreTracker:

    def __init__(self):

        self.db = DatabaseManager()

        self.anomaly_engine = AnomalyEngine()

        self.headers = {

            "User-Agent": "OASIOS-Research/0.5"

        }

        self.session = requests.Session()

        self.session.headers.update(
            self.headers
        )

        log.info(
            "tracker_initialized",
            sources=list(API_QUERIES.keys()),
            model_version=MODEL_VERSION
        )

    # ==================================================
    # UTILITIES
    # ==================================================

    @staticmethod
    def _generate_id(seed: str) -> str:

        return hashlib.sha256(
            seed.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    def _check_governance(
        self,
        base_url: str,
        full_url: str,
        source: str
    ) -> bool:

        """
        Enforces robots.txt policy.

        Failed governance checks fail closed.
        """

        try:

            allowed = can_fetch(
                base_url,
                full_url,
                self.headers["User-Agent"]
            )

            if not allowed:

                log.warning(
                    "robots_txt_blocked",
                    source=source,
                    url=full_url
                )

                return False

            return True

        except Exception as e:

            log.error(
                "governance_check_failed",
                source=source,
                error=str(e)
            )

            return False

    def _request(
        self,
        source: str,
        url: str,
        params: Dict[str, Any] = None
    ):

        RATE_MANAGER.wait_for_budget(
            source
        )

        response = self.session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        return response

    # ==================================================
    # FULL PIPELINE
    # ==================================================

    def run_full_sweep(self) -> Dict[str, Any]:

        start_time = time.time()

        log.info("sweep_start")

        results = {

            "successful": False,

            "duration_seconds": 0,

            "stages": {}

        }

        try:

            # STAGE 1: EVENT REGISTRATION

            erl_results = (
                self.run_event_registration_layer()
            )

            results["stages"]["ERL"] = erl_results

            # STAGE 2: FEATURE EXTRACTION

            fsal_count = (
                self.run_feature_analysis_layer()
            )

            results["stages"]["FSAL"] = fsal_count

            # STAGE 3: ANOMALY INFERENCE

            anomaly_count = (
                self.run_anomaly_inference_layer()
            )

            results["stages"]["APSL_INFERENCE"] = (
                anomaly_count
            )

            # STAGE 4: PATTERN SYNTHESIS

            group_count = (
                self.run_pattern_synthesis_layer()
            )

            results["stages"]["APSL_SYNTHESIS"] = (
                group_count
            )

            results["successful"] = True

        except Exception as e:

            log.exception(
                "sweep_failed",
                error=str(e)
            )

            results["error"] = str(e)

        finally:

            results["duration_seconds"] = round(
                time.time() - start_time,
                2
            )

            log.info(
                "sweep_complete",
                duration_seconds=results[
                    "duration_seconds"
                ],
                stages=results["stages"],
                successful=results["successful"]
            )

        return results

    # ==================================================
    # STAGE 1: ERL
    # ==================================================

    def run_event_registration_layer(
        self
    ) -> Dict[str, int]:

        log.info("erl_start")

        fetchers = {

            "github": self._register_github_events,

            "arxiv": self._register_arxiv_events,

            "huggingface": (
                self._register_huggingface_events
            ),

            "semantic_scholar": (
                self._register_semantic_scholar_events
            ),

            "openml": self._register_openml_events

        }

        results = {}

        for source, fetcher in fetchers.items():

            try:

                results[source] = fetcher()

            except Exception as e:

                results[source] = 0

                log.exception(
                    "erl_source_failed",
                    source=source,
                    error=str(e)
                )

        log.info(
            "erl_complete",
            results=results,
            successful_registrations=sum(
                results.values()
            )
        )

        return results

    # ==================================================
    # RAW EVENT REGISTRATION
    # ==================================================

    def _save_event(
        self,
        source: str,
        payload: Dict[str, Any],
        collection_method: str,
        retention_class: str
    ) -> bool:

        payload_string = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False
        )

        event_hash = self._generate_id(
            source + payload_string
        )

        event_id = event_hash

        try:

            inserted = self.db.insert_raw_event(
                event_id=event_id,
                source_system=source,
                raw_payload=payload,
                collection_method=collection_method,
                retention_class=retention_class,
                event_hash=event_hash
            )

            if inserted:

                self.db.record_provenance(
                    entity_type="raw_events",
                    entity_id=event_id,
                    process_description=(
                        f"Event ingestion from {source}"
                    ),
                    system_actor="ERL",
                    integrity_hash=event_hash
                )

                return True

            return False

        except Exception:

            log.exception(
                "event_registration_failed",
                source=source
            )

            return False

    # ==================================================
    # GITHUB
    # ==================================================

    def _register_github_events(self) -> int:

        source = "github"

        base_url = "https://api.github.com"

        url = (
            f"{base_url}/search/repositories"
        )

        params = {

            "q": API_QUERIES[source],

            "sort": "updated",

            "order": "desc",

            "per_page": MAX_RESULTS_PER_SOURCE

        }

        full_url = (
            f"{url}?{urlencode(params)}"
        )

        if not self._check_governance(
            base_url,
            full_url,
            source
        ):

            return 0

        response = self._request(
            source,
            url,
            params=params
        )

        data = response.json()

        registered = 0

        for item in data.get("items", []):

            payload = {

                "title": item.get(
                    "full_name",
                    ""
                ),

                "description": (
                    item.get("description") or ""
                ),

                "url": item.get(
                    "html_url",
                    ""
                ),

                "stars": item.get(
                    "stargazers_count",
                    0
                ),

                "forks": item.get(
                    "forks_count",
                    0
                ),

                "language": item.get("language"),

                "updated_at": item.get("updated_at")

            }

            if self._save_event(
                source,
                payload,
                "GitHub API",
                "repo_metadata"
            ):

                registered += 1

        log.info(
            "erl_source_complete",
            source=source,
            count=registered
        )

        return registered

    # ==================================================
    # ARXIV
    # ==================================================

    def _register_arxiv_events(self) -> int:

        source = "arxiv"

        base_url = "https://export.arxiv.org"

        url = f"{base_url}/api/query"

        params = {

            "search_query": API_QUERIES[source],

            "sortBy": "submittedDate",

            "sortOrder": "descending",

            "max_results": MAX_RESULTS_PER_SOURCE

        }

        full_url = (
            f"{url}?{urlencode(params)}"
        )

        if not self._check_governance(
            base_url,
            full_url,
            source
        ):

            return 0

        response = self._request(
            source,
            url,
            params=params
        )

        data = xmltodict.parse(
            response.content
        )

        entries = (
            data.get("feed", {}).get("entry", [])
        )

        if not isinstance(entries, list):

            entries = [entries]

        registered = 0

        for entry in entries:

            if not isinstance(entry, dict):
                continue

            authors = entry.get(
                "author",
                []
            )

            if not isinstance(authors, list):

                authors = [authors]

            author_names = []

            for author in authors:

                if isinstance(author, dict):

                    author_names.append(
                        author.get(
                            "name",
                            "Unknown"
                        )
                    )

            categories = entry.get(
                "category",
                []
            )

            if not isinstance(categories, list):

                categories = [categories]

            category_names = [

                category.get(
                    "@term",
                    "Unknown"
                )

                for category in categories

                if isinstance(category, dict)

            ]

            payload = {

                "title": (

                    entry.get("title", "")
                    .replace("\n", " ")
                    .strip()

                ),

                "description": (

                    entry.get("summary", "")
                    .replace("\n", " ")
                    .strip()

                ),

                "url": entry.get("id", ""),

                "authors": author_names,

                "categories": category_names,

                "published": entry.get("published"),

                "updated": entry.get("updated")

            }

            if self._save_event(
                source,
                payload,
                "arXiv API",
                "scientific_paper"
            ):

                registered += 1

        log.info(
            "erl_source_complete",
            source=source,
            count=registered
        )

        return registered

    # ==================================================
    # HUGGING FACE
    # ==================================================

    def _register_huggingface_events(self) -> int:

        log.info(
            "source_not_implemented",
            source="huggingface"
        )

        return 0

    # ==================================================
    # SEMANTIC SCHOLAR
    # ==================================================

    def _register_semantic_scholar_events(self) -> int:

        log.info(
            "source_not_implemented",
            source="semantic_scholar"
        )

        return 0

    # ==================================================
    # OPENML
    # ==================================================

    def _register_openml_events(self) -> int:

        log.info(
            "source_not_implemented",
            source="openml"
        )

        return 0

    # ==================================================
    # STAGE 2: FSAL
    # ==================================================

    def run_feature_analysis_layer(self) -> int:

        log.info("fsal_start")

        events = (
            self.db.get_raw_events_for_feature_extraction()
        )

        processed = 0

        processed_event_ids = []

        for event in events:

            event_id = event["event_id"]

            try:

                metadata = dict(
                    event["raw_payload"]
                )

                metadata["source"] = (
                    event["source_system"]
                )

                result = classify_and_score(
                    metadata
                )

                feature_id = str(uuid.uuid4())

                feature_tuple = (

                    feature_id,

                    event_id,

                    "ontological_vector",

                    result["score"],

                    json.dumps(
                        result["features"],
                        sort_keys=True
                    ),

                    MODEL_VERSION,

                    self._timestamp()

                )

                inserted = (
                    self.db.insert_extracted_features(
                        [feature_tuple]
                    )
                )

                if inserted:

                    self.db.record_provenance(
                        entity_type="extracted_features",
                        entity_id=feature_id,
                        process_description=(
                            f"Feature extraction via "
                            f"{MODEL_VERSION}"
                        ),
                        system_actor="FSAL",
                        integrity_hash=event.get("hash")
                    )

                    processed += 1

                    processed_event_ids.append(
                        event_id
                    )

                else:

                    log.warning(
                        "feature_insert_skipped",
                        event_id=event_id
                    )

            except Exception:

                log.exception(
                    "fsal_event_failed",
                    event_id=event_id
                )

        # Mark only successfully extracted events.

        if processed_event_ids:

            self.db.mark_events_processed(
                processed_event_ids
            )

        log.info(
            "fsal_complete",
            features=processed,
            processed=len(events),
            failed=len(events) - processed
        )

        return processed

    # ==================================================
    # STAGE 3: APSL INFERENCE
    # ==================================================

    def run_anomaly_inference_layer(self) -> int:

        log.info("inference_start")

        count = (
            self.anomaly_engine.infer_anomalies()
        )

        log.info(
            "inference_complete",
            anomalies=count
        )

        return count

    # ==================================================
    # STAGE 4: APSL SYNTHESIS
    # ==================================================

    def run_pattern_synthesis_layer(self) -> int:

        log.info("synthesis_start")

        count = (
            self.anomaly_engine.group_anomalies()
        )

        log.info(
            "synthesis_complete",
            groups_created=count
        )

        return count

    # ==================================================
    # CLEANUP
    # ==================================================

    def close(self):

        if self.session:

            self.session.close()

            log.info(
                "tracker_session_closed"
            )
