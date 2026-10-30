# oasios/tracker/rate_limiter_t.py

import time
import threading
import structlog

from collections import defaultdict

log = structlog.get_logger()


DEFAULT_RATE_LIMITS = {
    "github": 30,
    "arxiv": 10,
    "huggingface": 20,
    "semantic_scholar": 10,
    "openml": 10,
}


class RequestBudgetManager:

    def __init__(self, limits=None):

        self.limits = dict(
            limits or DEFAULT_RATE_LIMITS
        )

        self.last_request_time = defaultdict(float)

        self.lock = threading.Lock()

    def wait_for_budget(self, source: str):

        limit = self.limits.get(source, 5)

        if limit <= 0:
            raise ValueError("Rate limit must be positive")

        interval = 60.0 / limit

        with self.lock:

            now = time.monotonic()

            previous = self.last_request_time[source]

            delay = max(
                0.0,
                interval - (now - previous),
            )

            if delay:

                log.info(
                    "rate_limit_delay",
                    source=source,
                    delay_seconds=round(delay, 2),
                )

                time.sleep(delay)

            self.last_request_time[source] = time.monotonic()
