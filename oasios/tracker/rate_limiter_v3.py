# oasios/tracker/rate_limiter_v3.py
# the core compliance logic

import time
from collections import defaultdict
import structlog
import os

log = structlog.get_logger()

# Define default rate limits (requests per minute)
# These are conservative to ensure politeness and stay well within API limits.
DEFAULT_RATE_LIMITS = {
    "github": 60,  # High limit as GitHub API is generous (usually 5000/hr)
    "arxiv": 60,  # High limit as arXiv is academic
    "huggingface": 30,  # Conservative for searching the hub
    "semantic_scholar": 20,  # Semantic Scholar unauthenticated is 100/sec, but we stay very polite
    "openml": 20  # Conservative limit
}


class RequestBudgetManager:
    """
    Manages the request budget for all external API calls across all sources.
    Enforces a polite global rate limit to prevent IP bans and 429 errors.
    """

    def __init__(self, limits=None):
        self.limits = limits if limits is not None else DEFAULT_RATE_LIMITS
        # Stores the last time a request was made for a source (in seconds since epoch)
        self.last_request_time = defaultdict(float)

    def _calculate_delay(self, source: str) -> float:
        """Calculates the minimum required delay before the next request."""
        limit_per_min = self.limits.get(source, 10)  # Default to 10 RPM if source not found

        # Calculate the minimum time required between requests (in seconds)
        min_interval = 60.0 / limit_per_min

        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time[source]

        # If time since last request is less than the minimum interval, calculate the delay
        if time_since_last_request < min_interval:
            delay = min_interval - time_since_last_request
            return delay
        return 0.0

    def wait_for_budget(self, source: str):
        """
        Blocks execution until the rate limit budget allows a new request for the source.
        """
        delay = self._calculate_delay(source)

        if delay > 0.01:  # Only log/wait if delay is meaningful
            log.info("rate_limit_delay", source=source, delay_sec=f"{delay:.2f}")
            time.sleep(delay)

        # Update the last request time right before making the actual request
        self.last_request_time[source] = time.time()


# --- Example Usage (Optional: for testing the utility) ---
if __name__ == '__main__':
    # Simple setup for demonstration
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        logger_factory=structlog.PrintLoggerFactory()
    )

    manager = RequestBudgetManager()

    print("Testing GitHub (60 RPM, 1 req/sec):")
    for _ in range(3):
        start = time.time()
        manager.wait_for_budget("github")
        end = time.time()
        log.info("request_made", source="github", elapsed=f"{end - start:.3f}")

    print("\nTesting Semantic Scholar (20 RPM, 3 sec/req):")
    for _ in range(3):
        start = time.time()
        manager.wait_for_budget("semantic_scholar")
        end = time.time()
        log.info("request_made", source="semantic_scholar", elapsed=f"{end - start:.3f}")