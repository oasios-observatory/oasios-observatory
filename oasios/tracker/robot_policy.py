# oasios/common/robot_policy.py (UPDATED)

import requests
import structlog
from urllib.robotparser import RobotFileParser
from typing import Dict
# Removed urlparse and structlog import for brevity if they are already in scope
# import structlog
from requests.exceptions import RequestException, HTTPError

log = structlog.get_logger()

# Cache to store parsed robots.txt rules for efficiency
_ROBOT_PARSERS: Dict[str, RobotFileParser] = {}


def get_robot_parser(base_url: str) -> RobotFileParser:
    """Fetches or retrieves a cached RobotFileParser for a given base URL."""
    if base_url not in _ROBOT_PARSERS:
        parser = RobotFileParser()
        parser.set_url(f"{base_url}/robots.txt")

        try:
            # Use requests to fetch the robots.txt content, respecting timeouts
            robot_response = requests.get(parser.url, timeout=5)

            # **1. Check for 4XX errors without using raise_for_status() initially**
            if robot_response.status_code == 404:
                # Standard convention: 404 means no robots.txt, so everything is allowed.
                # We log this as a warning but proceed with a default permissive parser.
                log.warning("robots_txt_not_found_permissive", url=parser.url)
                # The parser remains empty/permissive by default, so no further action is needed here.
            elif robot_response.status_code >= 400:
                # For other 4XX errors (e.g., 403 Forbidden), some crawlers assume Disallow-All,
                # but the simplest/safest way is to log the error and let the empty parser be permissive.
                log.warning("robots_txt_fetch_failed_http", url=parser.url, status_code=robot_response.status_code)
            elif robot_response.status_code == 200:
                # 2. Success: Read the content into the parser
                parser.parse(robot_response.text.splitlines())
                log.info("robots_txt_fetched", base=base_url)

        except RequestException as e:
            # 3. Handle connection errors (timeouts, DNS failures, etc.)
            log.error("robots_txt_connection_error", url=parser.url, error=str(e))
            # On connection failure, the parser remains permissive.
        except Exception as e:
            # Handle parsing errors
            log.error("robots_txt_parse_error", url=parser.url, error=str(e))

        _ROBOT_PARSERS[base_url] = parser

    return _ROBOT_PARSERS[base_url]


def can_fetch(base_url: str, url: str, user_agent: str) -> bool:
    """Checks if the user_agent is permitted to fetch the specific URL."""
    parser = get_robot_parser(base_url)

    # Crucial: The parser checks the URL path against the rules for the User-Agent
    return parser.can_fetch(user_agent, url)