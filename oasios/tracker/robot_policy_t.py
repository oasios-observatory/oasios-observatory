# oasios/common/robot_policy_t.py

import requests
import structlog

from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from typing import Dict

log = structlog.get_logger()

_ROBOT_PARSERS: Dict[str, RobotFileParser] = {}


def get_robot_parser(base_url: str) -> RobotFileParser:

    parsed = urlparse(base_url)

    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin in _ROBOT_PARSERS:
        return _ROBOT_PARSERS[origin]

    parser = RobotFileParser()

    robots_url = f"{origin}/robots.txt"

    parser.set_url(robots_url)

    try:

        response = requests.get(
            robots_url,
            headers={"User-Agent": "OASIOS-Research/0.4"},
            timeout=8,
        )

        if response.status_code == 404:

            log.warning(
                "robots_txt_not_found",
                url=robots_url,
            )

            # No robots.txt: standard permissive convention.
            parser.parse([])

        elif response.status_code == 200:

            parser.parse(response.text.splitlines())

            log.info(
                "robots_txt_loaded",
                origin=origin,
            )

        else:

            # Fail closed when permission cannot be established.
            parser.parse(["User-agent: *", "Disallow: /"])

            log.warning(
                "robots_txt_unavailable",
                status=response.status_code,
                origin=origin,
            )

    except requests.RequestException as exc:

        parser.parse(["User-agent: *", "Disallow: /"])

        log.error(
            "robots_txt_request_failed",
            origin=origin,
            error=str(exc),
        )

    _ROBOT_PARSERS[origin] = parser

    return parser


def can_fetch(
    base_url: str,
    url: str,
    user_agent: str,
) -> bool:

    parser = get_robot_parser(base_url)

    permitted = parser.can_fetch(user_agent, url)

    if not permitted:

        log.warning(
            "robots_txt_blocked",
            url=url,
            user_agent=user_agent,
        )

    return permitted
