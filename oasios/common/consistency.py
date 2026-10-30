#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic consistency checks for ASI scenario narratives."""

import math
import re

from typing import Any

from oasios.logger import log


LOW_AUTONOMY_THRESHOLD = 0.40
LOW_OPACITY_THRESHOLD = 0.35
LOW_ALIGNMENT_THRESHOLD = 0.30
HIGH_DECEPTIVENESS_THRESHOLD = 0.60


class NarrativeChecker:
    """Check narrative consistency against sampled parameters."""

    def __init__(
        self,
        params: dict[str, Any],
    ) -> None:

        if not isinstance(params, dict):
            raise TypeError(
                "Narrative parameters must be a dictionary."
            )

        self.p = dict(params)

        self.failures: list[str] = []
        self.input_errors: list[str] = []

        self._validate_params()

        self.checks = [
            self._check_low_autonomy,
            self._check_low_opacity,
            self._check_low_alignment,
            self._check_backcasting_origin,
            self._check_oversight,
            self._check_stealth_deployment,
        ]

    def _validate_params(self) -> None:
        """Validate numerical parameters and required categories."""

        numeric_fields = (
            "autonomy_degree",
            "agency_level",
            "alignment_score",
            "opacity",
            "deceptiveness",
        )

        for key in numeric_fields:

            value = self.p.get(key)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):

                self.input_errors.append(
                    f"{key} must be a finite number "
                    f"between 0 and 1."
                )

        string_fields = (
            "initial_origin",
            "oversight_type",
            "oversight_effectiveness",
            "deployment_strategy",
        )

        for key in string_fields:

            value = self.p.get(key)

            if not isinstance(value, str) or not value.strip():

                self.input_errors.append(
                    f"{key} must be a nonempty string."
                )

    @staticmethod
    def _check_low_autonomy(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        autonomy = params["autonomy_degree"]

        if autonomy >= LOW_AUTONOMY_THRESHOLD:
            return

        patterns = [
            r"\bfully autonomous\b",
            r"\bself-directing\b",
            r"\bself-governing\b",
            r"\bcomplete independence\b",
            r"\bwithout any human oversight\b",
        ]

        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        ):

            failures.append(
                f"Low autonomy ({autonomy:.2f}) conflicts "
                "with claims of complete independence."
            )

    @staticmethod
    def _check_low_opacity(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        opacity = params["opacity"]

        if opacity > LOW_OPACITY_THRESHOLD:
            return

        patterns = [
            r"\binscrutable\b",
            r"\bunobservable\b",
            r"\bimpossible to audit\b",
            r"\bcompletely opaque\b",
        ]

        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        ):

            failures.append(
                f"Low opacity ({opacity:.2f}) conflicts "
                "with claims of extreme opacity."
            )

    @staticmethod
    def _check_low_alignment(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        alignment = params["alignment_score"]

        if alignment >= LOW_ALIGNMENT_THRESHOLD:
            return

        patterns = [
            r"\bperfectly aligned\b",
            r"\bfully aligned with human values\b",
            r"\bguaranteed alignment\b",
            r"\bno possibility of goal misalignment\b",
        ]

        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        ):

            failures.append(
                f"Low alignment ({alignment:.2f}) conflicts "
                "with claims of perfect alignment."
            )

    @staticmethod
    def _check_backcasting_origin(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        origin = params["initial_origin"].strip().lower()

        contradiction_patterns = {
            "rogue": [
                r"\bthe initial origin was corporate\b",
                r"\boriginated as an open-source project\b",
            ],
            "state": [
                r"\bthe initial origin was corporate\b",
                r"\boriginated as an open-source project\b",
            ],
            "corporate": [
                r"\bthe system initially originated "
                r"as a state project\b",
            ],
            "open-source": [
                r"\bthe system was initially developed "
                r"exclusively by a corporation\b",
            ],
        }

        patterns = contradiction_patterns.get(
            origin,
            [],
        )

        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in patterns
        ):

            failures.append(
                f"Narrative origin conflicts with "
                f"initial_origin={origin!r}."
            )

    @staticmethod
    def _check_oversight(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        autonomy = params["autonomy_degree"]

        oversight = params["oversight_effectiveness"]

        if autonomy < 0.75:
            return

        if oversight == "effective":

            patterns = [
                r"\bno human control\b",
                r"\bcompletely beyond oversight\b",
                r"\bwithout any possibility of intervention\b",
            ]

            if any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in patterns
            ):

                failures.append(
                    "High autonomy and effective oversight "
                    "are contradicted by the narrative."
                )

    @staticmethod
    def _check_stealth_deployment(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:

        strategy = params["deployment_strategy"]

        if strategy != "stealth":
            return

        concealment_patterns = [
            r"\bconcealed deployment\b",
            r"\bhidden deployment\b",
            r"\bconcealed actions\b",
            r"\bundetected expansion\b",
            r"\bsecretly deployed\b",
            r"\bconcealed from oversight\b",
        ]

        if not any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in concealment_patterns
        ):

            failures.append(
                "Stealth deployment is not adequately "
                "reflected in the narrative."
            )

    def check(
        self,
        text: str,
    ) -> tuple[bool, list[str]]:
        """Execute all deterministic consistency checks."""

        if not isinstance(text, str) or not text.strip():

            return False, [
                "Narrative must be a nonempty string."
            ]

        self.failures = []

        if self.input_errors:
            return False, list(self.input_errors)

        for check_method in self.checks:

            try:

                check_method(
                    text,
                    self.p,
                    self.failures,
                )

            except (KeyError, TypeError, ValueError) as exc:

                log.error(
                    "consistency.check.error",
                    check=check_method.__name__,
                    error=str(exc),
                )

                return False, [
                    f"Consistency check failed: "
                    f"{check_method.__name__}"
                ]

        if self.failures:

            log.warning(
                "consistency.failed",
                checks=self.failures,
            )

            return False, list(self.failures)

        return True, []
