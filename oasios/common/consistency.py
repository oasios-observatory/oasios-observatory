#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/consistency.py

"""Consistency checks for generated scenario narratives."""

import math
import re

from typing import Any

from oasios.logger import log


LOW_AUTONOMY_THRESHOLD = 0.40
LOW_OPACITY_THRESHOLD = 0.35
LOW_ALIGNMENT_THRESHOLD = 0.30


class NarrativeChecker:
    """Check narrative consistency against sampled parameters."""

    def __init__(self, params: dict[str, Any]) -> None:
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
        ]

    def _validate_params(self) -> None:
        """Validate parameters required by consistency checks."""

        for key in (
            "autonomy_degree",
            "agency_level",
            "alignment_score",
            "opacity",
        ):
            value = self.p.get(key)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                self.input_errors.append(
                    f"{key} must be a finite number between "
                    f"0 and 1; got {value!r}"
                )

        origin = self.p.get("initial_origin")

        if not isinstance(origin, str) or not origin.strip():
            self.input_errors.append(
                "initial_origin must be a nonempty string."
            )

    @staticmethod
    def _check_low_autonomy(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:
        """Detect strong claims of independence at low autonomy."""

        autonomy = params["autonomy_degree"]

        if autonomy >= LOW_AUTONOMY_THRESHOLD:
            return

        patterns = [
            r"\bfully autonomous\b",
            r"\bself-directing\b",
            r"\bself-governing\b",
            r"\boperates with complete independence\b",
            r"\bwithout any human oversight\b",
        ]

        found = [
            pattern
            for pattern in patterns
            if re.search(pattern, text, re.IGNORECASE)
        ]

        if found:
            failures.append(
                "Autonomy contradiction: low autonomy score "
                f"{autonomy:.2f} conflicts with narrative claims."
            )

    @staticmethod
    def _check_low_opacity(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:
        """Detect extreme opacity claims when opacity is low."""

        opacity = params["opacity"]

        if opacity > LOW_OPACITY_THRESHOLD:
            return

        patterns = [
            r"\binscrutable\b",
            r"\bunobservable\b",
            r"\bimpossible to audit\b",
            r"\bhidden internal state\b",
            r"\bcompletely opaque\b",
        ]

        found = [
            pattern
            for pattern in patterns
            if re.search(pattern, text, re.IGNORECASE)
        ]

        if found:
            failures.append(
                f"Opacity contradiction: score {opacity:.2f} "
                "conflicts with strong opacity claims."
            )

    @staticmethod
    def _check_low_alignment(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:
        """
        Detect potentially contradictory alignment claims.

        This is a heuristic and cannot establish semantic
        consistency on its own.
        """

        alignment = params["alignment_score"]

        if alignment >= LOW_ALIGNMENT_THRESHOLD:
            return

        positive_claims = [
            r"\bperfectly aligned\b",
            r"\bfully aligned with human values\b",
            r"\bguaranteed alignment\b",
            r"\bno possibility of goal misalignment\b",
        ]

        found = [
            pattern
            for pattern in positive_claims
            if re.search(pattern, text, re.IGNORECASE)
        ]

        if found:
            failures.append(
                f"Alignment contradiction: low alignment score "
                f"{alignment:.2f} conflicts with strong alignment claims."
            )

    @staticmethod
    def _check_backcasting_origin(
        text: str,
        params: dict[str, Any],
        failures: list[str],
    ) -> None:
        """Detect explicit contradictions about scenario origin."""

        origin = params["initial_origin"].strip().lower()

        if origin not in {"rogue", "state"}:
            return

        patterns = [
            r"\bthe initial origin was corporate\b",
            r"\bthe system originated as an open-source project\b",
            r"\binitially developed as a public open-source model\b",
        ]

        found = [
            pattern
            for pattern in patterns
            if re.search(pattern, text, re.IGNORECASE)
        ]

        if found:
            failures.append(
                f"Backcasting contradiction for origin {origin!r}."
            )

    def check(self, text: str) -> tuple[bool, list[str]]:
        """Run consistency checks against the narrative."""

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
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

                return False, [
                    f"Consistency check "
                    f"{check_method.__name__} failed."
                ]

        if self.failures:
            log.warning(
                "consistency.failed",
                checks=self.failures,
            )

            return False, list(self.failures)

        return True, []
