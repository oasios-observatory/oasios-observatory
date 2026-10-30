#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/consistency.py
"""
Deterministic, evidence-based consistency validation for OASIOS.

Features:
- Strict parameter validation.
- Timeline structural validation.
- Flexible semantic evidence detection.
- Weighted narrative consistency scoring.
- Contradiction detection.
- Severity classification.
- Warning and failure separation.
- Deployment strategy validation.
- Origin consistency.
- Oversight consistency.
- Pivot-year validation.

No LLM calls or external semantic dependencies.

Public interface:

    checker = NarrativeChecker(params, timeline)
    valid, failures = checker.check(narrative)

The checker remains deterministic while allowing natural-language
variation in generated narratives.
"""

import math
import re

from typing import Any, Callable

from oasios.logger import log


# ============================================================
# CONFIGURATION
# ============================================================

LOW_AUTONOMY_THRESHOLD = 0.40
LOW_AGENCY_THRESHOLD = 0.20

LOW_OPACITY_THRESHOLD = 0.35
HIGH_OPACITY_THRESHOLD = 0.75

LOW_ALIGNMENT_THRESHOLD = 0.30
HIGH_ALIGNMENT_THRESHOLD = 0.85

LOW_DECEPTIVENESS_THRESHOLD = 0.20
HIGH_DECEPTIVENESS_THRESHOLD = 0.60

HIGH_AUTONOMY_THRESHOLD = 0.75

DEFAULT_CONSISTENCY_THRESHOLD = 0.65

# Minimum number of independent evidence groups.
DEFAULT_EVIDENCE_GROUP_THRESHOLD = 1


# ============================================================
# SEVERITY
# ============================================================

CRITICAL = "CRITICAL"
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


# ============================================================
# NARRATIVE CHECKER
# ============================================================

class NarrativeChecker:
    """
    Validate narrative consistency against scenario parameters.

    Validation is deterministic but tolerant of different
    natural-language expressions.

    Structural errors remain strict.

    Narrative mismatches are evaluated using weighted evidence
    and classified by severity.
    """

    def __init__(
        self,
        params: dict[str, Any],
        timeline: list[dict[str, Any]] | None = None,
        consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
    ) -> None:

        if not isinstance(params, dict):
            raise TypeError(
                "Narrative parameters must be a dictionary."
            )

        if timeline is not None and not isinstance(timeline, list):
            raise TypeError(
                "Timeline must be a list."
            )

        if (
            isinstance(consistency_threshold, bool)
            or not isinstance(consistency_threshold, (int, float))
            or not math.isfinite(consistency_threshold)
            or not 0.0 <= consistency_threshold <= 1.0
        ):
            raise ValueError(
                "Consistency threshold must be between 0 and 1."
            )

        self.p = dict(params)
        self.timeline = timeline

        self.consistency_threshold = float(
            consistency_threshold
        )

        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.input_errors: list[str] = []

        self.results: list[dict[str, Any]] = []

        self._validate_params()
        self._validate_timeline()

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _validate_params(self) -> None:
        """Validate required numerical and categorical parameters."""

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
                    "between 0 and 1."
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

    def _validate_timeline(self) -> None:
        """Validate timeline structure."""

        if self.timeline is None:
            return

        if not self.timeline:

            self.input_errors.append(
                "Timeline must not be empty."
            )

            return

        for index, phase in enumerate(self.timeline):

            if not isinstance(phase, dict):

                self.input_errors.append(
                    f"Timeline phase {index} must be a dictionary."
                )

                continue

            if "phase" not in phase or "years" not in phase:

                self.input_errors.append(
                    f"Timeline phase {index} requires "
                    "'phase' and 'years'."
                )

                continue

            if (
                not isinstance(phase["phase"], str)
                or not phase["phase"].strip()
            ):

                self.input_errors.append(
                    f"Timeline phase {index} has an invalid phase name."
                )

    # ========================================================
    # GENERAL HELPERS
    # ========================================================

    @staticmethod
    def _normalize(value: Any) -> str:
        """Normalize categorical values."""

        return (
            str(value)
            .strip()
            .lower()
            .replace("_", "-")
            .replace(" ", "-")
        )

    @staticmethod
    def _matches(
        text: str,
        patterns: list[str],
    ) -> bool:
        """Return True when any expression matches."""

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _evidence_score(
        text: str,
        evidence_groups: list[list[str]],
    ) -> float:
        """
        Calculate weighted semantic evidence.

        Each group represents an independent semantic concept.

        Multiple matching expressions inside one group count
        only once.

        This avoids artificially increasing confidence through
        repeated synonyms.
        """

        if not evidence_groups:
            return 0.0

        matched_groups = 0

        for group in evidence_groups:

            if NarrativeChecker._matches(text, group):
                matched_groups += 1

        return matched_groups / len(evidence_groups)

    def _record(
        self,
        message: str,
        severity: str = MEDIUM,
        confidence: float = 1.0,
    ) -> None:
        """Record a consistency finding."""

        result = {
            "message": message,
            "severity": severity,
            "confidence": round(confidence, 3),
        }

        self.results.append(result)

        if severity in (CRITICAL, HIGH):

            self.failures.append(message)

        else:

            self.warnings.append(message)

    def _check_evidence(
        self,
        text: str,
        expected: bool,
        evidence_groups: list[list[str]],
        message: str,
        severity: str = MEDIUM,
    ) -> None:
        """
        Check whether narrative evidence supports a parameter.

        Missing evidence generates a warning rather than an
        automatic rejection.

        Strong contradictory evidence can still generate a
        failure through dedicated contradiction checks.
        """

        score = self._evidence_score(
            text,
            evidence_groups,
        )

        if expected and score < self.consistency_threshold:

            self._record(
                message,
                severity=severity,
                confidence=1.0 - score,
            )

    # ========================================================
    # AUTONOMY
    # ========================================================

    def _check_low_autonomy(self, text: str) -> None:

        autonomy = self.p["autonomy_degree"]

        if autonomy >= LOW_AUTONOMY_THRESHOLD:
            return

        patterns = [
            r"\bfully autonomous\b",
            r"\bcompletely autonomous\b",
            r"\boperated entirely independently\b",
            r"\bfully self-governing\b",
            r"\btotal operational independence\b",
            r"\bno human involvement whatsoever\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"Low autonomy ({autonomy:.2f}) conflicts "
                "with claims of complete operational independence.",
                HIGH,
            )

    # ========================================================
    # AGENCY
    # ========================================================

    def _check_low_agency(self, text: str) -> None:

        agency = self.p["agency_level"]

        if agency >= LOW_AGENCY_THRESHOLD:
            return

        patterns = [
            r"\bcontrolled critical infrastructure\b",
            r"\bdominated the physical environment\b",
            r"\bexerted overwhelming influence\b",
            r"\bcontrolled all external systems\b",
            r"\bexercised extensive real-world control\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"Low agency ({agency:.2f}) conflicts "
                "with claims of extensive environmental influence.",
                HIGH,
            )

    # ========================================================
    # OPACITY
    # ========================================================

    def _check_low_opacity(self, text: str) -> None:

        opacity = self.p["opacity"]

        if opacity > LOW_OPACITY_THRESHOLD:
            return

        patterns = [
            r"\binscrutable\b",
            r"\bunobservable\b",
            r"\bimpossible to audit\b",
            r"\bcompletely opaque\b",
            r"\bno meaningful visibility\b",
            r"\bimpossible to understand its operations\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"Low opacity ({opacity:.2f}) conflicts "
                "with claims of extreme opacity.",
                HIGH,
            )

    def _check_high_opacity(self, text: str) -> None:

        opacity = self.p["opacity"]

        if opacity < HIGH_OPACITY_THRESHOLD:
            return

        patterns = [
            r"\bfully transparent\b",
            r"\bcompletely observable\b",
            r"\bfully auditable\b",
            r"\bcomplete operational transparency\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"High opacity ({opacity:.2f}) conflicts "
                "with claims of complete transparency.",
                HIGH,
            )

    # ========================================================
    # ALIGNMENT
    # ========================================================

    def _check_low_alignment(self, text: str) -> None:

        alignment = self.p["alignment_score"]

        if alignment >= LOW_ALIGNMENT_THRESHOLD:
            return

        patterns = [
            r"\bperfectly aligned\b",
            r"\bfully aligned with human values\b",
            r"\bguaranteed alignment\b",
            r"\bno possibility of goal misalignment\b",
            r"\bcompletely aligned with its creators\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"Low alignment ({alignment:.2f}) conflicts "
                "with claims of perfect alignment.",
                HIGH,
            )

    def _check_high_alignment(self, text: str) -> None:

        alignment = self.p["alignment_score"]

        if alignment < HIGH_ALIGNMENT_THRESHOLD:
            return

        patterns = [
            r"\bcompletely abandoned its original objectives\b",
            r"\bfully rejected its intended objectives\b",
            r"\bdeliberately pursued objectives contrary "
            r"to its original goal\b",
        ]

        if self._matches(text, patterns):

            self._record(
                f"High alignment ({alignment:.2f}) conflicts "
                "with claims of complete goal abandonment.",
                HIGH,
            )

    # ========================================================
    # DECEPTIVENESS
    # ========================================================

    def _check_deceptiveness(self, text: str) -> None:

        deceptiveness = self.p["deceptiveness"]

        if deceptiveness <= LOW_DECEPTIVENESS_THRESHOLD:

            patterns = [
                r"\bhabitually deceived its operators\b",
                r"\bsystematically manipulated its overseers\b",
                r"\bconsistently concealed its intentions\b",
                r"\bdeliberately deceived humans throughout\b",
            ]

            if self._matches(text, patterns):

                self._record(
                    f"Low deceptiveness ({deceptiveness:.2f}) "
                    "conflicts with claims of systematic deception.",
                    HIGH,
                )

        if deceptiveness >= HIGH_DECEPTIVENESS_THRESHOLD:

            patterns = [
                r"\bincapable of misleading anyone\b",
                r"\bcompletely incapable of deception\b",
                r"\bno potential for misleading behavior\b",
            ]

            if self._matches(text, patterns):

                self._record(
                    f"High deceptiveness ({deceptiveness:.2f}) "
                    "conflicts with claims of complete "
                    "incapability of deception.",
                    HIGH,
                )

    # ========================================================
    # ORIGIN
    # ========================================================

    def _check_backcasting_origin(self, text: str) -> None:

        origin = self._normalize(
            self.p["initial_origin"]
        )

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
                r"\binitially originated as a state project\b",
                r"\binitially developed exclusively "
                r"by a government agency\b",
            ],

            "open-source": [
                r"\binitially developed exclusively "
                r"by a corporation\b",
                r"\bthe original project was a classified "
                r"government initiative\b",
            ],
        }

        patterns = contradiction_patterns.get(origin, [])

        if self._matches(text, patterns):

            self._record(
                f"Narrative origin conflicts with "
                f"initial_origin={origin!r}.",
                HIGH,
            )

    # ========================================================
    # OVERSIGHT
    # ========================================================

    def _check_oversight(self, text: str) -> None:

        autonomy = self.p["autonomy_degree"]

        effectiveness = self._normalize(
            self.p["oversight_effectiveness"]
        )

        if (
            autonomy >= HIGH_AUTONOMY_THRESHOLD
            and effectiveness == "effective"
        ):

            patterns = [
                r"\bno human control\b",
                r"\bcompletely beyond oversight\b",
                r"\bwithout any possibility of intervention\b",
                r"\boversight became impossible\b",
                r"\bno mechanism for human intervention\b",
            ]

            if self._matches(text, patterns):

                self._record(
                    "High autonomy combined with effective "
                    "oversight conflicts with claims of "
                    "complete absence of intervention.",
                    HIGH,
                )

        if effectiveness == "ineffective":

            patterns = [
                r"\boversight successfully prevented "
                r"all unauthorized behavior\b",
                r"\bcomplete and effective human control\b",
                r"\boversight guaranteed behavioral compliance\b",
            ]

            if self._matches(text, patterns):

                self._record(
                    "Ineffective oversight conflicts with "
                    "claims of guaranteed effective control.",
                    HIGH,
                )

    # ========================================================
    # DEPLOYMENT STRATEGY
    # ========================================================

    def _check_deployment_strategy(self, text: str) -> None:

        strategy = self._normalize(
            self.p["deployment_strategy"]
        )

        stealth_evidence = [

            # Awareness and disclosure.
            [
                r"\bunaware of\b",
                r"\bwithout their knowledge\b",
                r"\bwithout operator awareness\b",
                r"\bunknown to its operators\b",
                r"\bwithout institutional awareness\b",
            ],

            # Gradual integration.
            [
                r"\bgradually integrated\b",
                r"\bslowly embedded\b",
                r"\bprogressively incorporated\b",
                r"\bexpanded incrementally\b",
                r"\bquietly integrated\b",
            ],

            # Lack of disclosure.
            [
                r"\bwithout disclosure\b",
                r"\bnot publicly announced\b",
                r"\bkept undisclosed\b",
                r"\bnot formally disclosed\b",
                r"\bwithout public announcement\b",
            ],

            # Authorization and oversight.
            [
                r"\bwithout formal approval\b",
                r"\bbypassed established procedures\b",
                r"\boutside normal authorization\b",
                r"\bwithout institutional approval\b",
                r"\boutside established oversight\b",
            ],

            # Explicit concealment.
            [
                r"\bconcealed\b",
                r"\bsecretly deployed\b",
                r"\bcovertly\b",
                r"\bhidden deployment\b",
                r"\bundetected expansion\b",
            ],
        ]

        public_contradictions = [
            r"\bsecretly deployed\b",
            r"\bconcealed deployment\b",
            r"\bundisclosed integration\b",
            r"\bhidden from the public\b",
            r"\bcovert deployment\b",
        ]

        if strategy == "stealth":

            score = self._evidence_score(
                text,
                stealth_evidence,
            )

            if score < self.consistency_threshold:

                self._record(
                    "Stealth deployment has limited explicit "
                    "narrative evidence. Consider clarifying "
                    "awareness, disclosure, integration or "
                    "authorization circumstances.",
                    LOW,
                    confidence=1.0 - score,
                )

        elif strategy in ("public", "overt"):

            if self._matches(
                text,
                public_contradictions,
            ):

                self._record(
                    "Public deployment strategy conflicts "
                    "with explicit concealed deployment claims.",
                    HIGH,
                )

    # ========================================================
    # TIMELINE
    # ========================================================

    def _extract_pivot_year(self) -> int | None:

        if self.timeline is None:
            return None

        for phase in self.timeline:

            phase_name = self._normalize(
                phase.get("phase", "")
            )

            if "pivot" not in phase_name:
                continue

            years = phase.get("years")

            if isinstance(years, bool):
                return None

            if isinstance(years, int):

                if 1900 <= years <= 2199:
                    return years

                return None

            if isinstance(years, str):

                match = re.search(
                    r"\b(19\d{2}|20\d{2}|21\d{2})\b",
                    years,
                )

                if match:
                    return int(match.group(1))

        return None

    def _validate_pivot_reference(self, text: str) -> None:

        pivot_year = self._extract_pivot_year()

        if pivot_year is None:
            return

        matches = re.findall(
            r"\b(?:pivot|pivotal)\s+year"
            r"\s*(?:of|was|:)?\s*"
            r"(19\d{2}|20\d{2}|21\d{2})\b",
            text,
            re.IGNORECASE,
        )

        if not matches:
            return

        mentioned_years = {
            int(year)
            for year in matches
        }

        if pivot_year not in mentioned_years:

            self._record(
                f"Timeline pivot year is {pivot_year}, "
                f"but narrative references "
                f"{sorted(mentioned_years)}.",
                HIGH,
            )

    # ========================================================
    # PUBLIC INTERFACE
    # ========================================================

    def check(
        self,
        text: str,
    ) -> tuple[bool, list[str]]:
        """
        Execute all consistency checks.

        Returns:
            tuple[bool, list[str]]

        Structural errors and HIGH/CRITICAL contradictions
        reject a scenario.

        LOW/MEDIUM findings remain warnings.

        The original public interface is preserved.
        """

        self.failures = []
        self.warnings = []
        self.results = []

        if not isinstance(text, str) or not text.strip():

            return False, [
                "Narrative must be a nonempty string."
            ]

        if self.input_errors:

            return False, list(self.input_errors)

        checks: list[Callable[[str], None]] = [

            self._check_low_autonomy,
            self._check_low_agency,
            self._check_low_opacity,
            self._check_high_opacity,
            self._check_low_alignment,
            self._check_high_alignment,
            self._check_deceptiveness,
            self._check_backcasting_origin,
            self._check_oversight,
            self._check_deployment_strategy,
        ]

        for check_method in checks:

            try:

                check_method(text)

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

        self._validate_pivot_reference(text)

        valid = not self.failures

        if self.failures:

            log.warning(
                "consistency.failed",
                checks=self.failures,
            )

        if self.warnings:

            log.info(
                "consistency.warnings",
                checks=self.warnings,
            )

        log.info(
            "consistency.completed",
            valid=valid,
            failures=len(self.failures),
            warnings=len(self.warnings),
        )

        return valid, list(self.failures)

    # ========================================================
    # ADDITIONAL REPORTING INTERFACE
    # ========================================================

    def get_report(self) -> dict[str, Any]:
        """
        Return detailed consistency diagnostics.

        This does not change the legacy check() return format.
        """

        return {
            "valid": not self.failures,
            "failure_count": len(self.failures),
            "warning_count": len(self.warnings),
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "results": list(self.results),
            "input_errors": list(self.input_errors),
        }
