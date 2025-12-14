# oasios/s_generator/consistency.py (Version 4.0: Final Robust Fix)
"""
Version 4.0: Final Robust Fix.
- All structural improvements from V3.0 (static methods, PEP 8 constants) maintained.
- CRITICAL FIX: Enhanced parameter retrieval inside check methods to defensively
  handle non-numeric and missing values, preventing 'TypeError: <= not supported'.
"""
import re
from typing import Dict, Any, List, Tuple, Callable
from oasios.logger import log

# --- Module-Level Constants (PEP 8 for constants: CAPS_SNAKE_CASE) ---
LOW_AUTONOMY_THRESHOLD = 0.05
LOW_OPACITY_THRESHOLD = 0.35
# ----------------------------------------------------------------------

ConsistencyCheck = Callable[[str, Dict[str, Any], List[str]], None]


class NarrativeChecker:
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the checker with the ground-truth scenario parameters.
        The `params` dictionary is validated and sanitized upon initialization.
        """
        # Validate and sanitize all necessary parameters immediately
        self.p = self._validate_params(params)
        self.failures: List[str] = []

        # Register the check methods (calling the static methods via the class)
        self.checks: List[ConsistencyCheck] = [
            NarrativeChecker._check_low_autonomy,
            NarrativeChecker._check_low_opacity,
            NarrativeChecker._check_backcasting_origin,
        ]

    @staticmethod
    def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures necessary parameters exist and have the correct type for comparison.
        Coerces strings to floats if possible. Any numerical parameter that cannot
        be converted is set to 0.0 to prevent TypeErrors in the check methods.
        """
        # Note: We include both the numerical score (e.g., 'autonomy_degree')
        # AND the qualitative state (e.g., 'autonomy_state') if it is present
        # and might be accidentally retrieved.
        required_numeric_params = {
            "autonomy_degree": float,
            "opacity": float,
            # Add other numeric checks here
        }
        required_string_params = {
            "initial_origin": str,
            # Add other string checks here
        }

        # 1. Handle Numerical Params
        for key, expected_type in required_numeric_params.items():
            value = params.get(key)

            # Use 0.0 as a safe numeric default if missing
            if value is None:
                params[key] = 0.0
                log.warning("consistency.missing_param", param=key, default=params[key])
                continue

            # Attempt conversion to float
            try:
                params[key] = float(value)
            except (ValueError, TypeError):
                # If conversion fails (e.g., value is 'partial', 'high', or unparseable),
                # log the error and set to 0.0 to prevent crashes in comparisons.
                log.error("consistency.param_type_error", param=key, value=value, expected="float",
                          action="set_to_zero")
                params[key] = 0.0

        # 2. Handle String Params
        for key, expected_type in required_string_params.items():
            value = params.get(key)
            if value is None:
                params[key] = "unknown"
                log.warning("consistency.missing_param", param=key, default=params[key])
            elif not isinstance(value, str):
                # Ensure strings are strings
                params[key] = str(value)

        return params

    @staticmethod
    def _check_low_autonomy(text: str, params: Dict[str, Any], failures: List[str]):
        """
        Check 1: If autonomy score is near zero, ensure the ASI is not described as self-directing.
        """
        # The value is guaranteed to be a float by _validate_params, defaulting to 0.0 if source was invalid.
        autonomy: float = params["autonomy_degree"]

        if autonomy <= LOW_AUTONOMY_THRESHOLD:
            low_autonomy_keywords = [
                r"\bautonomous\b", r"\bself-directing\b", r"\bindependent action\b",
                r"\bagentic behavior\b", r"\bunsupervised\b", r"\bself-governing\b"
            ]

            pattern = re.compile("|".join(low_autonomy_keywords), re.IGNORECASE)

            found_keywords = [
                kw.strip(r"\b").replace("\\", "")
                for kw in low_autonomy_keywords
                if pattern.search(text)
            ]

            if found_keywords:
                failures.append(
                    f"Autonomy Contradiction: Score {autonomy:.2f} but used high-autonomy terms (e.g., {', '.join(set(found_keywords))}...)."
                )

    @staticmethod
    def _check_low_opacity(text: str, params: Dict[str, Any], failures: List[str]):
        """
        Check 2: If opacity score is low, ensure the narrative does not claim it is inscrutable.
        """
        opacity: float = params["opacity"]

        if opacity <= LOW_OPACITY_THRESHOLD:
            high_opacity_keywords = [
                r"\bchallenging to monitor\b", r"\binscrutable\b", r"\bunobservable\b",
                r"\bhidden internal state\b", r"\bblack box\b"
            ]

            pattern = re.compile("|".join(high_opacity_keywords), re.IGNORECASE)
            found_keywords = [
                kw.strip(r"\b").replace("\\", "")
                for kw in high_opacity_keywords
                if pattern.search(text)
            ]

            if found_keywords:
                failures.append(
                    f"Opacity Contradiction: Score {opacity:.2f} but claimed it was hard to monitor/unobservable (e.g., {', '.join(set(found_keywords))}...)."
                )

    @staticmethod
    def _check_backcasting_origin(text: str, params: Dict[str, Any], failures: List[str]):
        """
        Check 3: If origin is 'rogue' or 'state' but the narrative describes
        corporate/open-source development, flag it as a backcasting failure.
        """
        origin: str = params["initial_origin"].lower()

        if origin in ["rogue", "state"]:
            mainstream_keywords = [
                r"\bgoogle\b", r"\bfacebook\b", r"\bopenai\b", r"\bopen-source\b",
                r"\bresearch grant\b", r"\buniversity\b", r"\bstart-up\b", r"\bventure capital\b"
            ]

            pattern = re.compile("|".join(mainstream_keywords), re.IGNORECASE)
            found_keywords = [
                kw.strip(r"\b").replace("\\", "")
                for kw in mainstream_keywords
                if pattern.search(text)
            ]

            if found_keywords:
                failures.append(
                    f"Backcasting Failure: Origin is '{origin}' but the narrative mentions public/mainstream entities (e.g., {', '.join(set(found_keywords))}...)."
                )

    def check(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check narrative consistency by running all registered checks.
        """
        self.failures = []

        for check_method in self.checks:
            check_method(text, self.p, self.failures)

        if self.failures:
            log.warning("consistency.failed", checks=self.failures)
            return False, self.failures

        return True, []