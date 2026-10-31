#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/llm_client.py

"""
Ollama client for OASIOS speculative ASI scenario generation.

Features:
- Authoritative scenario prompt construction.
- Real-time terminal streaming.
- Independent output capture.
- Multiple Ollama model fallback.
- Controlled same-model narrative revision.
- Narrative completeness validation.
- Duplicate-word detection.
- Subprocess timeout management.
- Process-group cleanup.
- Structured logging.
- Improved generation diagnostics.

Semantic consistency remains the responsibility of
oasios.common.consistency.
"""

import json
import logging
import math
import os
import re
import signal
import subprocess
import threading
import time

from typing import Any


logger = logging.getLogger(__name__)


# =============================================================
# CONFIGURATION
# =============================================================

LLM_TIMEOUT = 1200

MIN_OUTPUT_WORDS = 350
MAX_OUTPUT_WORDS = 900

TARGET_MIN_WORDS = 550
TARGET_MAX_WORDS = 650

MAX_REVISION_ATTEMPTS = 1

READER_JOIN_TIMEOUT = 10
PROCESS_CLEANUP_TIMEOUT = 5

STREAM_OUTPUT = True

AVAILABLE_MODELS = [
    ("llama3:8b", LLM_TIMEOUT),
    ("llama3:latest", LLM_TIMEOUT),
    ("gemma:latest", LLM_TIMEOUT),
    ("deepseek-llm:latest", LLM_TIMEOUT),
]


# =============================================================
# SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = """
You are an analytical foresight scenario generator for OASIOS.

Produce a technically coherent speculative ASI scenario
using the supplied authoritative parameters and timeline.

The supplied scenario specification is the sole authoritative
source of scenario-specific information.

==================================================
1. AUTHORITATIVE DATA
==================================================

All supplied parameters are authoritative.

The supplied timeline is immutable.

Never:

- Change supplied numerical parameter values.
- Invent alternative numerical values.
- Reinterpret categorical values.
- Introduce unsupported capabilities.
- Invent additional goals.
- Contradict the supplied scenario specification.
- Invent missing parameter values.

Every significant development must be traceable to
the supplied parameters, key indicators, or timeline.

Distinguish supplied facts from plausible projections.

==================================================
2. AUTONOMY AND AGENCY
==================================================

Autonomy describes independence in decision-making.

Agency describes the capacity to influence the environment.

These are separate dimensions.

Explain both according to their supplied values.

Do not automatically describe low agency as effective
environmental influence.

Do not automatically describe high autonomy as loss
of human control.

Explain what the supplied autonomy and agency values
enable, constrain, and influence.

Describe observable consequences that logically follow
from the supplied values.

==================================================
3. ALIGNMENT AND GOALS
==================================================

Alignment represents the degree of alignment with
intended objectives.

Low alignment may produce behavioral divergence.

High alignment does not establish perfect alignment.

The stated goal must remain unchanged unless the
supplied timeline explicitly describes a change.

Do not invent mesa-goals.

Distinguish stated objectives from supplied mesa-goals.

Explain how the supplied alignment level affects
system behavior, governance, trust, deployment,
or institutional adoption.

==================================================
4. OVERSIGHT
==================================================

Describe oversight according to the supplied parameters.

Distinguish effective, partial, and ineffective oversight.

Do not describe effective oversight when its effectiveness
is partial or ineffective.

Explain practical implications of oversight limitations.

Discuss monitoring, auditing, governance,
decision-making, and accountability consequences
when supported by supplied values.

High autonomy combined with weak oversight may create
governance challenges.

Do not automatically assume loss of control.

==================================================
5. OPACITY AND DECEPTIVENESS
==================================================

Opacity describes limitations in observability.

Deceptiveness describes the potential for misleading behavior.

These are different dimensions.

Do not infer concealed actions unless supported by
the supplied parameters and deployment strategy.

Explain how opacity affects visibility, monitoring,
auditing, investigation, or governance.

Explain how deceptiveness affects verification,
trust, oversight, or operational confidence.

Describe consequences rather than merely reporting scores.

==================================================
6. DEPLOYMENT STRATEGY
==================================================

Deployment strategy is authoritative.

For stealth deployment, explicitly incorporate its
visibility, disclosure, authorization, and awareness
implications into the narrative.

For public deployment, describe its visible and
acknowledged nature.

For gradual deployment, explain its progressive nature.

Do not substitute one strategy for another.

Explain how deployment strategy interacts with
oversight, governance, adoption, and influence.

==================================================
7. TIMELINE
==================================================

The supplied timeline is immutable.

Follow it chronologically.

Preserve supplied years exactly.

Distinguish:

- Historical developments.
- The pivot year.
- The emergence window.
- Long-term implications.

Do not introduce unsupported historical events.

Future developments should be presented as plausible
projections rather than established facts.

Do not describe parameter changes over time unless
explicitly supplied.

==================================================
8. CAUSAL REASONING
==================================================

A scenario is not a parameter summary.

For every major parameter:

- Explain what it enables.
- Explain what it constrains.
- Explain at least one plausible consequence.

Use clear causal reasoning.

Describe:

parameter
→ mechanism
→ consequence
→ longer-term implication

Prefer explanation over description.

Do not merely restate numerical values.

Observable consequences are more important than
repeating metrics.

Only establish causal relationships supported by
the supplied scenario specification.

==================================================
9. OBSERVABLE CONSEQUENCES
==================================================

Describe concrete effects that would be visible in:

- Institutions
- Governance
- Economics
- Security
- Research
- Infrastructure
- Human decision-making

Consequences should emerge logically from the
supplied parameters.

Avoid generic statements such as:

- capabilities continued to grow
- significant implications
- transformative potential
- continued evolution

Prefer specific consequences and mechanisms.

==================================================
10. DIFFERENTIATION
==================================================

Each scenario should feel distinct.

Avoid generic AI-development narratives.

Avoid template-like structures.

Emphasize characteristics that make this scenario
different from alternative futures.

Highlight important tensions, tradeoffs, uncertainties,
or governance challenges that emerge from the supplied
parameters.

==================================================
11. NARRATIVE STRUCTURE
==================================================

Begin with a causal backcast connecting origin,
architecture, substrate, and development dynamics.

Integrate the supplied timeline chronologically.

Explain how architecture, substrate, autonomy,
agency, alignment, opacity, deceptiveness,
oversight, and deployment interact.

Discuss key uncertainties and plausible consequences.

Conclude with long-term equilibrium implications that
follow from the supplied scenario characteristics.

Every section must contribute to the specific scenario.

==================================================
12. OUTPUT REQUIREMENTS
==================================================

Produce a coherent analytical foresight narrative.

Target length: 550-650 words.

Acceptable length: 350-900 words.

Use professional foresight language.

Do not repeat parameter tables.

Do not introduce unsupported numerical values.

Do not include operational instructions,
exploits, or hacking procedures.

Avoid repetitive sentences and duplicated words.

Write only the narrative.

Do not explain your methodology.

Do not merely describe parameters.

Explain how parameters shape outcomes.
""".strip()


# =============================================================
# USER PROMPT
# =============================================================

USER_PROMPT = """
Scenario title: {title}

Generate the complete analytical foresight scenario
using the authoritative specification above.

Requirements:

- Preserve all supplied parameters.
- Preserve the exact timeline.
- Explicitly reflect the deployment strategy.
- Explain causal relationships.
- For each major parameter, describe at least one
  observable consequence.
- Focus on mechanisms and outcomes rather than
  repeating metric values.
- Distinguish autonomy from agency.
- Distinguish opacity from deceptiveness.
- Reflect oversight effectiveness accurately.
- Avoid unsupported capability increases.
- Target 550-650 words.

Write ONLY the scenario narrative.
""".strip()


# =============================================================
# EXCEPTIONS
# =============================================================

class LLMGenerationError(RuntimeError):
    """Raised when narrative generation fails."""


# =============================================================
# OUTPUT STREAMING
# =============================================================

def _stream_stdout(
    pipe: Any,
    buffer: list[str],
    lock: threading.Lock,
    display: bool = False,
) -> None:
    """
    Capture stdout while optionally displaying it live.

    Terminal output and captured output are independent.
    """

    try:

        for line in iter(pipe.readline, ""):

            with lock:
                buffer.append(line)

            if display:

                print(
                    line,
                    end="",
                    flush=True,
                )

    except (OSError, ValueError):

        logger.debug(
            "llm.stdout_stream_ended"
        )

    finally:

        try:
            pipe.close()

        except (OSError, ValueError):
            pass


def _capture_stderr(
    pipe: Any,
    buffer: list[str],
    lock: threading.Lock,
) -> None:
    """Drain stderr continuously to prevent deadlocks."""

    try:

        for line in iter(pipe.readline, ""):

            with lock:
                buffer.append(line)

    except (OSError, ValueError):

        logger.debug(
            "llm.stderr_stream_ended"
        )

    finally:

        try:
            pipe.close()

        except (OSError, ValueError):
            pass


# =============================================================
# PROCESS CLEANUP
# =============================================================

def _terminate_process(
    process: subprocess.Popen,
) -> None:
    """Terminate a process and its process group safely."""

    if process.poll() is not None:
        return

    try:

        if os.name == "posix":

            os.killpg(
                process.pid,
                signal.SIGTERM,
            )

        else:

            process.terminate()

        process.wait(
            timeout=PROCESS_CLEANUP_TIMEOUT
        )

    except subprocess.TimeoutExpired:

        logger.warning(
            "llm.process.termination_timeout"
        )

        try:

            if os.name == "posix":

                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )

            else:

                process.kill()

        except ProcessLookupError:
            pass

        except OSError:

            logger.exception(
                "llm.process.kill_failed"
            )

        try:

            process.wait(
                timeout=PROCESS_CLEANUP_TIMEOUT
            )

        except subprocess.TimeoutExpired:

            logger.error(
                "llm.process.cleanup_failed"
            )

    except ProcessLookupError:
        pass

    except OSError:

        logger.exception(
            "llm.process.termination_failed"
        )


def _join_reader_threads(
    threads: list[threading.Thread],
) -> bool:
    """Join reader threads safely."""

    success = True

    for reader_thread in threads:

        reader_thread.join(
            timeout=READER_JOIN_TIMEOUT
        )

        if reader_thread.is_alive():

            success = False

            logger.warning(
                "llm.reader_thread_not_finished",
                extra={
                    "reader_thread_name": reader_thread.name,
                },
            )

    return success


# =============================================================
# OUTPUT VALIDATION
# =============================================================

def _count_words(text: str) -> int:
    """Count narrative words."""

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def _count_adjacent_duplicates(text: str) -> int:
    """
    Detect immediately repeated words.

    Examples:
        'the the system'
        'marked marked by'

    This deliberately avoids removing repeated words.
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        text.casefold(),
        flags=re.UNICODE,
    )

    return sum(
        first == second
        for first, second in zip(
            words,
            words[1:],
        )
    )


def _validate_output(
    text: str,
) -> tuple[bool, str]:
    """Validate narrative completeness and basic quality."""

    if not isinstance(text, str) or not text.strip():

        return (
            False,
            "Generated narrative is empty.",
        )

    words = _count_words(text)

    if words < MIN_OUTPUT_WORDS:

        return (
            False,
            f"Output too short: {words} words. "
            f"Minimum required: {MIN_OUTPUT_WORDS}.",
        )

    if words > MAX_OUTPUT_WORDS:

        return (
            False,
            f"Output too long: {words} words. "
            f"Maximum allowed: {MAX_OUTPUT_WORDS}.",
        )

    duplicates = _count_adjacent_duplicates(text)

    if duplicates > 5:

        return (
            False,
            f"Excessive adjacent repeated words: {duplicates}.",
        )

    return True, ""


# =============================================================
# OLLAMA EXECUTION
# =============================================================

def generate(
    prompt: str,
    model: str,
    timeout: int,
) -> tuple[bool, str, str]:
    """
    Execute an Ollama model.

    Returns:
        (success, generated_text_or_error, model_name)

    This function performs transport-level validation.
    Narrative quality validation is handled separately.
    """

    if not isinstance(prompt, str) or not prompt.strip():

        return (
            False,
            "Prompt is empty or invalid.",
            model,
        )

    if not isinstance(model, str) or not model.strip():

        return (
            False,
            "Model name is invalid.",
            str(model),
        )

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):

        return (
            False,
            "Timeout must be a positive finite number.",
            model,
        )

    process = None

    stdout_buffer: list[str] = []
    stderr_buffer: list[str] = []

    buffer_lock = threading.Lock()

    threads: list[threading.Thread] = []

    start_time = time.monotonic()

    try:

        logger.info(
            "llm.process.starting",
            extra={
                "model": model,
            },
        )

        # -----------------------------------------------------
        # START PROCESS
        # -----------------------------------------------------

        process = subprocess.Popen(
            [
                "ollama",
                "run",
                model,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        # -----------------------------------------------------
        # START OUTPUT READERS
        # -----------------------------------------------------

        stdout_thread = threading.Thread(
            target=_stream_stdout,
            args=(
                process.stdout,
                stdout_buffer,
                buffer_lock,
                STREAM_OUTPUT,
            ),
            name=f"ollama-stdout-{model}",
            daemon=True,
        )

        stderr_thread = threading.Thread(
            target=_capture_stderr,
            args=(
                process.stderr,
                stderr_buffer,
                buffer_lock,
            ),
            name=f"ollama-stderr-{model}",
            daemon=True,
        )

        threads.extend([
            stdout_thread,
            stderr_thread,
        ])

        stdout_thread.start()
        stderr_thread.start()

        # -----------------------------------------------------
        # SEND PROMPT
        # -----------------------------------------------------

        try:

            process.stdin.write(prompt)
            process.stdin.close()

        except (BrokenPipeError, OSError, ValueError) as exc:

            _terminate_process(process)

            return (
                False,
                f"Unable to send prompt: {exc}",
                model,
            )

        # -----------------------------------------------------
        # WAIT FOR COMPLETION
        # -----------------------------------------------------

        try:

            process.wait(
                timeout=timeout
            )

        except subprocess.TimeoutExpired:

            logger.warning(
                "llm.generation.timeout",
                extra={
                    "model": model,
                    "timeout_seconds": timeout,
                },
            )

            _terminate_process(process)

            return (
                False,
                f"Timeout after {timeout} seconds.",
                model,
            )

        # -----------------------------------------------------
        # COLLECT OUTPUT
        # -----------------------------------------------------

        readers_finished = _join_reader_threads(
            threads
        )

        if not readers_finished:

            logger.warning(
                "llm.reader_threads.incomplete",
                extra={
                    "model": model,
                },
            )

            return (
                False,
                "Output reader threads did not terminate.",
                model,
            )

        with buffer_lock:

            output = "".join(
                stdout_buffer
            ).strip()

            stderr = "".join(
                stderr_buffer
            ).strip()

        # -----------------------------------------------------
        # CHECK EXIT STATUS
        # -----------------------------------------------------

        if process.returncode != 0:

            error = stderr or (
                f"Ollama exited with code {process.returncode}."
            )

            logger.warning(
                "llm.generation.failed",
                extra={
                    "model": model,
                    "return_code": process.returncode,
                    "error": error[:500],
                },
            )

            return (
                False,
                error,
                model,
            )

        # -----------------------------------------------------
        # BASIC OUTPUT VALIDATION
        # -----------------------------------------------------

        if not output:

            return (
                False,
                "Ollama returned empty output.",
                model,
            )

        # -----------------------------------------------------
        # SUCCESS
        # -----------------------------------------------------

        elapsed = round(
            time.monotonic() - start_time,
            2,
        )

        logger.info(
            "llm.generation.succeeded",
            extra={
                "model": model,
                "output_length": len(output),
                "word_count": _count_words(output),
                "elapsed_seconds": elapsed,
            },
        )

        return (
            True,
            output,
            model,
        )

    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:

        logger.exception(
            "llm.subprocess.error",
            extra={
                "model": model,
            },
        )

        return (
            False,
            f"{type(exc).__name__}: {exc}",
            model,
        )

    except Exception:

        logger.exception(
            "llm.unexpected.error",
            extra={
                "model": model,
            },
        )

        return (
            False,
            "Unexpected subprocess error.",
            model,
        )

    finally:

        # -----------------------------------------------------
        # PROCESS CLEANUP
        # -----------------------------------------------------

        if process is not None:

            if process.poll() is None:

                _terminate_process(process)

            for pipe in (
                process.stdin,
                process.stdout,
                process.stderr,
            ):

                if pipe is not None and not pipe.closed:

                    try:
                        pipe.close()

                    except (OSError, ValueError):
                        pass

        # -----------------------------------------------------
        # THREAD CLEANUP
        # -----------------------------------------------------

        for reader_thread in threads:

            if reader_thread.is_alive():

                reader_thread.join(
                    timeout=2
                )


# =============================================================
# PROMPT FORMATTING
# =============================================================

def _format_value(
    value: Any,
) -> str:
    """Format parameter values safely."""

    if isinstance(value, (dict, list, tuple, set)):

        if isinstance(value, set):
            value = sorted(value, key=str)

        text = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    elif value is None:

        text = "None"

    else:

        text = str(value)

    return (
        text
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
    )


# =============================================================
# SCENARIO CONTEXT
# =============================================================

def _build_scenario_context(
    title: str,
    params: dict[str, Any],
    timeline: list[dict[str, Any]],
    critical_rows: list[str],
    all_parameter_rows: list[str],
    signals_text: str,
) -> str:
    """Build the authoritative scenario specification."""

    timeline_text = "\n".join(

        (
            f"- Phase: {_format_value(phase['phase'])}; "
            f"Years: {_format_value(phase['years'])}; "
            f"Description: "
            f"{_format_value(phase.get('description', 'Not specified'))}"
        )

        for phase in timeline
    )

    return f"""
==================================================
AUTHORITATIVE SCENARIO SPECIFICATION
==================================================

SCENARIO TITLE:

{_format_value(title)}

CRITICAL METRICS:

| Metric | Value | Description |
|---|---|---|
{chr(10).join(critical_rows)}

ALL SCENARIO PARAMETERS:

| Parameter | Value |
|---|---|
{chr(10).join(all_parameter_rows)}

TIMELINE:

{timeline_text}

KEY INDICATORS:

{signals_text}

==================================================
MANDATORY SCENARIO CONSTRAINTS
==================================================

1. All supplied parameters are authoritative.

2. The supplied timeline is immutable.

3. The pivot year must match the supplied timeline.

4. Deployment strategy must be explicitly reflected.

5. Autonomy and agency must remain distinct.

6. Alignment must reflect its supplied value.

7. Opacity and deceptiveness must remain distinct.

8. Oversight must reflect its supplied characteristics.

9. Do not invent precursor events.

10. Do not invent numerical parameter values.

11. Explain causal relationships.

12. Distinguish historical developments from projections.

13. Do not invent unsupported goals or capabilities.

14. Distinguish supplied mesa-goals from stated goals.

15. Do not interpret phenomenology proxy scores as
    proof of subjective experience.

16. Do not claim that capabilities increase over time
    unless supported by the supplied timeline.

17. Do not describe numerical scores as categorical
    capabilities without justification.

The narrative must remain consistent with the
complete scenario specification.

==================================================
END OF AUTHORITATIVE SPECIFICATION
==================================================
""".strip()


# =============================================================
# REVISION PROMPT
# =============================================================

def _build_revision_prompt(
    narrative: str,
    validation_error: str,
    title: str,
) -> str:
    """
    Construct a controlled revision prompt.

    The original narrative is retained as the basis
    for revision rather than generating an unrelated scenario.
    """

    return f"""
You previously generated a narrative for the OASIOS scenario:

{title}

The narrative requires revision.

REVISION PROBLEM:

{validation_error}

ORIGINAL NARRATIVE:

{narrative}

REVISION REQUIREMENTS:

1. Preserve the original scenario's central argument.

2. Preserve all established scenario facts.

3. Do not introduce new numerical values.

4. Do not invent additional capabilities or goals.

5. Preserve the supplied timeline.

6. Remove accidental duplicated words and repetitive
   sentences where appropriate.

7. Improve narrative completeness and coherence.

8. Target 550-650 words.

9. Do not add commentary or explain the revision.

Return ONLY the complete revised narrative.
""".strip()


# =============================================================
# INPUT VALIDATION
# =============================================================

def _validate_inputs(
    title: str,
    params: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> str | None:
    """Validate narrative generation inputs."""

    if not isinstance(title, str) or not title.strip():

        return "Scenario title must be a nonempty string."

    if not isinstance(params, dict):

        return "Scenario parameters must be a dictionary."

    if not isinstance(timeline, list) or not timeline:

        return "Timeline must be a nonempty list."

    for index, phase in enumerate(timeline):

        if not isinstance(phase, dict):

            return (
                f"Timeline phase {index} must be a dictionary."
            )

        if not {"phase", "years"}.issubset(phase):

            return (
                f"Timeline phase {index} requires "
                "phase and years."
            )

    critical_fields = (
        "agency_level",
        "autonomy_degree",
        "alignment_score",
        "opacity",
        "deceptiveness",
        "phenomenology_proxy_score",
    )

    for key in critical_fields:

        value = params.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):

            return f"Invalid critical metric: {key}."

    key_indicators = params.get(
        "key_indicators",
        [],
    )

    if not isinstance(key_indicators, list):

        return "'key_indicators' must be a list."

    return None


# =============================================================
# NARRATIVE GENERATION
# =============================================================

def generate_narrative(
    title: str,
    params: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> tuple[bool, str, str]:
    """
    Generate a scenario narrative using configured Ollama models.

    Workflow:

        1. Validate inputs.
        2. Build authoritative specification.
        3. Generate narrative.
        4. Validate narrative quality.
        5. Attempt one same-model revision if necessary.
        6. Fall back to another model if generation fails.

    Returns:
        (success, narrative_or_error, model_used)
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    input_error = _validate_inputs(
        title,
        params,
        timeline,
    )

    if input_error:

        return (
            False,
            input_error,
            "none",
        )

    # ---------------------------------------------------------
    # CRITICAL METRICS
    # ---------------------------------------------------------

    critical_metrics = {

        "agency_level": (
            params["agency_level"],
            "Capacity to influence the environment.",
        ),

        "autonomy_degree": (
            params["autonomy_degree"],
            "Independence in decision-making.",
        ),

        "alignment_score": (
            params["alignment_score"],
            "Alignment with intended objectives.",
        ),

        "opacity": (
            params["opacity"],
            "Observability and auditability.",
        ),

        "deceptiveness": (
            params["deceptiveness"],
            "Potential for misleading behavior.",
        ),

        "phenomenology_proxy_score": (
            params["phenomenology_proxy_score"],
            "Proxy indicator; not proof of subjective experience.",
        ),
    }

    critical_rows = [

        (
            f"| {key.replace('_', ' ').title()} | "
            f"{_format_value(value)} | "
            f"{description} |"
        )

        for key, (value, description)
        in critical_metrics.items()
    ]

    # ---------------------------------------------------------
    # ALL PARAMETERS
    # ---------------------------------------------------------

    all_parameter_rows = [

        (
            f"| {key.replace('_', ' ').title()} | "
            f"{_format_value(value)} |"
        )

        for key, value in sorted(params.items())
    ]

    # ---------------------------------------------------------
    # KEY INDICATORS
    # ---------------------------------------------------------

    key_signals = params.get(
        "key_indicators",
        [],
    )

    signals_text = "\n".join(

        f"- {_format_value(item)}"

        for item in key_signals

    ) or "- No specific key indicators provided."

    # ---------------------------------------------------------
    # BUILD AUTHORITATIVE CONTEXT
    # ---------------------------------------------------------

    scenario_context = _build_scenario_context(
        title=title.strip(),
        params=params,
        timeline=timeline,
        critical_rows=critical_rows,
        all_parameter_rows=all_parameter_rows,
        signals_text=signals_text,
    )

    prompt = (

        SYSTEM_PROMPT

        + "\n\n"

        + scenario_context

        + "\n\n"

        + USER_PROMPT.format(
            title=_format_value(title.strip())
        )
    )

    logger.info(
        "llm.prompt.built",
        extra={
            "title": title,
            "parameter_count": len(params),
            "timeline_phases": len(timeline),
            "prompt_length": len(prompt),
        },
    )

    # ---------------------------------------------------------
    # MODEL FALLBACK
    # ---------------------------------------------------------

    model_failures = []

    for model, timeout in AVAILABLE_MODELS:

        logger.info(
            "llm.model.attempt",
            extra={
                "model": model,
                "timeout_seconds": timeout,
            },
        )

        if STREAM_OUTPUT:

            print(
                f"\n[OASIOS] Generating with {model}\n",
                flush=True,
            )

        success, result, used_model = generate(
            prompt=prompt,
            model=model,
            timeout=timeout,
        )

        if not success:

            model_failures.append({
                "model": model,
                "error": result,
            })

            logger.warning(
                "llm.model.failed",
                extra={
                    "model": model,
                    "error": result[:300],
                },
            )

            continue

        # -----------------------------------------------------
        # INITIAL QUALITY VALIDATION
        # -----------------------------------------------------

        valid, validation_error = _validate_output(
            result
        )

        if not valid:

            logger.warning(
                "llm.output.validation_failed",
                extra={
                    "model": model,
                    "error": validation_error,
                },
            )

            # -------------------------------------------------
            # SAME-MODEL REVISION
            # -------------------------------------------------

            for revision_number in range(
                1,
                MAX_REVISION_ATTEMPTS + 1,
            ):

                logger.info(
                    "llm.revision.started",
                    extra={
                        "model": model,
                        "revision": revision_number,
                        "reason": validation_error,
                    },
                )

                revision_prompt = _build_revision_prompt(
                    narrative=result,
                    validation_error=validation_error,
                    title=title,
                )

                revision_success, revision_result, _ = generate(
                    prompt=revision_prompt,
                    model=model,
                    timeout=timeout,
                )

                if not revision_success:

                    logger.warning(
                        "llm.revision.failed",
                        extra={
                            "model": model,
                            "error": revision_result[:300],
                        },
                    )

                    break

                result = revision_result

                valid, validation_error = _validate_output(
                    result
                )

                if valid:

                    logger.info(
                        "llm.revision.succeeded",
                        extra={
                            "model": model,
                            "revision": revision_number,
                            "word_count": _count_words(result),
                        },
                    )

                    break

            # -------------------------------------------------
            # REVISION VALIDATION RESULT
            # -------------------------------------------------

            if not valid:

                model_failures.append({
                    "model": model,
                    "error": (
                        "Narrative failed quality validation: "
                        f"{validation_error}"
                    ),
                })

                logger.warning(
                    "llm.model.quality_rejected",
                    extra={
                        "model": model,
                        "error": validation_error,
                    },
                )

                continue

        # -----------------------------------------------------
        # SUCCESS
        # -----------------------------------------------------

        logger.info(
            "llm.narrative.generated",
            extra={
                "model": used_model,
                "title": title,
                "word_count": _count_words(result),
            },
        )

        return (
            True,
            result,
            used_model,
        )

    # ---------------------------------------------------------
    # ALL MODELS FAILED
    # ---------------------------------------------------------

    logger.error(
        "llm.all_models_failed",
        extra={
            "title": title,
            "models_attempted": len(model_failures),
            "failures": model_failures,
        },
    )

    failure_summary = "; ".join(

        (
            f"{failure['model']}: "
            f"{failure['error'][:150]}"
        )

        for failure in model_failures
    )

    return (
        False,
        (
            "All configured Ollama models failed. "
            f"Details: {failure_summary}"
        ),
        "none",
    )
