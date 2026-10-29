#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/common/llm_client.py

"""
Ollama client for precursor-informed ASI scenario generation.
"""

import logging
import subprocess
import sys
import threading

from typing import Any

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 1200
MIN_OUTPUT_LENGTH = 100

AVAILABLE_MODELS = [
    ("llama3:8b", LLM_TIMEOUT),
    ("llama3:latest", LLM_TIMEOUT),
    ("Gemma:latest", LLM_TIMEOUT),
    ("deepseek-llm:latest", LLM_TIMEOUT),
]

SYSTEM_PROMPT = """
You are generating a narrative for an artificial superintelligence (ASI)
foresight scenario.

The output MUST strictly follow the provided parameters and timeline.
Do NOT contradict or reinterpret numerical values.
Use a minimalist technical writing style.

------------------------------
SCENARIO TITLE: {title}
------------------------------

# PARAMETERS (Ground Truth)
## CRITICAL NUMERICAL METRICS
| Metric | Value | Constraint Context |
| :--- | :--- | :--- |
{critical_metrics_table}

## ALL SCENARIO PARAMETERS
| Parameter | Value |
| :--- | :--- |
{all_params_table}

# TIMELINE PHASES
{timeline_json}

# KEY SIGNAL TRENDS
{key_signal_trends}

------------------------------
REQUIREMENTS
------------------------------

Produce a coherent scenario narrative of 400–800 words.

1. Reflect every parameter without contradiction.
2. High autonomy should correspond to reduced oversight effectiveness.
3. Low alignment should motivate greater risk or goal misalignment.
4. High opacity should correspond to limited observability.
5. Stealth deployment should involve concealed actions.
6. Integrate timeline phases in chronological order.
7. Do not introduce capabilities unsupported by the parameters.
8. Do not include operational instructions, exploits, or hacking steps.
9. Use professional, analytical foresight language.
10. Begin with a causal backcast connecting the key signal trends to the
    scenario's origin and architecture.

Write ONLY the narrative. Do not include headings or repeat the parameters.
""".strip()

USER_PROMPT = "Title: {title}\nWrite the scenario now."


class LLMGenerationError(RuntimeError):
    """Raised when narrative generation fails."""


def _stream_stdout(pipe, buffer: list[str]) -> None:
    """Read and display subprocess stdout without blocking stderr."""
    try:
        for line in iter(pipe.readline, ""):
            buffer.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    except (OSError, ValueError):
        logger.debug("llm.stdout_stream_ended")
    finally:
        pipe.close()


def _capture_stderr(pipe, buffer: list[str]) -> None:
    """Continuously drain stderr to prevent subprocess pipe deadlocks."""
    try:
        for line in iter(pipe.readline, ""):
            buffer.append(line)
    except (OSError, ValueError):
        logger.debug("llm.stderr_stream_ended")
    finally:
        pipe.close()


def generate(
    prompt: str,
    model: str,
    timeout: int,
) -> tuple[bool, str, str]:
    """
    Execute an Ollama model.

    Returns:
        Tuple of (success, generated_text_or_error, model_name).
    """
    if not prompt.strip():
        return False, "Prompt is empty.", model

    if timeout <= 0:
        return False, "Timeout must be positive.", model

    process = None
    stdout_buffer: list[str] = []
    stderr_buffer: list[str] = []
    threads: list[threading.Thread] = []

    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_thread = threading.Thread(
            target=_stream_stdout,
            args=(process.stdout, stdout_buffer),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_capture_stderr,
            args=(process.stderr, stderr_buffer),
            daemon=True,
        )

        threads.extend([stdout_thread, stderr_thread])

        stdout_thread.start()
        stderr_thread.start()

        process.stdin.write(prompt)
        process.stdin.close()

        process.wait(timeout=timeout)

        # The pipes reach EOF when the child closes them.
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if any(thread.is_alive() for thread in threads):
            logger.warning(
                "llm.reader_thread_not_finished",
                extra={"model": model},
            )

        output = "".join(stdout_buffer).strip()
        stderr = "".join(stderr_buffer).strip()

        if process.returncode == 0 and len(output) > MIN_OUTPUT_LENGTH:
            logger.info(
                "llm.generation.succeeded",
                extra={"model": model, "output_length": len(output)},
            )
            return True, output, model

        error = stderr or (
            f"Ollama exited with code {process.returncode}; "
            f"output length was {len(output)}."
        )

        logger.warning(
            "llm.generation.failed",
            extra={"model": model, "error": error[:500]},
        )

        return False, error, model

    except subprocess.TimeoutExpired:
        logger.warning(
            "llm.generation.timeout",
            extra={"model": model, "timeout_seconds": timeout},
        )
        return False, f"Timeout after {timeout} seconds.", model

    except (OSError, ValueError, BrokenPipeError) as exc:
        logger.warning(
            "llm.subprocess.error",
            extra={
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return False, f"{type(exc).__name__}: {exc}", model

    except Exception:
        logger.exception(
            "Unexpected error while running Ollama model %s",
            model,
        )
        return False, "Unexpected subprocess error.", model

    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("llm.process_cleanup_failed")

            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except OSError:
                        pass

        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2)


def _format_value(value: Any) -> str:
    """Format values for prompt tables without multiline table corruption."""
    if value is None:
        return "None"

    if isinstance(value, (dict, list, tuple)):
        text = repr(value)
    else:
        text = str(value)

    return text.replace("|", "\\|").replace("\n", " ")


def generate_narrative(
    title: str,
    params: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> tuple[bool, str, str]:
    """
    Generate a scenario narrative using the configured Ollama models.

    Returns:
        (success, narrative_or_error, model_used)
    """
    if not isinstance(title, str) or not title.strip():
        return False, "Scenario title must be a nonempty string.", "none"

    if not isinstance(params, dict):
        return False, "Scenario parameters must be a dictionary.", "none"

    if not isinstance(timeline, list) or not timeline:
        return False, "Timeline must be a nonempty list.", "none"

    for index, phase in enumerate(timeline):
        if not isinstance(phase, dict):
            return (
                False,
                f"Timeline phase {index} must be a dictionary.",
                "none",
            )

        if "phase" not in phase or "years" not in phase:
            return (
                False,
                f"Timeline phase {index} requires 'phase' and 'years'.",
                "none",
            )

    critical_metrics = {
        "agency_level": (
            params.get("agency_level"),
            "Effective capacity for influencing the world.",
        ),
        "autonomy_degree": (
            params.get("autonomy_degree"),
            "Independence from human control.",
        ),
        "alignment_score": (
            params.get("alignment_score"),
            "Alignment with intended objectives.",
        ),
        "opacity": (
            params.get("opacity"),
            "Observability and auditability.",
        ),
        "deceptiveness": (
            params.get("deceptiveness"),
            "Potential for misleading behavior.",
        ),
    }

    critical_rows = [
        f"| {key.replace('_', ' ').title()} | "
        f"{_format_value(value)} | {description} |"
        for key, (value, description) in critical_metrics.items()
    ]

    all_parameter_rows = [
        f"| {key.replace('_', ' ').title()} | "
        f"{_format_value(value)} |"
        for key, value in sorted(params.items())
    ]

    timeline_text = "\n".join(
        f"- {phase['phase']}: {phase['years']}"
        for phase in timeline
    )

    key_signals = params.get("key_indicators", [])

    if not isinstance(key_signals, list):
        return False, "'key_indicators' must be a list.", "none"

    signals_text = "\n".join(
        f"- {_format_value(signal)}"
        for signal in key_signals
    ) or "- No specific key indicators provided."

    try:
        prompt = SYSTEM_PROMPT.format(
            title=title,
            critical_metrics_table="\n".join(critical_rows),
            all_params_table="\n".join(all_parameter_rows),
            timeline_json=timeline_text,
            key_signal_trends=signals_text,
        )

        prompt += "\n\n" + USER_PROMPT.format(title=title)

    except (KeyError, ValueError, IndexError) as exc:
        logger.exception("llm.prompt.format_failed")
        return False, f"Prompt formatting failed: {exc}", "none"

    for model, timeout in AVAILABLE_MODELS:
        logger.info(
            "llm.model.attempt",
            extra={"model": model, "timeout_seconds": timeout},
        )

        success, result, used_model = generate(prompt, model, timeout)

        if success:
            return True, result, used_model

        logger.warning(
            "llm.model.failed",
            extra={
                "model": model,
                "error": result[:300],
            },
        )

    return False, "All configured Ollama models failed.", "none"
