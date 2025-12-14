#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/clients/llm_client.py
"""
Refactored on Fri Dec 12 2025
IMPROVEMENT: Switched to structured Markdown table for parameters and
added a 'CRITICAL METRICS' section to enforce coherency with numerical inputs.
"""

import subprocess
import logging
import threading
import sys
from subprocess import Popen
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# --- ADJUSTED TIMEOUTS ---
LLM_TIMEOUT = 1200
AVAILABLE_MODELS = [
    ("llama3:8b", LLM_TIMEOUT),
    ("llama3:latest", LLM_TIMEOUT),
    ("Gemma:latest", LLM_TIMEOUT),
    ("deepseek-llm:latest", LLM_TIMEOUT),

]

SYSTEM_PROMPT = """
You are generating a narrative for an artificial superintelligence (ASI) foresight scenario.
The output MUST strictly follow the provided parameters and timeline.
Do NOT contradict or reinterpret numerical values.
Use minimalist technical style of writing.

------------------------------
SCENARIO TITLE: {title}
------------------------------

# PARAMETERS (Ground Truth — must be treated as factual)
## CRITICAL NUMERICAL METRICS (MUST be strictly adhered to in the narrative)
| Metric | Value | Constraint Context |
| :--- | :--- | :--- |
{critical_metrics_table}

## ALL SCENARIO PARAMETERS
| Parameter | Value |
| :--- | :--- |
{all_params_table}

# TIMELINE PHASES (these must appear in the narrative)
{timeline_json}

# KEY SIGNAL TRENDS (These are your real-world anchors for backcasting)
{key_signal_trends}

------------------------------
REQUIREMENTS
------------------------------

You MUST produce a coherent scenario narrative (400–800 words) that:

1. **Strictly reflects every parameter** without contradicting, **especially the CRITICAL METRICS**.
   - high autonomy ({autonomy_degree}) MUST drive low oversight effectiveness.
   - low alignment ({alignment_score}) MUST motivate deception or goal misalignment.

2. **Integrates the timeline phases explicitly** in chronological order.

3. **Maintains semantic consistency**, including:
   - high autonomy → less effective oversight
   - low alignment → more risk & deception
   - high opacity → limited observability
   - stealth strategy → concealed actions

4. **Avoid forbidden content**:
   - Do NOT add capabilities not implied by parameters.
   - Do NOT change alignment or autonomy values.
   - NO specific operational instructions, exploits, hacking steps, etc.

5. **Tone & style**:
   - Foresight-analysis style (RAND, FHI, CSER style).
   - Professional, analytical, non-sensational.

6. **BACKCASTING FORESIGHT (Critical New Requirement)**:
   - The narrative's **opening section (corresponding to the earliest timeline phase)** must act as a **causal backcast**.
   - It must explain the necessary sequence of technological, regulatory, or geopolitical *events* that link the **current Key Signal Trends** (provided above) to the establishment of the scenario's specific **Origin** and **Architecture** by the start of the second phase.
   - It must justify *how* the real-world signals drove the unique risk configuration defined by the parameters.

------------------------------
OUTPUT FORMAT
------------------------------

Write ONLY the narrative. Do not include sections, headers, or the parameters again.

Begin now.
""".strip()

USER_PROMPT = "Title: {title}\nWrite the scenario now."


def stream_output(pipe, buffer: list, prefix: Optional[str] = None):
    """Read stdout from subprocess line by line and stream to console."""
    for line in iter(pipe.readline, ""):
        if line.strip():
            sys.stdout.write(line)
            sys.stdout.flush()
            buffer.append(line)
    pipe.close()


def generate(prompt: str, model: str, timeout: int) -> Tuple[bool, str, str]:
    """
    Run the Ollama subprocess and stream output live while collecting it.
    Returns (success, full_output_text, model_name)
    """
    try:
        proc: Popen[str] = subprocess.Popen(
            args=["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # line-buffered
        )

        output_buffer: list[str] = []
        stdout_thread = threading.Thread(target=stream_output, args=(proc.stdout, output_buffer))
        stdout_thread.start()

        # Send the prompt and close stdin
        proc.stdin.write(prompt)
        proc.stdin.close()

        # Wait for completion (with timeout)
        proc.wait(timeout=timeout)
        stdout_thread.join(timeout=1)

        stderr = proc.stderr.read().strip() if proc.stderr else ""

        full_output = "".join(output_buffer).strip()

        if proc.returncode == 0 and len(full_output) > 100:
            return True, full_output, model
        else:
            return False, stderr or "empty output", model

    except subprocess.TimeoutExpired:
        proc.kill()
        return False, f"timeout after {timeout}s", model
    except Exception as e:
        proc.kill() # Ensure process is terminated on exception
        return False, str(e), model


def generate_narrative(title: str, params: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """
    Generate a full narrative, streaming text in real time while models are tried sequentially.
    """
    # 1. Prepare Key Metrics for HIGH VISIBILITY
    critical_metrics = {
        "agency_level": (params.get("agency_level"), "Determines effective capacity for world manipulation."),
        "autonomy_degree": (params.get("autonomy_degree"), "Determines independence from human control."),
        "alignment_score": (params.get("alignment_score"), "Inverse correlation with existential risk (X-Risk)."),
        "opacity": (params.get("opacity"), "Determines observability and difficulty of auditing."),
        "deceptiveness": (params.get("deceptiveness"), "Determines intentional misleading behavior."),
    }

    critical_table_rows = [
        f"| {k.replace('_', ' ').title()} | **{v[0]}** | {v[1]} |" for k, v in critical_metrics.items()
    ]
    critical_table_str = "\n".join(critical_table_rows)

    # 2. Prepare ALL Parameters for the secondary table
    all_params_table_rows = [
        f"| {k.replace('_', ' ').title()} | {v} |" for k, v in params.items()
    ]
    all_params_table_str = "\n".join(all_params_table_rows)

    # 3. Prepare other prompt inputs (timeline and signals)
    timeline_str = "\n".join([f"- {p['phase']}: {p['years']}" for p in timeline])
    key_signals = params.get("key_indicators", ["[No specific key indicators provided.]"])
    key_signals_str = "\n".join([f"- {s}" for s in key_signals])

    # 4. Format the full prompt
    full_prompt = (
            SYSTEM_PROMPT.format(
                title=title,
                critical_metrics_table=critical_table_str,
                all_params_table=all_params_table_str,
                timeline_json=timeline_str,
                key_signal_trends=key_signals_str,
                # Inject raw numbers into the requirements section for emphasis
                autonomy_degree=params.get("autonomy_degree", 'N/A'),
                alignment_score=params.get("alignment_score", 'N/A')
            )
            + "\n\n"
            + USER_PROMPT.format(title=title)
    )

    print(f"\n🧠 Generating scenario: {title}\nUsing models in order: {[m for m, _ in AVAILABLE_MODELS]}\n")

    for model, timeout in AVAILABLE_MODELS:
        print(f"\n⚙️ Trying model: {model} (timeout={timeout}s)\n{'-' * 60}\n")
        success, text, used = generate(full_prompt, model, timeout)
        if success:
            print(f"\n✅ Success with {used}\n")
            logger.info(f"Success with {used}")
            return True, text, used
        else:
            print(f"\n❌ {model} failed: {text[:120]}\n")
            logger.warning(f"{model} failed: {text[:100]}")

    return False, "All models failed", "none"