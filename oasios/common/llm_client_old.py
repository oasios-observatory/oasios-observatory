#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# oasios/s_generator/clients/llm_client.py
"""
Refactored on Sun Nov 10 2025
Adds real-time streaming output from Ollama subprocess.
"""

import subprocess
import logging
import threading
import sys
from subprocess import Popen
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    ("llama3:latest", 600),
    ("deepseek-llm:latest", 600),
    ("Gemma:latest", 600),
    ("llama3:8b", 600),
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
{params_json}

# TIMELINE PHASES (these must appear in the narrative)
{timeline_json}

------------------------------
REQUIREMENTS
------------------------------

You MUST produce a coherent scenario narrative (400–800 words) that:

1. **Strictly reflects every parameter** without contradicting:
   - agency_level
   - autonomy_degree
   - alignment_score
   - phenomenology_proxy_score
   - oversight structure
   - goals & deceptiveness levels
   - impact domains
   - development dynamics
   - origin
   - substrate & deployment medium
   - deployment strategy

2. **Integrates the timeline phases explicitly** in chronological order.

3. **Maintains semantic consistency**, including:
   - high autonomy → less effective oversight
   - low alignment → more risk & deception
   - high opacity → limited observability
   - stealth strategy → concealed actions
   - corporate origin → economic incentives
   - state origin → geopolitical framing

4. **Avoid forbidden content**:
   - Do NOT add capabilities not implied by parameters.
   - Do NOT change alignment or autonomy values.
   - Do NOT claim the ASI has motivations outside the stated_goal or mesa_goals.
   - NO specific operational instructions, exploits, hacking steps, etc.

5. **Tone & style**:
   - Foresight-analysis style (RAND, FHI, CSER style).
   - Professional, analytical, non-sensational.
   - All claims must logically follow from parameters.

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
            # Print to live console in real time
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

        # Capture stderr (if any)
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
        return False, str(e), model


def generate_narrative(title: str, params: Dict[str, Any], timeline: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """
    Generate a full narrative, streaming text in real time while models are tried sequentially.
    """
    params_str = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in params.items()])
    timeline_str = "\n".join([f"- {p['phase']}: {p['years']}" for p in timeline])

    full_prompt = (
            SYSTEM_PROMPT.format(
                title=title,
                params_json=params_str,
                timeline_json=timeline_str
            )
            + "\n\n"
            + USER_PROMPT.format(title=title)
    )

    print(f"\n🧠 Generating scenario: {title}\nUsing models in order: {[m for m, _ in AVAILABLE_MODELS]}\n")

    for model, timeout in AVAILABLE_MODELS:
        print(f"\n⚙️ Trying model: {model} (timeout={timeout}s)\n{'-'*60}\n")
        success, text, used = generate(full_prompt, model, timeout)
        if success:
            print(f"\n✅ Success with {used}\n")
            logger.info(f"Success with {used}")
            return True, text, used
        else:
            print(f"\n❌ {model} failed: {text[:120]}\n")
            logger.warning(f"{model} failed: {text[:100]}")

    return False, "All models failed", "none"