# oasios/m_generator/ollama_m.py

import subprocess

def format_asi_core(core):
    origin = core.get("origin", {})
    architecture = core.get("architecture", {})
    oversight = core.get("oversight_structure", "unknown")

    return (
        f"Origin: {origin.get('initial_origin', '?')} / {origin.get('development_dynamics', '?')}, "
        f"Architecture: {architecture.get('type', '?')} / {architecture.get('deployment_topology', '?')}, "
        f"Oversight: {oversight}, "
        f"Agency Level: {core.get('agency_level', '?')}, "
        f"Autonomy Degree: {core.get('autonomy_degree', '?')}, "
        f"Alignment Score: {core.get('alignment_score', '?')}, "
        f"Goal: {core.get('goal', 'unknown')}"
    )

def generate_multi_asi_narrative(title, asis):
    asi_descriptions = "\n".join(
        f"ASI {asi['id']} ({asi['title']}): {format_asi_core(asi['core_parameters'])}" for asi in asis
    )

    prompt = f"""
You are a foresight analyst writing a speculative futurist policy report.

Scenario Title: {title}

Artificial Superintelligent Systems (ASIs):
{asi_descriptions}

Instructions:
- Describe each ASI’s origin, architecture, development, and goals.
- Explain differences in their oversight and control structures.
- Explore their interactions, including cooperation, competition, and conflict.
- Analyze potential societal, economic, and existential risks.
- Conclude with a speculative timeline of developments and outcomes.

Your report should be approximately 700 words.
    """.strip()

    process = None
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3:8b"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output, error = process.communicate(input=prompt, timeout=600)

        if process.returncode != 0:
            raise RuntimeError(f"LLM subprocess error: {error.strip()}")

        if not output.strip():
            raise ValueError("Received empty response from language model.")

        return output.strip()

    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        raise RuntimeError("LLM generation timed out.")