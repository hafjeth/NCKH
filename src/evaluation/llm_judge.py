"""
LLM Judge Module for Multi-Agent Policy Debate Evaluation
========================================================

This module implements a research-grade LLM-as-a-Judge framework
for evaluating multi-agent policy debates on carbon tax implementation
in the Vietnamese textile industry.

Evaluation Metrics:
- coherence (1–10)
- factuality (1–10)

Key Research Features:
- Deterministic decoding (temperature = 0)
- Robust JSON extraction and validation
- Multi-run evaluation for variance estimation
- Statistical reporting (mean, std, median, IQR)
- Full logging for reproducibility

Author: Research Team
Version: 3.0 (Conference-ready)
"""

import os
import json
import re
import time
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

load_dotenv()
print(">>> LLM Judge module initialized")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Configuration for LLM-based evaluation."""
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0
    log_dir: str = "logs/evaluation"
    enable_logging: bool = True
    n_runs_default: int = 5


# ---------------------------------------------------------------------
# Prompt (Research-grade)
# ---------------------------------------------------------------------

PROMPT_TEMPLATE = """
You are an independent expert reviewer evaluating the quality of a multi-agent
policy debate on carbon tax implementation in the Vietnamese textile industry.

Your task:
- Carefully read the full conversation below.
- Assign integer scores from 1 to 10 for each criterion.

Scoring guidelines:
- Use the full 1–10 scale.
- A score of 5 represents an average-quality debate.
- Avoid overly generous scoring.

Evaluation criteria:

1. coherence:
   The logical consistency, clarity, and structured flow of arguments
   across the entire debate.

2. factuality:
   The degree to which arguments are accurate, grounded in policy knowledge,
   economic reasoning, and domain-specific facts.

STRICT REQUIREMENTS:
- Output ONLY valid JSON.
- Do NOT include any text outside the JSON object.
- Use the exact format below.

{
  "coherence": <int>,
  "factuality": <int>,
  "explanation": "<brief explanation, maximum 3 sentences>"
}

Conversation:
<<<
{conversation_log}
>>>
"""


# ---------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------
# Parsing & Validation (TẦNG 1)
# ---------------------------------------------------------------------

def _extract_json(text: str) -> Dict:
    """
    Safely extract a JSON object from LLM output.
    Robust against markdown or additional text.
    """
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in LLM output.")
    return json.loads(match.group())


def _validate_scores(scores: Dict) -> Dict:
    """
    Validate required keys, score ranges, and types.
    """
    required_keys = ["coherence", "factuality", "explanation"]

    for key in required_keys:
        if key not in scores:
            raise ValueError(f"Missing required key: {key}")

    for metric in ["coherence", "factuality"]:
        if not isinstance(scores[metric], int):
            raise TypeError(f"{metric} must be an integer")
        if not 1 <= scores[metric] <= 10:
            raise ValueError(f"{metric} out of valid range [1,10]")

    if not isinstance(scores["explanation"], str) or not scores["explanation"].strip():
        raise ValueError("Explanation must be a non-empty string")

    return scores


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def _log_evaluation(
    experiment_id: str,
    conversation_log: str,
    raw_output: str,
    scores: Dict,
    config: EvaluationConfig,
    error: Optional[str] = None
) -> None:
    if not config.enable_logging:
        return

    os.makedirs(config.log_dir, exist_ok=True)
    path = os.path.join(config.log_dir, f"{experiment_id}.json")

    log_entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": asdict(config),
        "conversation_log": conversation_log,
        "raw_output": raw_output,
        "scores": scores,
        "error": error
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------

def evaluate_conversation(
    conversation_log: str,
    config: Optional[EvaluationConfig] = None,
    experiment_id: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> Dict:
    """
    Perform a single LLM-based evaluation.
    """
    config = config or EvaluationConfig()
    client = client or get_openai_client()
    experiment_id = experiment_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    prompt = PROMPT_TEMPLATE.format(conversation_log=conversation_log)
    raw_output = ""
    last_error = None

    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature
            )

            raw_output = response.choices[0].message.content
            scores = _extract_json(raw_output)
            scores = _validate_scores(scores)

            _log_evaluation(
                experiment_id,
                conversation_log,
                raw_output,
                scores,
                config
            )
            return scores

        except (OpenAIError, ValueError, TypeError, json.JSONDecodeError) as e:
            last_error = str(e)
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2 ** attempt))
            else:
                _log_evaluation(
                    experiment_id,
                    conversation_log,
                    raw_output,
                    {},
                    config,
                    error=last_error
                )
                raise RuntimeError(f"Evaluation failed: {last_error}")


# ---------------------------------------------------------------------
# Multi-run evaluation with statistics 
# ---------------------------------------------------------------------

def evaluate_with_confidence(
    conversation_log: str,
    n_runs: Optional[int] = None,
    config: Optional[EvaluationConfig] = None,
    experiment_id: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> Dict:
    """
    Run multiple evaluations and report statistical stability.
    """
    config = config or EvaluationConfig()
    client = client or get_openai_client()
    n_runs = n_runs or config.n_runs_default
    base_id = experiment_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    for i in range(n_runs):
        run_id = f"{base_id}_run{i+1:02d}"
        try:
            score = evaluate_conversation(
                conversation_log,
                config,
                run_id,
                client
            )
            results.append(score)
        except Exception as e:
            print(f"Run {i+1} failed: {e}")

    if not results:
        raise RuntimeError("All evaluation runs failed")

    coherence = [r["coherence"] for r in results]
    factuality = [r["factuality"] for r in results]

    summary = {
        "coherence_mean": float(np.mean(coherence)),
        "coherence_std": float(np.std(coherence)),
        "coherence_median": float(np.median(coherence)),
        "coherence_iqr": float(np.percentile(coherence, 75) - np.percentile(coherence, 25)),
        "factuality_mean": float(np.mean(factuality)),
        "factuality_std": float(np.std(factuality)),
        "factuality_median": float(np.median(factuality)),
        "factuality_iqr": float(np.percentile(factuality, 75) - np.percentile(factuality, 25)),
        "coherence_scores": coherence,
        "factuality_scores": factuality,
        "explanations": [r["explanation"] for r in results],
        "n_runs": len(results),
        "success_rate": len(results) / n_runs
    }

    if config.enable_logging:
        os.makedirs(config.log_dir, exist_ok=True)
        path = os.path.join(config.log_dir, f"{base_id}_aggregated.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    dummy_conversation = """
    Agent A: A carbon tax will increase production costs for textile firms.
    Agent B: However, it encourages cleaner technology adoption and long-term competitiveness.
    Agent A: SMEs may struggle with upfront investment costs.
    Agent B: Government subsidies and phased implementation could mitigate this issue.
    """

    result = evaluate_with_confidence(dummy_conversation, n_runs=3)
    print(json.dumps(result, indent=2))
