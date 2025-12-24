"""
LLM Judge Module for Multi-Agent Policy Debate Evaluation
===========================================================

This module provides a research-grade evaluation framework for assessing
the quality of multi-agent policy debates using LLM-as-a-Judge methodology.

Author: Research Team
Version: 2.0 (Production & Research Ready)
"""

import json
import re
import os
import time
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
from openai import OpenAI, OpenAIError


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for LLM-based evaluation."""
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0
    log_dir: str = "logs/evaluation"
    enable_logging: bool = True


# ============================================================================
# Prompt Template
# ============================================================================

PROMPT_TEMPLATE = """
You are an independent expert reviewer evaluating the quality of a multi-agent
policy debate on carbon tax implementation in the Vietnamese textile industry.

Your task:
- Carefully read the full conversation below.
- Assign integer scores from 1 to 10 for each criterion:

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

{{
  "coherence": <int>,
  "factuality": <int>,
  "explanation": "<brief explanation, maximum 3 sentences>"
}}

Conversation:
<<<
{conversation_log}
>>>
"""


# ============================================================================
# OpenAI Client Initialization
# ============================================================================

def get_openai_client() -> OpenAI:
    """
    Initialize OpenAI client with API key from environment.
    
    Returns:
        OpenAI: Configured client instance
        
    Raises:
        ValueError: If OPENAI_API_KEY not found in environment
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it before running evaluation."
        )
    return OpenAI(api_key=api_key)


# ============================================================================
# Core Functions
# ============================================================================

def _extract_json(text: str) -> Dict:
    """
    Safely extract JSON object from LLM output.
    
    This function makes the evaluation robust against malformed responses,
    such as those containing markdown code blocks or explanatory text.
    
    Args:
        text: Raw text output from LLM
        
    Returns:
        dict: Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON found
        json.JSONDecodeError: If JSON parsing fails
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Extract JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(
            f"No valid JSON object found in LLM output.\n"
            f"Raw output: {text[:200]}..."
        )
    
    return json.loads(match.group())


def _validate_scores(scores: Dict) -> Dict:
    """
    Validate score ranges and required keys.
    
    This ensures that the LLM judge returns valid, interpretable scores
    that can be used for quantitative analysis.
    
    Args:
        scores: Dictionary containing evaluation scores
        
    Returns:
        dict: Validated scores (same as input if valid)
        
    Raises:
        ValueError: If required keys missing or scores out of range
        TypeError: If scores are not integers
    """
    required_keys = ["coherence", "factuality", "explanation"]
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in scores]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Validate score types and ranges
    for metric in ["coherence", "factuality"]:
        score = scores[metric]
        
        if not isinstance(score, int):
            raise TypeError(
                f"{metric} must be integer, got {type(score).__name__}"
            )
        
        if not (1 <= score <= 10):
            raise ValueError(
                f"{metric} score {score} out of valid range [1, 10]"
            )
    
    # Validate explanation
    if not isinstance(scores["explanation"], str):
        raise TypeError("explanation must be string")
    
    if len(scores["explanation"].strip()) == 0:
        raise ValueError("explanation cannot be empty")
    
    return scores


def _log_evaluation(
    experiment_id: str,
    conversation_log: str,
    raw_output: str,
    scores: Dict,
    config: EvaluationConfig,
    error: Optional[str] = None
) -> None:
    """
    Log evaluation results for reproducibility and debugging.
    
    Args:
        experiment_id: Unique identifier for this evaluation
        conversation_log: Input conversation
        raw_output: Raw LLM response
        scores: Parsed and validated scores
        config: Evaluation configuration
        error: Error message if evaluation failed
    """
    if not config.enable_logging:
        return
    
    os.makedirs(config.log_dir, exist_ok=True)
    log_file = os.path.join(config.log_dir, f"{experiment_id}.json")
    
    log_entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": asdict(config),
        "conversation_log": conversation_log,
        "raw_output": raw_output,
        "scores": scores,
        "error": error
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)


def evaluate_conversation(
    conversation_log: str,
    config: Optional[EvaluationConfig] = None,
    experiment_id: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> Dict:
    """
    Evaluate a debate conversation using an LLM-based judge.
    
    This function implements a single-run evaluation with robust error
    handling and retry logic for production use.
    
    Args:
        conversation_log: Full debate transcript
        config: Evaluation configuration (uses defaults if None)
        experiment_id: Unique ID for logging (auto-generated if None)
        client: OpenAI client (creates new if None)
        
    Returns:
        dict: {
            "coherence": int (1-10),
            "factuality": int (1-10),
            "explanation": str
        }
        
    Raises:
        OpenAIError: If API calls fail after all retries
        ValueError: If output parsing or validation fails
    """
    if config is None:
        config = EvaluationConfig()
    
    if client is None:
        client = get_openai_client()
    
    if experiment_id is None:
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    prompt = PROMPT_TEMPLATE.format(conversation_log=conversation_log)
    raw_output = None
    last_error = None
    
    # Retry loop for API calls
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
            
            # Log successful evaluation
            _log_evaluation(
                experiment_id=experiment_id,
                conversation_log=conversation_log,
                raw_output=raw_output,
                scores=scores,
                config=config
            )
            
            return scores
            
        except OpenAIError as e:
            last_error = str(e)
            if attempt < config.max_retries - 1:
                wait_time = config.retry_delay * (2 ** attempt)
                print(f"⚠️  API error on attempt {attempt + 1}/{config.max_retries}")
                print(f"   Retrying in {wait_time:.1f}s... ({e})")
                time.sleep(wait_time)
            else:
                print(f"❌ API error: Failed after {config.max_retries} attempts")
                
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            last_error = str(e)
            print(f"❌ Parsing/validation error: {e}")
            if raw_output:
                print(f"Raw output: {raw_output[:300]}...")
            
            # Log failed evaluation
            _log_evaluation(
                experiment_id=experiment_id,
                conversation_log=conversation_log,
                raw_output=raw_output or "",
                scores={},
                config=config,
                error=last_error
            )
            raise
    
    # If we get here, all retries failed
    _log_evaluation(
        experiment_id=experiment_id,
        conversation_log=conversation_log,
        raw_output=raw_output or "",
        scores={},
        config=config,
        error=last_error
    )
    raise OpenAIError(f"Evaluation failed after {config.max_retries} attempts: {last_error}")


def evaluate_with_confidence(
    conversation_log: str,
    n_runs: int = 5,
    config: Optional[EvaluationConfig] = None,
    experiment_id: Optional[str] = None,
    client: Optional[OpenAI] = None
) -> Dict:
    """
    Evaluate conversation multiple times to estimate confidence intervals.
    
    This is the recommended method for research publications, as it provides
    statistical measures of evaluation stability.
    
    Args:
        conversation_log: Full debate transcript
        n_runs: Number of independent evaluation runs
        config: Evaluation configuration
        experiment_id: Base ID for logging (run number appended)
        client: OpenAI client
        
    Returns:
        dict: {
            "coherence_mean": float,
            "coherence_std": float,
            "coherence_scores": list[int],
            "factuality_mean": float,
            "factuality_std": float,
            "factuality_scores": list[int],
            "explanations": list[str],
            "n_runs": int
        }
    """
    if config is None:
        config = EvaluationConfig()
    
    if client is None:
        client = get_openai_client()
    
    if experiment_id is None:
        base_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        base_id = experiment_id
    
    all_scores = []
    print(f"🔄 Running {n_runs} evaluation passes...")
    
    for run_idx in range(n_runs):
        run_id = f"{base_id}_run{run_idx + 1:02d}"
        
        try:
            score = evaluate_conversation(
                conversation_log=conversation_log,
                config=config,
                experiment_id=run_id,
                client=client
            )
            all_scores.append(score)
            print(f"   ✅ Run {run_idx + 1}/{n_runs}: "
                  f"coherence={score['coherence']}, "
                  f"factuality={score['factuality']}")
            
        except Exception as e:
            print(f"   ❌ Run {run_idx + 1}/{n_runs} failed: {e}")
            continue
    
    if not all_scores:
        raise RuntimeError(f"All {n_runs} evaluation runs failed")
    
    # Extract scores
    coherence_scores = [s['coherence'] for s in all_scores]
    factuality_scores = [s['factuality'] for s in all_scores]
    explanations = [s['explanation'] for s in all_scores]
    
    # Compute statistics
    result = {
        "coherence_mean": float(np.mean(coherence_scores)),
        "coherence_std": float(np.std(coherence_scores)),
        "coherence_scores": coherence_scores,
        "factuality_mean": float(np.mean(factuality_scores)),
        "factuality_std": float(np.std(factuality_scores)),
        "factuality_scores": factuality_scores,
        "explanations": explanations,
        "n_runs": len(all_scores),
        "success_rate": len(all_scores) / n_runs
    }
    
    # Log aggregated results
    agg_log_file = os.path.join(config.log_dir, f"{base_id}_aggregated.json")
    with open(agg_log_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Results: coherence={result['coherence_mean']:.2f}±{result['coherence_std']:.2f}, "
          f"factuality={result['factuality_mean']:.2f}±{result['factuality_std']:.2f}")
    
    return result


def batch_evaluate(
    conversations: List[Dict[str, str]],
    n_runs: int = 5,
    config: Optional[EvaluationConfig] = None
) -> List[Dict]:
    """
    Evaluate multiple conversations in batch.
    
    Args:
        conversations: List of dicts with keys "id" and "log"
        n_runs: Number of evaluation runs per conversation
        config: Evaluation configuration
        
    Returns:
        list[dict]: Evaluation results for each conversation
    """
    if config is None:
        config = EvaluationConfig()
    
    client = get_openai_client()
    results = []
    
    print(f"📋 Batch evaluating {len(conversations)} conversations...")
    print(f"   {n_runs} runs per conversation\n")
    
    for idx, conv in enumerate(conversations, 1):
        conv_id = conv.get("id", f"conv_{idx:03d}")
        conv_log = conv["log"]
        
        print(f"[{idx}/{len(conversations)}] Evaluating {conv_id}...")
        
        try:
            result = evaluate_with_confidence(
                conversation_log=conv_log,
                n_runs=n_runs,
                config=config,
                experiment_id=conv_id,
                client=client
            )
            result["conversation_id"] = conv_id
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}\n")
            continue
    
    print(f"\n✅ Batch evaluation complete: {len(results)}/{len(conversations)} succeeded")
    return results


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example 1: Single evaluation
    print("="*70)
    print("EXAMPLE 1: Single Evaluation")
    print("="*70)
    
    dummy_conversation = """
    Agent A: Implementing a carbon tax in Vietnam's textile industry will significantly increase production costs. This could reduce competitiveness against countries without such regulations.
    
    Agent B: While costs may increase initially, the carbon tax incentivizes adoption of cleaner technologies. Long-term benefits include improved sustainability, access to green markets, and reduced environmental damage costs.
    
    Agent A: However, many Vietnamese textile firms are SMEs with limited capital for technology upgrades. The tax burden could force closures or relocation to less regulated regions.
    
    Agent B: That's a valid concern. A phased implementation with support programs—such as tax credits for green investments and technical assistance—could help SMEs transition without severe disruption.
    """
    
    try:
        result = evaluate_conversation(dummy_conversation)
        print(f"\n✅ Single evaluation result:")
        print(f"   Coherence: {result['coherence']}/10")
        print(f"   Factuality: {result['factuality']}/10")
        print(f"   Explanation: {result['explanation']}\n")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}\n")
    
    
    # Example 2: Multi-run evaluation with confidence intervals
    print("="*70)
    print("EXAMPLE 2: Multi-Run Evaluation (Research-Grade)")
    print("="*70)
    
    try:
        result = evaluate_with_confidence(
            conversation_log=dummy_conversation,
            n_runs=3,  # Use 5+ for actual research
            experiment_id="demo_multirun"
        )
        print(f"\n✅ Multi-run evaluation complete!")
        print(f"   Check logs/evaluation/ for detailed results")
    except Exception as e:
        print(f"\n❌ Multi-run evaluation failed: {e}\n")
    
    
    # Example 3: Batch evaluation
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Evaluation")
    print("="*70)
    
    batch_conversations = [
        {
            "id": "debate_001",
            "log": dummy_conversation
        },
        {
            "id": "debate_002",
            "log": "Agent A: Carbon tax is necessary. Agent B: I agree, but implementation matters."
        }
    ]
    
    try:
        batch_results = batch_evaluate(
            conversations=batch_conversations,
            n_runs=2,  # Use 5+ for actual research
        )
        print(f"\n✅ Batch evaluation complete: {len(batch_results)} conversations evaluated")
    except Exception as e:
        print(f"\n❌ Batch evaluation failed: {e}\n")