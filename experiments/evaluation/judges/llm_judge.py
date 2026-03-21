"""
LLM Judge
=========
FIX: client.responses.create() → client.chat.completions.create()
FIX: response.output_text      → response.choices[0].message.content
"""

import os
import json
import re
import time
import datetime
import hashlib
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from experiments.evaluation.judges.base_judge import BaseJudge  # FIX: inherit

project_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(project_root / ".env")


# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class EvaluationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 3.0
    max_conversation_chars: int = 8000
    enable_cache: bool = True
    cache_dir: str = "cache/llm_judge"


# =====================================================================
# PROMPT
# =====================================================================
PROMPT_TEMPLATE = """
You are an independent academic evaluator assessing the quality of a
stakeholder-based policy debate on carbon tax implementation
in Vietnam's textile industry.

The debate involves:
- Government representatives
- Business representatives

You are NOT a participant in the debate.
Assign integer scores from 1 to 10.

STRICT:
- Output ONLY valid JSON
- No markdown, no explanation outside JSON

Format:
{{
  "coherence": <int>,
  "factuality": <int>,
  "explanation": "<max 3 sentences>"
}}

Conversation:
<<<
{conversation_log}
>>>
"""

# =====================================================================
# CACHE
# =====================================================================
class EvaluationCache:
    def __init__(self, cache_dir: str):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text, model, temp):
        return hashlib.md5(f"{text}{model}{temp}".encode()).hexdigest()

    def get(self, text, model, temp):
        path = self.dir / f"{self._key(text, model, temp)}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = datetime.datetime.fromisoformat(data["timestamp"])
        if (datetime.datetime.now() - ts).days < 7:
            return data["scores"]
        return None

    def set(self, text, model, temp, scores):
        path = self.dir / f"{self._key(text, model, temp)}.json"
        path.write_text(
            json.dumps(
                {"timestamp": datetime.datetime.now().isoformat(), "scores": scores},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )


# =====================================================================
# HELPERS
# =====================================================================
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=key)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:2000] + "\n...[TRUNCATED]...\n" + text[-2000:]


def extract_json(text: str) -> Dict:
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    return json.loads(match.group())


def validate(scores: Dict) -> Dict:
    for key in ("coherence", "factuality"):
        if not isinstance(scores.get(key), int) or not 1 <= scores[key] <= 10:
            raise ValueError(f"Invalid score for {key}")
    if not isinstance(scores.get("explanation"), str) or not scores["explanation"].strip():
        raise ValueError("Missing or invalid explanation")
    return scores


# =====================================================================
# CORE  (FIXED API CALL)
# =====================================================================
def evaluate_conversation(
    conversation_log: str,
    config: Optional[EvaluationConfig] = None,
    client: Optional[OpenAI] = None,
) -> Dict:

    config = config or EvaluationConfig()
    client = client or get_openai_client()
    cache  = EvaluationCache(config.cache_dir)
    start  = time.time()

    if config.enable_cache:
        cached = cache.get(conversation_log, config.model, config.temperature)
        if cached:
            cached["_cache_hit"]    = True
            cached["_elapsed_time"] = time.time() - start
            return cached

    conversation_log = truncate(conversation_log, config.max_conversation_chars)
    prompt = PROMPT_TEMPLATE.format(conversation_log=conversation_log)

    last_error = None

    for attempt in range(config.max_retries):
        try:
            # FIX: correct OpenAI Chat Completions API
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a neutral academic evaluator. Output only valid JSON."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=400,
            )

            # FIX: correct response access
            raw_text = response.choices[0].message.content.strip()
            scores   = validate(extract_json(raw_text))

            scores["_cache_hit"]    = False
            scores["_attempts"]     = attempt + 1
            scores["_elapsed_time"] = time.time() - start

            if config.enable_cache:
                cache.set(conversation_log, config.model, config.temperature, scores)

            return scores

        except (OpenAIError, ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
            time.sleep(config.retry_delay * (attempt + 1))

    raise RuntimeError(f"LLM Judge failed after retries: {last_error}")


# =====================================================================
# LLMJudge class (for use with Evaluator)
# =====================================================================
class LLMJudge(BaseJudge):
    """Wrapper class for use with experiments/evaluation/evaluator.py"""
    


    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(name="LLMJudge (GPT-4o-mini)")
        self.config = config or EvaluationConfig()
        self.client = get_openai_client()

    def judge(self, input_data: Dict) -> Dict:
        """
        Accept debate output dict, convert to conversation log, evaluate.
        """
        # Build conversation log from debate history
        if isinstance(input_data, dict):
            history = input_data.get("debate_history", [])
            if history:
                log = "\n".join(
                    f"{h.get('agent', 'Agent')}: {h.get('content', '')}"
                    for h in history
                )
            else:
                # fallback: stringify the whole dict
                log = str(input_data)
        else:
            log = str(input_data)

        return evaluate_conversation(log, self.config, self.client)


# =====================================================================
# DEBUG
# =====================================================================
if __name__ == "__main__":
    dummy = """
    Agent (Government): Carbon tax aligns with national green growth strategy.
    Agent (Business): It increases cost pressure on textile exporters.
    Agent (Expert): Transitional subsidies may mitigate short-term impacts.
    """
    print(evaluate_conversation(dummy))