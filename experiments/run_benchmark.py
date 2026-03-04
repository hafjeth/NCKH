"""
run_benchmark.py

Benchmark runner for multi-agent policy debate system.
Research-grade, reproducible, and evaluation-ready.
"""

import sys
from pathlib import Path
import time
import json
import datetime
import logging

# ======================================================
# PATH SETUP
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================
# IMPORTS
# ======================================================

from src.core.debate_manager import DebateManager
from src.evaluation.llm_judge import evaluate_conversation

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================================================
# CONFIG
# ======================================================

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 2

# ======================================================
# LOAD QUESTIONS
# ======================================================

def load_questions(lang="vi"):
    question_file = PROJECT_ROOT / "experiments" / "benchmarks" / f"questions_{lang}.json"
    if not question_file.exists():
        logger.error(f"Missing question file: {question_file}")
        return []

    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f).get("questions", [])

# ======================================================
# RUN SINGLE DEBATE
# ======================================================

def run_single_debate(question_text: str, question_id: int):
    logger.info(f"Running debate for question {question_id}")
    start_time = time.time()

    try:
        manager = DebateManager()

        final_summary, history = manager.run_debate(
            topic=question_text,
            max_rounds=N_ROUNDS
        )

        elapsed = time.time() - start_time

        # Format full conversation
        conversation_log = "\n\n".join(
            f"[{h['agent'].upper()}]\n{h['content']}"
            for h in history
        )

        logger.info(f"Evaluating conversation for Q{question_id}")
        evaluation = evaluate_conversation(conversation_log)

        return {
            "status": "success",
            "question_id": question_id,
            "question": question_text,
            "conversation": conversation_log,
            "final_summary": final_summary,
            "evaluation": evaluation,
            "time_seconds": round(elapsed, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in debate Q{question_id}: {e}")
        return {
            "status": "error",
            "question_id": question_id,
            "question": question_text,
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

# ======================================================
# RUN FULL BENCHMARK
# ======================================================

def run_full_benchmark(lang="vi"):
    questions = load_questions(lang)
    if not questions:
        return

    results = []

    for i, q in enumerate(questions, 1):
        logger.info("=" * 60)
        logger.info(f"Question {i}/{len(questions)}")
        logger.info("=" * 60)

        result = run_single_debate(
            question_text=q.get("question", ""),
            question_id=q.get("id", i)
        )

        results.append(result)

        # Save intermediate result
        with open(OUTPUT_DIR / f"intermediate_{i}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    output = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful": sum(r["status"] == "success" for r in results),
            "failed": sum(r["status"] == "error" for r in results),
            "language": lang,
            "rounds": N_ROUNDS
        },
        "results": results
    }

    out_path = OUTPUT_DIR / f"benchmark_results_{lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Benchmark completed → {out_path}")

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent debate benchmark")
    parser.add_argument("--lang", default="vi", choices=["vi", "en"])
    parser.add_argument("--single", type=int, default=None)

    args = parser.parse_args()

    if args.single is not None:
        qs = load_questions(args.lang)
        q = next((x for x in qs if x.get("id") == args.single), None)

        if q:
            res = run_single_debate(q["question"], q["id"])
            with open(OUTPUT_DIR / f"single_{args.single}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        else:
            logger.error("Question not found")
    else:
        run_full_benchmark(args.lang)
