"""
run_baseline.py
===============
Baseline runner: Single-agent (no debate, no expert synthesis)
Dùng để so sánh với multi-agent system trong analyze_results.py

Usage:
    python experiments/run_baseline.py --lang vi
"""

import sys
import csv
import json
import time
import datetime
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.base_agent import BaseAgent
from src.knowledge.retrieval.retriever import KnowledgeRetriever
from experiments.evaluation.judges.llm_judge import evaluate_conversation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================
# SINGLE AGENT SETUP
# ======================================================
BASELINE_ROLE = """
You are a neutral policy analyst specializing in carbon tax policy
and the Vietnamese textile and garment industry.

Your task is to provide a comprehensive, evidence-based analysis
of the given policy question.

Requirements:
- Academic tone
- Evidence-based reasoning
- Cover multiple perspectives (government, business, environmental)
- 300–400 words
- Cite relevant legal documents using format: [Nguồn: <tên văn bản>]
- Minimum 2 citations per response
"""


def load_questions(lang="vi"):
    question_file = PROJECT_ROOT / "experiments" / "benchmarks" / f"questions_{lang}.json"
    if not question_file.exists():
        logger.error(f"Missing question file: {question_file}")
        return []
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f).get("questions", [])


def run_single_baseline(question_text: str, question_id: int, agent: BaseAgent):
    logger.info(f"Running baseline for question {question_id}")
    start_time = time.time()

    try:
        prompt = f"""
POLICY QUESTION:
{question_text}

Provide a comprehensive analysis covering:
1. Legal/regulatory context
2. Economic implications
3. Environmental considerations
4. Key stakeholder perspectives
5. Policy recommendations

Remember to cite relevant documents using [Nguồn: <tên văn bản>]
"""
        response = agent.chat(prompt)
        elapsed = time.time() - start_time

        # Metrics
        word_count      = len(response.split())
        total_citations = response.count("[Nguồn:")

        # LLM Judge evaluation
        logger.info(f"Evaluating baseline Q{question_id}")
        evaluation = evaluate_conversation(response)

        return {
            "status":          "success",
            "id":              question_id,
            "question_id":     question_id,
            "question":        question_text,
            "response":        response,
            "word_count":      word_count,
            "total_citations": total_citations,
            "diversity_score": 0.0,          # N/A for single agent
            "coherence":       evaluation.get("coherence", 0),
            "factuality":      evaluation.get("factuality", 0),
            "time_seconds":    round(elapsed, 2),
            "timestamp":       datetime.datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in baseline Q{question_id}: {e}")
        return {
            "status":      "error",
            "id":          question_id,
            "question_id": question_id,
            "question":    question_text,
            "error":       str(e),
            "timestamp":   datetime.datetime.now().isoformat()
        }


def save_csv(results: list, path: Path):
    success_results = [r for r in results if r["status"] == "success"]
    if not success_results:
        return

    fieldnames = [
        "id", "question_id", "question", "word_count", "total_citations",
        "diversity_score", "coherence", "factuality", "time_seconds",
        "status", "timestamp"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(success_results)

    logger.info(f"CSV saved → {path}")


def run_baseline(lang="vi"):
    questions = load_questions(lang)
    if not questions:
        return

    # Initialize single agent with RAG
    retriever = KnowledgeRetriever()
    agent = BaseAgent(
        name="Baseline Agent",
        role=BASELINE_ROLE,
        retriever=retriever
    )
    logger.info("✅ Baseline agent initialized")

    results = []

    for i, q in enumerate(questions, 1):
        logger.info("=" * 60)
        logger.info(f"Baseline Q{i}/{len(questions)}: {q.get('question','')[:60]}...")
        logger.info("=" * 60)

        result = run_single_baseline(
            question_text=q.get("question", ""),
            question_id=q.get("id", i),
            agent=agent
        )
        results.append(result)

        # Save intermediate
        with open(OUTPUT_DIR / f"baseline_intermediate_{i}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # Save full JSON
    output = {
        "metadata": {
            "timestamp":       datetime.datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful":      sum(r["status"] == "success" for r in results),
            "failed":          sum(r["status"] == "error"   for r in results),
            "language":        lang,
            "mode":            "single-agent baseline"
        },
        "results": results
    }

    json_path = OUTPUT_DIR / f"baseline_results_{lang}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved → {json_path}")

    # Save CSV (for analyze_results.py)
    csv_path = OUTPUT_DIR / "baseline.csv"
    save_csv(results, csv_path)

    successful = output["metadata"]["successful"]
    logger.info(f"\n✅ Baseline completed: {successful}/{len(questions)} successful")

    # Print quick summary
    success_results = [r for r in results if r["status"] == "success"]
    if success_results:
        avg_words = sum(r["word_count"] for r in success_results) / len(success_results)
        avg_cit   = sum(r["total_citations"] for r in success_results) / len(success_results)
        avg_coh   = sum(r["coherence"] for r in success_results) / len(success_results)
        avg_fact  = sum(r["factuality"] for r in success_results) / len(success_results)
        avg_time  = sum(r["time_seconds"] for r in success_results) / len(success_results)

        print("\n" + "=" * 50)
        print("📊 BASELINE SUMMARY")
        print("=" * 50)
        print(f"  Success rate : {successful}/{len(questions)} (100%)")
        print(f"  Avg words    : {avg_words:.1f}")
        print(f"  Avg citations: {avg_cit:.1f}")
        print(f"  Coherence    : {avg_coh:.2f}/10")
        print(f"  Factuality   : {avg_fact:.2f}/10")
        print(f"  Avg time     : {avg_time:.1f}s")
        print("=" * 50)
        print(f"💾 Results saved: experiments/results/baseline.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single-agent baseline benchmark")
    parser.add_argument("--lang",   default="vi", choices=["vi", "en"])
    parser.add_argument("--single", type=int, default=None,
                        help="Run only 1 question by ID (for testing)")
    args = parser.parse_args()

    if args.single is not None:
        # Test mode: chạy 1 câu
        qs = load_questions(args.lang)
        q  = next((x for x in qs if x.get("id") == args.single), None)
        if q:
            retriever = KnowledgeRetriever()
            agent = BaseAgent(name="Baseline Agent", role=BASELINE_ROLE, retriever=retriever)
            res = run_single_baseline(q["question"], q["id"], agent)
            out = OUTPUT_DIR / f"baseline_single_{args.single}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved → {out}")
            print(f"\nWord count : {res.get('word_count', 0)}")
            print(f"Citations  : {res.get('total_citations', 0)}")
            print(f"Coherence  : {res.get('coherence', 0)}/10")
            print(f"Factuality : {res.get('factuality', 0)}/10")
        else:
            logger.error("Question not found")
    else:
        run_baseline(args.lang)