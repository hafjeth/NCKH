"""
run_benchmark.py
================
ĐẶT FILE NÀY TẠI: experiments/run_benchmark.py  (thay file cũ)

THAY ĐỔI SO VỚI BẢN CŨ:
  1. run_debate() trả về 3 giá trị → unpack đúng
  2. Lưu tier3_output vào intermediate JSON và full JSON
  3. CSV có thêm 3 cột Tier 3 flat (policy_options_count, overall_risk, recommended_option)
  4. --single mode in preview Tier 3 ra console
  5. N_ROUNDS = 3 (đủ để Tier 3 có ngữ cảnh tốt)
"""

import sys
import csv
from pathlib import Path
import time
import json
import datetime
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.debate_manager import DebateManager
from experiments.evaluation.judges.llm_judge import evaluate_conversation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 3


def load_questions(lang="vi"):
    question_file = PROJECT_ROOT / "experiments" / "benchmarks" / f"questions_{lang}.json"
    if not question_file.exists():
        logger.error(f"Missing question file: {question_file}")
        return []
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f).get("questions", [])


def run_single_debate(question_text: str, question_id: int):
    logger.info(f"Running debate for question {question_id}")
    start_time = time.time()

    try:
        manager = DebateManager()

        # ── THAY ĐỔI: unpack 3 giá trị (thêm tier3_output) ──────────────────
        expert_text, history, tier3_output = manager.run_debate(
            topic=question_text,
            max_rounds=N_ROUNDS,
        )

        elapsed = time.time() - start_time

        conversation_log = "\n\n".join(
            f"[{h['agent'].upper()}]\n{h['content']}"
            for h in history
        )

        logger.info(f"Evaluating conversation for Q{question_id}")
        evaluation = evaluate_conversation(conversation_log)

        word_count      = len(conversation_log.split())
        total_citations = (
            conversation_log.count("[Nguồn:") +
            conversation_log.count("[Source:")
        )

        agent_texts = [h["content"] for h in history if h["agent"] != "Moderator"]
        all_words   = [set(t.lower().split()) for t in agent_texts]
        if len(all_words) >= 2:
            inter = all_words[0] & all_words[1]
            union = all_words[0] | all_words[1]
            diversity_score = round(1 - len(inter) / max(len(union), 1), 4)
        else:
            diversity_score = 0.0

        # Lấy final_summary từ Moderator turn cuối
        final_summary = ""
        for turn in reversed(history):
            if turn.get("agent") == "Moderator":
                final_summary = turn.get("content", "")
                break

        return {
            "status":          "success",
            "question_id":     question_id,
            "question":        question_text,
            "conversation":    conversation_log,
            "final_summary":   final_summary,
            "evaluation":      evaluation,
            "word_count":      word_count,
            "total_citations": total_citations,
            "diversity_score": diversity_score,
            "coherence":       evaluation.get("coherence", 0),
            "factuality":      evaluation.get("factuality", 0),
            "time_seconds":    round(elapsed, 2),
            "timestamp":       datetime.datetime.now().isoformat(),
            # ── NEW: Tier 3 output ──────────────────────────────────────────
            "tier3_output":    tier3_output,
        }

    except Exception as e:
        logger.error(f"Error in debate Q{question_id}: {e}", exc_info=True)
        return {
            "status":      "error",
            "question_id": question_id,
            "question":    question_text,
            "error":       str(e),
            "timestamp":   datetime.datetime.now().isoformat(),
        }


def save_csv(results: list, path: Path):
    """Lưu CSV flat — bao gồm 3 cột Tier 3 summary"""
    success_results = [r for r in results if r["status"] == "success"]
    if not success_results:
        return

    fieldnames = [
        "question_id", "question", "word_count", "total_citations",
        "diversity_score", "coherence", "factuality", "time_seconds",
        "status", "timestamp",
        # Tier 3 flat summary
        "tier3_policy_options_count",
        "tier3_overall_risk",
        "tier3_recommended_option",
    ]

    rows = []
    for r in success_results:
        t3  = r.get("tier3_output", {})
        row = {k: r.get(k, "") for k in fieldnames if k in r}
        row["status"]    = r.get("status", "")
        row["timestamp"] = r.get("timestamp", "")

        opts = t3.get("policy_options", {}).get("options", [])
        row["tier3_policy_options_count"] = len(opts)

        impact = t3.get("impact_analysis", {})
        row["tier3_overall_risk"] = impact.get("overall_risk", "")

        rec = t3.get("decision_support", {}).get("recommended_option", {})
    if isinstance(rec, list):
        rec = rec[0] if rec else {}
        row["tier3_recommended_option"] = rec.get("option_id", "")

        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV saved → {path}")


def _print_tier3_preview(t3: dict):
    """In Tier 3 preview ra console sau mỗi câu hỏi."""
    print("\n" + "─" * 55)
    print("  TIER 3 PREVIEW")
    print("─" * 55)

    opts = t3.get("policy_options", {}).get("options", [])
    print(f"  Policy Options ({len(opts)}):")
    for o in opts:
        print(f"    [{o.get('id','?')}] {o.get('title','')} "
              f"— feasibility: {o.get('feasibility','?')}")

    impact = t3.get("impact_analysis", {})
    print(f"  Overall Risk  : {impact.get('overall_risk', 'N/A')}")

    rec = t3.get("decision_support", {}).get("recommended_option", {})
    if isinstance(rec, list):
        rec = rec[0] if rec else {}
    rationale = rec.get("rationale", "")[:90]
    print(f"  Recommended   : {rec.get('option_id','?')} — {rationale}...")

    gaps = t3.get("evidence", {}).get("evidence_gaps", [])
    if gaps:
        print(f"  Evidence Gaps : {gaps[0]}")
    print("─" * 55 + "\n")


def run_full_benchmark(lang="vi"):
    questions = load_questions(lang)
    if not questions:
        return

    results = []

    for i, q in enumerate(questions, 1):
        logger.info("=" * 60)
        logger.info(f"Question {i}/{len(questions)}: {q.get('question','')[:60]}...")
        logger.info("=" * 60)

        result = run_single_debate(
            question_text=q.get("question", ""),
            question_id=q.get("id", i),
        )
        results.append(result)

        # Lưu intermediate JSON (bao gồm tier3_output đầy đủ)
        with open(OUTPUT_DIR / f"intermediate_{i}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if result["status"] == "success":
            _print_tier3_preview(result.get("tier3_output", {}))

    # Full JSON
    output = {
        "metadata": {
            "timestamp":       datetime.datetime.now().isoformat(),
            "total_questions": len(questions),
            "successful":      sum(r["status"] == "success" for r in results),
            "failed":          sum(r["status"] == "error"   for r in results),
            "language":        lang,
            "rounds":          N_ROUNDS,
            "tier3_enabled":   True,
        },
        "results": results,
    }

    json_path = OUTPUT_DIR / f"benchmark_results_{lang}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved → {json_path}")

    csv_path = OUTPUT_DIR / "multiagent_results.csv"
    save_csv(results, csv_path)

    logger.info(
        f"\n✅ Benchmark completed: "
        f"{output['metadata']['successful']}/{len(questions)} successful"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent debate benchmark")
    parser.add_argument("--lang",   default="vi", choices=["vi", "en"])
    parser.add_argument("--single", type=int, default=None,
                        help="Test 1 câu hỏi theo ID")
    args = parser.parse_args()

    if args.single is not None:
        qs = load_questions(args.lang)
        q  = next((x for x in qs if x.get("id") == args.single), None)
        if q:
            res = run_single_debate(q["question"], q["id"])
            out = OUTPUT_DIR / f"single_{args.single}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved → {out}")
            if res["status"] == "success":
                _print_tier3_preview(res.get("tier3_output", {}))
        else:
            logger.error("Question not found")
    else:
        run_full_benchmark(args.lang)