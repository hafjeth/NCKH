"""
run_benchmark.py
================
ĐẶT FILE NÀY TẠI: experiments/run_benchmark.py  (thay file cũ)

THAY ĐỔI SO VỚI BẢN CŨ:
  1. run_debate() trả về 3 giá trị → unpack đúng
  2. Lưu tier3_output vào intermediate JSON và full JSON
  3. CSV có thêm 3 cột Tier 3 flat (policy_options_count, overall_risk, recommended_option)
  4. --single mode in preview Tier 3 ra console
  5. N_ROUNDS đọc từ Config.BENCHMARK_ROUNDS (không hardcode)

FIXES SO VỚI BẢN LỖI:
  [FIX-1] save_csv(): sửa indentation sai khiến tier3_recommended_option không được gán
  [FIX-2] save_csv(): sửa rows.append(row) nằm ngoài vòng lặp for → chỉ lưu 1 row cuối
  [FIX-3] diversity_score: tính trên toàn bộ các vòng debate thay vì chỉ 2 agent đầu
  [FIX-4] N_ROUNDS: đọc từ Config thay vì hardcode = 3
  [FIX-5] run_full_benchmark(): thêm progress tracking và error summary cuối
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

# [FIX-4] Đọc N_ROUNDS từ Config thay vì hardcode
try:
    from config.settings import Config
    N_ROUNDS = Config.BENCHMARK_ROUNDS
except Exception:
    N_ROUNDS = 3  # fallback an toàn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# LOAD QUESTIONS
# ══════════════════════════════════════════════════════════════

def load_questions(lang: str = "vi") -> list:
    question_file = (
        PROJECT_ROOT / "experiments" / "benchmarks" / f"questions_{lang}.json"
    )
    if not question_file.exists():
        logger.error(f"Missing question file: {question_file}")
        return []
    with open(question_file, "r", encoding="utf-8") as f:
        return json.load(f).get("questions", [])


# ══════════════════════════════════════════════════════════════
# SINGLE DEBATE
# ══════════════════════════════════════════════════════════════

def run_single_debate(question_text: str, question_id: int) -> dict:
    logger.info(f"Running debate for question {question_id}")
    start_time = time.time()

    try:
        manager = DebateManager()

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

        word_count = len(conversation_log.split())
        total_citations = (
            conversation_log.count("[Nguồn:")
            + conversation_log.count("[Source:")
        )

        # [FIX-3] Tính diversity_score trên TẤT CẢ các turn của agent
        # (không chỉ 2 agent đầu tiên)
        diversity_score = _compute_diversity(history)

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


# ══════════════════════════════════════════════════════════════
# DIVERSITY SCORE  [FIX-3]
# ══════════════════════════════════════════════════════════════

def _compute_diversity(history: list) -> float:
    """
    Tính diversity_score dựa trên TẤT CẢ các turn của Government Agent
    và Enterprise Agent qua toàn bộ các vòng — không chỉ vòng đầu.

    Trả về: 1 - (Jaccard similarity trung bình giữa gov và biz)
    Giá trị càng cao = ngôn ngữ càng khác nhau = tranh luận càng đa dạng.
    """
    gov_words = set()
    biz_words = set()

    for turn in history:
        agent = turn.get("agent", "")
        words = set(turn.get("content", "").lower().split())
        if "Government" in agent:
            gov_words |= words
        elif "Enterprise" in agent or "Business" in agent:
            biz_words |= words

    if not gov_words or not biz_words:
        return 0.0

    intersection = gov_words & biz_words
    union = gov_words | biz_words
    jaccard = len(intersection) / max(len(union), 1)
    return round(1 - jaccard, 4)


# ══════════════════════════════════════════════════════════════
# SAVE CSV  [FIX-1] [FIX-2]
# ══════════════════════════════════════════════════════════════

def save_csv(results: list, path: Path) -> None:
    """
    Lưu CSV flat — bao gồm 3 cột Tier 3 summary.

    [FIX-1] tier3_recommended_option được gán đúng ngoài if block
    [FIX-2] rows.append(row) nằm đúng trong vòng lặp for
    """
    success_results = [r for r in results if r.get("status") == "success"]
    if not success_results:
        logger.warning("save_csv: Không có kết quả thành công để lưu.")
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

    for r in success_results:                           # [FIX-2] rows.append ở đây
        t3  = r.get("tier3_output", {}) or {}
        row = {k: r.get(k, "") for k in fieldnames if k in r}
        row["status"]    = r.get("status", "")
        row["timestamp"] = r.get("timestamp", "")

        # Tier 3 — policy options count
        opts = t3.get("policy_options", {}).get("options", [])
        if not isinstance(opts, list):
            opts = []
        row["tier3_policy_options_count"] = len(opts)

        # Tier 3 — overall risk
        impact = t3.get("impact_analysis", {}) or {}
        row["tier3_overall_risk"] = impact.get("overall_risk", "")

        # [FIX-1] Gán recommended_option NGOÀI if block
        rec = t3.get("decision_support", {}).get("recommended_option", {}) or {}
        if isinstance(rec, list):
            rec = rec[0] if rec else {}
        row["tier3_recommended_option"] = rec.get("option_id", "")  # luôn được gán

        rows.append(row)                                # [FIX-2] nằm đúng trong for loop

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV saved → {path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════
# TIER 3 PREVIEW
# ══════════════════════════════════════════════════════════════

def _print_tier3_preview(t3: dict) -> None:
    """In Tier 3 preview ra console sau mỗi câu hỏi."""
    if not t3:
        print("  [Tier 3] Không có output.")
        return

    print("\n" + "─" * 55)
    print("  TIER 3 PREVIEW")
    print("─" * 55)

    opts = t3.get("policy_options", {}).get("options", [])
    if not isinstance(opts, list):
        opts = []
    print(f"  Policy Options ({len(opts)}):")
    for o in opts:
        print(
            f"    [{o.get('id', '?')}] {o.get('title', '')} "
            f"— feasibility: {o.get('feasibility', '?')}"
        )

    impact = t3.get("impact_analysis", {}) or {}
    print(f"  Overall Risk  : {impact.get('overall_risk', 'N/A')}")

    rec = t3.get("decision_support", {}).get("recommended_option", {}) or {}
    if isinstance(rec, list):
        rec = rec[0] if rec else {}
    rationale = rec.get("rationale", "")[:90]
    print(f"  Recommended   : {rec.get('option_id', '?')} — {rationale}...")

    gaps = t3.get("evidence", {}).get("evidence_gaps", [])
    if gaps:
        print(f"  Evidence Gaps : {gaps[0]}")

    # Hiển thị lỗi tier3 nếu có
    for section in ["policy_options", "pros_cons", "impact_analysis", "evidence"]:
        sec = t3.get(section, {})
        if isinstance(sec, dict) and "_error" in sec:
            print(f"  ⚠️  {section}: {sec['_error']}")

    print("─" * 55 + "\n")


# ══════════════════════════════════════════════════════════════
# FULL BENCHMARK  [FIX-5]
# ══════════════════════════════════════════════════════════════

def run_full_benchmark(lang: str = "vi") -> None:
    questions = load_questions(lang)
    if not questions:
        return

    results      = []
    failed_ids   = []
    total        = len(questions)

    logger.info(f"Bắt đầu benchmark: {total} câu hỏi | {N_ROUNDS} vòng/câu")

    for i, q in enumerate(questions, 1):
        logger.info("=" * 60)
        logger.info(f"Question {i}/{total}: {q.get('question', '')[:60]}...")
        logger.info("=" * 60)

        result = run_single_debate(
            question_text=q.get("question", ""),
            question_id=q.get("id", i),
        )
        results.append(result)

        if result["status"] == "error":
            failed_ids.append(result["question_id"])

        # Lưu intermediate JSON ngay sau mỗi câu (an toàn khi crash giữa chừng)
        intermediate_path = OUTPUT_DIR / f"intermediate_{i}.json"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if result["status"] == "success":
            _print_tier3_preview(result.get("tier3_output", {}))

    # ── Full JSON ────────────────────────────────────────────────────────────
    n_success = sum(r["status"] == "success" for r in results)
    n_failed  = sum(r["status"] == "error"   for r in results)

    output = {
        "metadata": {
            "timestamp":       datetime.datetime.now().isoformat(),
            "total_questions": total,
            "successful":      n_success,
            "failed":          n_failed,
            "failed_ids":      failed_ids,   # thêm để dễ debug
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

    # [FIX-5] In summary cuối rõ ràng hơn
    logger.info("\n" + "=" * 60)
    logger.info(f" Benchmark hoàn thành")
    logger.info(f"   Thành công : {n_success}/{total}")
    logger.info(f"   Thất bại   : {n_failed}/{total}")
    if failed_ids:
        logger.info(f"   ID lỗi     : {failed_ids}")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

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
                logger.error(f"Debate failed: {res.get('error')}")
        else:
            logger.error(f"Question ID {args.single} không tìm thấy.")
    else:
        run_full_benchmark(args.lang)