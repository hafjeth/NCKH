"""
Step 2: Tính Precision / Recall / F1
=====================================
Chạy SAU KHI đã gán nhãn thủ công vào samples_to_label.json

Usage:
    python scripts/evaluation/step2_evaluate_accuracy.py

Output:
    experiments/evaluation/ground_truth/accuracy_report.json
    experiments/evaluation/ground_truth/accuracy_report.txt
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "ground_truth"
INPUT_FILE  = GROUND_TRUTH_DIR / "samples_to_label.json"
OUTPUT_JSON = GROUND_TRUTH_DIR / "accuracy_report.json"
OUTPUT_TXT  = GROUND_TRUTH_DIR / "accuracy_report.txt"


# ======================================================
# METRICS
# ======================================================
def exact_match(pred, gold) -> bool:
    """So sánh chính xác cho single-label"""
    return str(pred).strip().lower() == str(gold).strip().lower()


def set_match(pred_list, gold_list):
    """Tính Precision/Recall/F1 cho multi-label"""
    pred = set(str(x).strip().lower() for x in pred_list) if pred_list else set()
    gold = set(str(x).strip().lower() for x in gold_list) if gold_list else set()

    if not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall    = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ======================================================
# LEGAL EVALUATION
# ======================================================
def evaluate_legal(samples: list) -> dict:
    """
    Đánh giá độ chính xác legal semantic tagging:
    - clause_type: exact match
    - subjects: multi-label F1
    - domains: multi-label F1
    """
    labeled = [s for s in samples if s.get("human_clause_type", "").strip()]

    if not labeled:
        return {"error": "Chưa có mẫu nào được gán nhãn thủ công"}

    print(f"Legal: {len(labeled)}/{len(samples)} mẫu đã gán nhãn")

    # clause_type accuracy
    clause_correct = sum(
        1 for s in labeled
        if exact_match(s["auto_clause_type"], s["human_clause_type"])
    )
    clause_accuracy = clause_correct / len(labeled)

    # subjects F1
    subj_scores = [
        set_match(s["auto_subjects"], s["human_subjects"])
        for s in labeled if s.get("human_subjects")
    ]
    subj_p = sum(x["precision"] for x in subj_scores) / len(subj_scores) if subj_scores else 0
    subj_r = sum(x["recall"]    for x in subj_scores) / len(subj_scores) if subj_scores else 0
    subj_f = sum(x["f1"]        for x in subj_scores) / len(subj_scores) if subj_scores else 0

    # domains F1
    dom_scores = [
        set_match(s["auto_domains"], s["human_domains"])
        for s in labeled if s.get("human_domains")
    ]
    dom_p = sum(x["precision"] for x in dom_scores) / len(dom_scores) if dom_scores else 0
    dom_r = sum(x["recall"]    for x in dom_scores) / len(dom_scores) if dom_scores else 0
    dom_f = sum(x["f1"]        for x in dom_scores) / len(dom_scores) if dom_scores else 0

    # Overall F1 (average of all 3)
    overall_f1 = (clause_accuracy + subj_f + dom_f) / 3

    # Per-class clause_type breakdown
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for s in labeled:
        auto  = str(s["auto_clause_type"]).strip().lower()
        human = str(s["human_clause_type"]).strip().lower()
        if auto == human:
            per_class[human]["tp"] += 1
        else:
            per_class[auto]["fp"]  += 1
            per_class[human]["fn"] += 1

    return {
        "n_labeled":        len(labeled),
        "clause_type": {
            "accuracy":   round(clause_accuracy, 4),
            "correct":    clause_correct,
            "per_class":  dict(per_class)
        },
        "subjects": {
            "precision":  round(subj_p, 4),
            "recall":     round(subj_r, 4),
            "f1":         round(subj_f, 4),
        },
        "domains": {
            "precision":  round(dom_p, 4),
            "recall":     round(dom_r, 4),
            "f1":         round(dom_f, 4),
        },
        "overall_f1": round(overall_f1, 4)
    }


# ======================================================
# BUSINESS EVALUATION
# ======================================================
def evaluate_business(samples: list) -> dict:
    """
    Đánh giá độ chính xác business semantic tagging:
    - stance: exact match
    - focus: multi-label F1
    - cbam_relevance: exact match
    """
    labeled = [
        s for s in samples
        if s.get("human_stance", "").strip() or s.get("human_cbam_relevance") is not None
    ]

    if not labeled:
        return {"error": "Chưa có mẫu nào được gán nhãn thủ công"}

    print(f"Business: {len(labeled)}/{len(samples)} mẫu đã gán nhãn")

    # stance accuracy
    stance_labeled = [s for s in labeled if s.get("human_stance", "").strip()]
    stance_correct = sum(
        1 for s in stance_labeled
        if exact_match(s["auto_stance"], s["human_stance"])
    )
    stance_accuracy = stance_correct / len(stance_labeled) if stance_labeled else 0

    # focus F1
    focus_labeled = [s for s in labeled if s.get("human_focus")]
    focus_scores = [
        set_match(s["auto_focus"], s["human_focus"])
        for s in focus_labeled
    ]
    focus_p = sum(x["precision"] for x in focus_scores) / len(focus_scores) if focus_scores else 0
    focus_r = sum(x["recall"]    for x in focus_scores) / len(focus_scores) if focus_scores else 0
    focus_f = sum(x["f1"]        for x in focus_scores) / len(focus_scores) if focus_scores else 0

    # cbam_relevance accuracy
    cbam_labeled = [s for s in labeled if s.get("human_cbam_relevance") is not None]
    cbam_correct = sum(
        1 for s in cbam_labeled
        if bool(s["auto_cbam_relevance"]) == bool(s["human_cbam_relevance"])
    )
    cbam_accuracy = cbam_correct / len(cbam_labeled) if cbam_labeled else 0

    overall_f1 = (stance_accuracy + focus_f + cbam_accuracy) / 3

    return {
        "n_labeled":     len(labeled),
        "stance": {
            "accuracy":  round(stance_accuracy, 4),
            "correct":   stance_correct,
            "n":         len(stance_labeled)
        },
        "focus": {
            "precision": round(focus_p, 4),
            "recall":    round(focus_r, 4),
            "f1":        round(focus_f, 4),
            "n":         len(focus_labeled)
        },
        "cbam_relevance": {
            "accuracy":  round(cbam_accuracy, 4),
            "correct":   cbam_correct,
            "n":         len(cbam_labeled)
        },
        "overall_f1": round(overall_f1, 4)
    }


# ======================================================
# MAIN
# ======================================================
def main():
    if not INPUT_FILE.exists():
        print(f"File không tồn tại: {INPUT_FILE}")
        print("Chạy step1 trước: python scripts/evaluation/step1_create_samples.py")
        return

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    legal_samples    = data.get("legal_samples", [])
    business_samples = data.get("business_samples", [])

    print("=" * 60)
    print("ĐÁNH GIÁ ĐỘ CHÍNH XÁC SEMANTIC TAGGING")
    print("=" * 60)

    legal_result    = evaluate_legal(legal_samples)
    business_result = evaluate_business(business_samples)

    # ---- REPORT ----
    report = {
        "legal_semantic_tagging":    legal_result,
        "business_semantic_tagging": business_result
    }

    # Save JSON
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Save TXT report
    lines = []
    lines.append("SEMANTIC TAGGING ACCURACY REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("LEGAL SEMANTIC TAGGING")
    lines.append("-" * 40)
    if "error" not in legal_result:
        ct = legal_result["clause_type"]
        su = legal_result["subjects"]
        do = legal_result["domains"]
        lines.append(f"Samples labeled   : {legal_result['n_labeled']}")
        lines.append(f"Clause type accuracy: {ct['accuracy']*100:.1f}%  ({ct['correct']}/{legal_result['n_labeled']})")
        lines.append(f"Subjects  P/R/F1  : {su['precision']*100:.1f}% / {su['recall']*100:.1f}% / {su['f1']*100:.1f}%")
        lines.append(f"Domains   P/R/F1  : {do['precision']*100:.1f}% / {do['recall']*100:.1f}% / {do['f1']*100:.1f}%")
        lines.append(f"Overall F1        : {legal_result['overall_f1']*100:.1f}%")
    else:
        lines.append(f"  {legal_result['error']}")

    lines.append("")
    lines.append("BUSINESS SEMANTIC TAGGING")
    lines.append("-" * 40)
    if "error" not in business_result:
        st = business_result["stance"]
        fo = business_result["focus"]
        cb = business_result["cbam_relevance"]
        lines.append(f"Samples labeled    : {business_result['n_labeled']}")
        lines.append(f"Stance accuracy    : {st['accuracy']*100:.1f}%  ({st['correct']}/{st['n']})")
        lines.append(f"Focus     P/R/F1   : {fo['precision']*100:.1f}% / {fo['recall']*100:.1f}% / {fo['f1']*100:.1f}%")
        lines.append(f"CBAM accuracy      : {cb['accuracy']*100:.1f}%  ({cb['correct']}/{cb['n']})")
        lines.append(f"Overall F1         : {business_result['overall_f1']*100:.1f}%")
    else:
        lines.append(f"  {business_result['error']}")

    report_text = "\n".join(lines)
    OUTPUT_TXT.write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    print()
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()