"""
business_semantic.py — FIXED
==============================
Sửa so với bản cũ:
  1. Giữ lại toàn bộ metadata từ chunker mới:
     document_title, language, context_prefix, para_id, char_len
  2. Bỏ filter len > 200 — chunker mới đã lọc >= 150 rồi,
     filter lại ở đây loại oan ~30% chunk hợp lệ
  3. Forward document_title và language vào output JSON
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ================= CONFIG =================
INPUT_DIR  = Path("data/processed/chunks/business_paragraphs")
OUTPUT_DIR = Path("data/processed/chunks/business_semantic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= RULES =================

FOCUS_RULES = {
    "cost": [
        "cost", "expense", "investment", "capital", "financial burden",
        "profit margin", "revenue",
        "chi phí", "đầu tư", "tài chính", "ngân sách",
        "vốn", "lợi nhuận", "biên lợi nhuận",
        "gánh nặng tài chính", "tốn kém", "phí"
    ],
    "compliance": [
        "compliance", "regulation", "reporting", "requirement",
        "standard", "certification", "audit",
        "tuân thủ", "quy định", "báo cáo", "yêu cầu",
        "tiêu chuẩn", "chứng nhận", "kiểm toán",
        "kiểm kê", "kê khai", "nghĩa vụ pháp lý"
    ],
    "competitiveness": [
        "competitiveness", "competition", "export", "market",
        "trade", "advantage", "market share",
        "cạnh tranh", "xuất khẩu", "thị trường",
        "thương mại", "lợi thế", "thị phần",
        "năng lực cạnh tranh", "hội nhập"
    ],
    "risk": [
        "risk", "penalty", "burden", "threat", "uncertainty",
        "rủi ro", "phạt", "gánh nặng", "mối đe dọa",
        "bất định", "nguy cơ", "tác động tiêu cực"
    ],
    "opportunity": [
        "opportunity", "innovation", "green", "sustainable",
        "growth", "potential", "benefit",
        "cơ hội", "đổi mới", "xanh", "bền vững",
        "tăng trưởng", "tiềm năng", "lợi ích",
        "phát triển bền vững", "công nghệ xanh"
    ],
}

STANCE_CONCERN = [
    "challenge", "burden", "difficult", "costly", "obstacle",
    "concern", "problem", "issue", "barrier", "constraint",
    "impact negatively", "disadvantage",
    "khó khăn", "gánh nặng", "thách thức", "lo ngại",
    "tốn kém", "rào cản", "hạn chế", "bất lợi",
    "áp lực", "ảnh hưởng tiêu cực", "không khả thi",
    "thiếu năng lực", "không đủ"
]

STANCE_SUPPORT = [
    "benefit", "advantage", "support", "enhance", "improve",
    "opportunity", "positive", "promote", "facilitate",
    "lợi ích", "lợi thế", "hỗ trợ", "nâng cao", "cải thiện",
    "cơ hội", "tích cực", "thúc đẩy", "tạo điều kiện",
    "phát triển", "bền vững", "hiệu quả hơn"
]

CBAM_KEYWORDS = [
    "cbam", "carbon border", "carbon border adjustment",
    "eu cbam", "carbon leakage", "embedded carbon",
    "cơ chế điều chỉnh carbon", "thuế carbon biên giới",
    "điều chỉnh biên giới carbon", "cbam của eu",
    "xuất khẩu sang eu", "thị trường eu",
    "quy định carbon", "tín chỉ carbon eu"
]


# ================= TAGGING =================

def infer_semantic(text: str) -> dict:
    t = text.lower()

    # Focus: tối thiểu 2 keyword, lấy top 3
    focus_scores = {
        f: sum(1 for kw in kws if kw in t)
        for f, kws in FOCUS_RULES.items()
    }
    focus = sorted(
        [f for f, s in focus_scores.items() if s >= 2],
        key=lambda f: focus_scores[f], reverse=True
    )[:3]
    if not focus:
        best = max(FOCUS_RULES, key=lambda f: focus_scores[f])
        if focus_scores[best] > 0:
            focus = [best]

    # Stance
    concern_score = sum(1 for kw in STANCE_CONCERN if kw in t)
    support_score = sum(1 for kw in STANCE_SUPPORT if kw in t)
    if concern_score > support_score:
        stance = "concern"
    elif support_score > concern_score:
        stance = "support"
    elif concern_score > 0:
        stance = "concern"
    else:
        stance = "neutral"

    return {
        "role":           "business",
        "stance":         stance,
        "focus":          focus,
        "cbam_relevance": any(kw in t for kw in CBAM_KEYWORDS),
    }


# ================= MAIN =================

def main():
    files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"Tìm thấy {len(files)} file business paragraph")

    total_paragraphs = 0
    stance_counter   = {"concern": 0, "support": 0, "neutral": 0}

    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"❌ Lỗi đọc {file.name}: {e}")
            continue

        # FIX: lấy toàn bộ paragraph object, không filter lại len > 200
        raw_paragraphs = data.get("paragraphs", [])

        semantic_chunks = []
        for para in raw_paragraphs:
            text = para.get("text", "").strip()
            if not text:
                continue

            sem = infer_semantic(text)
            stance_counter[sem["stance"]] = (
                stance_counter.get(sem["stance"], 0) + 1
            )

            # FIX: giữ lại toàn bộ fields từ chunker mới
            semantic_chunks.append({
                "para_id":        para.get("para_id", ""),
                "document_title": para.get("document_title", ""),
                "language":       para.get("language", ""),
                "source_file":    para.get("source_file", file.name),
                "source_type":    para.get("source_type", "BUSINESS"),
                "position":       para.get("position", 0),
                "char_len":       para.get("char_len", len(text)),
                "text":           text,
                "context_prefix": para.get("context_prefix", ""),
                "semantic":       sem,
            })

        #  FIX: forward document_title và language vào output
        output = {
            "source_file":     file.name,
            "document_title":  data.get("document_title", ""),
            "language":        data.get("language", ""),
            "agent":           "business",
            "paragraph_count": len(semantic_chunks),
            "paragraphs":      semantic_chunks,
        }

        out_path = OUTPUT_DIR / file.with_suffix(".json").name
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_paragraphs += len(semantic_chunks)
        logging.info(f" {file.name}: {len(semantic_chunks)} semantic paragraphs")

    logging.info(f"\n DONE — Tổng BUSINESS semantic paragraphs: {total_paragraphs}")
    logging.info(f" Stance distribution: {stance_counter}")


if __name__ == "__main__":
    main()