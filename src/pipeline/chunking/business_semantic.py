"""
business_semantic.py — IMPROVED
=================================
ĐẶT FILE NÀY TẠI: src/pipeline/chunking/business_semantic.py

CẢI THIỆN SO VỚI BẢN CŨ:
  1. Thêm keyword tiếng Việt — bản cũ chỉ có tiếng Anh → bỏ sót
  2. Stance: thêm nhiều từ biểu thị concern/support rõ hơn
  3. Focus: thêm keyword tiếng Việt cho cost, compliance, risk...
  4. CBAM: thêm các từ liên quan CBAM tiếng Việt
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================
INPUT_DIR  = Path("data/processed/chunks/business_paragraphs")
OUTPUT_DIR = Path("data/processed/chunks/business_semantic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= IMPROVED RULES =================

FOCUS_RULES = {
    "cost": [
        # tiếng Anh
        "cost", "expense", "investment", "capital", "financial burden",
        "profit margin", "revenue",
        # tiếng Việt
        "chi phí", "đầu tư", "tài chính", "ngân sách",
        "vốn", "lợi nhuận", "biên lợi nhuận",
        "gánh nặng tài chính", "tốn kém", "phí"
    ],
    "compliance": [
        # tiếng Anh
        "compliance", "regulation", "reporting", "requirement",
        "standard", "certification", "audit",
        # tiếng Việt
        "tuân thủ", "quy định", "báo cáo", "yêu cầu",
        "tiêu chuẩn", "chứng nhận", "kiểm toán",
        "kiểm kê", "kê khai", "nghĩa vụ pháp lý"
    ],
    "competitiveness": [
        # tiếng Anh
        "competitiveness", "competition", "export", "market",
        "trade", "advantage", "market share",
        # tiếng Việt
        "cạnh tranh", "xuất khẩu", "thị trường",
        "thương mại", "lợi thế", "thị phần",
        "năng lực cạnh tranh", "hội nhập"
    ],
    "risk": [
        # tiếng Anh
        "risk", "penalty", "burden", "threat", "uncertainty",
        # tiếng Việt
        "rủi ro", "phạt", "gánh nặng", "mối đe dọa",
        "bất định", "nguy cơ", "tác động tiêu cực"
    ],
    "opportunity": [
        # tiếng Anh
        "opportunity", "innovation", "green", "sustainable",
        "growth", "potential", "benefit",
        # tiếng Việt
        "cơ hội", "đổi mới", "xanh", "bền vững",
        "tăng trưởng", "tiềm năng", "lợi ích",
        "phát triển bền vững", "công nghệ xanh"
    ],
}

STANCE_CONCERN = [
    # tiếng Anh
    "challenge", "burden", "difficult", "costly", "obstacle",
    "concern", "problem", "issue", "barrier", "constraint",
    "impact negatively", "disadvantage",
    # tiếng Việt
    "khó khăn", "gánh nặng", "thách thức", "lo ngại",
    "tốn kém", "rào cản", "hạn chế", "bất lợi",
    "áp lực", "ảnh hưởng tiêu cực", "không khả thi",
    "thiếu năng lực", "không đủ"
]

STANCE_SUPPORT = [
    # tiếng Anh
    "benefit", "advantage", "support", "enhance", "improve",
    "opportunity", "positive", "promote", "facilitate",
    # tiếng Việt
    "lợi ích", "lợi thế", "hỗ trợ", "nâng cao", "cải thiện",
    "cơ hội", "tích cực", "thúc đẩy", "tạo điều kiện",
    "phát triển", "bền vững", "hiệu quả hơn"
]

CBAM_KEYWORDS = [
    # tiếng Anh
    "cbam", "carbon border", "carbon border adjustment",
    "eu cbam", "carbon leakage", "embedded carbon",
    # tiếng Việt
    "cơ chế điều chỉnh carbon", "thuế carbon biên giới",
    "điều chỉnh biên giới carbon", "cbam của eu",
    "xuất khẩu sang eu", "thị trường eu",
    "quy định carbon", "tín chỉ carbon eu"
]


# ================= HEURISTIC SEMANTIC TAG (CẢI THIỆN) =================

def infer_semantic(paragraph: str) -> dict:
    text = paragraph.lower()

    # Focus — yêu cầu tối thiểu 2 keyword khớp, lấy top 3
    focus_scores = {}
    for f, kws in FOCUS_RULES.items():
        score = sum(1 for kw in kws if kw in text)
        if score >= 2:
            focus_scores[f] = score
    # Lấy top 3 focus có score cao nhất
    focus = sorted(focus_scores, key=focus_scores.get, reverse=True)[:3]
    # Nếu không có gì khớp đủ 2, lấy label có score cao nhất
    if not focus:
        best = max(FOCUS_RULES, key=lambda f: sum(1 for kw in FOCUS_RULES[f] if kw in text))
        if any(kw in text for kw in FOCUS_RULES[best]):
            focus = [best]

    # Stance — concern ưu tiên hơn support nếu cả hai xuất hiện
    concern_score = sum(1 for kw in STANCE_CONCERN if kw in text)
    support_score = sum(1 for kw in STANCE_SUPPORT if kw in text)

    if concern_score > support_score:
        stance = "concern"
    elif support_score > concern_score:
        stance = "support"
    elif concern_score > 0:
        stance = "concern"   # tie → concern (conservative)
    else:
        stance = "neutral"

    # CBAM — thêm keyword tiếng Việt
    cbam_related = any(kw in text for kw in CBAM_KEYWORDS)

    return {
        "role":           "business",
        "stance":         stance,
        "focus":          focus,
        "cbam_relevance": cbam_related,
    }


# ================= MAIN =================

def main():
    files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"📁 Found {len(files)} business paragraph files")

    total_paragraphs = 0
    stance_counter = {"concern": 0, "support": 0, "neutral": 0}

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        paragraphs = [
            p["text"].strip()
            for p in data.get("paragraphs", [])
            if len(p.get("text", "").strip()) > 200
        ]

        semantic_chunks = []
        for p in paragraphs:
            sem = infer_semantic(p)
            semantic_chunks.append({
                "text":     p,
                "semantic": sem,
            })
            stance_counter[sem["stance"]] = \
                stance_counter.get(sem["stance"], 0) + 1

        output = {
            "source_file":     file.name,
            "agent":           "business",
            "paragraph_count": len(semantic_chunks),
            "paragraphs":      semantic_chunks,
        }

        out_path = OUTPUT_DIR / file.with_suffix(".json").name
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_paragraphs += len(semantic_chunks)
        logging.info(f"✅ {file.name}: {len(semantic_chunks)} semantic paragraphs")

    logging.info(f"🎉 DONE — Total BUSINESS semantic paragraphs: {total_paragraphs}")
    logging.info(f"📊 Stance distribution: {stance_counter}")


if __name__ == "__main__":
    main()