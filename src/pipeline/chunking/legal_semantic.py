from pathlib import Path
import json
import re

# ================== PATH ==================
SRC_DIR = Path("data/processed/legal_chunks")
OUT_DIR = Path("data/processed/legal_chunks_semantic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== RULE SET ==================
CLAUSE_TYPE_RULES = {
    "definition": [
        "được hiểu là", "là việc", "là quá trình", "giải thích từ ngữ"
    ],
    "obligation": [
        "phải", "có trách nhiệm", "có nghĩa vụ"
    ],
    "prohibition": [
        "nghiêm cấm", "không được phép", "bị cấm"
    ],
    "permission": [
        "được phép", "được thực hiện"
    ],
    "procedure": [
        "trình tự", "thủ tục", "thực hiện theo", "hồ sơ gồm"
    ],
    "condition": [
        "điều kiện", "trường hợp", "khi đáp ứng"
    ],
    "responsibility": [
        "chịu trách nhiệm", "có trách nhiệm tổ chức"
    ],
    "sanction": [
        "xử phạt", "bị xử lý", "mức phạt"
    ]
}

SUBJECT_RULES = {
    "state_agency": ["cơ quan", "ủy ban", "bộ", "sở"],
    "enterprise": ["doanh nghiệp", "cơ sở sản xuất", "cơ sở kinh doanh"],
    "organization": ["tổ chức"],
    "individual": ["cá nhân"],
    "household": ["hộ gia đình"],
    "producer": ["nhà sản xuất"],
    "importer": ["nhập khẩu"]
}

DOMAIN_RULES = {
    "water": ["nước thải", "nguồn nước"],
    "air": ["khí thải", "không khí"],
    "waste": ["chất thải", "rác"],
    "plastic": ["nhựa", "bao bì nhựa", "vi nhựa"],
    "emission": ["phát thải", "khí nhà kính"],
    "environmental_permit": ["giấy phép môi trường"],
    "EIA": ["đánh giá tác động môi trường"],
    "recycling": ["tái chế", "tái sử dụng"]
}

# ================== TAGGING FUNCTIONS ==================
def detect_clause_type(text: str) -> str:
    text_l = text.lower()
    for tag, kws in CLAUSE_TYPE_RULES.items():
        if any(kw in text_l for kw in kws):
            return tag
    return "general"


def detect_subjects(text: str):
    text_l = text.lower()
    subjects = [
        s for s, kws in SUBJECT_RULES.items()
        if any(kw in text_l for kw in kws)
    ]
    return subjects or ["unspecified"]


def detect_domains(text: str):
    text_l = text.lower()
    domains = [
        d for d, kws in DOMAIN_RULES.items()
        if any(kw in text_l for kw in kws)
    ]
    return domains or ["general_environment"]


# ================== MAIN ==================
def main():
    files = list(SRC_DIR.glob("*_clauses.json"))

    if not files:
        print("❌ Không tìm thấy file clause chunk")
        return

    for file in files:
        print(f"🏷️ Semantic tagging: {file.name}")

        data = json.loads(file.read_text(encoding="utf-8"))
        new_chunks = []

        for chunk in data["chunks"]:
            text = chunk["text"]

            chunk["clause_type"] = detect_clause_type(text)
            chunk["subjects"] = detect_subjects(text)
            chunk["domains"] = detect_domains(text)

            new_chunks.append(chunk)

        out_file = OUT_DIR / file.name.replace("_clauses", "_semantic")
        out_file.write_text(
            json.dumps({
                "source_file": data["source_file"],
                "num_chunks": len(new_chunks),
                "chunks": new_chunks
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"✅ {len(new_chunks)} chunks → {out_file}")

    print("\n🎯 Hoàn tất SEMANTIC TAGGING")


if __name__ == "__main__":
    main()
