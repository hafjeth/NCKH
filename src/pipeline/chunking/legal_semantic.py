"""
legal_semantic.py — FIXED
==========================
Sửa so với bản cũ:
  1. Giữ lại field "law" trong output JSON
     (chunker mới thêm field này nhưng bản cũ không forward)
  2. Logic tagging giữ nguyên — đã tốt
"""

from pathlib import Path
import json

# ================== PATH ==================
SRC_DIR = Path("data/processed/chunks/legal_chunks")
OUT_DIR = Path("data/processed/chunks/legal_chunks_semantic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== RULE SET ==================

CLAUSE_TYPE_RULES = [
    ("sanction", [
        "phạt tiền", "xử phạt", "bị xử lý", "mức phạt",
        "tịch thu", "buộc khôi phục", "buộc tháo dỡ",
        "buộc di dời", "buộc nộp lại", "cưỡng chế",
        "xử lý vi phạm", "hành vi vi phạm",
        "phạt cảnh cáo", "đình chỉ hoạt động",
        "tước quyền sử dụng", "trục xuất"
    ]),
    ("obligation", [
        "phải ", "có nghĩa vụ", "bắt buộc",
        "được yêu cầu", "cần phải", "có trách nhiệm thực hiện",
        "phải thực hiện", "phải đảm bảo", "phải tuân thủ",
        "phải báo cáo", "phải nộp", "phải lập",
        "phải duy trì", "phải kiểm tra"
    ]),
    ("responsibility", [
        "chịu trách nhiệm", "chủ trì",
        "có trách nhiệm tổ chức", "có trách nhiệm quản lý",
        "có trách nhiệm hướng dẫn", "có trách nhiệm phối hợp",
        "chịu sự quản lý", "chịu sự giám sát",
        "phối hợp với", "chủ trì, phối hợp"
    ]),
    ("procedure", [
        "trình tự", "thủ tục", "hồ sơ gồm",
        "hồ sơ bao gồm", "nộp hồ sơ", "gửi hồ sơ",
        "thực hiện theo các bước", "bước 1", "bước 2",
        "thông báo cho", "thông báo tới",
        "đăng ký với", "xin phép", "cấp phép",
        "quy trình", "kê khai", "đề nghị cấp"
    ]),
    ("permission", [
        "được phép", "được thực hiện", "được phép thực hiện",
        "có quyền", "được quyền", "được sử dụng",
        "được áp dụng", "được lựa chọn", "được miễn"
    ]),
    ("prohibition", [
        "nghiêm cấm", "không được phép", "bị cấm",
        "cấm", "không được", "không cho phép"
    ]),
    ("definition", [
        "được hiểu là", "là việc", "là quá trình",
        "giải thích từ ngữ", "theo quy định này",
        "được định nghĩa", "có nghĩa là",
        "thuật ngữ", "khái niệm"
    ]),
    ("condition", [
        "trong trường hợp", "trường hợp ",
        "khi đáp ứng", "điều kiện để",
        "nếu ", "trừ khi", "trừ trường hợp",
        "đối với trường hợp", "khi có",
        "khi xảy ra", "khi phát hiện"
    ]),
]

SUBJECT_RULES = {
    "state_agency": [
        "cơ quan", "ủy ban nhân dân", "bộ ", "sở ",
        "cục ", "tổng cục", "phòng ", "ban ",
        "chính phủ", "thủ tướng", "bộ trưởng",
        "hội đồng", "thanh tra", "kiểm tra nhà nước",
        "cơ quan nhà nước", "cơ quan chuyên môn"
    ],
    "enterprise": [
        "doanh nghiệp", "cơ sở sản xuất", "cơ sở kinh doanh",
        "công ty", "tập đoàn", "xí nghiệp",
        "cơ sở", "đơn vị sản xuất", "nhà máy",
        "cơ sở dịch vụ", "chủ đầu tư"
    ],
    "organization": [
        "tổ chức", "hiệp hội", "liên đoàn",
        "trung tâm", "viện ", "trường "
    ],
    "individual": [
        "cá nhân", "người ", "chủ hộ",
        "công dân", "người dân"
    ],
    "household": [
        "hộ gia đình", "hộ dân", "hộ kinh doanh"
    ],
    "producer": [
        "nhà sản xuất", "cơ sở sản xuất",
        "đơn vị sản xuất", "người sản xuất"
    ],
    "importer": [
        "nhập khẩu", "đơn vị nhập khẩu",
        "tổ chức nhập khẩu", "cá nhân nhập khẩu"
    ],
    "unspecified": []
}

DOMAIN_RULES = {
    "water": [
        "nước thải", "nguồn nước", "nước mặt",
        "nước ngầm", "xả thải", "ô nhiễm nước"
    ],
    "air": [
        "khí thải", "không khí", "bụi ", "ô nhiễm không khí",
        "chất lượng không khí", "phát thải khí"
    ],
    "waste": [
        "chất thải", "rác thải", "rác ", "chôn lấp",
        "thu gom rác", "xử lý chất thải", "chất thải rắn",
        "chất thải nguy hại", "phế liệu"
    ],
    "plastic": [
        "nhựa", "bao bì nhựa", "vi nhựa",
        "túi ni lông", "sản phẩm nhựa"
    ],
    "emission": [
        "phát thải", "khí nhà kính", "carbon",
        "co2", "metan", "kiểm kê khí nhà kính",
        "tín chỉ carbon", "thị trường carbon",
        "giảm phát thải", "phát thải ròng"
    ],
    "environmental_permit": [
        "giấy phép môi trường", "giấy chứng nhận môi trường",
        "giấy phép xả thải"
    ],
    "EIA": [
        "đánh giá tác động môi trường", "báo cáo đánh giá",
        "đánh giá môi trường chiến lược", "ĐTM"
    ],
    "recycling": [
        "tái chế", "tái sử dụng", "thu hồi",
        "tái sinh", "kinh tế tuần hoàn"
    ],
}


# ================== TAGGING ==================

def detect_clause_type(text: str) -> str:
    text_l = text.lower()
    for tag, keywords in CLAUSE_TYPE_RULES:
        if any(kw in text_l for kw in keywords):
            return tag
    return "general"


def detect_subjects(text: str):
    text_l = text.lower()
    found = []
    for subject, keywords in SUBJECT_RULES.items():
        if subject == "unspecified":
            continue
        if any(kw in text_l for kw in keywords):
            found.append(subject)
    seen, result = set(), []
    for s in found:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result if result else ["unspecified"]


def detect_domains(text: str):
    text_l = text.lower()
    found = [d for d, kws in DOMAIN_RULES.items()
             if any(kw in text_l for kw in kws)]
    return found if found else ["general_environment"]


# ================== MAIN ==================

def main():
    files = list(SRC_DIR.glob("*_clauses.json"))
    if not files:
        files = list(SRC_DIR.glob("*.json"))

    if not files:
        print(f"❌ Không tìm thấy file JSON trong {SRC_DIR}")
        return

    print(f"📁 Tìm thấy {len(files)} file trong {SRC_DIR}")

    total_chunks = 0
    type_counter = {}
    skipped = 0

    for file in files:
        if file.stat().st_size < 200:
            print(f"   ⚠️  Bỏ qua file rỗng: {file.name}")
            skipped += 1
            continue

        print(f"🏷️  Semantic tagging: {file.name}")

        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   ❌ Lỗi đọc file {file.name}: {e}")
            skipped += 1
            continue

        if not data.get("chunks"):
            print(f"   ⚠️  Không có chunks: {file.name}")
            skipped += 1
            continue

        new_chunks = []
        for chunk in data["chunks"]:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            # Tag thêm vào chunk object — giữ nguyên tất cả fields cũ
            # (article_title, sub_part, char_len... từ chunker mới)
            chunk["clause_type"] = detect_clause_type(text)
            chunk["subjects"]    = detect_subjects(text)
            chunk["domains"]     = detect_domains(text)
            new_chunks.append(chunk)
            type_counter[chunk["clause_type"]] = (
                type_counter.get(chunk["clause_type"], 0) + 1
            )

        out_name = file.name.replace("_clauses", "_semantic")
        if out_name == file.name:
            out_name = file.stem + "_semantic.json"

        out_file = OUT_DIR / out_name
        out_file.write_text(
            json.dumps({
                "source_file": data.get("source_file", file.name),
                "law":         data.get("law", "Unknown"),  #  FIX: forward field law
                "num_chunks":  len(new_chunks),
                "chunks":      new_chunks,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_chunks += len(new_chunks)
        print(f"   {len(new_chunks)} chunks → {out_file.name}")

    print(f"\n Hoàn tất SEMANTIC TAGGING — {total_chunks} chunks")
    if skipped:
        print(f"  Đã bỏ qua {skipped} file rỗng/lỗi")
    print(" Phân phối clause_type:")
    for t, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"   {t:<20}: {cnt}")


if __name__ == "__main__":
    main()