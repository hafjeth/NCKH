"""
legal_semantic.py — IMPROVED
=============================
ĐẶT FILE NÀY TẠI: src/pipeline/chunking/legal_semantic.py

CẢI THIỆN SO VỚI BẢN CŨ:
  1. CLAUSE TYPE: Thêm nhiều keyword, fix priority (sanction > condition),
     "general" chỉ dùng khi không khớp gì khác
  2. SUBJECTS: Thêm keyword, detect tất cả subjects không bỏ sót,
     fix "tổ chức, cá nhân" → gán cả hai
  3. DOMAINS: Thêm keyword emission/air liên quan carbon
  4. Priority logic: sanction > obligation > responsibility > procedure
     > permission > prohibition > definition > condition > general
"""

from pathlib import Path
import json
import re

# ================== PATH ==================
SRC_DIR = Path("data/processed/chunks/legal_chunks")
OUT_DIR = Path("data/processed/chunks/legal_chunks_semantic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== RULE SET (CẢI THIỆN) ==================

# Thứ tự ưu tiên: index nhỏ hơn = ưu tiên cao hơn
# Giải quyết vấn đề "general" chiếm 28/48 lỗi
CLAUSE_TYPE_RULES = [
    # 1. SANCTION — phải check trước CONDITION vì "phạt tiền nếu..."
    #    vừa có điều kiện vừa có chế tài → ưu tiên sanction
    ("sanction", [
        "phạt tiền", "xử phạt", "bị xử lý", "mức phạt",
        "tịch thu", "buộc khôi phục", "buộc tháo dỡ",
        "buộc di dời", "buộc nộp lại", "cưỡng chế",
        "xử lý vi phạm", "hành vi vi phạm",
        "phạt cảnh cáo", "đình chỉ hoạt động",
        "tước quyền sử dụng", "trục xuất"
    ]),

    # 2. OBLIGATION — "phải" xuất hiện rất nhiều
    ("obligation", [
        "phải ", "có nghĩa vụ", "bắt buộc",
        "được yêu cầu", "cần phải", "có trách nhiệm thực hiện",
        "phải thực hiện", "phải đảm bảo", "phải tuân thủ",
        "phải báo cáo", "phải nộp", "phải lập",
        "phải duy trì", "phải kiểm tra"
    ]),

    # 3. RESPONSIBILITY — phân biệt với obligation:
    #    responsibility = ai chịu trách nhiệm/chủ trì
    #    obligation = phải làm gì
    ("responsibility", [
        "chịu trách nhiệm", "chủ trì",
        "có trách nhiệm tổ chức", "có trách nhiệm quản lý",
        "có trách nhiệm hướng dẫn", "có trách nhiệm phối hợp",
        "chịu sự quản lý", "chịu sự giám sát",
        "phối hợp với", "chủ trì, phối hợp"
    ]),

    # 4. PROCEDURE — trình tự, thủ tục
    ("procedure", [
        "trình tự", "thủ tục", "hồ sơ gồm",
        "hồ sơ bao gồm", "nộp hồ sơ", "gửi hồ sơ",
        "thực hiện theo các bước", "bước 1", "bước 2",
        "thông báo cho", "thông báo tới",
        "đăng ký với", "xin phép", "cấp phép",
        "quy trình", "kê khai", "đề nghị cấp"
    ]),

    # 5. PERMISSION — được phép làm gì
    ("permission", [
        "được phép", "được thực hiện", "được phép thực hiện",
        "có quyền", "được quyền", "được sử dụng",
        "được áp dụng", "được lựa chọn", "được miễn"
    ]),

    # 6. PROHIBITION — nghiêm cấm
    ("prohibition", [
        "nghiêm cấm", "không được phép", "bị cấm",
        "cấm", "không được", "không cho phép"
    ]),

    # 7. DEFINITION — định nghĩa thuật ngữ
    ("definition", [
        "được hiểu là", "là việc", "là quá trình",
        "giải thích từ ngữ", "theo quy định này",
        "được định nghĩa", "có nghĩa là",
        "thuật ngữ", "khái niệm"
    ]),

    # 8. CONDITION — điều kiện, trường hợp
    #    (sau sanction để tránh nhầm "phạt X nếu..." → condition)
    ("condition", [
        "trong trường hợp", "trường hợp ",
        "khi đáp ứng", "điều kiện để",
        "nếu ", "trừ khi", "trừ trường hợp",
        "đối với trường hợp", "khi có",
        "khi xảy ra", "khi phát hiện"
    ]),
]
# "general" là fallback cuối cùng — không có trong list


# SUBJECTS — thêm nhiều keyword, detect TẤT CẢ không bỏ sót
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
    "unspecified": []  # fallback
}

# DOMAINS — thêm keyword carbon/emission liên quan NCKH
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


# ================== TAGGING FUNCTIONS (CẢI THIỆN) ==================

def detect_clause_type(text: str) -> str:
    """
    Priority-based detection — không dùng general trừ khi không khớp gì.
    Thứ tự: sanction > obligation > responsibility > procedure >
            permission > prohibition > definition > condition > general
    """
    text_l = text.lower()
    for tag, keywords in CLAUSE_TYPE_RULES:
        if any(kw in text_l for kw in keywords):
            return tag
    return "general"


def detect_subjects(text: str):
    """
    Detect TẤT CẢ subjects — không bỏ sót.
    Fix: "tổ chức, cá nhân" → gán cả organization lẫn individual.
    """
    text_l = text.lower()
    found = []
    for subject, keywords in SUBJECT_RULES.items():
        if subject == "unspecified":
            continue
        if any(kw in text_l for kw in keywords):
            found.append(subject)

    # Loại bỏ trùng lặp, giữ thứ tự
    seen = set()
    result = []
    for s in found:
        if s not in seen:
            seen.add(s)
            result.append(s)

    return result if result else ["unspecified"]


def detect_domains(text: str):
    """
    Detect tất cả domains liên quan.
    """
    text_l = text.lower()
    found = [
        d for d, kws in DOMAIN_RULES.items()
        if any(kw in text_l for kw in kws)
    ]
    return found if found else ["general_environment"]


# ================== MAIN ==================
def main():
    # Tìm cả *_clauses.json lẫn *.json (phòng trường hợp tên file khác)
    files = list(SRC_DIR.glob("*_clauses.json"))
    if not files:
        files = list(SRC_DIR.glob("*.json"))

    if not files:
        print(f"❌ Không tìm thấy file JSON trong {SRC_DIR}")
        print("   → Hãy chạy legal_chunker.py trước")
        return

    print(f"📁 Tìm thấy {len(files)} file trong {SRC_DIR}")

    total_chunks = 0
    type_counter = {}
    skipped = 0

    for file in files:
        # Bỏ qua file rỗng hoặc quá nhỏ (< 200 bytes = không có chunk thật)
        if file.stat().st_size < 200:
            print(f"   ⚠️  Bỏ qua file rỗng: {file.name} ({file.stat().st_size} bytes)")
            skipped += 1
            continue

        print(f"🏷️  Semantic tagging: {file.name}")

        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   ❌ Lỗi đọc file {file.name}: {e}")
            skipped += 1
            continue

        # Bỏ qua nếu không có chunks hoặc chunks rỗng
        if not data.get("chunks"):
            print(f"   ⚠️  Không có chunks: {file.name}")
            skipped += 1
            continue

        new_chunks = []

        for chunk in data["chunks"]:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            clause_type = detect_clause_type(text)
            subjects    = detect_subjects(text)
            domains     = detect_domains(text)

            chunk["clause_type"] = clause_type
            chunk["subjects"]    = subjects
            chunk["domains"]     = domains

            new_chunks.append(chunk)
            type_counter[clause_type] = type_counter.get(clause_type, 0) + 1

        # Tên output: thay _clauses → _semantic, hoặc thêm _semantic
        out_name = file.name.replace("_clauses", "_semantic")
        if out_name == file.name:
            out_name = file.stem + "_semantic.json"

        out_file = OUT_DIR / out_name
        out_file.write_text(
            json.dumps({
                "source_file": data.get("source_file", file.name),
                "num_chunks":  len(new_chunks),
                "chunks":      new_chunks
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_chunks += len(new_chunks)
        print(f"   ✅ {len(new_chunks)} chunks → {out_file.name}")

    print(f"\n🎯 Hoàn tất SEMANTIC TAGGING — {total_chunks} chunks")
    if skipped:
        print(f"⚠️  Đã bỏ qua {skipped} file rỗng/lỗi")
    print("📊 Phân phối clause_type:")
    for t, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"   {t:<20}: {cnt}")


if __name__ == "__main__":
    main()