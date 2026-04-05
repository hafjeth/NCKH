"""
legal_chunker.py — FIXED v9
==========================
Sửa so với v8:
  - [FIX-20] Thêm is_toc_line() để lọc mục lục trước khi chunk
  - [FIX-21] Sửa ARTICLE_RE chỉ bắt "Điều X." ở đầu dòng
  - [FIX-22] Lọc bỏ chunk có article_title rác (chứa nhiều "khoản", "Điều")
  - [FIX-23] Tăng MIN_CLAUSE_LEN lên 100 cho legal documents
  - [FIX-24] Sửa lỗi regex look-behind trong CLAUSE_RE
"""

from pathlib import Path
import re
import json

# ================== PATH CONFIG ==================
SRC_DIR = Path("data/processed/normalized")
OUT_DIR = Path("data/processed/chunks/legal_chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== CHUNK SIZE LIMITS ==================
MAX_CHUNK_CHARS = 3000
MIN_CLAUSE_LEN  = 100   # tăng từ 80 lên 100

# ================== LAW NAME MAP ==================
LAW_PATTERNS = [
    ("luật bảo vệ môi trường 2020",    "Luật Bảo vệ Môi trường 2020"),
    ("nghị định 06 2022",              "Nghị định 06/2022/NĐ-CP"),
    ("nghị định 08 2022",              "Nghị định 08/2022/NĐ-CP"),
    ("nghị định 45 2022",              "Nghị định 45/2022/NĐ-CP"),
    ("thông tư 02 2022",               "Thông tư 02/2022/TT-BTNMT"),
    ("thông tư 17 2022",               "Thông tư 17/2022/TT-BTNMT"),
    ("quyết định 01 2022",             "Quyết định 01/2022/QĐ-TTg"),
    ("quyết định 148",                 "Quyết định 148/QĐ-TTg"),
    ("quyết định 232",                 "Quyết định 232/QĐ-TTg"),
    ("quyết định 448",                 "Quyết định 448/QĐ-TTg"),
    ("quyết định 450",                 "Quyết định 450/QĐ-TTg"),
    ("quyết định 888",                 "Quyết định 888/QĐ-TTg"),
    ("quyết định 896",                 "Quyết định 896/QĐ-TTg"),
    ("quyết định 1055",                "Quyết định 1055/QĐ-TTg"),
    ("quyết định 1658",                "Quyết định 1658/QĐ-TTg"),
    ("quyết định 1973",                "Quyết định 1973/QĐ-TTg"),
]

def detect_law_name(filename: str) -> str:
    name_lower = filename.lower()
    for pattern, law_name in LAW_PATTERNS:
        if pattern in name_lower:
            return law_name
    return f"Unknown ({filename})"


# ================== REGEX ==================
# [FIX-21] Chỉ bắt "Điều X." ở đầu dòng
ARTICLE_RE = re.compile(r"^Điều\s+(\d+)\.?\s*(.*?)(?=^Điều\s+\d+\.?|\Z)", re.MULTILINE | re.DOTALL)
# [FIX-24] Sửa lỗi look-behind: dùng (?:^|\n) thay vì (?<=\n|^)
CLAUSE_RE = re.compile(r"(?:^|\n)\s*(\d+)\.\s+", re.MULTILINE)
CHAPTER_RE = re.compile(r"^Chương\s+[IVXLC]+", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+\d+", re.IGNORECASE)


# ================== TOC FILTER [FIX-20] ==================
def is_toc_line(line: str) -> bool:
    """Phát hiện dòng mục lục: 'Điều 14; khoản 4 Điều 15;' hoặc 'khoản 2 Điều 24;'"""
    line = line.strip()
    if not line:
        return False
    # Dòng chỉ gồm "Điều X; khoản Y Điều Z; ..."
    toc_pattern = re.compile(r'^(?:Điều\s+\d+[.;]?\s*|khoản\s+\d+\s+Điều\s+\d+[.;]?\s*)+$')
    if toc_pattern.match(line):
        return True
    # Dòng chỉ gồm "khoản 2 Điều 24, khoản 3 Điều 25" (dấu phẩy)
    toc_pattern2 = re.compile(r'^(?:khoản\s+\d+\s+Điều\s+\d+[,;]?\s*)+$')
    return bool(toc_pattern2.match(line))


def is_garbage_article_title(title: str) -> bool:
    """Phát hiện article_title rác (chứa nhiều 'Điều' hoặc 'khoản')"""
    title = title.strip()
    # Nếu title chứa nhiều hơn 2 từ "Điều" → rác
    if len(re.findall(r'Điều', title)) > 1:
        return True
    # Nếu title chứa "khoản" và không bắt đầu bằng "Điều" → rác
    if 'khoản' in title.lower() and not title.startswith('Điều'):
        return True
    return False


# ================== CORE FUNCTIONS ==================

def split_articles(text: str):
    """Tách văn bản thành các điều, trả về list (article_num, article_title, article_body)"""
    articles = []
    # Tìm tất cả các điều
    for match in ARTICLE_RE.finditer(text):
        article_num = int(match.group(1))
        # Tách title và body
        remaining = match.group(2).strip()
        # Dòng đầu tiên sau số điều là title
        lines = remaining.split('\n')
        title = lines[0].strip() if lines else ""
        body = '\n'.join(lines[1:]).strip()
        articles.append((article_num, title, body))
    return articles


def split_clauses(article_text: str):
    """Tách thân điều thành các khoản dựa trên pattern '1. '"""
    clauses = []
    # Tìm vị trí của tất cả các khoản
    parts = CLAUSE_RE.split(article_text)
    
    if len(parts) <= 1:
        # Không có khoản nào → coi là khoản 0
        return [(0, article_text.strip())]
    
    # parts[0] là text trước khoản đầu tiên (thường là rỗng hoặc preamble)
    for i in range(1, len(parts), 2):
        clause_num = int(parts[i])
        clause_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        clauses.append((clause_num, clause_text))
    
    return clauses


def is_structural_noise(line: str) -> bool:
    return bool(CHAPTER_RE.match(line) or SECTION_RE.match(line))


def split_long_chunk(text: str, max_chars: int = MAX_CHUNK_CHARS):
    """Nếu chunk quá dài, tách theo câu"""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.;!?])\s+', text)
    parts, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        parts.append(current.strip())

    # Fallback: cắt cứng nếu vẫn còn câu quá dài
    result = []
    for part in parts:
        if len(part) > max_chars:
            for start in range(0, len(part), max_chars):
                result.append(part[start:start + max_chars])
        else:
            result.append(part)
    return result


# ================== MAIN ==================

def main():
    txt_files = list(SRC_DIR.glob("*.txt"))

    if not txt_files:
        print("Không có file normalized text trong", SRC_DIR)
        return

    legal_keywords = ["luật", "nghị định", "thông tư", "quyết định"]
    legal_files = [
        f for f in txt_files
        if any(kw in f.name.lower() for kw in legal_keywords)
    ]

    if not legal_files:
        print("Không tìm thấy file legal nào")
        return

    print(f"Tìm thấy {len(legal_files)} file legal")

    for txt_file in legal_files:
        law_name = detect_law_name(txt_file.stem)
        print(f"\n{txt_file.name}")
        print(f"  └─ Tên văn bản: {law_name}")

        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")

        # [FIX-20] Lọc bỏ dòng mục lục trước khi xử lý
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if is_toc_line(stripped):
                continue  # bỏ qua dòng mục lục
            if not is_structural_noise(stripped):
                lines.append(line)
        text = "\n".join(lines)

        articles = split_articles(text)

        if not articles:
            print("  Không tìm thấy Điều nào — bỏ qua")
            continue

        chunks = []
        chunk_id = 1
        toc_skipped = 0

        for article_num, art_title, art_body in articles:
            # [FIX-22] Bỏ qua article_title rác
            if is_garbage_article_title(art_title):
                toc_skipped += 1
                continue

            # Nếu article_body quá ngắn, có thể là heading, bỏ qua
            if len(art_body) < MIN_CLAUSE_LEN:
                continue

            clauses = split_clauses(art_body)

            for clause_num, clause_text in clauses:
                if len(clause_text) < MIN_CLAUSE_LEN:
                    continue

                sub_texts = split_long_chunk(clause_text)

                for sub_idx, sub_text in enumerate(sub_texts):
                    if len(sub_text) < MIN_CLAUSE_LEN:
                        continue

                    chunks.append({
                        "chunk_id":      chunk_id,
                        "law":           law_name,
                        "source_file":   txt_file.name,
                        "article":       article_num,
                        "article_title": art_title[:200],  # giới hạn độ dài
                        "clause":        clause_num,
                        "sub_part":      sub_idx,
                        "text":          sub_text,
                        "char_len":      len(sub_text),
                    })
                    chunk_id += 1

        if toc_skipped > 0:
            print(f"  Đã bỏ qua {toc_skipped} mục lục rác")

        if not chunks:
            print("  Không có chunk nào sau khi lọc — bỏ qua")
            continue

        out_path = OUT_DIR / f"{txt_file.stem}_clauses.json"
        out_path.write_text(
            json.dumps({
                "source_file": txt_file.name,
                "law":         law_name,
                "num_chunks":  len(chunks),
                "chunks":      chunks,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        long_chunks = [c for c in chunks if c["char_len"] > MAX_CHUNK_CHARS]
        print(f"  {len(chunks)} chunks → {out_path.name}")
        if long_chunks:
            print(f"    ⚠️ {len(long_chunks)} chunk > {MAX_CHUNK_CHARS} chars")

    print("\n✅ Hoàn tất chunking văn bản pháp luật")


if __name__ == "__main__":
    main()