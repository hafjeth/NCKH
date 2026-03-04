from pathlib import Path
import re
import json

# ================== PATH CONFIG ==================
SRC_DIR = Path("data/processed/normalized_text")
OUT_DIR = Path("data/processed/legal_chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAW_NAME = "Nghị định 08/2022/NĐ-CP"

# ================== REGEX ==================
ARTICLE_RE = re.compile(r"(Điều\s+\d+\.?.*)", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\n\s*(\d+)\s*[.\)]\s+", re.MULTILINE)

CHAPTER_RE = re.compile(r"^Chương\s+[IVXLC]+", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+\d+", re.IGNORECASE)


# ================== CORE FUNCTIONS ==================
def split_articles(text: str):
    parts = ARTICLE_RE.split(text)
    articles = []

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        articles.append((title, body.strip()))

    return articles


def extract_article_number(title: str) -> int:
    m = re.search(r"Điều\s+(\d+)", title)
    return int(m.group(1)) if m else -1


def split_clauses(article_text: str):
    splits = CLAUSE_RE.split(article_text)

    clauses = []
    for i in range(1, len(splits), 2):
        clause_num = int(splits[i])
        clause_text = splits[i + 1].strip()
        clauses.append((clause_num, clause_text))

    return clauses


def is_structural_noise(line: str) -> bool:
    return bool(
        CHAPTER_RE.match(line)
        or SECTION_RE.match(line)
    )


# ================== MAIN ==================
def main():
    txt_files = list(SRC_DIR.glob("*.txt"))

    if not txt_files:
        print("❌ Không có file normalized text")
        return

    for txt_file in txt_files:
        print(f"📄 Chunking clauses: {txt_file.name}")

        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")

        # remove chapter / section headers
        lines = [
            l for l in raw_text.splitlines()
            if not is_structural_noise(l.strip())
        ]
        text = "\n".join(lines)

        articles = split_articles(text)

        chunks = []
        chunk_id = 1

        for art_title, art_body in articles:
            article_num = extract_article_number(art_title)
            clauses = split_clauses(art_body)

            for clause_num, clause_text in clauses:
                if len(clause_text) < 30:
                    continue

                chunks.append({
                    "chunk_id": chunk_id,
                    "law": LAW_NAME,
                    "source_file": txt_file.name,
                    "article": article_num,
                    "clause": clause_num,
                    "text": clause_text
                })
                chunk_id += 1

        out_path = OUT_DIR / f"{txt_file.stem}_clauses.json"
        out_path.write_text(
            json.dumps({
                "source_file": txt_file.name,
                "num_chunks": len(chunks),
                "chunks": chunks
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"✅ {len(chunks)} clauses → {out_path}")

    print("\n🎯 Hoàn tất chunking theo KHOẢN")


if __name__ == "__main__":
    main()
