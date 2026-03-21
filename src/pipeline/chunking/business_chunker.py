import json
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================
NORMALIZED_DIR = Path("data/processed/normalized")
OUTPUT_DIR     = Path("data/processed/chunks/business_paragraphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_PARAGRAPH_LENGTH = 50

# ================= UTILS =================
def is_heading(text: str) -> bool:
    """
    Heuristic: coi là heading nếu:
    - Toàn chữ hoa
    - Hoặc rất ngắn và không có dấu chấm
    """
    text = text.strip()
    if len(text) < 40 and "." not in text:
        return True
    if text.isupper():
        return True
    return False


def split_paragraphs(text: str):
    """
    Tách paragraph theo 2 dòng trống trở lên
    """
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]


# ================= MAIN =================
def main():
    business_files = []

    # Chỉ lấy file BUSINESS (đã được gán nhãn trước đó)
    for txt_file in NORMALIZED_DIR.glob("*.txt"):
        name = txt_file.name.lower()
        if any(key in name for key in [
            "luật", "nghị định", "thông tư", "quyết định"
        ]):
            continue
        business_files.append(txt_file)

    logging.info(f"Found {len(business_files)} BUSINESS files")

    total_paragraphs = 0

    for txt_file in business_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_paragraphs(text)

        para_chunks = []
        para_counter = 1

        for p in paragraphs:
            if len(p) < MIN_PARAGRAPH_LENGTH:
                continue
            if is_heading(p):
                continue

            para_chunks.append({
                "para_id": f"{txt_file.stem}_{para_counter:03d}",
                "text": p,
                "position": para_counter
            })
            para_counter += 1

        if not para_chunks:
            logging.warning(f" No valid paragraphs: {txt_file.name}")
            continue

        output = {
            "source_file": txt_file.name,
            "source_type": "BUSINESS",
            "paragraphs": para_chunks
        }

        out_path = OUTPUT_DIR / f"{txt_file.stem}_paragraphs.json"
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_paragraphs += len(para_chunks)
        logging.info(f" {txt_file.name}: {len(para_chunks)} paragraphs")

    logging.info(f"DONE – Total BUSINESS paragraphs: {total_paragraphs}")


# ================= RUN =================
if __name__ == "__main__":
    main()