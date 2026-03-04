import json
import unicodedata
from pathlib import Path

import pdfplumber

RAW_PDF_DIR = Path("data/raw/pdfs")
REPORT_PATH = Path("data/raw/pdf_type_report.json")

# ---------- helper functions ----------

def count_vietnamese_chars(text: str) -> int:
    vietnamese_chars = "ăâđêôơưĂÂĐÊÔƠƯáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    return sum(1 for c in text if c in vietnamese_chars)


def analyze_text_quality(text: str):
    total_chars = len(text)
    if total_chars == 0:
        return {
            "non_ascii_ratio": 0,
            "vietnamese_ratio": 0
        }

    non_ascii = sum(1 for c in text if ord(c) > 127)
    viet_chars = count_vietnamese_chars(text)

    return {
        "non_ascii_ratio": round(non_ascii / total_chars, 3),
        "vietnamese_ratio": round(viet_chars / total_chars, 3)
    }


# ---------- main processing ----------

def classify_pdf(pdf_path: Path):
    total_text = ""
    pages_with_text = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:5]:  # chỉ đọc 5 trang đầu là đủ
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_with_text += 1
                    total_text += page_text
    except Exception as e:
        return {
            "type": "error",
            "error": str(e)
        }

    total_length = len(total_text)
    avg_chars_per_page = round(
        total_length / pages_with_text, 2
    ) if pages_with_text > 0 else 0

    quality = analyze_text_quality(total_text)

    # ---- classification rules ----
    if total_length == 0:
        pdf_type = "scan_pdf"
    elif avg_chars_per_page < 300:
        pdf_type = "low_quality_text_pdf"
    elif quality["non_ascii_ratio"] > 0.3:
        pdf_type = "embedded_font_pdf"
    else:
        pdf_type = "clean_text_pdf"

    return {
        "type": pdf_type,
        "total_text_length": total_length,
        "pages_with_text": pages_with_text,
        "avg_chars_per_page": avg_chars_per_page,
        **quality
    }


def main():
    results = {
        "clean_text_pdf": [],
        "embedded_font_pdf": [],
        "scan_pdf": [],
        "low_quality_text_pdf": [],
        "details": {}
    }

    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        info = classify_pdf(pdf)
        pdf_type = info["type"]

        if pdf_type in results:
            results[pdf_type].append(pdf.name)

        results["details"][pdf.name] = info

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ PDF classification completed.")
    for k in ["clean_text_pdf", "embedded_font_pdf", "scan_pdf", "low_quality_text_pdf"]:
        print(f"{k}: {len(results[k])}")
    print(f"📄 Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
