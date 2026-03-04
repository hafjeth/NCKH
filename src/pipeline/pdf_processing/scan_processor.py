from pathlib import Path
import json
import pytesseract
from pdf2image import convert_from_path
from cleaning import clean_text, extract_metadata

RAW_DIR = Path("data/raw_pdfs/pdf_scan")
OUT_DIR = Path("data/processed_text/ocr_pdf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf in RAW_DIR.glob("*.pdf"):
    print(f"📸 OCR PDF: {pdf.name}")

    images = convert_from_path(pdf, dpi=300)
    raw_text = ""

    for img in images:
        raw_text += pytesseract.image_to_string(img, lang="vie+eng")

    if len(raw_text.strip()) < 100:
        print("⚠️ OCR quá ít text – skip")
        continue

    cleaned = clean_text(raw_text)
    metadata = extract_metadata(raw_text, pdf.name)

    (OUT_DIR / f"{pdf.stem}.txt").write_text(cleaned, encoding="utf-8")
    (OUT_DIR / f"{pdf.stem}.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
