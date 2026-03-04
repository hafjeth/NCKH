from pathlib import Path
import json
import fitz
from cleaning import clean_text, extract_metadata

RAW_DIR = Path("data/raw_pdfs/pdf_text")
OUT_DIR = Path("data/processed_text/text_pdf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text

for pdf in RAW_DIR.glob("*.pdf"):
    print(f"📄 TEXT PDF: {pdf.name}")
    text = extract_text(pdf)

    cleaned = clean_text(text)
    metadata = extract_metadata(text, pdf.name)

    (OUT_DIR / f"{pdf.stem}.txt").write_text(cleaned, encoding="utf-8")
    (OUT_DIR / f"{pdf.stem}.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
