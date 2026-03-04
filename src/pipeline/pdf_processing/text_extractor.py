from pathlib import Path
import fitz  # PyMuPDF

# ===== PATH CONFIG =====
PDF_DIR = Path("data/intermediate/pdf_classified/text_pdf")
OUT_DIR = Path("data/intermediate/extracted_text/from_text_pdf")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF with text layer using PyMuPDF
    """
    doc = fitz.open(pdf_path)
    texts = []

    for page in doc:
        text = page.get_text("text")
        if text:
            texts.append(text)

    doc.close()
    return "\n".join(texts).strip()


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ Không tìm thấy PDF nào trong text_pdf")
        return

    success = 0
    empty = 0
    failed = 0

    for pdf_file in pdf_files:
        print(f"📄 Extracting: {pdf_file.name}")

        try:
            text = extract_text_from_pdf(pdf_file)

            if not text:
                print(f"⚠️  Text rỗng: {pdf_file.name}")
                empty += 1
                continue

            out_file = OUT_DIR / f"{pdf_file.stem}.txt"
            out_file.write_text(text, encoding="utf-8")
            success += 1

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {pdf_file.name}: {e}")
            failed += 1

    print("\n✅ Hoàn tất extract text")
    print(f"  ✔️ Thành công : {success}")
    print(f"  ⚠️ Text rỗng  : {empty}")
    print(f"  ❌ Lỗi        : {failed}")
    print(f"\n📂 Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
