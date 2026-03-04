from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from pytesseract import Output

# ===== FIX TESSERACT (KHÔNG PHỤ THUỘC PATH) =====
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== PATH CONFIG =====
PDF_DIR = Path("data/intermediate/pdf_classified/scan_pdf")
OUT_DIR = Path("data/intermediate/extracted_text/from_scan_pdf")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def ocr_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    all_text = []

    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        # ---- Detect orientation ----
        try:
            osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
            angle = osd.get("rotate", 0)
        except Exception:
            angle = 0

        if angle != 0:
            image = image.rotate(360 - angle, expand=True)

        # ---- OCR ----
        text = pytesseract.image_to_string(
            image,
            lang="vie+eng",
            config="--oem 3 --psm 6"
        )

        all_text.append(f"\n===== PAGE {page_index + 1} =====\n{text}")

    return "\n".join(all_text)



def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ Không tìm thấy PDF scan nào")
        return

    count = 0
    for pdf_file in pdf_files:
        print(f"📄 OCR: {pdf_file.name}")

        try:
            text = ocr_pdf(pdf_file)
            out_file = OUT_DIR / f"{pdf_file.stem}.txt"
            out_file.write_text(text, encoding="utf-8")
            count += 1
        except Exception as e:
            print(f"❌ Lỗi OCR {pdf_file.name}: {e}")

    print(f"\n✅ OCR hoàn tất cho {count} PDF scan")


if __name__ == "__main__":
    main()

