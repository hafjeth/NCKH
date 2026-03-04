import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import cv2
import numpy as np
import sys

# =========================
# CONFIG
# =========================
PDF_PATH = Path("data/raw/pdfs/Thông tư 01 2022 TT-BTNMT.pdf")

OUTPUT_DIR = Path(
    "data/intermediate/extracted_text/from_text_pdf"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT = OUTPUT_DIR / "Thông tư 01 2022 TT-BTNMT.txt"

DPI = 300
LANG = "vie"
MIN_TEXT_LENGTH = 3000


# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold – rất quan trọng cho PDF pháp luật
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )
    return th


# =========================
# OCR PDF
# =========================
def ocr_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")

    print("🖨️  Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=DPI)

    all_text = []

    for i, page in enumerate(pages, start=1):
        print(f"🔍 OCR trang {i}/{len(pages)}")

        img = preprocess_image(page)

        text = pytesseract.image_to_string(
            img,
            lang=LANG,
            config="--psm 6"
        )

        if not text.strip():
            print(f"⚠️ Trang {i}: OCR rỗng")

        all_text.append(text.strip())

    return "\n\n".join(all_text)


# =========================
# MAIN
# =========================
def main():
    print("📄 OCR:", PDF_PATH.name)

    try:
        text = ocr_pdf(PDF_PATH)

        length = len(text)
        print(f"🔍 Tổng số ký tự OCR: {length}")

        if length < MIN_TEXT_LENGTH:
            print("❌ OCR THẤT BẠI – TEXT QUÁ NGẮN")
            sys.exit(1)

        OUTPUT_TXT.write_text(text, encoding="utf-8")
        print("✅ OCR thành công")
        print("📁 Output:", OUTPUT_TXT)

    except Exception as e:
        print("❌ LỖI:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
