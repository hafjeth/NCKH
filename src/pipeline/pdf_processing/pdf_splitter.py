import json
import shutil
from pathlib import Path

# ===== PATH CONFIG (ĐÚNG THEO PROJECT CỦA BẠN) =====
RAW_PDF_DIR = Path("data/raw/pdfs")
REPORT_PATH = Path("data/raw/pdf_type_report.json")

OUT_TEXT_DIR = Path("data/intermediate/pdf_classified/text_pdf")
OUT_SCAN_DIR = Path("data/intermediate/pdf_classified/scan_pdf")

OUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SCAN_DIR.mkdir(parents=True, exist_ok=True)


def copy_files(file_list, src_dir, dst_dir):
    count = 0
    for fname in file_list:
        src = src_dir / fname
        dst = dst_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            count += 1
        else:
            print(f"⚠️ Không tìm thấy file: {fname}")
    return count


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy {REPORT_PATH}")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    text_pdfs = report.get("text_pdf", [])
    scan_pdfs = report.get("scan_pdf", [])

    print("\n📂 Đang copy CLEAN TEXT PDFs...")
    text_count = copy_files(text_pdfs, RAW_PDF_DIR, OUT_TEXT_DIR)

    print("\n📂 Đang copy SCAN PDFs...")
    scan_count = copy_files(scan_pdfs, RAW_PDF_DIR, OUT_SCAN_DIR)

    print("\n🎉 Hoàn tất tách PDF theo loại!")
    print(f"Text PDFs : {text_count}")
    print(f"Scan PDFs       : {scan_count}")


if __name__ == "__main__":
    main()
