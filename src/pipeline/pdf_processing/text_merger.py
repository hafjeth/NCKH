from pathlib import Path
import shutil

# ===== PATH CONFIG =====
TEXT_PDF_DIR = Path("data/intermediate/extracted_text/from_text_pdf")
SCAN_PDF_DIR = Path("data/intermediate/extracted_text/from_scan_pdf")

OUT_DIR = Path("data/processed/clean_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def copy_all(src_dir: Path, dst_dir: Path, overwrite=False):
    count = 0
    for txt_file in src_dir.glob("*.txt"):
        dst_file = dst_dir / txt_file.name

        if dst_file.exists() and not overwrite:
            continue

        shutil.copy2(txt_file, dst_file)
        count += 1
    return count


def main():
    print("🔁 MERGING extracted text...\n")

    # 1️⃣ Copy text từ PDF có text layer (ưu tiên)
    print("📄 Copy from_text_pdf → processed/clean_text")
    text_count = copy_all(TEXT_PDF_DIR, OUT_DIR, overwrite=True)

    # 2️⃣ Copy OCR text nếu chưa tồn tại
    print("📄 Copy from_scan_pdf → processed/clean_text (if not exists)")
    scan_count = copy_all(SCAN_PDF_DIR, OUT_DIR, overwrite=False)

    print("\n✅ MERGE HOÀN TẤT")
    print(f"  Text PDFs copied : {text_count}")
    print(f"  Scan PDFs copied : {scan_count}")
    print(f"  Total files      : {len(list(OUT_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()
