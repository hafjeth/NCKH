import shutil
from pathlib import Path
import csv

# ===== CONFIG =====
BASE_DIR = Path("data/raw_pdfs")
TARGET_DIR = BASE_DIR
LOG_FILE = TARGET_DIR / "_move_log.csv"
MODE = "move"  # "move" hoặc "copy"
# ==================

def safe_name(target_dir, filename):
    """Tránh overwrite khi trùng tên"""
    target_path = target_dir / filename
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    i = 1
    while True:
        new_name = f"{stem}_{i}{suffix}"
        new_path = target_dir / new_name
        if not new_path.exists():
            return new_path
        i += 1


def collect_pdfs(base_dir):
    pdfs = []
    for p in base_dir.rglob("*.pdf"):
        if p.parent != TARGET_DIR:
            pdfs.append(p)
    return pdfs


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = collect_pdfs(BASE_DIR)

    if not pdf_files:
        print("❌ Không tìm thấy file PDF nào để gom.")
        return

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        writer.writerow(["original_name", "old_path", "new_path"])

        for pdf in pdf_files:
            new_path = safe_name(TARGET_DIR, pdf.name)

            if MODE == "move":
                shutil.move(str(pdf), new_path)
            else:
                shutil.copy2(str(pdf), new_path)

            writer.writerow([pdf.name, str(pdf), str(new_path)])
            print(f"✔ {pdf.name} → {new_path.name}")

    print(f"\n✅ Hoàn tất. Log lưu tại: {LOG_FILE}")


if __name__ == "__main__":
    main()
