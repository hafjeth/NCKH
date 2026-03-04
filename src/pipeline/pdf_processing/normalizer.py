from pathlib import Path
import re

# ===== PATH CONFIG =====
SRC_DIR = Path("data/processed/clean_text")
OUT_DIR = Path("data/processed/normalized_text")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== REGEX RULES =====
WEIRD_CHARS = re.compile(r"[�□■]")
MULTI_SPACE = re.compile(r"[ \t]{2,}")

SEPARATOR_LINE = re.compile(r"^[\|\-_,.`\s]{2,}$")
LONELY_NOISE_TOKEN = re.compile(r"(?<=\s)[\|`]{1,2}(?=\s|$)")

# Marker do OCR / extract sinh ra
PAGE_MARKER = re.compile(r"=+\s*PAGE\s*\d+\s*=+", re.IGNORECASE)

# Dòng nhiễu: chỉ có ký tự lạ / số lẻ / dấu
NOISE_LINE = re.compile(r"^[^a-zA-ZÀ-ỹ0-9]{0,5}\s*\d{0,3}\s*$")

# Dòng gãy do PDF
BROKEN_LINE = re.compile(r"(?<![.!?:;])\n(?!\n)")


def normalize_text(text: str) -> str:
    # 0️⃣ Normalize newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1️⃣ Remove PAGE markers
    text = PAGE_MARKER.sub("", text)

    # 2️⃣ Remove weird OCR chars
    text = WEIRD_CHARS.sub("", text)

    # 3️⃣ Join broken lines (KHÔNG làm mất đoạn)
    text = BROKEN_LINE.sub(" ", text)

    # 4️⃣ Line-level cleaning
    clean_lines = []
    for line in text.split("\n"):
        line = line.strip()

        # bỏ dòng trống dư
        if not line:
            clean_lines.append("")
            continue

        # bỏ dòng nhiễu kiểu: "` : 3", "— 7", "§"
        if NOISE_LINE.match(line):
            continue

        clean_lines.append(line)

    text = "\n".join(clean_lines)

    # 5️⃣ Normalize spaces
    text = MULTI_SPACE.sub(" ", text)

    # 6️⃣ Normalize paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    txt_files = list(SRC_DIR.glob("*.txt"))

    if not txt_files:
        print("⚠️ Không có file .txt để normalize")
        return

    count = 0
    for txt_file in txt_files:
        print(f"🧹 Normalizing: {txt_file.name}")

        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        clean_text = normalize_text(raw_text)

        out_file = OUT_DIR / txt_file.name
        out_file.write_text(clean_text, encoding="utf-8")

        count += 1

    print(f"\n✅ Hoàn tất normalize {count} files")
    print(f"📂 Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
