from pathlib import Path
import re
import json

# =========================
# PATH CONFIG
# =========================
SRC_DIR = Path("data/processed/normalized_text")
REPORT_PATH = Path("data/processed/text_quality_report.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# REGEX RULES
# =========================
SEVERE_BAD_CHARS = re.compile(r"[�□■]")
NOISY_TOKENS = re.compile(r"(?<=\s)[`|]{1,2}(?=\s|$)")
MULTI_SPACE = re.compile(r"\s{3,}")

VALID_CHAR = re.compile(r"[a-zA-ZÀ-ỹ0-9]")
TOTAL_CHAR = re.compile(r"\S")

# =========================
# THRESHOLDS
# =========================
MIN_LENGTH = 1000
MIN_VALID_RATIO = 0.85


def analyze_text(text: str) -> dict:
    total_chars = len(TOTAL_CHAR.findall(text))
    valid_chars = len(VALID_CHAR.findall(text))

    valid_ratio = valid_chars / total_chars if total_chars else 0

    return {
        "length": len(text),
        "total_non_space_chars": total_chars,
        "valid_char_ratio": round(valid_ratio, 4),
        "severe_bad_chars": len(SEVERE_BAD_CHARS.findall(text)),
        "noisy_tokens": len(NOISY_TOKENS.findall(text)),
        "multi_spaces": len(MULTI_SPACE.findall(text)),
    }


def quality_label(metrics: dict) -> str:
    if metrics["length"] < MIN_LENGTH:
        return "FAIL"

    if metrics["severe_bad_chars"] > 0:
        return "FAIL"

    if metrics["valid_char_ratio"] < MIN_VALID_RATIO:
        return "FAIL"

    if metrics["noisy_tokens"] > 0 or metrics["multi_spaces"] > 0:
        return "WARN"

    return "PASS"


def main():
    txt_files = list(SRC_DIR.glob("*.txt"))

    if not txt_files:
        print("⚠️ Không có file normalized_text để kiểm tra")
        return

    report = {
        "summary": {
            "total_files": len(txt_files),
            "pass": 0,
            "warn": 0,
            "fail": 0,
        },
        "files": {}
    }

    for txt_file in txt_files:
        print(f"🔍 Checking: {txt_file.name}")

        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        metrics = analyze_text(text)
        status = quality_label(metrics)

        report["files"][txt_file.name] = {
            "status": status,
            **metrics
        }

        report["summary"][status.lower()] += 1

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n Hoàn tất kiểm tra chất lượng text")
    print(f"Report: {REPORT_PATH}")
    print("Summary:", report["summary"])


if __name__ == "__main__":
    main()
