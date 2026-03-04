import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
# ================= CONFIG =================
INPUT_DIR = Path("data/processed/business_paragraphs")
OUTPUT_DIR = Path("data/processed/business_paragraphs_semantic")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= HEURISTIC SEMANTIC TAG =================
def infer_semantic(paragraph: str):
    text = paragraph.lower()

    focus = []
    if any(k in text for k in ["cost", "expense", "investment", "capital"]):
        focus.append("cost")
    if any(k in text for k in ["compliance", "regulation", "reporting", "requirement"]):
        focus.append("compliance")
    if any(k in text for k in ["competitiveness", "competition", "export", "market"]):
        focus.append("competitiveness")
    if any(k in text for k in ["risk", "penalty", "burden"]):
        focus.append("risk")
    if any(k in text for k in ["opportunity", "innovation", "green", "sustainable"]):
        focus.append("opportunity")

    stance = "neutral"
    if any(k in text for k in ["challenge", "burden", "difficult", "costly"]):
        stance = "concern"
    elif any(k in text for k in ["benefit", "advantage", "support", "enhance"]):
        stance = "support"

    cbam_related = "cbam" in text or "carbon border" in text

    return {
        "role": "business",
        "stance": stance,
        "focus": focus,
        "cbam_relevance": cbam_related
    }

# ================= MAIN =================
def main():
    files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"📁 Found {len(files)} business paragraph files")

    total_paragraphs = 0

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        paragraphs = [
            p["text"].strip()
            for p in data.get("paragraphs", [])
            if len(p.get("text", "").strip()) > 200
        ]

        semantic_chunks = []
        for p in paragraphs:
            semantic_chunks.append({
                "text": p,
                "semantic": infer_semantic(p)
            })

        output = {
            "source_file": file.name,
            "agent": "business",
            "paragraph_count": len(semantic_chunks),
            "paragraphs": semantic_chunks
        }

        out_path = OUTPUT_DIR / file.with_suffix(".json").name
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_paragraphs += len(semantic_chunks)
        logging.info(f"✅ {file.name}: {len(semantic_chunks)} semantic paragraphs")

    logging.info(f"🎉 DONE – Total BUSINESS semantic paragraphs: {total_paragraphs}")

if __name__ == "__main__":
    main()
