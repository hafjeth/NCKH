"""
Step 1: Lấy mẫu để gán nhãn thủ công
Usage: python scripts/evaluation/step1_create_samples.py
"""
import json, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LEGAL_DIR    = PROJECT_ROOT / "data" / "processed" / "chunks" / "legal_semantic"
BUSINESS_DIR = PROJECT_ROOT / "data" / "processed" / "chunks" / "business_semantic"
OUTPUT_DIR   = PROJECT_ROOT / "experiments" / "evaluation" / "ground_truth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N = 100
random.seed(42)

def sample_legal(n):
    all_chunks = []
    for f in LEGAL_DIR.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        for chunk in data.get("chunks", []):
            text = chunk.get("text", "").strip()
            if len(text) < 50:
                continue
            all_chunks.append({
                "id": f"legal_{len(all_chunks):04d}",
                "type": "legal",
                "text": text[:500],
                "source_file": data.get("source_file", f.name),
                "article": chunk.get("article", ""),
                "clause": chunk.get("clause", ""),
                "auto_clause_type": chunk.get("clause_type", ""),
                "auto_subjects": chunk.get("subjects", []),
                "auto_domains": chunk.get("domains", []),
                "human_clause_type": "",
                "human_subjects": [],
                "human_domains": [],
                "notes": ""
            })
    sampled = random.sample(all_chunks, min(n, len(all_chunks)))
    print(f"Legal:    {len(sampled)}/{len(all_chunks)} mau")
    return sampled

def sample_business(n):
    all_chunks = []
    for f in BUSINESS_DIR.glob("*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        for para in data.get("paragraphs", []):
            text = para.get("text", "").strip()
            sem  = para.get("semantic", {})
            if len(text) < 100:
                continue
            all_chunks.append({
                "id": f"biz_{len(all_chunks):04d}",
                "type": "business",
                "text": text[:500],
                "source_file": data.get("source_file", f.name),
                "auto_stance": sem.get("stance", ""),
                "auto_focus": sem.get("focus", []),
                "auto_cbam_relevance": sem.get("cbam_relevance", False),
                "human_stance": "",
                "human_focus": [],
                "human_cbam_relevance": None,
                "notes": ""
            })
    sampled = random.sample(all_chunks, min(n, len(all_chunks)))
    print(f"Business: {len(sampled)}/{len(all_chunks)} mau")
    return sampled

def main():
    print("Lay mau de gan nhan thu cong...")
    legal    = sample_legal(N)
    business = sample_business(N)
    output = {
        "instructions": {
            "legal": {
                "human_clause_type": "Chon 1: obligation|definition|prohibition|permission|procedure|condition|responsibility|sanction|general",
                "human_subjects": "Chon nhieu: state_agency|enterprise|organization|individual|household|producer|importer|unspecified",
                "human_domains": "Chon nhieu: water|air|waste|plastic|emission|environmental_permit|EIA|recycling|general_environment"
            },
            "business": {
                "human_stance": "Chon 1: concern|support|neutral",
                "human_focus": "Chon nhieu: cost|compliance|competitiveness|risk|opportunity",
                "human_cbam_relevance": "true neu lien quan CBAM, false neu khong"
            },
            "note": "Dien nhan vao cac truong human_*. Giu nguyen auto_*. Chi can dien 50 mau."
        },
        "legal_samples": legal,
        "business_samples": business
    }
    out = OUTPUT_DIR / "samples_to_label.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFile: {out}")
    print(f"Total: {len(legal)+len(business)} mau")
    print("\nBUOC TIEP THEO:")
    print("  1. Mo: experiments/evaluation/ground_truth/samples_to_label.json")
    print("  2. Dien nhan vao cac truong human_* (chi can 50 mau)")
    print("  3. Chay: python scripts/evaluation/step2_evaluate_accuracy.py")

if __name__ == "__main__":
    main()