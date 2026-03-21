"""
Quick inspection - chạy để kiểm tra cấu trúc data
python scripts/evaluation/inspect_data.py
"""
import json
from pathlib import Path

chunks_dir = Path("data/processed/chunks")
print("=== data/processed/chunks ===")
for sub in sorted(chunks_dir.iterdir()):
    files = list(sub.glob("*.json"))
    print(f"\n  {sub.name}: {len(files)} files")
    if files:
        d = json.loads(files[0].read_text(encoding="utf-8"))
        print(f"    Top-level keys: {list(d.keys())}")
        # Check chunks
        if "chunks" in d and d["chunks"]:
            print(f"    Chunk[0] keys: {list(d['chunks'][0].keys())}")
        # Check paragraphs
        if "paragraphs" in d and d["paragraphs"]:
            print(f"    Para[0] keys: {list(d['paragraphs'][0].keys())}")
        # Sample file name
        print(f"    Example file: {files[0].name}")