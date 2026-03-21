"""
ingest_business_paragraphs.py
==============================
Ingest business paragraphs vào ChromaDB.
Collection: business_cbam_textile_vn

Fix so với script cũ:
  1. Import đúng: src.knowledge.vector_db.chroma_client
  2. Path đúng: data/processed/chunks/business_paragraphs/
  3. sys.path để import từ root project

Cách chạy (từ D:/NCKH):
    python scripts/data_processing/ingest_business_paragraphs.py
"""

import sys
import json
import logging
from pathlib import Path

# ── Setup import path ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # D:\NCKH
sys.path.insert(0, str(ROOT))

from src.knowledge.vector_db.chroma_client import ChromaDBClient  # FIX: import đúng

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR = ROOT / "data" / "processed" / "chunks" / "business_paragraphs"  # FIX: path đúng
COLLECTION_NAME = "business_cbam_textile_vn"
BATCH_SIZE = 32


def main():
    # ── Kiểm tra thư mục tồn tại ──
    if not INPUT_DIR.exists():
        logging.error(f"❌ Không tìm thấy thư mục: {INPUT_DIR}")
        logging.error("   Kiểm tra lại path hoặc chạy business_chunker trước")
        sys.exit(1)

    # ── Kết nối ChromaDB ──
    logging.info("🔌 Kết nối ChromaDB...")
    try:
        client = ChromaDBClient(host="localhost", port=8000)
    except ConnectionError as e:
        logging.error(str(e))
        logging.error("👉 Hãy chắc chắn Docker ChromaDB đang chạy: docker-compose up -d")
        sys.exit(1)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",
            "agent":  "business",
            "sector": "textile_garment",
            "topic":  "CBAM",
        }
    )

    # ── Load business paragraphs ──
    files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"📂 Found {len(files)} business files in {INPUT_DIR}")

    if not files:
        logging.warning("❌ Không có file JSON nào. Abort.")
        return

    documents, metadatas, ids = [], [], []
    idx = 0

    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"❌ Lỗi đọc {file.name}: {e}")
            continue

        # Hỗ trợ cả 2 format:
        # Format A (dict): {"source_file": "...", "paragraphs": [...]}
        # Format B (list): [{"text": ..., "semantic": {...}}, ...]
        if isinstance(data, dict):
            paragraphs = data.get("paragraphs", [])
            source = data.get("source_file", file.name)
        else:
            paragraphs = data
            source = file.name

        for p in paragraphs:
            text = p.get("text", "").strip()
            if not text:
                continue

            # Hỗ trợ cả semantic lồng {"semantic": {...}} và flat {"stance": ...}
            sem = p.get("semantic", p)

            cbam = sem.get("cbam_relevance", False)

            documents.append(text)
            metadatas.append({
                "agent":          "business",
                "stance":         sem.get("stance", "neutral"),
                "focus":          ",".join(sem.get("focus", [])),
                "cbam_relevance": str(cbam),   # ChromaDB HTTP API: bool → str
                "source_file":    source,
                "domains":        ",".join(sem.get("domains", [])),
            })
            ids.append(f"biz_{idx:05d}")
            idx += 1

    total = len(documents)
    logging.info(f"📢 Total business paragraphs to ingest: {total}")

    if total == 0:
        logging.warning("❌ Không có paragraph hợp lệ nào. Abort.")
        return

    # ── Batch ingest ──
    for i in range(0, total, BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]
        batch_ids  = ids[i : i + BATCH_SIZE]

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            logging.info(f"✅ Batch {i // BATCH_SIZE + 1}: ingested {i + len(batch_docs)}/{total}")
        except Exception as e:
            logging.error(f"❌ Batch {i // BATCH_SIZE + 1} thất bại: {e}")

    final_count = collection.count()
    logging.info(f"🎉 Business ingestion completed — collection hiện có {final_count} documents")


if __name__ == "__main__":
    main()
