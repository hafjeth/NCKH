"""
ingest_legal_semantic_chunks.py
================================
Ingest legal semantic chunks vào ChromaDB.
Collection: carbon_policy_textile_vn

Fix so với script cũ:
  1. Import đúng: src.knowledge.vector_db.chroma_client
  2. Path đúng: data/processed/chunks/legal_semantic/
  3. sys.path để import từ root project

Cách chạy (từ D:/NCKH):
    python scripts/data_processing/ingest_legal_semantic_chunks.py
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
SEMANTIC_DIR = ROOT / "data" / "processed" / "chunks" / "legal_semantic"  # FIX: path đúng
COLLECTION_NAME = "carbon_policy_textile_vn"
BATCH_SIZE = 32


def main():
    # ── Kiểm tra thư mục tồn tại ──
    if not SEMANTIC_DIR.exists():
        logging.error(f"❌ Không tìm thấy thư mục: {SEMANTIC_DIR}")
        logging.error("   Kiểm tra lại path hoặc chạy legal_chunker trước")
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
            "domain": "carbon_policy",
            "country": "Vietnam",
            "sector": "textile_garment",
            "chunk_level": "clause"
        }
    )

    # ── Load semantic chunks ──
    semantic_files = list(SEMANTIC_DIR.glob("*.json"))
    logging.info(f"📂 Found {len(semantic_files)} semantic files in {SEMANTIC_DIR}")

    if not semantic_files:
        logging.warning("❌ Không có file JSON nào. Abort.")
        return

    documents, metadatas, ids = [], [], []
    doc_counter = 0

    for file in semantic_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"❌ Lỗi đọc {file.name}: {e}")
            continue

        # Hỗ trợ cả 2 format: {"chunks": [...]} hoặc list trực tiếp
        chunks = data.get("chunks", data) if isinstance(data, dict) else data
        source_file = data.get("source_file", file.name) if isinstance(data, dict) else file.name

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            documents.append(text)
            metadatas.append({
                "law":         chunk.get("law", ""),
                "article":     chunk.get("article", ""),
                "clause":      chunk.get("clause", ""),
                "clause_type": chunk.get("clause_type", ""),
                "domains":     ",".join(chunk.get("domains", [])),
                "subjects":    ",".join(chunk.get("subjects", [])),
                "source_file": source_file,
                "agent":       "legal",
            })
            ids.append(f"legal_{source_file}_{doc_counter:05d}")
            doc_counter += 1

    total = len(documents)
    logging.info(f"📢 Total legal chunks to ingest: {total}")

    if total == 0:
        logging.warning("❌ Không có chunk hợp lệ nào. Abort.")
        return

    # ── Batch ingest ──
    for i in range(0, total, BATCH_SIZE):
        batch_docs  = documents[i : i + BATCH_SIZE]
        batch_meta  = metadatas[i : i + BATCH_SIZE]
        batch_ids   = ids[i : i + BATCH_SIZE]

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
    logging.info(f"🎉 Legal ingestion completed — collection hiện có {final_count} documents")


if __name__ == "__main__":
    main()
