"""
ingest_legal_semantic_chunks.py
================================
ĐẶT FILE NÀY TẠI: src/knowledge/ingestion/ingest_legal_semantic_chunks.py

Nạp legal semantic chunks vào ChromaDB với multilingual embedding.

FIX SO VỚI BẢN CŨ:
  [FIX-1] Bỏ chromadb_client → dùng chromadb.HttpClient chuẩn
  [FIX-2] Thêm doc_counter += 1 (bản cũ thiếu → tất cả ID bị trùng _0)
  [FIX-3] Thêm kiểm tra duplicate ID trước khi add
  [FIX-4] Metadata: ép kiểu article/clause về str
  [FIX-5] Dùng paraphrase-multilingual-mpnet-base-v2 thay default embedding
          → hỗ trợ song ngữ Anh-Việt, dim=768
"""

import json
import logging
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= CONFIG =================
SEMANTIC_DIR    = Path("data/processed/chunks/legal_chunks_semantic")
COLLECTION_NAME = "carbon_policy_textile_vn"
BATCH_SIZE      = 32
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ================= MAIN =================
def main():
    client = chromadb.HttpClient(host="localhost", port=8000)
    print(f" ChromaDB connected — heartbeat: {client.heartbeat()}")

    # [FIX-5] Multilingual embedding function
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    logging.info(f" Embedding model: {EMBEDDING_MODEL}")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={
            "domain":          "carbon_policy",
            "country":         "Vietnam",
            "sector":          "textile_garment",
            "chunk_level":     "clause",
            "embedding_model": EMBEDDING_MODEL,
        }
    )
    logging.info(f" Collection '{COLLECTION_NAME}' ready")

    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])
    logging.info(f" Existing documents in collection: {len(existing_ids)}")

    semantic_files = sorted(SEMANTIC_DIR.glob("*_semantic.json"))
    logging.info(f" Found {len(semantic_files)} semantic files in {SEMANTIC_DIR}")

    documents, metadatas, ids = [], [], []
    doc_counter  = 0
    skip_counter = 0

    for file in semantic_files:
        data        = json.loads(file.read_text(encoding="utf-8"))
        source_file = data.get("source_file", file.stem)

        for chunk in data.get("chunks", []):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            doc_counter += 1

            chunk_id = (
                f"{str(chunk.get('law', 'unknown'))}"
                f"_Art{str(chunk.get('article', ''))}"
                f"_Cl{str(chunk.get('clause', ''))}"
                f"_{doc_counter}"
            )

            if chunk_id in existing_ids:
                skip_counter += 1
                continue

            subjects_raw = chunk.get("subjects", [])
            domains_raw  = chunk.get("domains", [])
            subjects_str = (
                ",".join(subjects_raw)
                if isinstance(subjects_raw, list)
                else str(subjects_raw)
            )
            domains_str = (
                ",".join(domains_raw)
                if isinstance(domains_raw, list)
                else str(domains_raw)
            )

            documents.append(text)
            metadatas.append({
                "law":         str(chunk.get("law", "")),
                "article":     str(chunk.get("article", "")),
                "clause":      str(chunk.get("clause", "")),
                "clause_type": str(chunk.get("clause_type", "")),
                "domains":     domains_str,
                "subjects":    subjects_str,
                "source_file": source_file,
                "agent":       "government",
            })
            ids.append(chunk_id)

    total = len(documents)
    logging.info(f"New chunks to ingest: {total} (skipped existing: {skip_counter})")

    if total == 0:
        logging.warning("No new chunks to add. Collection may already be up to date.")
        return

    for i in range(0, total, BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
        logging.info(
            f"Batch {i//BATCH_SIZE + 1} ingested "
            f"({min(i+BATCH_SIZE, total)}/{total})"
        )

    logging.info("Legal ingestion completed successfully")
    logging.info(f"Total in collection now: {collection.count()}")


if __name__ == "__main__":
    main()