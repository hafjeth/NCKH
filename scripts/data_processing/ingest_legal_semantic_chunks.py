import json
import logging
from pathlib import Path
from chromadb_client import ChromaDBClient

logging.basicConfig(level=logging.INFO)

# ================= CONFIG =================
SEMANTIC_DIR = Path("data/processed/legal_chunks_semantic")
COLLECTION_NAME = "carbon_policy_textile_vn"

BATCH_SIZE = 32  # an toàn cho Chroma + CPU

# ================= MAIN =================
def main():
    # 1️⃣ Connect ChromaDB (KHÔNG truyền collection_name)
    client = ChromaDBClient(
        host="localhost",
        port=8000
    )

    # 2️⃣ Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "domain": "carbon_policy",
            "country": "Vietnam",
            "sector": "textile_garment",
            "chunk_level": "clause"
        }
    )

    # 3️⃣ Load semantic chunks
    semantic_files = list(SEMANTIC_DIR.glob("*_semantic.json"))
    logging.info(f"📁 Found {len(semantic_files)} semantic files")

    documents, metadatas, ids = [], [], []
    doc_counter = 0

    for file in semantic_files:
        data = json.loads(file.read_text(encoding="utf-8"))

        source_file = data.get("source_file", file.name)

        for chunk in data.get("chunks", []):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            documents.append(text)

            # ⚠️ Metadata PHẢI là scalar (string / number / bool)
            metadatas.append({
                "law": chunk.get("law", ""),
                "article": chunk.get("article", ""),
                "clause": chunk.get("clause", ""),
                "clause_type": chunk.get("clause_type", ""),
                "domains": ",".join(chunk.get("domains", [])),
                "subjects": ",".join(chunk.get("subjects", [])),
                "source_file": source_file
            })

            ids.append(f"{source_file}_{doc_counter}")
            doc_counter += 1

    total = len(documents)
    logging.info(f"🔢 Total chunks to ingest: {total}")

    if total == 0:
        logging.warning("❌ No valid chunks found. Abort.")
        return

    # 4️⃣ Batch ingest (tránh lỗi 422)
    for i in range(0, total, BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )

        logging.info(f"✅ Ingested batch {i//BATCH_SIZE + 1} "
                     f"({i+len(batch_docs)}/{total})")

    logging.info("🎉 Ingestion completed successfully")

# ================= RUN =================
if __name__ == "__main__":
    main()
