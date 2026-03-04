import json
import logging
from pathlib import Path
from chromadb_client import ChromaDBClient

logging.basicConfig(level=logging.INFO)

# ============ CONFIG ============
INPUT_DIR = Path("data/processed/business_paragraphs_semantic")
COLLECTION_NAME = "business_cbam_textile_vn"
BATCH_SIZE = 32

def main():
    client = ChromaDBClient(host="localhost", port=8000)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "agent": "business",
            "sector": "textile_garment",
            "topic": "CBAM"
        }
    )

    files = list(INPUT_DIR.glob("*.json"))
    logging.info(f"📁 Found {len(files)} business semantic files")

    documents, metadatas, ids = [], [], []
    idx = 0

    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        source = data.get("source_file", file.name)

        for p in data.get("paragraphs", []):
            text = p.get("text", "").strip()
            if not text:
                continue

            sem = p.get("semantic", {})

            documents.append(text)
            metadatas.append({
                "agent": "business",
                "stance": sem.get("stance", "neutral"),
                "focus": ",".join(sem.get("focus", [])),
                "cbam_relevance": sem.get("cbam_relevance", False),
                "source_file": source
            })
            ids.append(f"biz_{idx}")
            idx += 1

    total = len(documents)
    logging.info(f"🔢 Total BUSINESS paragraphs to ingest: {total}")

    for i in range(0, total, BATCH_SIZE):
        collection.add(
            documents=documents[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE]
        )
        logging.info(f"✅ Ingested {min(i+BATCH_SIZE, total)}/{total}")

    logging.info("🎉 BUSINESS ingestion completed")

if __name__ == "__main__":
    main()
