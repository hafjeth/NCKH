from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= PATH =================
SRC_DIR = Path("data/processed/legal_chunks_semantic")
OUT_DIR = Path("data/processed/vector_store")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= MODEL =================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================= MAIN =================
def main():
    texts = []
    metadata = []

    for file in SRC_DIR.glob("*_semantic.json"):
        data = json.loads(file.read_text(encoding="utf-8"))

        for chunk in data["chunks"]:
            texts.append(chunk["text"])
            metadata.append({
                "source": data["source_file"],
                "article": chunk.get("article"),
                "clause": chunk.get("clause"),
                "clause_type": chunk.get("clause_type"),
                "domains": chunk.get("domains")
            })

    if not texts:
        print("❌ No chunks to embed")
        return

    print(f"🔢 Embedding {len(texts)} legal clauses...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(OUT_DIR / "legal.index"))
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ Embedding + FAISS store created")

if __name__ == "__main__":
    main()
