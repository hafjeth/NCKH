"""
Knowledge Ingestion Engine
Nạp dữ liệu từ processed text vào ChromaDB với OpenAI embeddings
"""

import os
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================
# CONFIGURATION
# ==========================
PROCESSED_TEXT_DIR = "data/processed_text"
CHROMA_DB_DIR = "data/chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"


# ==========================
# 1) ĐỌC TẤT CẢ FILE TEXT
# ==========================
def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Đọc tất cả file .txt từ thư mục processed_text.
    
    Args:
        directory: Đường dẫn đến thư mục chứa file text
        
    Returns:

        List các Document objects
    """
    documents = []
    text_dir = Path(directory)
    
    if not text_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {directory}")
        return documents
    
    txt_files = list(text_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️ Không có file .txt nào trong {directory}")
        return documents
    
    print(f"📂 Tìm thấy {len(txt_files)} file text")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            if content.strip():
                # Tạo Document với metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": txt_file.name,
                        "file_path": str(txt_file),
                        "length": len(content)
                    }
                )
                documents.append(doc)
                print(f"  ✓ Đọc: {txt_file.name} ({len(content):,} chars)")
            else:
                print(f"  ⚠️ Bỏ qua file rỗng: {txt_file.name}")
                
        except Exception as e:
            print(f"  ❌ Lỗi đọc {txt_file.name}: {e}")
            continue
    
    print(f"\n✓ Đã đọc {len(documents)} documents\n")
    return documents


# ==========================
# 2) CHUNKING VỚI LANGCHAIN
# ==========================
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chia documents thành chunks nhỏ hơn.
    
    Args:
        documents: List các Document cần chia
        
    Returns:
        List các Document chunks
    """
    print(f"✂️ Bắt đầu chunking với chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Thêm metadata về chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    print(f"✓ Đã tạo {len(chunks)} chunks\n")
    return chunks


# ==========================
# 3) KHỞI TẠO EMBEDDING MODEL
# ==========================
def initialize_embeddings() -> OpenAIEmbeddings:
    """
    Khởi tạo OpenAI Embeddings model.
    
    Returns:
        OpenAIEmbeddings object
    """
    # Set API key nếu chưa có trong env
    if OPENAI_API_KEY and OPENAI_API_KEY != "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":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    print(f"🔑 Khởi tạo embedding model: {EMBEDDING_MODEL}")
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    print("✓ Embedding model đã sẵn sàng\n")
    return embeddings


# ==========================
# 4) LƯU VÀO CHROMADB
# ==========================
def save_to_chromadb(chunks: List[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Lưu chunks vào ChromaDB với embeddings.
    
    Args:
        chunks: List các Document chunks
        embeddings: Embedding model
        
    Returns:
        Chroma vectorstore object
    """
    print(f"💾 Đang lưu {len(chunks)} chunks vào ChromaDB...")
    print(f"📁 Persist directory: {CHROMA_DB_DIR}\n")
    
    # Tạo thư mục nếu chưa có
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    
    try:
        # Khởi tạo hoặc load existing ChromaDB
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name="knowledge_base"
        )
        
        # Thêm documents vào vectorstore
        # Chia nhỏ để tránh timeout với batch lớn
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            vectorstore.add_documents(batch)
        
        print(f"\n✓ Đã lưu thành công {len(chunks)} chunks vào ChromaDB!")
        print(f"✓ Collection: knowledge_base")
        print(f"✓ Location: {os.path.abspath(CHROMA_DB_DIR)}\n")
        
        return vectorstore
        
    except Exception as e:
        print(f"\n❌ Lỗi khi lưu vào ChromaDB: {e}")
        raise


# ==========================
# 5) VERIFY DATABASE
# ==========================
def verify_database(vectorstore: Chroma):
    """
    Kiểm tra xem database đã được tạo đúng chưa.
    
    Args:
        vectorstore: Chroma vectorstore object
    """
    print("🔍 Đang verify database...")
    
    try:
        # Lấy collection
        collection = vectorstore._collection
        count = collection.count()
        
        print(f"✓ Database verification:")
        print(f"  - Total vectors: {count:,}")
        print(f"  - Collection name: {collection.name}")
        
        # Test query
        if count > 0:
            print(f"\n🧪 Test query...")
            results = vectorstore.similarity_search("test", k=1)
            if results:
                print(f"✓ Query thành công! Sample result:")
                print(f"  - Source: {results[0].metadata.get('source', 'N/A')}")
                print(f"  - Content preview: {results[0].page_content[:100]}...")
        
        print("\n✅ Database đã sẵn sàng sử dụng!")
        
    except Exception as e:
        print(f"⚠️ Verify warning: {e}")


# ==========================
# 6) MAIN PIPELINE
# ==========================
def main():
    """
    Main function - chạy toàn bộ pipeline ingestion.
    """
    print("="*70)
    print("🚀 KNOWLEDGE INGESTION PIPELINE")
    print("="*70)
    print()
    
    # Step 1: Load documents
    print("📖 STEP 1: Loading documents...")
    print("-" * 70)
    documents = load_documents_from_directory(PROCESSED_TEXT_DIR)
    
    if not documents:
        print("❌ Không có documents để xử lý. Thoát.")
        return
    
    # Step 2: Chunk documents
    print("✂️ STEP 2: Chunking documents...")
    print("-" * 70)
    chunks = chunk_documents(documents)
    
    # Step 3: Initialize embeddings
    print("🔑 STEP 3: Initializing embeddings...")
    print("-" * 70)
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "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":
        print("❌ CHƯA CÓ OPENAI_API_KEY!")
        print("   Vui lòng:")
        print("   1. Set env: export OPENAI_API_KEY=sk-...")
        print("   2. Hoặc sửa trong code: OPENAI_API_KEY = 'sk-...'")
        return
    
    embeddings = initialize_embeddings()
    
    # Step 4: Save to ChromaDB
    print("💾 STEP 4: Saving to ChromaDB...")
    print("-" * 70)
    vectorstore = save_to_chromadb(chunks, embeddings)
    
    # Step 5: Verify
    print("🔍 STEP 5: Verifying database...")
    print("-" * 70)
    verify_database(vectorstore)
    
    print("\n" + "="*70)
    print("✅ INGESTION HOÀN TẤT!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Database location: {os.path.abspath(CHROMA_DB_DIR)}")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print()


# ==========================
# 7) RUN
# ==========================
if __name__ == "__main__":
    main()