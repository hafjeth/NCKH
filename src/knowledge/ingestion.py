import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging with UTF-8 encoding
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Force UTF-8 for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class DataIngestionEngine:
    """
    Engine để nạp dữ liệu text đã cleaned vào ChromaDB
    """
    
    def __init__(
        self,
        processed_text_dir: str = "data/processed_text",
        chroma_db_dir: str = "data/chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Khởi tạo Ingestion Engine
        
        Args:
            processed_text_dir: Thư mục chứa text đã cleaned
            chroma_db_dir: Thư mục lưu ChromaDB
            collection_name: Tên collection trong ChromaDB
            embedding_model: Model để tạo embeddings
            chunk_size: Kích thước mỗi chunk (số ký tự)
            chunk_overlap: Số ký tự overlap giữa các chunks
        """
        self.processed_text_dir = Path(processed_text_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Knowledge base from PDF documents"}
        )
        
        logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        logger.info(f"Current documents in collection: {self.collection.count()}")
    
    def chunk_text(self, text: str, filename: str) -> List[Dict]:
        """
        Chia text thành các chunks nhỏ hơn
        
        Args:
            text: Nội dung text cần chia
            filename: Tên file gốc
            
        Returns:
            List các chunks với metadata
        """
        chunks = []
        
        # Chia theo đoạn văn trước
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Nếu đoạn văn quá dài, chia nhỏ hơn
            if len(para) > self.chunk_size:
                # Lưu chunk hiện tại nếu có
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'filename': filename
                    })
                    chunk_id += 1
                    current_chunk = ""
                
                # Chia đoạn văn dài thành các chunks
                words = para.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= self.chunk_size:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append({
                                'text': temp_chunk.strip(),
                                'chunk_id': chunk_id,
                                'filename': filename
                            })
                            chunk_id += 1
                        
                        # Overlap: giữ lại một số từ cuối
                        overlap_words = temp_chunk.split()[-10:] if len(temp_chunk.split()) > 10 else []
                        temp_chunk = " ".join(overlap_words) + " " + word + " "
                
                if temp_chunk:
                    current_chunk = temp_chunk
            
            # Đoạn văn bình thường
            elif len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Lưu chunk hiện tại
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'filename': filename
                    })
                    chunk_id += 1
                
                current_chunk = para + "\n\n"
        
        # Lưu chunk cuối cùng
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'filename': filename
            })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho list texts
        
        Args:
            texts: List các text cần embedding
            
        Returns:
            List các embedding vectors
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def ingest_file(self, filepath: Path) -> Dict:
        """
        Nạp một file text vào ChromaDB
        
        Args:
            filepath: Đường dẫn đến file text
            
        Returns:
            Dict chứa thông tin về quá trình ingest
        """
        try:
            # Đọc file
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"File {filepath.name} is empty, skipping...")
                return {'status': 'skipped', 'reason': 'empty file'}
            
            # Chunk text
            chunks = self.chunk_text(text, filepath.name)
            
            if not chunks:
                logger.warning(f"No chunks generated from {filepath.name}")
                return {'status': 'skipped', 'reason': 'no chunks'}
            
            # Prepare data for ChromaDB
            ids = [f"{filepath.stem}_chunk_{chunk['chunk_id']}" for chunk in chunks]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'filename': chunk['filename'],
                    'chunk_id': chunk['chunk_id'],
                    'total_chunks': len(chunks),
                    'ingestion_date': datetime.now().isoformat(),
                    'source': 'pdf'
                }
                for chunk in chunks
            ]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(documents)
            
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Ingested {filepath.name}: {len(chunks)} chunks")
            
            return {
                'status': 'success',
                'filename': filepath.name,
                'chunks': len(chunks),
                'total_chars': len(text)
            }
            
        except Exception as e:
            logger.error(f"✗ Error ingesting {filepath.name}: {str(e)}")
            return {
                'status': 'error',
                'filename': filepath.name,
                'error': str(e)
            }
    
    def ingest_all(self, file_pattern: str = "*.txt") -> Dict:
        """
        Nạp tất cả files từ thư mục processed_text
        
        Args:
            file_pattern: Pattern để filter files (default: *.txt)
            
        Returns:
            Dict chứa thống kê về quá trình ingest
        """
        logger.info(f"Starting ingestion from {self.processed_text_dir}")
        
        # Get all text files
        files = list(self.processed_text_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files found in {self.processed_text_dir}")
            return {'status': 'no files found'}
        
        logger.info(f"Found {len(files)} files to ingest")
        
        # Statistics
        stats = {
            'total_files': len(files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_chunks': 0,
            'start_time': datetime.now().isoformat(),
            'details': []
        }
        
        # Process each file
        for filepath in tqdm(files, desc="Ingesting files"):
            result = self.ingest_file(filepath)
            stats['details'].append(result)
            
            if result['status'] == 'success':
                stats['successful'] += 1
                stats['total_chunks'] += result['chunks']
            elif result['status'] == 'error':
                stats['failed'] += 1
            else:
                stats['skipped'] += 1
        
        stats['end_time'] = datetime.now().isoformat()
        stats['total_documents_in_db'] = self.collection.count()
        
        # Save stats to file
        stats_file = self.chroma_db_dir / 'ingestion_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Log summary
        logger.info("=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {stats['total_files']}")
        logger.info(f"✓ Successful: {stats['successful']}")
        logger.info(f"✗ Failed: {stats['failed']}")
        logger.info(f"⊝ Skipped: {stats['skipped']}")
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Total documents in DB: {stats['total_documents_in_db']}")
        logger.info(f"Stats saved to: {stats_file}")
        logger.info("=" * 60)
        
        return stats
    
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Tìm kiếm trong knowledge base
        
        Args:
            query: Câu hỏi/query cần tìm
            n_results: Số kết quả trả về
            
        Returns:
            Dict chứa kết quả tìm kiếm
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return {
            'query': query,
            'results': [
                {
                    'document': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        }
    
    def get_stats(self) -> Dict:
        """
        Lấy thống kê về database
        
        Returns:
            Dict chứa thông tin thống kê
        """
        count = self.collection.count()
        
        # Get sample metadata
        sample = self.collection.get(limit=10)
        
        unique_files = set()
        if sample['metadatas']:
            unique_files = set(meta.get('filename', '') for meta in sample['metadatas'])
        
        return {
            'total_documents': count,
            'collection_name': self.collection_name,
            'sample_files': list(unique_files),
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
    
    def reset_collection(self):
        """
        Xóa toàn bộ collection và tạo lại
        CẢNH BÁO: Thao tác này không thể hoàn tác!
        """
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Knowledge base from PDF documents"}
        )
        logger.info("Collection reset complete")


def main():
    """
    Main function để chạy ingestion pipeline
    """
    # Lấy thư mục gốc của project (lên 2 cấp từ knowledge/)
    project_root = Path(__file__).parent.parent.parent
    
    # Khởi tạo engine
    engine = DataIngestionEngine(
        processed_text_dir=str(project_root / "data/processed_text"),
        chroma_db_dir=str(project_root / "data/chroma_db"),
        collection_name="knowledge_base",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Hiển thị stats hiện tại
    logger.info("Current database stats:")
    stats = engine.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Nạp dữ liệu
    logger.info("\nStarting data ingestion...")
    results = engine.ingest_all()
    
    # Test search
    logger.info("\nTesting search functionality...")
    test_query = "nghiên cứu khoa học"
    search_results = engine.search(test_query, n_results=3)
    
    logger.info(f"\nSearch results for: '{test_query}'")
    for i, result in enumerate(search_results['results'], 1):
        logger.info(f"\n--- Result {i} ---")
        logger.info(f"File: {result['metadata']['filename']}")
        logger.info(f"Distance: {result['distance']:.4f}")
        logger.info(f"Text preview: {result['document'][:200]}...")


if __name__ == "__main__":
    main()