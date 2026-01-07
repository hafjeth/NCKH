import sys
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Force UTF-8 for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


class RetrievalSystem:
    """
    Hệ thống truy xuất thông tin từ ChromaDB
    Nhận query, trả về top-k documents liên quan nhất
    """
    
    def __init__(
        self,
        chroma_db_dir: str = "data/chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3
    ):
        """
        Khởi tạo Retrieval System
        
        Args:
            chroma_db_dir: Thư mục chứa ChromaDB
            collection_name: Tên collection
            embedding_model: Model embedding (phải giống với lúc ingest)
            top_k: Số lượng documents trả về (mặc định 3)
        """
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at: {self.chroma_db_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Connected to collection: {self.collection_name}")
            logger.info(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to load collection '{self.collection_name}': {str(e)}")
            raise
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Truy xuất top-k documents liên quan nhất với query
        
        Args:
            query: Câu hỏi/query từ người dùng
            top_k: Số lượng documents trả về (None = dùng default)
            filter_metadata: Điều kiện filter (VD: {'filename': 'abc.txt'})
        
        Returns:
            List các documents với metadata và score
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Generate query embedding
            logger.info(f"Processing query: '{query}'")
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                where=filter_metadata  # Filter nếu có
            )
            
            # Format results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, meta, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ), 1):
                    retrieved_docs.append({
                        'rank': i,
                        'content': doc,
                        'metadata': meta,
                        'distance': distance,
                        'similarity_score': self._distance_to_similarity(distance)
                    })
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents")
            else:
                logger.warning("No documents found for the query")
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Chuyển đổi distance thành similarity score (0-1)
        Distance càng nhỏ => Similarity càng cao
        
        Args:
            distance: Distance từ ChromaDB (L2 distance)
        
        Returns:
            Similarity score (0-1), 1 là giống nhất
        """
        # Sử dụng công thức: similarity = 1 / (1 + distance)
        return 1.0 / (1.0 + distance)
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_window: int = 1
    ) -> List[Dict]:
        """
        Truy xuất documents kèm context (chunks trước/sau)
        
        Args:
            query: Câu hỏi
            top_k: Số documents trả về
            context_window: Số chunks trước/sau cần lấy thêm
        
        Returns:
            List documents với context mở rộng
        """
        # Lấy kết quả thông thường trước
        base_results = self.retrieve(query, top_k)
        
        if not base_results or context_window == 0:
            return base_results
        
        # Mở rộng context cho mỗi result
        enhanced_results = []
        
        for result in base_results:
            filename = result['metadata']['filename']
            chunk_id = result['metadata']['chunk_id']
            
            # Lấy các chunks lân cận
            context_chunks = self._get_context_chunks(
                filename, 
                chunk_id, 
                context_window
            )
            
            result['context_before'] = context_chunks['before']
            result['context_after'] = context_chunks['after']
            result['full_context'] = (
                '\n'.join(context_chunks['before']) + 
                '\n' + result['content'] + '\n' + 
                '\n'.join(context_chunks['after'])
            )
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _get_context_chunks(
        self,
        filename: str,
        chunk_id: int,
        window: int
    ) -> Dict[str, List[str]]:
        """
        Lấy các chunks lân cận của một chunk
        
        Args:
            filename: Tên file
            chunk_id: ID của chunk hiện tại
            window: Số chunks trước/sau cần lấy
        
        Returns:
            Dict với 'before' và 'after' chunks
        """
        context = {'before': [], 'after': []}
        
        try:
            # Lấy chunks trước
            for i in range(chunk_id - window, chunk_id):
                if i >= 0:
                    chunk_results = self.collection.get(
                        ids=[f"{Path(filename).stem}_chunk_{i}"],
                        include=['documents']
                    )
                    if chunk_results['documents']:
                        context['before'].append(chunk_results['documents'][0])
            
            # Lấy chunks sau
            for i in range(chunk_id + 1, chunk_id + window + 1):
                chunk_results = self.collection.get(
                    ids=[f"{Path(filename).stem}_chunk_{i}"],
                    include=['documents']
                )
                if chunk_results['documents']:
                    context['after'].append(chunk_results['documents'][0])
        
        except Exception as e:
            logger.warning(f"Error getting context chunks: {str(e)}")
        
        return context
    
    def retrieve_by_filename(
        self,
        filename: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Lấy tất cả chunks từ một file cụ thể
        
        Args:
            filename: Tên file cần lấy
            top_k: Giới hạn số chunks (None = lấy tất cả)
        
        Returns:
            List các chunks từ file đó
        """
        try:
            results = self.collection.get(
                where={"filename": filename},
                limit=top_k if top_k else 10000,
                include=['documents', 'metadatas']
            )
            
            chunks = []
            if results['documents']:
                for doc, meta in zip(results['documents'], results['metadatas']):
                    chunks.append({
                        'content': doc,
                        'metadata': meta
                    })
                
                # Sắp xếp theo chunk_id
                chunks.sort(key=lambda x: x['metadata']['chunk_id'])
                logger.info(f"Retrieved {len(chunks)} chunks from '{filename}'")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving from file '{filename}': {str(e)}")
            return []
    
    def get_all_filenames(self) -> List[str]:
        """
        Lấy danh sách tất cả filenames trong database
        
        Returns:
            List tên files
        """
        try:
            # Lấy sample để extract filenames
            results = self.collection.get(
                limit=10000,
                include=['metadatas']
            )
            
            filenames = set()
            if results['metadatas']:
                for meta in results['metadatas']:
                    filenames.add(meta.get('filename', ''))
            
            return sorted(list(filenames))
        
        except Exception as e:
            logger.error(f"Error getting filenames: {str(e)}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Lấy thống kê về database
        
        Returns:
            Dict chứa thông tin thống kê
        """
        try:
            total_docs = self.collection.count()
            filenames = self.get_all_filenames()
            
            return {
                'total_documents': total_docs,
                'total_files': len(filenames),
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'sample_files': filenames[:10]
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def format_results_for_display(
        self,
        results: List[Dict],
        show_metadata: bool = True,
        max_content_length: int = 300
    ) -> str:
        """
        Format kết quả retrieval để hiển thị đẹp
        
        Args:
            results: List kết quả từ retrieve()
            show_metadata: Có hiển thị metadata không
            max_content_length: Độ dài tối đa của content hiển thị
        
        Returns:
            String formatted để print
        """
        if not results:
            return "Không tìm thấy kết quả nào."
        
        output = []
        output.append("=" * 80)
        output.append(f"RETRIEVAL RESULTS - Found {len(results)} documents")
        output.append("=" * 80)
        
        for result in results:
            output.append(f"\n📄 Rank {result['rank']}")
            output.append(f"📊 Similarity Score: {result['similarity_score']:.4f}")
            
            if show_metadata:
                meta = result['metadata']
                output.append(f"📁 File: {meta.get('filename', 'N/A')}")
                output.append(f"🔢 Chunk: {meta.get('chunk_id', 'N/A')}/{meta.get('total_chunks', 'N/A')}")
            
            # Content (truncate if too long)
            content = result['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            output.append(f"\n📝 Content:\n{content}")
            output.append("-" * 80)
        
        return "\n".join(output)


def main():
    """
    Demo sử dụng Retrieval System
    """
    # Khởi tạo retrieval system
    project_root = Path(__file__).parent.parent.parent
    
    retriever = RetrievalSystem(
        chroma_db_dir=str(project_root / "data/chroma_db"),
        collection_name="knowledge_base",
        top_k=3
    )
    
    # Hiển thị stats
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    stats = retriever.get_stats()
    for key, value in stats.items():
        if key == 'sample_files':
            print(f"\n{key}:")
            for f in value:
                print(f"  - {f}")
        else:
            print(f"{key}: {value}")
    
    # Test queries
    test_queries = [
        "CBAM là gì?",
        "Quy định về phát thải khí nhà kính",
        "Nghị định về bảo vệ môi trường",
        "Chuyển đổi xanh ngành dệt may",
        "Industry 4.0"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL WITH SAMPLE QUERIES")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f"🔍 QUERY: {query}")
        print(f"{'='*80}")
        
        # Retrieve
        results = retriever.retrieve(query, top_k=3)
        
        # Display
        formatted_output = retriever.format_results_for_display(
            results,
            show_metadata=True,
            max_content_length=200
        )
        print(formatted_output)
        
        # Thêm dòng phân cách
        print("\n" + "─" * 80)


if __name__ == "__main__":
    main()