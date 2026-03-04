from config.paths import ensure_directories
from config.settings import CHROMA_PERSIST_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def init_database():
    """Khởi tạo database và directories"""
    logger.info("Initializing project structure...")
    
    try:
        # Tạo directories
        ensure_directories()
        logger.info("✓ All directories created")
        
        # Kiểm tra ChromaDB directory
        if CHROMA_PERSIST_DIR.exists():
            logger.info(f"✓ ChromaDB directory exists: {CHROMA_PERSIST_DIR}")
        else:
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ ChromaDB directory created: {CHROMA_PERSIST_DIR}")
        
        logger.info("Database initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_database()