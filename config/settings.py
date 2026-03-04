"""
System Configuration - NCKH Project
=====================================
Centralized configuration for:
- Paths
- API Keys
- Models
- Vector Database
- Logging
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================================================
# BASE PATHS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# API KEYS
# ======================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Determine primary API provider
if OPENAI_API_KEY:
    API_PROVIDER = "openai"
    API_KEY = OPENAI_API_KEY
elif ANTHROPIC_API_KEY:
    API_PROVIDER = "anthropic"
    API_KEY = ANTHROPIC_API_KEY
elif GEMINI_API_KEY:
    API_PROVIDER = "gemini"
    API_KEY = GEMINI_API_KEY
else:
    API_PROVIDER = None
    API_KEY = None

# ======================================================
# MODEL SETTINGS
# ======================================================

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")

# OpenAI Models
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")

# Anthropic Models
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Gemini Models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Active model based on provider
if API_PROVIDER == "openai":
    MODEL_NAME = OPENAI_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING
elif API_PROVIDER == "anthropic":
    MODEL_NAME = ANTHROPIC_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING  # Fallback for embeddings
elif API_PROVIDER == "gemini":
    MODEL_NAME = GEMINI_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING  # Fallback for embeddings
else:
    MODEL_NAME = "gpt-4o-mini"  # Default
    EMBEDDING_MODEL = "text-embedding-3-small"

# Generation parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# ======================================================
# VECTOR DATABASE (ChromaDB)
# ======================================================
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = DATA_DIR / "vector_stores" / "chroma"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "carbon_policy_textile_vn")

# ======================================================
# RAG CONFIGURATION
# ======================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

RAG_RETRIEVAL = {
    "adaptive": True,
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
    "candidate_pool_size": int(os.getenv("CANDIDATE_POOL_SIZE", "15")),
    "min_results": int(os.getenv("MIN_RESULTS", "2")),
    "max_results": int(os.getenv("MAX_RESULTS", "7")),
}

# ======================================================
# CITATION POLICY
# ======================================================
CITATION_POLICY = {
    "min_citations_base": 2,
    "min_citations_if_5_docs": 3,
    "min_citations_if_7_docs": 4,
    "strict_mode": False,
}

# ======================================================
# LOGGING
# ======================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ======================================================
# DEBUG MODE
# ======================================================
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# ======================================================
# BENCHMARK SETTINGS
# ======================================================
BENCHMARK_ROUNDS = int(os.getenv("BENCHMARK_ROUNDS", "2"))
BENCHMARK_TIMEOUT = int(os.getenv("BENCHMARK_TIMEOUT", "300"))  # seconds


# ======================================================
# Config CLASS (Backward Compatibility)
# ======================================================
class Config:
    """
    Unified configuration class
    
    Usage:
        from src.core.config import Config
        
        print(Config.MODEL_NAME)
        print(Config.OPENAI_API_KEY)
    """
    
    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    SRC_DIR = SRC_DIR
    LOGS_DIR = LOGS_DIR
    
    # API Keys
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    OPENAI_API_KEY = OPENAI_API_KEY
    GEMINI_API_KEY = GEMINI_API_KEY
    API_PROVIDER = API_PROVIDER
    API_KEY = API_KEY
    
    # Models
    DEFAULT_MODEL = DEFAULT_MODEL
    MODEL_NAME = MODEL_NAME
    EMBEDDING_MODEL = EMBEDDING_MODEL
    TEMPERATURE = TEMPERATURE
    MAX_TOKENS = MAX_TOKENS
    
    # OpenAI specific
    OPENAI_MODEL = OPENAI_MODEL
    OPENAI_EMBEDDING = OPENAI_EMBEDDING
    
    # Anthropic specific
    ANTHROPIC_MODEL = ANTHROPIC_MODEL
    
    # Gemini specific
    GEMINI_MODEL = GEMINI_MODEL
    
    # ChromaDB
    CHROMA_HOST = CHROMA_HOST
    CHROMA_PORT = CHROMA_PORT
    CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIR
    COLLECTION_NAME = COLLECTION_NAME
    
    # RAG
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    RAG_RETRIEVAL = RAG_RETRIEVAL
    CITATION_POLICY = CITATION_POLICY
    
    # Logging
    LOG_LEVEL = LOG_LEVEL
    LOG_FORMAT = LOG_FORMAT
    
    # Debug
    DEBUG_MODE = DEBUG_MODE
    
    # Benchmark
    BENCHMARK_ROUNDS = BENCHMARK_ROUNDS
    BENCHMARK_TIMEOUT = BENCHMARK_TIMEOUT
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration"""
        errors = []
        
        if not cls.API_KEY:
            errors.append("No API key found (OPENAI/ANTHROPIC/GEMINI)")
        
        if not cls.DATA_DIR.exists():
            errors.append(f"Data directory not found: {cls.DATA_DIR}")
        
        if errors:
            for error in errors:
                print(f"❌ Config Error: {error}")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    @classmethod
    def summary(cls):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("📋 SYSTEM CONFIGURATION SUMMARY")
        print("="*70)
        print(f"API Provider:     {cls.API_PROVIDER or 'NOT SET'}")
        print(f"Model:            {cls.MODEL_NAME}")
        print(f"Embedding:        {cls.EMBEDDING_MODEL}")
        print(f"Temperature:      {cls.TEMPERATURE}")
        print(f"ChromaDB:         {cls.CHROMA_HOST}:{cls.CHROMA_PORT}")
        print(f"Collection:       {cls.COLLECTION_NAME}")
        print(f"Debug Mode:       {cls.DEBUG_MODE}")
        print(f"Log Level:        {cls.LOG_LEVEL}")
        print("="*70 + "\n")


# ======================================================
# MODULE-LEVEL FUNCTIONS
# ======================================================
def get_config() -> Config:
    """Get configuration instance"""
    return Config


def validate_config() -> bool:
    """Validate configuration"""
    return Config.validate()


# ======================================================
# AUTO-VALIDATION ON IMPORT
# ======================================================
if __name__ == "__main__":
    Config.summary()
    Config.validate()