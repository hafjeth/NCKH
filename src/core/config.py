"""
System Configuration (Research-Grade)
====================================
FIX: Added FALLBACK_CONFIG (was missing, caused KeyError in base_agent.py)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    # ======================================================
    # API
    # ======================================================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if OPENAI_API_KEY:
        API_PROVIDER = "openai"
        API_KEY = OPENAI_API_KEY
    elif GEMINI_API_KEY:
        API_PROVIDER = "gemini"
        API_KEY = GEMINI_API_KEY
    else:
        raise RuntimeError(
            "❌ No API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
        )

    # ======================================================
    # MODEL
    # ======================================================
    OPENAI_MODEL  = "gpt-4o-mini"
    GEMINI_MODEL  = "gemini-2.0-flash-exp"
    MODEL_NAME    = OPENAI_MODEL if API_PROVIDER == "openai" else GEMINI_MODEL
    TEMPERATURE   = 0.3
    MAX_TOKENS    = 2000

    # ======================================================
    # RAG
    # ======================================================
    CHUNK_SIZE    = 1000
    CHUNK_OVERLAP = 200

    # ======================================================
    # CHROMADB
    # ======================================================
    CHROMADB_HOST  = "localhost"
    CHROMADB_PORT  = 8000
    COLLECTION_NAME = "carbon_policy_textile_vn"

    # ======================================================
    # RAG RETRIEVAL
    # ======================================================
    RAG_RETRIEVAL = {
        "adaptive": True,
        "similarity_threshold": 0.75,
        "candidate_pool_size": 15,
        "min_results": 2,
        "max_results": 7,
        "prefer_policy_documents": True,
        "use_metadata_filter": True,
    }

    # ======================================================
    # CITATION POLICY
    # ======================================================
    CITATION_POLICY = {
        "min_citations_base": 2,
        "min_citations_if_5_docs": 3,
        "min_citations_if_7_docs": 4,
        "strict_mode": False,
        "penalty_weight": 0.15,
    }

    # ======================================================
    # FIX: FALLBACK_CONFIG (was missing — caused KeyError in base_agent.py)
    # ======================================================
    FALLBACK_CONFIG = {
        "enabled": True,
        "max_retries_per_model": 3,
        "use_cache": True,
        "cache_ttl_days": 7,
    }

    # ======================================================
    # SYSTEM
    # ======================================================
    DEBUG_MODE = True