"""
settings.py
=====================================
ĐẶT FILE NÀY TẠI: config/settings.py  (thay file cũ)

Centralized configuration cho NCKH Project:
- Paths
- API Keys & Provider
- Models
- Vector Database (ChromaDB)
- RAG
- Fallback & Cache
- Logging & Debug
- Benchmark

FIXES SO VỚI BẢN CŨ:
  [FIX-1] Thêm FALLBACK_CONFIG — base_agent.py gọi Config.FALLBACK_CONFIG.get(...)
          nhưng bản cũ không định nghĩa → AttributeError khi chạy
  [FIX-2] Sửa ANTHROPIC_MODEL default từ 'claude-sonnet-4-20250514' (sai)
          → 'claude-sonnet-4-5' (đúng theo Anthropic API hiện tại)
  [FIX-3] Sửa DEFAULT_MODEL đồng bộ với ANTHROPIC_MODEL
  [FIX-4] validate() mở rộng: kiểm tra FALLBACK_CONFIG, model name, BENCHMARK_ROUNDS
  [FIX-5] summary() mở rộng: hiển thị thêm FALLBACK_CONFIG và BENCHMARK info
  [FIX-6] BENCHMARK_ROUNDS default tăng từ 2 → 3 (đồng bộ với run_benchmark.py)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# BASE PATHS
# ══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR  = BASE_DIR / "src"
LOGS_DIR = BASE_DIR / "logs"

# Tạo thư mục cần thiết nếu chưa có
for _dir in [LOGS_DIR, DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════════════
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")

# Xác định provider theo thứ tự ưu tiên
# Có thể override bằng env var: API_PROVIDER=anthropic
_env_provider = os.getenv("API_PROVIDER", "").lower()

if _env_provider in ("openai", "anthropic", "gemini"):
    API_PROVIDER = _env_provider
    API_KEY = {
        "openai":    OPENAI_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "gemini":    GEMINI_API_KEY,
    }.get(API_PROVIDER)
elif OPENAI_API_KEY:
    API_PROVIDER = "openai"
    API_KEY      = OPENAI_API_KEY
elif ANTHROPIC_API_KEY:
    API_PROVIDER = "anthropic"
    API_KEY      = ANTHROPIC_API_KEY
elif GEMINI_API_KEY:
    API_PROVIDER = "gemini"
    API_KEY      = GEMINI_API_KEY
else:
    API_PROVIDER = None
    API_KEY      = None

# ══════════════════════════════════════════════════════════════
# MODEL SETTINGS
# ══════════════════════════════════════════════════════════════

# OpenAI
OPENAI_MODEL     = os.getenv("OPENAI_MODEL",     "gpt-4o-mini")
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING", "text-embedding-3-small")

# [FIX-2] Sửa model name Anthropic — bản cũ dùng 'claude-sonnet-4-20250514' (sai)
ANTHROPIC_MODEL  = os.getenv("ANTHROPIC_MODEL",  "claude-sonnet-4-5")

# Gemini
GEMINI_MODEL     = os.getenv("GEMINI_MODEL",     "gemini-2.0-flash-exp")

# [FIX-3] DEFAULT_MODEL đồng bộ với provider đang dùng
_DEFAULT_MODEL_MAP = {
    "openai":    OPENAI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
    "gemini":    GEMINI_MODEL,
}
DEFAULT_MODEL = os.getenv(
    "DEFAULT_MODEL",
    _DEFAULT_MODEL_MAP.get(API_PROVIDER, "gpt-4o-mini")
)

# Model và embedding đang active
if API_PROVIDER == "openai":
    MODEL_NAME      = OPENAI_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING
elif API_PROVIDER == "anthropic":
    MODEL_NAME      = ANTHROPIC_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING   # Anthropic không có embedding riêng → dùng OpenAI
elif API_PROVIDER == "gemini":
    MODEL_NAME      = GEMINI_MODEL
    EMBEDDING_MODEL = OPENAI_EMBEDDING
else:
    MODEL_NAME      = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"

# Generation parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS",    "2000"))

# ══════════════════════════════════════════════════════════════
# VECTOR DATABASE (ChromaDB)
# ══════════════════════════════════════════════════════════════
CHROMA_HOST        = os.getenv("CHROMA_HOST",        "localhost")
CHROMA_PORT        = int(os.getenv("CHROMA_PORT",    "8000"))
CHROMA_PERSIST_DIR = DATA_DIR / "vector_stores" / "chroma"
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "carbon_policy_textile_vn")

# ══════════════════════════════════════════════════════════════
# RAG CONFIGURATION
# ══════════════════════════════════════════════════════════════
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

RAG_RETRIEVAL = {
    "adaptive":             True,
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.75")),
    "candidate_pool_size":  int(os.getenv("CANDIDATE_POOL_SIZE",    "15")),
    "min_results":          int(os.getenv("MIN_RESULTS",            "2")),
    "max_results":          int(os.getenv("MAX_RESULTS",            "7")),
}

# ══════════════════════════════════════════════════════════════
# CITATION POLICY
# ══════════════════════════════════════════════════════════════
CITATION_POLICY = {
    "min_citations_base":       2,
    "min_citations_if_5_docs":  3,
    "min_citations_if_7_docs":  4,
    "strict_mode":              False,
}

# ══════════════════════════════════════════════════════════════
# [FIX-1] FALLBACK & CACHE CONFIG  ← MỚI HOÀN TOÀN
# base_agent.py gọi Config.FALLBACK_CONFIG.get(...) → cần định nghĩa ở đây
# ══════════════════════════════════════════════════════════════
FALLBACK_CONFIG = {
    # Cache: tránh gọi API lặp lại với cùng prompt
    "use_cache":              os.getenv("USE_CACHE", "true").lower() == "true",

    # Số lần retry mỗi model trước khi chuyển sang model tiếp theo
    "max_retries_per_model":  int(os.getenv("MAX_RETRIES_PER_MODEL", "3")),

    # Thời gian chờ giữa các retry (giây) — exponential backoff base
    "retry_backoff_base":     int(os.getenv("RETRY_BACKOFF_BASE",    "2")),

    # Timeout cho mỗi LLM call (giây)
    "request_timeout":        int(os.getenv("REQUEST_TIMEOUT",       "60")),

    # Danh sách model fallback theo thứ tự ưu tiên
    # Nếu primary model fail, thử lần lượt các model trong list này
    "fallback_model_chain": [
        {"provider": "openai",    "model": OPENAI_MODEL,    "priority": 1},
        {"provider": "anthropic", "model": ANTHROPIC_MODEL, "priority": 2},
    ],
}

# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
LOG_LEVEL  = os.getenv("LOG_LEVEL",  "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ══════════════════════════════════════════════════════════════
# DEBUG MODE
# ══════════════════════════════════════════════════════════════
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# ══════════════════════════════════════════════════════════════
# BENCHMARK SETTINGS
# [FIX-6] BENCHMARK_ROUNDS default 2 → 3 (đồng bộ với run_benchmark.py)
# ══════════════════════════════════════════════════════════════
BENCHMARK_ROUNDS  = int(os.getenv("BENCHMARK_ROUNDS",  "3"))
BENCHMARK_TIMEOUT = int(os.getenv("BENCHMARK_TIMEOUT", "300"))  # giây


# ══════════════════════════════════════════════════════════════
# Config CLASS
# ══════════════════════════════════════════════════════════════
class Config:
    """
    Unified configuration class — single source of truth.

    Import từ hai nơi đều được (backward compat):
        from config.settings import Config
        from src.core.config import Config  (nếu có alias)

    Usage:
        Config.MODEL_NAME
        Config.FALLBACK_CONFIG.get("use_cache", True)
        Config.validate()
        Config.summary()
    """

    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    SRC_DIR  = SRC_DIR
    LOGS_DIR = LOGS_DIR

    # API
    OPENAI_API_KEY    = OPENAI_API_KEY
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    GEMINI_API_KEY    = GEMINI_API_KEY
    API_PROVIDER      = API_PROVIDER
    API_KEY           = API_KEY

    # Models
    DEFAULT_MODEL    = DEFAULT_MODEL
    MODEL_NAME       = MODEL_NAME
    EMBEDDING_MODEL  = EMBEDDING_MODEL
    TEMPERATURE      = TEMPERATURE
    MAX_TOKENS       = MAX_TOKENS
    OPENAI_MODEL     = OPENAI_MODEL
    OPENAI_EMBEDDING = OPENAI_EMBEDDING
    ANTHROPIC_MODEL  = ANTHROPIC_MODEL
    GEMINI_MODEL     = GEMINI_MODEL

    # ChromaDB
    CHROMA_HOST        = CHROMA_HOST
    CHROMA_PORT        = CHROMA_PORT
    CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIR
    COLLECTION_NAME    = COLLECTION_NAME

    # RAG
    CHUNK_SIZE      = CHUNK_SIZE
    CHUNK_OVERLAP   = CHUNK_OVERLAP
    RAG_RETRIEVAL   = RAG_RETRIEVAL
    CITATION_POLICY = CITATION_POLICY

    # [FIX-1] Fallback & Cache — bắt buộc phải có
    FALLBACK_CONFIG = FALLBACK_CONFIG

    # Logging & Debug
    LOG_LEVEL  = LOG_LEVEL
    LOG_FORMAT = LOG_FORMAT
    DEBUG_MODE = DEBUG_MODE

    # Benchmark
    BENCHMARK_ROUNDS  = BENCHMARK_ROUNDS
    BENCHMARK_TIMEOUT = BENCHMARK_TIMEOUT

    # ── validate() [FIX-4] ────────────────────────────────────────────────
    @classmethod
    def validate(cls) -> bool:
        """
        Kiểm tra toàn bộ config bắt buộc.
        Trả về True nếu hợp lệ, False nếu có lỗi.
        """
        errors   = []
        warnings = []

        # API Key
        if not cls.API_KEY:
            errors.append("Không tìm thấy API key nào (OPENAI / ANTHROPIC / GEMINI)")
        elif cls.API_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("API_PROVIDER=anthropic nhưng ANTHROPIC_API_KEY trống")
        elif cls.API_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("API_PROVIDER=openai nhưng OPENAI_API_KEY trống")

        # [FIX-2] Kiểm tra model name Anthropic
        known_anthropic = [
            "claude-opus-4-5", "claude-sonnet-4-5",
            "claude-haiku-4-5", "claude-opus-4-5-20251101",
        ]
        if cls.API_PROVIDER == "anthropic" and cls.MODEL_NAME not in known_anthropic:
            warnings.append(
                f"ANTHROPIC_MODEL='{cls.MODEL_NAME}' có thể không đúng. "
                f"Các model hợp lệ: {known_anthropic}"
            )

        # Data directory
        if not cls.DATA_DIR.exists():
            warnings.append(f"Data directory chưa tồn tại: {cls.DATA_DIR}")

        # [FIX-4] FALLBACK_CONFIG
        if not isinstance(cls.FALLBACK_CONFIG, dict):
            errors.append("FALLBACK_CONFIG phải là dict")
        else:
            for key in ("use_cache", "max_retries_per_model"):
                if key not in cls.FALLBACK_CONFIG:
                    warnings.append(f"FALLBACK_CONFIG thiếu key: '{key}'")

        # BENCHMARK_ROUNDS hợp lệ
        if not (1 <= cls.BENCHMARK_ROUNDS <= 10):
            warnings.append(
                f"BENCHMARK_ROUNDS={cls.BENCHMARK_ROUNDS} nằm ngoài khoảng khuyến nghị [1, 10]"
            )

        # In kết quả
        for w in warnings:
            print(f"⚠️  Config Warning: {w}")
        for e in errors:
            print(f"❌ Config Error  : {e}")

        if errors:
            return False

        print("✅ Configuration validated successfully")
        return True

    # ── summary() [FIX-5] ─────────────────────────────────────────────────
    @classmethod
    def summary(cls) -> None:
        """In toàn bộ config ra console để debug."""
        w = 70
        print("\n" + "=" * w)
        print("  📋 SYSTEM CONFIGURATION SUMMARY")
        print("=" * w)

        print(f"  {'API Provider':<22}: {cls.API_PROVIDER or 'NOT SET'}")
        print(f"  {'Model':<22}: {cls.MODEL_NAME}")
        print(f"  {'Embedding':<22}: {cls.EMBEDDING_MODEL}")
        print(f"  {'Temperature':<22}: {cls.TEMPERATURE}")
        print(f"  {'Max Tokens':<22}: {cls.MAX_TOKENS}")
        print("-" * w)
        print(f"  {'ChromaDB':<22}: {cls.CHROMA_HOST}:{cls.CHROMA_PORT}")
        print(f"  {'Collection':<22}: {cls.COLLECTION_NAME}")
        print(f"  {'Chroma Persist':<22}: {cls.CHROMA_PERSIST_DIR}")
        print("-" * w)
        print(f"  {'Benchmark Rounds':<22}: {cls.BENCHMARK_ROUNDS}")
        print(f"  {'Benchmark Timeout':<22}: {cls.BENCHMARK_TIMEOUT}s")
        print("-" * w)
        fc = cls.FALLBACK_CONFIG
        print(f"  {'Cache Enabled':<22}: {fc.get('use_cache')}")
        print(f"  {'Max Retries/Model':<22}: {fc.get('max_retries_per_model')}")
        print(f"  {'Request Timeout':<22}: {fc.get('request_timeout')}s")
        print("-" * w)
        print(f"  {'Debug Mode':<22}: {cls.DEBUG_MODE}")
        print(f"  {'Log Level':<22}: {cls.LOG_LEVEL}")
        print(f"  {'Base Dir':<22}: {cls.BASE_DIR}")
        print("=" * w + "\n")


# ══════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════

def get_config() -> type:
    """Trả về class Config (singleton pattern)."""
    return Config


def validate_config() -> bool:
    """Shortcut để validate từ ngoài module."""
    return Config.validate()


# ══════════════════════════════════════════════════════════════
# RUN TRỰC TIẾP
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Config.summary()
    Config.validate()