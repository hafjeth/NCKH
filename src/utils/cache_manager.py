"""
Cache Manager for LLM Responses
===============================
Lưu kết quả API để:
- Giảm số lần gọi API
- Tăng tốc benchmark
- Fallback khi API lỗi
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Quản lý cache cho LLM responses
    - Lưu theo hash của messages
    - Có TTL (time to live)
    - Support cả file và memory cache
    """

    def __init__(self, cache_dir: str = "cache/llm_responses", ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        
        # Memory cache cho những lần gọi gần đây
        self.memory_cache: Dict[str, Dict] = {}
        self.memory_cache_size = 100
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "file_hits": 0
        }

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Chuẩn hóa messages để tạo hash nhất quán"""
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", "").strip(),
                "content": msg.get("content", "").strip()
            })
        return normalized

    def _create_key(self, messages: List[Dict[str, str]], model: str, temperature: float) -> str:
        """Tạo cache key đơn giản và nhất quán"""
        norm_msgs = self._normalize_messages(messages)
        
        key_parts = []
        for msg in norm_msgs:
            key_parts.append(f"{msg['role']}:{msg['content']}")
        
        messages_str = "|".join(key_parts)
        content = f"{messages_str}|{model}|{temperature:.2f}"
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def get(self, messages: List[Dict[str, str]], model: str, temperature: float) -> Optional[str]:
        """
        Lấy response từ cache nếu còn hạn
        """
        key = self._create_key(messages, model, temperature)
        
        logger.debug(f"🔑 GET key: {key}")
        
        # Check memory cache trước
        if key in self.memory_cache:
            cached = self.memory_cache[key]
            if datetime.now() < cached["expires"]:
                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1
                logger.info(f"✅ Cache HIT (memory): {key[:8]}...")
                return cached["response"]
            else:
                del self.memory_cache[key]
        
        # Check file cache
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                
                if datetime.now() < cached["expires"]:
                    self._add_to_memory_cache(key, cached)
                    self.stats["hits"] += 1
                    self.stats["file_hits"] += 1
                    logger.info(f"✅ Cache HIT (file): {key[:8]}...")
                    return cached["response"]
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Cache read error: {e}")
        
        self.stats["misses"] += 1
        logger.info(f"❌ Cache MISS: {key[:8]}...")
        return None

    def set(self, messages: List[Dict[str, str]], model: str, temperature: float, response: str):
        """
        Lưu response vào cache - CẢI TIẾN VỚI DEBUG CHI TIẾT
        """
        key = self._create_key(messages, model, temperature)
        expires = datetime.now() + self.ttl
        
        logger.info(f"💾 Attempting to save cache for key: {key[:8]}...")
        
        cached_data = {
            "response": response,
            "expires": expires,
            "created": datetime.now(),
            "model": model,
            "temperature": temperature,
            "key": key
        }
        
        # Lưu memory cache
        self._add_to_memory_cache(key, cached_data)
        logger.info(f"✅ Memory cache saved for key: {key[:8]}...")
        
        # Lưu file cache với try-catch chi tiết
        cache_file = self._get_cache_path(key)
        try:
            # Kiểm tra quyền ghi
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created cache directory: {self.cache_dir}")
            
            # Ghi file
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Kiểm tra file đã được tạo chưa
            if cache_file.exists():
                file_size = cache_file.stat().st_size
                logger.info(f"✅ File cache saved: {key[:8]}.pkl ({file_size} bytes)")
            else:
                logger.error(f"❌ File not created after write: {cache_file}")
                
        except PermissionError as e:
            logger.error(f"❌ Permission error when writing cache: {e}")
            logger.info(f"   Current directory: {Path.cwd()}")
            logger.info(f"   Cache path: {cache_file.absolute()}")
        except Exception as e:
            logger.error(f"❌ Unexpected error when writing cache: {e}")
            import traceback
            traceback.print_exc()

    def _add_to_memory_cache(self, key: str, data: Dict):
        """Thêm vào memory cache với LRU đơn giản"""
        if len(self.memory_cache) >= self.memory_cache_size:
            removed_key = next(iter(self.memory_cache))
            del self.memory_cache[removed_key]
            logger.debug(f"🗑️ Removed oldest from memory cache: {removed_key[:8]}...")
        
        self.memory_cache[key] = data

    def get_stats(self) -> Dict:
        """Lấy thống kê cache"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        # Đếm file thực tế
        file_count = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 1),
            "memory_cache_size": len(self.memory_cache),
            "file_cache_size": file_count,
            "cache_dir": str(self.cache_dir.absolute())
        }

    def debug_key(self, messages: List[Dict[str, str]], model: str, temperature: float):
        """Helper function để debug key"""
        key = self._create_key(messages, model, temperature)
        cache_file = self._get_cache_path(key)
        
        print(f"\n🔑 DEBUG KEY:")
        print(f"  Model: {model}")
        print(f"  Temperature: {temperature}")
        print(f"  Key: {key}")
        print(f"  Cache file path: {cache_file.absolute()}")
        print(f"  Cache file exists: {cache_file.exists()}")
        
        if cache_file.exists():
            print(f"  File size: {cache_file.stat().st_size} bytes")
        
        return key

    def clear_all(self):
        """Xóa toàn bộ cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.memory_cache.clear()
        self.stats = {"hits": 0, "misses": 0, "memory_hits": 0, "file_hits": 0}
        logger.info("🧹 Cache cleared")


# Singleton instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get or create cache manager singleton"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager