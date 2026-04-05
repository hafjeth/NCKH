"""
Fallback Manager for LLM Calls
===============================
Quản lý retry, fallback models, và error handling

THAY ĐỔI SO VỚI BẢN TRƯỚC:
  [FIX-14] Tăng backoff time (2 → 2^(attempt+1), tối đa 30s)
           Bản cũ: wait_time = 2 ** attempt (2, 4, 8s)
           Bản mới: wait_time = min(30, 2 ** (attempt + 1)) (4, 8, 16, 30s)
  [FIX-14] Giảm max_retries mặc định từ 3 → 2
"""

import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI, OpenAIError

from src.utils.cache_manager import get_cache_manager
from src.core.config import Config

logger = logging.getLogger(__name__)


class FallbackManager:
    """
    Quản lý chiến lược fallback cho LLM calls
    - Retry với exponential backoff
    - Fallback qua các models khác nhau
    - Cache responses
    """

    def __init__(self, config: Config):
        self.config = config
        self.cache = get_cache_manager()
        
        # Định nghĩa các models theo thứ tự ưu tiên
        self.model_chain = self._build_model_chain()
        
        # Khởi tạo OpenAI client
        if config.OPENAI_API_KEY:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self.client = None

    def _build_model_chain(self) -> List[Dict[str, Any]]:
        """
        Xây dựng chuỗi fallback models
        Ưu tiên: primary → secondary → tertiary
        """
        chain = []
        
        # Primary: GPT-4o-mini (nhanh, rẻ)
        if self.config.OPENAI_API_KEY:
            chain.append({
                "provider": "openai",
                "model": "gpt-4o-mini",
                "priority": 1,
            })
            
            # Secondary: GPT-3.5-turbo (rất ổn định)
            chain.append({
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "priority": 2,
            })
        
        return sorted(chain, key=lambda x: x["priority"])

    def _extract_user_content(self, messages: List[Dict[str, str]]) -> str:
        """Trích xuất nội dung user từ messages (bỏ qua system prompt)"""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _normalize_messages_for_cache(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Chuẩn hóa messages cho cache:
        - Giữ nguyên user content
        - Giữ nguyên system prompt (vì nó quan trọng)
        """
        normalized = []
        for msg in messages:
            normalized.append({
                "role": msg.get("role", ""),
                "content": msg.get("content", "").strip()
            })
        return normalized

    def call_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        max_retries: int = 2,  # [FIX-14] Giảm từ 3 → 2
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Gọi LLM với fallback và retry
        """
        
        # Chuẩn hóa messages cho cache
        cache_messages = self._normalize_messages_for_cache(messages)
        
        # Debug: In messages để xem có gì khác không
        logger.debug(f"📨 Messages for cache: {cache_messages}")
        
        # Kiểm tra cache cho TẤT CẢ models
        if use_cache:
            for model_info in self.model_chain:
                cached = self.cache.get(cache_messages, model_info["model"], temperature)
                if cached:
                    logger.info(f"✅ Cache HIT: {model_info['model']}")
                    return {
                        "response": cached,
                        "model_used": model_info["model"],
                        "from_cache": True,
                        "attempts": 0
                    }
        
        # Thử từng model trong chain
        last_error = None
        total_attempts = 0
        
        for model_info in self.model_chain:
            for attempt in range(max_retries):
                total_attempts += 1
                try:
                    logger.info(f"🔄 Attempt {attempt+1}/{max_retries} with {model_info['model']}")
                    
                    response = self.client.chat.completions.create(
                        model=model_info["model"],
                        messages=messages,  # Dùng messages gốc để gọi API
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=30
                    )
                    
                    response_text = response.choices[0].message.content.strip()
                    
                    # Lưu cache với messages đã chuẩn hóa
                    if use_cache:
                        self.cache.set(cache_messages, model_info["model"], temperature, response_text)
                        logger.info(f"💾 Saved to cache for {model_info['model']}")
                    
                    return {
                        "response": response_text,
                        "model_used": model_info["model"],
                        "from_cache": False,
                        "attempts": total_attempts
                    }
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ Failed with {model_info['model']}: {e}")
                    
                    # [FIX-14] Exponential backoff với max 30 giây
                    wait_time = min(30, 2 ** (attempt + 1))  # 4, 8, 16, 30s
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            
            logger.info(f"➡️ Moving to next model: {model_info['model']} exhausted")
        
        error_msg = f"All models failed. Last error: {last_error}"
        logger.error(f"❌ {error_msg}")
        
        return {
            "response": f"[ERROR] Không thể kết nối đến dịch vụ AI. Vui lòng thử lại sau.",
            "model_used": None,
            "from_cache": False,
            "attempts": total_attempts,
            "error": str(last_error)
        }


# Singleton instance
_fallback_manager = None

def get_fallback_manager(config: Config) -> FallbackManager:
    """Get or create fallback manager singleton"""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackManager(config)
    return _fallback_manager