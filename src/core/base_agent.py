"""
BaseAgent
=====================================
Agent nen tang cho he thong tranh luan da agent
Tich hop RAG (KnowledgeRetriever) + citation-aware
+ Fallback & Cache support
"""

from typing import List, Dict, Optional
import logging

from openai import OpenAI
from config.settings import Config
from src.knowledge.retrieval.retriever import KnowledgeRetriever
from src.core.fallback_manager import get_fallback_manager

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base Agent cho:
    - Government
    - Business
    - Expert
    """

    def __init__(
        self,
        name: str,
        role: str,
        retriever: Optional[KnowledgeRetriever] = None,
        model_name: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.retriever = retriever
        self.model_name = model_name or Config.MODEL_NAME

        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Khởi tạo fallback manager
        self.fallback_manager = get_fallback_manager(Config)

        # Luu phuc vu debug / evaluation
        self.last_rag_context: str = ""
        self.last_rag_metadata: Dict = {}
        self.last_call_metadata: Dict = {}

    # ======================================================
    # MAIN CHAT FUNCTION (with fallback & cache)
    # ======================================================
    def chat(self, user_prompt: str) -> str:
        """
        Sinh phan hoi cua agent voi RAG
        """

        rag_context = ""
        retrieval_result = None

        # ============================
        # RAG RETRIEVAL
        # ============================
        if self.retriever:
            try:
                retrieval_result = self.retriever.retrieve(
                    query=user_prompt,
                    agent=self._agent_key()
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Retrieval failed: {e}")

        # Handle both list and dict return types
        if retrieval_result:
            if isinstance(retrieval_result, list):
                if retrieval_result:
                    rag_context = self._build_rag_context(retrieval_result)
                    self.last_rag_metadata = {"retrieved_count": len(retrieval_result)}
            elif isinstance(retrieval_result, dict):
                docs = retrieval_result.get("documents", [])
                if docs:
                    rag_context = self._build_rag_context(docs)
                    self.last_rag_metadata = retrieval_result.get("retrieval_metadata", {})

        if not rag_context:
            self.last_rag_metadata = {}

        self.last_rag_context = rag_context

        # ============================
        # BUILD PROMPT
        # ============================
        messages = self._build_messages(user_prompt, rag_context)

        # ============================
        # CALL LLM WITH FALLBACK
        # ============================
        use_cache = Config.FALLBACK_CONFIG.get("use_cache", True)
        
        result = self.fallback_manager.call_with_fallback(
            messages=messages,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            max_retries=Config.FALLBACK_CONFIG.get("max_retries_per_model", 3),
            use_cache=use_cache
        )
        
        # Lưu metadata cho debug/evaluation
        self.last_call_metadata = {
            "model_used": result.get("model_used"),
            "from_cache": result.get("from_cache", False),
            "attempts": result.get("attempts", 1),
            "error": result.get("error")
        }
        
        if result.get("error"):
            logger.error(f"[{self.name}] All models failed: {result['error']}")
            return (
                f"[ERROR] Agent {self.name} khong the tra loi "
                f"do loi he thong."
            )
        
        return result.get("response", "")

    def _build_messages(self, user_prompt: str, rag_context: str) -> List[Dict]:
        """Xây dựng messages cho LLM call"""
        
        messages = [
            {"role": "system", "content": self.role},
            {
                "role": "system",
                "content": (
                    "YEU CAU HOC THUAT:\n"
                    "- Lap luan chat che, logic\n"
                    "- Neu su dung thong tin tu tai lieu tham khao, hay trich dan ro rang\n"
                    "- Neu tai lieu khong du, can neu ro gioi han thong tin\n"
                )
            }
        ]

        if rag_context:
            messages.append({
                "role": "system",
                "content": f"TAI LIEU THAM KHAO:\n{rag_context}"
            })

        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        return messages

    # ======================================================
    # HELPERS
    # ======================================================
    def _build_rag_context(self, docs: List[Dict]) -> str:
        """
        Gop cac doan truy xuat thanh context chuan hoa
        """
        if not docs:
            return ""

        context_blocks = []

        for i, item in enumerate(docs, start=1):
            # Handle both formats
            text = item.get("text", "")
            meta = item.get("metadata", {})
            
            # Fallback if no text key
            if not text and isinstance(item, dict):
                text = str(item)

            source = meta.get("source_file", meta.get("filename", "unknown"))
            agent = meta.get("agent", "unknown")

            block = (
                f"[{i}] Nguon: {source} | Nhom: {agent}\n"
                f"{text[:500]}"  # Limit to 500 chars per doc
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks[:5])  # Max 5 documents

    def _agent_key(self) -> str:
        """
        Chuan hoa agent name -> retrieval filter
        """
        name = self.name.lower()

        if "chinh phu" in name or "government" in name:
            return "government"
        if "doanh nghiep" in name or "business" in name:
            return "business"
        if "chuyen gia" in name or "expert" in name:
            return "expert"

        return "expert"  # fallback an toan