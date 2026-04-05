"""
BaseAgent
=====================================
Agent nền tảng cho hệ thống tranh luận đa agent
Tích hợp RAG (KnowledgeRetriever) + citation-aware
+ Fallback & Cache support

THAY ĐỔI SO VỚI BẢN TRƯỚC:
  [FIX-6] _build_rag_context(): tăng giới hạn text từ 500 → 1200 chars/doc
          và tăng số doc tối đa từ 5 → 7
  [FIX-8] Hiển thị rõ tên nguồn từng chunk — tránh gán nhầm citation
  [FIX-A] __init__(): hỗ trợ cả OpenAI và Anthropic dựa theo Config.PROVIDER
  [FIX-B] _build_messages(): bỏ dòng cho phép 'theo thực tiễn quốc tế'
  [FIX-C] _build_rag_context(): sort docs theo relevance score trước khi cắt
  [FIX-D] _build_rag_context(): clean đuôi file (.txt/.pdf/.docx) khỏi tên nguồn
  [FIX-13] Thêm delay 1-2s trước mỗi API call để tránh rate limit
  [FIX-15] Mapping tên nguồn đặc biệt (WWF, IFC, v.v.) để citation đẹp hơn
"""

import re
import time
import random
from typing import List, Dict, Optional
import logging

from src.core.config import Config
from src.knowledge.retrieval.retriever import KnowledgeRetriever
from src.core.fallback_manager import get_fallback_manager

logger = logging.getLogger(__name__)


# [FIX-15] Mapping tên nguồn đặc biệt → tên học thuật
SOURCE_NAME_MAPPING = {
    # WWF reports
    "greening-textile-industry-in-vietnam-5922q9q6cg": "Greening Textile Industry in Vietnam (WWF)",
    "greening-textile-industry-in-vietnam": "Greening Textile Industry in Vietnam (WWF)",
    "Greening Textile Sector in Vietnam (WWF)": "Greening Textile Sector in Vietnam (WWF)",
    "Greening Textile Sector in Vietnam (WWF).eng": "Greening Textile Sector in Vietnam (WWF)",
    
    # IFC / World Bank
    "Báo cáo của IFC World Bank về Xanh hóa ngành dệt may Việt Nam": "IFC World Bank: Greening Vietnam's Textile Sector",
    "Báo cáo của IFC World Bank về Xanh hóa ngành dệt may Việt Nam.": "IFC World Bank: Greening Vietnam's Textile Sector",
    
    # Legal documents
    "Luật Bảo vệ Môi trường 2020": "Luật Bảo vệ Môi trường 2020",
    "Nghị định 06 2022 NĐ-CP": "Nghị định 06/2022/NĐ-CP",
    "Nghị định 08 2022 NĐ-CP": "Nghị định 08/2022/NĐ-CP",
    "Nghị định 45 2022 NĐ‑CP": "Nghị định 45/2022/NĐ-CP",
    "Quyết định 232 QĐ‑TTg": "Quyết định 232/QĐ-TTg",
    "Quyết định 01 2022 QĐ-TTg": "Quyết định 01/2022/QĐ-TTg",
    "Quyết định 888 QĐ‑TTg": "Quyết định 888/QĐ-TTg",
    "Quyết định 450 QĐ‑TTg": "Quyết định 450/QĐ-TTg",
    
    # CBAM / EU documents
    "CBAM_Questions and Answers": "EU CBAM: Questions and Answers",
    "Guidance document on CBAM implementation for importers": "EU CBAM Guidance for Importers",
    "Guidance document on CBAM implementation for installation operators": "EU CBAM Guidance for Installation Operators",
    "Regulation (EU) 2023 956": "EU CBAM Regulation (2023/956)",
    
    # Business reports
    "deloitte-nl-tax-cbam-compliance-manager_two-pager": "Deloitte: CBAM Compliance Manager",
    "KPMG CBAM Report (PDF)": "KPMG: CBAM Report",
}


def _clean_source_name(source: str) -> str:
    """
    [FIX-D] Bỏ đuôi file khỏi tên nguồn để citation trông học thuật hơn.
    [FIX-15] Áp dụng mapping cho các tên đặc biệt.
    """
    # Kiểm tra mapping trước
    source_lower = source.lower()
    for key, value in SOURCE_NAME_MAPPING.items():
        if key.lower() in source_lower or source_lower in key.lower():
            return value
    
    # Bỏ đuôi file
    cleaned = re.sub(
        r'\.(txt|pdf|docx|doc|xlsx|csv|json)$', '',
        source,
        flags=re.IGNORECASE
    ).strip()
    
    # Bỏ _paragraphs, _clauses, _semantic
    cleaned = re.sub(r'_(paragraphs|clauses|semantic)$', '', cleaned, flags=re.IGNORECASE)
    
    # Bỏ hash ID (dạng -5922q9q6cg)
    cleaned = re.sub(r'-[a-z0-9]{8,}$', '', cleaned, flags=re.IGNORECASE)
    
    # Thay dấu gạch ngang bằng khoảng trắng
    cleaned = cleaned.replace('-', ' ').replace('_', ' ')
    
    # Viết hoa chữ đầu mỗi từ
    cleaned = ' '.join(word.capitalize() for word in cleaned.split())
    
    return cleaned


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

        # [FIX-A] Hỗ trợ cả OpenAI và Anthropic theo Config.PROVIDER
        provider = getattr(Config, "PROVIDER", "openai").lower()
        self._provider = provider

        if provider == "anthropic":
            if not getattr(Config, "ANTHROPIC_API_KEY", None):
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Hãy set PROVIDER=anthropic và cung cấp ANTHROPIC_API_KEY."
                )
            from anthropic import Anthropic
            self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            logger.info(f"[{self.name}] Khởi tạo với Anthropic client")
        else:
            if not getattr(Config, "OPENAI_API_KEY", None):
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Hãy set OPENAI_API_KEY hoặc chuyển PROVIDER=anthropic."
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info(f"[{self.name}] Khởi tạo với OpenAI client")

        self.fallback_manager = get_fallback_manager(Config)

        # Lưu phục vụ debug / evaluation
        self.last_rag_context: str = ""
        self.last_rag_metadata: Dict = {}
        self.last_call_metadata: Dict = {}

    # ======================================================
    # MAIN CHAT FUNCTION (with fallback & cache)
    # ======================================================
    def chat(self, user_prompt: str) -> str:
        """Sinh phản hồi của agent với RAG"""
        
        # [FIX-13] Anti rate limiting - random delay 1-2 seconds
        delay = random.uniform(1.0, 2.0)
        logger.debug(f"[{self.name}] Waiting {delay:.2f}s before API call")
        time.sleep(delay)

        rag_context = ""
        retrieval_result = None

        if self.retriever:
            try:
                retrieval_result = self.retriever.retrieve(
                    query=user_prompt,
                    agent=self._agent_key()
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Retrieval failed: {e}")

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
        messages = self._build_messages(user_prompt, rag_context)

        use_cache = Config.FALLBACK_CONFIG.get("use_cache", True)
        result = self.fallback_manager.call_with_fallback(
            messages=messages,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            max_retries=Config.FALLBACK_CONFIG.get("max_retries_per_model", 2),  # [FIX-14] Giảm từ 3 → 2
            use_cache=use_cache
        )

        self.last_call_metadata = {
            "model_used": result.get("model_used"),
            "from_cache": result.get("from_cache", False),
            "attempts":   result.get("attempts", 1),
            "error":      result.get("error")
        }

        if result.get("error"):
            logger.error(f"[{self.name}] All models failed: {result['error']}")
            return f"[ERROR] Agent {self.name} không thể trả lời do lỗi hệ thống."

        return result.get("response", "")

    def _build_messages(self, user_prompt: str, rag_context: str) -> List[Dict]:
        """
        Xây dựng messages cho LLM call.

        [FIX-B] Bỏ dòng cho phép 'theo thực tiễn quốc tế' —
                nếu không có tài liệu phù hợp thì trình bày bình thường,
                không dùng bất kỳ cụm giả nguồn nào.
        """
        messages = [
            {"role": "system", "content": self.role},
            {
                "role": "system",
                "content": (
                    "YÊU CẦU HỌC THUẬT:\n"
                    "- Lập luận chặt chẽ, logic\n"
                    "- Nếu sử dụng thông tin từ tài liệu tham khảo, hãy trích dẫn rõ ràng\n"
                    "- Nếu tài liệu không đủ, cần nêu rõ giới hạn thông tin\n"
                )
            }
        ]

        if rag_context:
            messages.append({
                "role": "system",
                "content": (
                    f"TÀI LIỆU THAM KHẢO (truy xuất từ knowledge base):\n\n"
                    f"{rag_context}\n\n"
                    "QUY TẮC CITATION BẮT BUỘC:\n"
                    "- CHỈ cite nguồn có trong danh sách TÀI LIỆU THAM KHẢO bên trên\n"
                    "- Format bắt buộc: [Nguồn: <tên nguồn chính xác như trong TÀI LIỆU>]\n"
                    "- Trước khi cite [Nguồn: X], kiểm tra: nội dung bạn trình bày "
                    "có XUẤT HIỆN TRONG ĐOẠN VĂN của [TÀI LIỆU X] không?\n"
                    "- Nếu KHÔNG tìm thấy trong đoạn văn đó → KHÔNG cite, "
                    "dù bạn biết X có liên quan\n"
                    "- Nếu không có tài liệu phù hợp → trình bày bình thường, "
                    "KHÔNG dùng bất kỳ cụm nào giả dạng nguồn như "
                    "'theo thực tiễn quốc tế', 'theo nghiên cứu quốc tế', v.v.\n"
                    "- KHÔNG tự bịa tên tài liệu không có trong danh sách\n"
                    "- Tối thiểu 2 citations/lượt nếu tài liệu có nội dung liên quan"
                )
            })
        else:
            messages.append({
                "role": "system",
                "content": (
                    "KHÔNG có tài liệu tham khảo nào được truy xuất cho câu hỏi này.\n"
                    "QUY TẮC: KHÔNG cite bất kỳ nguồn nào. KHÔNG dùng cụm "
                    "'theo thực tiễn quốc tế' hay tương tự như một dạng citation. "
                    "Trình bày dựa trên lập luận logic thuần túy."
                )
            })

        messages.append({"role": "user", "content": user_prompt})
        return messages

    # ======================================================
    # HELPERS
    # ======================================================
    def _build_rag_context(self, docs: List[Dict]) -> str:
        """
        Gộp các đoạn truy xuất thành context chuẩn hóa.

        [FIX-6] Tăng giới hạn text: 500 → 1200 chars/doc, tối đa 7 docs
        [FIX-8] Hiển thị rõ tên nguồn từng chunk
        [FIX-C] Sort docs theo relevance score giảm dần
        [FIX-D] Clean đuôi file (.txt/.pdf) khỏi tên nguồn
        [FIX-15] Áp dụng mapping tên nguồn
        """
        if not docs:
            return ""

        # [FIX-C] Sort theo score giảm dần — ưu tiên docs liên quan nhất
        docs_sorted = sorted(
            docs,
            key=lambda x: x.get("score",
                         x.get("relevance_score",
                         x.get("similarity_score", 0.0))),
            reverse=True
        )

        context_blocks = []

        for i, item in enumerate(docs_sorted[:7], start=1):
            text = item.get("text", "")
            meta = item.get("metadata", {})

            if not text and isinstance(item, dict):
                text = str(item)

            raw_source = (
                meta.get("source_file")
                or meta.get("filename")
                or meta.get("title")
                or "unknown"
            )

            # [FIX-D] + [FIX-15] Clean đuôi file + mapping → citation học thuật
            source = _clean_source_name(raw_source)

            score = item.get("score",
                    item.get("relevance_score",
                    item.get("similarity_score", None)))
            score_str = f" | Relevance: {score:.3f}" if score is not None else ""

            block = (
                f"[TÀI LIỆU {i}{score_str}]\n"
                f"Nguồn: {source}\n"
                f"Nội dung: {text[:1200]}\n"
                f"⚠️ Chỉ cite [{source}] cho nội dung CÓ TRONG đoạn văn trên."
            )
            context_blocks.append(block)

        logger.debug(
            f"[RAG] Built context: {len(context_blocks)} docs | "
            f"sources: {[b.split(chr(10))[1].replace('Nguồn: ', '') for b in context_blocks]}"
        )

        return "\n\n---\n\n".join(context_blocks)

    def _agent_key(self) -> str:
        """Chuẩn hóa agent name → retrieval filter key"""
        name = self.name.lower()
        if "chinh phu" in name or "government" in name:
            return "government"
        if "doanh nghiep" in name or "business" in name:
            return "business"
        if "chuyen gia" in name or "expert" in name:
            return "expert"
        return "expert"