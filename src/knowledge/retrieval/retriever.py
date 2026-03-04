"""
Retrieval System
=====================================
Semantic + Agent-aware Retrieval for RAG
"""

import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Wrapper cho ChromaDB retrieval
    Có hỗ trợ agent + semantic hint
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "carbon_policy_textile_vn"
    ):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    # ======================================================
    # MAIN RETRIEVAL
    # ======================================================
    def retrieve(
        self,
        query: str,
        agent: str,
        k: int = 5,
        semantic_hint: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve documents với filter thông minh

        Args:
            query: câu hỏi
            agent: government | business | expert
            k: số chunk cần lấy
            semantic_hint: {
                "focus": [...],
                "stance": "...",
                "cbam_relevance": bool
            }
        """

        # ----------------------------
        # STEP 1: build WHERE filter
        # ----------------------------
        base_filter = {"agent": agent}

        # thử full semantic filter trước
        where = dict(base_filter)

        if semantic_hint:
            where = self._extend_where(where, semantic_hint)

        # ----------------------------
        # STEP 2: query with fallback
        # ----------------------------
        results = self._query_with_fallback(
            query=query,
            where=where,
            base_filter=base_filter,
            k=k
        )

        return self._format_results(results)

    # ======================================================
    # INTERNAL METHODS
    # ======================================================
    def _extend_where(self, where: Dict, semantic_hint: Dict) -> Dict:
        """
        Gộp semantic hint vào where clause
        """
        extended = dict(where)

        if semantic_hint.get("focus"):
            extended["focus"] = {
                "$in": semantic_hint["focus"]
            }

        if semantic_hint.get("stance"):
            extended["stance"] = semantic_hint["stance"]

        if semantic_hint.get("cbam_relevance") is True:
            extended["cbam_relevance"] = True

        return extended

    def _query_with_fallback(
        self,
        query: str,
        where: Dict,
        base_filter: Dict,
        k: int
    ):
        """
        Thử truy vấn từ chặt → lỏng
        """

        # 1️⃣ Full filter
        results = self._query(query, where, k)
        if self._has_results(results):
            logger.info("🔎 Retrieval: full semantic filter")
            return results

        # 2️⃣ Remove stance
        relaxed = dict(where)
        relaxed.pop("stance", None)
        results = self._query(query, relaxed, k)
        if self._has_results(results):
            logger.info("🔎 Retrieval: relaxed stance")
            return results

        # 3️⃣ Remove focus
        relaxed = dict(base_filter)
        results = self._query(query, relaxed, k)
        if self._has_results(results):
            logger.info("🔎 Retrieval: agent-only filter")
            return results

        # 4️⃣ Absolute fallback: no filter
        logger.warning("⚠️ Retrieval fallback: no filter")
        return self._query(query, None, k)

    def _query(self, query: str, where: Optional[Dict], k: int):
        return self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )

    def _has_results(self, results) -> bool:
        docs = results.get("documents", [])
        return bool(docs and docs[0])

    def _format_results(self, results) -> List[Dict]:
        """
        Chuẩn hóa output cho Agent
        """
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        formatted = []
        for doc, meta in zip(documents, metadatas):
            formatted.append({
                "text": doc,
                "metadata": meta
            })

        return formatted

    # ======================================================
    # STATS (OPTIONAL – dùng cho logging / debug)
    # ======================================================
    def get_collection_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count()
        }
