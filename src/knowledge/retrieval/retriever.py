"""
Retrieval System
=====================================
Semantic + Agent-aware Retrieval for RAG
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# ================= CONFIG =================
# [FIX] Dùng đúng model với collection đã ingest
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Map agent type → list of matching 'subjects' values in ChromaDB
AGENT_SUBJECTS_MAP = {
    "government": [
        "state_agency",
        "state_agency,enterprise",
        "state_agency,individual",
        "state_agency,organization",
        "state_agency,individual,household",
        "state_agency,organization,individual",
        "state_agency,enterprise,organization,producer,importer",
        "organization",
        "unspecified",
    ],
    "business": [
        "enterprise",
        "state_agency,enterprise",
        "state_agency,enterprise,organization,producer,importer",
        "organization,individual,household,importer",
        "unspecified",
    ],
    "expert": None,
}

# Source boost map
SOURCE_BOOST_MAP = [
    (r"quyết định 232|QĐ-TTg.*232|232.*QĐ-TTg|\b232/QĐ",
     "Quyết định 232 QĐ‑TTg.txt", 3),
    (r"quyết định 888|QĐ-TTg.*888|888.*QĐ-TTg|\b888/QĐ",
     "Quyết định 888 QĐ‑TTg.txt", 3),
    (r"quyết định 450|QĐ-TTg.*450|450.*QĐ-TTg|\b450/QĐ",
     "Quyết định 450 QĐ‑TTg.txt", 2),
    (r"quyết định 1658|1658.*QĐ-TTg|\b1658/QĐ",
     "Quyết định 1658 QĐ-TTg.txt", 2),
    (r"quyết định 896|896.*QĐ-TTg|\b896/QĐ",
     "Quyết định 896 QĐ-TTg.txt", 2),
    (r"nghị định 06|06/2022/NĐ-CP",
     "Nghị định 06 2022 NĐ-CP.txt", 2),
    (r"nghị định 08|08/2022/NĐ-CP",
     "Nghị định 08 2022 NĐ-CP.txt", 2),
    (r"luật bảo vệ môi trường 2020|LBVMT|luật BVMT",
     "Luật Bảo vệ Môi trường 2020.txt", 2),
]


class KnowledgeRetriever:
    """Wrapper cho ChromaDB retrieval với multilingual embedding và source boosting."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "carbon_policy_textile_vn"
    ):
        self.client = chromadb.HttpClient(host=host, port=port)

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )
        logger.info(f"✅ ChromaDB connected: {host}:{port}")
        logger.info(f"   Embedding: {EMBEDDING_MODEL}")
        logger.info(f"   Collection: {collection_name} | Docs: {self.collection.count()}")

    # ======================================================
    # MAIN RETRIEVAL
    # ======================================================
    def retrieve(
        self,
        query: str,
        agent: str,
        k: int = 10,
        semantic_hint: Optional[Dict] = None
    ) -> List[Dict]:
        where = self._build_filter(agent)

        # [FIX] Thêm filter từ semantic_hint nếu có
        if semantic_hint:
            where = self._merge_filters(where, semantic_hint)

        results = self._query_with_fallback(query=query, where=where, k=k)
        formatted = self._format_results(results)

        # Apply similarity threshold
        try:
            from src.core.config import Config
            threshold = Config.RAG_RETRIEVAL.get("similarity_threshold", 0.0)
        except Exception:
            threshold = 0.0

        if threshold > 0:
            before = len(formatted)
            formatted = [item for item in formatted if item.get("score", 0.0) >= threshold]
            after = len(formatted)
            if before != after:
                logger.info(f"🔽 Threshold filter ({threshold}): {before} → {after} chunks")

            min_results = 2
            try:
                min_results = Config.RAG_RETRIEVAL.get("min_results", 2)
            except Exception:
                pass
            if len(formatted) < min_results and before > 0:
                all_sorted = sorted(
                    self._format_results(results),
                    key=lambda x: x.get("score", 0.0),
                    reverse=True
                )
                formatted = all_sorted[:min_results]
                logger.warning(f"⚠️ Threshold too strict — fallback to top-{min_results} chunks")

        # Source boosting
        boosted = self._apply_source_boost(query=query, existing=formatted, k=k)

        return boosted

    def _merge_filters(self, existing: Optional[Dict], hint: Dict) -> Dict:
        """Merge existing filter với semantic_hint."""
        result = existing.copy() if existing else {}
        
        if "domains" in hint and hint["domains"]:
            result["domains"] = {"$in": hint["domains"]}
        
        if "clause_type" in hint and hint["clause_type"]:
            result["clause_type"] = {"$eq": hint["clause_type"]}
        
        return result

    # ======================================================
    # SOURCE BOOSTING
    # ======================================================
    def _apply_source_boost(
        self,
        query: str,
        existing: List[Dict],
        k: int
    ) -> List[Dict]:
        boost_source, boost_n = self._detect_boost_source(query)

        if not boost_source or boost_n == 0:
            return existing

        existing_ids = set(item["text"][:100] for item in existing)

        try:
            boost_results = self.collection.query(
                query_texts=[query],
                n_results=boost_n,
                where={"source_file": {"$eq": boost_source}}
            )
            boost_formatted = self._format_results(boost_results)

            new_chunks = [
                item for item in boost_formatted
                if item["text"][:100] not in existing_ids
            ]

            if new_chunks:
                logger.info(f"🎯 Source boost: +{len(new_chunks)} chunks từ '{boost_source}'")
                combined = new_chunks + existing
                return combined[:k]

        except Exception as e:
            logger.warning(f"⚠️ Source boost failed for '{boost_source}': {e}")

        return existing

    def _detect_boost_source(self, query: str) -> tuple:
        query_lower = query.lower()
        for pattern, source_file, n in SOURCE_BOOST_MAP:
            if source_file and re.search(pattern, query, re.IGNORECASE):
                return source_file, n
        return None, 0

    # ======================================================
    # FILTER BUILDER
    # ======================================================
    def _build_filter(self, agent: str) -> Optional[Dict]:
        subjects_list = AGENT_SUBJECTS_MAP.get(agent.lower())
        if not subjects_list:
            return None
        return {"subjects": {"$in": subjects_list}}

    # ======================================================
    # QUERY WITH FALLBACK
    # ======================================================
    def _query_with_fallback(self, query: str, where: Optional[Dict], k: int):
        if where:
            results = self._query(query, where, k)
            if self._has_results(results):
                logger.info("🔎 Retrieval: filter matched")
                return results
            logger.warning("⚠️ Filter returned no results, falling back to no filter")

        logger.info("🔎 Retrieval: no filter (fallback)")
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
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            formatted.append({
                "text":     doc,
                "metadata": meta,
                "score":    round(score, 4),
            })
        return formatted

    # ======================================================
    # UTILITY METHODS
    # ======================================================
    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Lấy chunk theo ID (debugging)."""
        results = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if results["ids"]:
            return {
                "text": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def get_collection_stats(self) -> Dict:
        return {"total_documents": self.collection.count()}