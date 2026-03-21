"""
Retrieval System
=====================================
Semantic + Agent-aware Retrieval for RAG

FIX 1: HttpClient → PersistentClient
FIX 2: metadata key 'agent' → 'subjects'
FIX 3: subjects values mapped correctly per agent type
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import chromadb

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "vector_stores" / "chroma"
)

# FIX: Map agent type → list of matching 'subjects' values in ChromaDB
AGENT_SUBJECTS_MAP = {
    "government": [
        "state_agency",
        "state_agency,enterprise",
        "state_agency,individual",
        "state_agency,organization",
        "state_agency,individual,household",
        "state_agency,organization,individual",
        "state_agency,enterprise,organization,producer,importer",
    ],
    "business": [
        "enterprise",
        "state_agency,enterprise",
        "state_agency,enterprise,organization,producer,importer",
        "organization,individual,household,importer",
    ],
    "expert": None,  # No filter — retrieve from all documents
}


class KnowledgeRetriever:
    """
    Wrapper cho ChromaDB retrieval
    """

    def __init__(
        self,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        collection_name: str = "carbon_policy_textile_vn"
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        logger.info(f"✅ ChromaDB connected: {persist_dir}")
        logger.info(f"   Collection: {collection_name} | Docs: {self.collection.count()}")

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

        # FIX: Build filter using correct subjects values
        where = self._build_filter(agent)

        results = self._query_with_fallback(
            query=query,
            where=where,
            k=k
        )

        return self._format_results(results)

    # ======================================================
    # FILTER BUILDER
    # ======================================================
    def _build_filter(self, agent: str) -> Optional[Dict]:
        """
        Build ChromaDB WHERE filter based on agent type.
        Uses $in operator with known subjects values.
        """
        subjects_list = AGENT_SUBJECTS_MAP.get(agent.lower())

        if not subjects_list:
            return None  # expert or unknown → no filter

        return {"subjects": {"$in": subjects_list}}

    # ======================================================
    # QUERY WITH FALLBACK
    # ======================================================
    def _query_with_fallback(
        self,
        query: str,
        where: Optional[Dict],
        k: int
    ):
        # 1️⃣ Try with filter
        if where:
            results = self._query(query, where, k)
            if self._has_results(results):
                logger.info(f"🔎 Retrieval: filter matched")
                return results
            logger.warning(f"⚠️ Filter returned no results, falling back to no filter")

        # 2️⃣ Fallback: no filter
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

        formatted = []
        for doc, meta in zip(documents, metadatas):
            formatted.append({
                "text": doc,
                "metadata": meta
            })

        return formatted

    # ======================================================
    # STATS
    # ======================================================
    def get_collection_stats(self) -> Dict:
        return {
            "total_documents": self.collection.count()
        }