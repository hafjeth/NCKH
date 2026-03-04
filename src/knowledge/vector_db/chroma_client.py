"""
ChromaDB HTTP Client Wrapper (V2 API – Multi-tenant)
- Compatible with ChromaDB Docker server
- Auto-embedding using SentenceTransformers
- Research-grade: timeout, validation, retrieval metadata
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================
# MAIN CLIENT
# =========================
class ChromaDBClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        tenant: str = "default_tenant",
        database: str = "default_database",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.host = host
        self.port = port
        self.tenant = tenant
        self.database = database
        self.base_url = f"http://{host}:{port}/api/v2"

        logger.info(f"🔹 Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/heartbeat", timeout=5)
            r.raise_for_status()
            logger.info(f"✅ Connected to ChromaDB ({self.host}:{self.port})")
            logger.info(f"   Tenant={self.tenant} | DB={self.database}")
        except Exception as e:
            raise ConnectionError(f"❌ Cannot connect to ChromaDB: {e}")

    def list_collections(self) -> List[Dict]:
        try:
            url = f"{self.base_url}/tenants/{self.tenant}/databases/{self.database}/collections"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"List collections failed: {e}")
            return []

    def get_or_create_collection(self, name: str, metadata: Optional[Dict] = None):
        return ChromaCollection(self, name, metadata)


# =========================
# COLLECTION
# =========================
class ChromaCollection:
    def __init__(self, client: ChromaDBClient, name: str, metadata: Optional[Dict]):
        self.client = client
        self.name = name
        self.metadata = metadata
        self.embedder = client.embedder
        self.base_url = client.base_url
        self.tenant = client.tenant
        self.database = client.database
        self.collection_id = None

        self._ensure_collection()

    def _ensure_collection(self):
        try:
            url = f"{self.base_url}/tenants/{self.tenant}/databases/{self.database}/collections"
            payload = {
                "name": self.name,
                "get_or_create": True,
            }
            if self.metadata:
                payload["metadata"] = self.metadata

            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            self.collection_id = r.json()["id"]

            logger.info(f"✅ Collection ready: {self.name} ({self.collection_id})")
        except Exception as e:
            raise RuntimeError(f"Create/get collection failed: {e}")

    # =========================
    # ADD DOCUMENTS
    # =========================
    def add(
        self,
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
    ):
        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError("documents, metadatas, and ids must have the same length")

        try:
            if embeddings is None:
                logger.info(f"🔹 Embedding {len(documents)} documents")
                embeddings = self.embedder.encode(documents).tolist()

            payload = {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "embeddings": embeddings,
            }

            url = (
                f"{self.base_url}/tenants/{self.tenant}/databases/"
                f"{self.database}/collections/{self.collection_id}/add"
            )
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()

            logger.info(f"✅ Added {len(documents)} documents")

        except Exception as e:
            raise RuntimeError(f"Add documents failed: {e}")

    # =========================
    # QUERY (ADAPTIVE)
    # =========================
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        similarity_threshold: float = 0.75,
        adaptive: bool = True,
        min_results: int = 2,
        max_results: int = 7,
    ) -> Dict[str, Any]:

        if not query_texts and not query_embeddings:
            raise ValueError("Provide query_texts or query_embeddings")

        if query_texts:
            query_embeddings = self.embedder.encode(query_texts).tolist()

        candidate_pool = max(n_results, 15) if adaptive else n_results

        payload = {
            "query_embeddings": query_embeddings,
            "n_results": candidate_pool,
        }
        if where:
            payload["where"] = where

        url = (
            f"{self.base_url}/tenants/{self.tenant}/databases/"
            f"{self.database}/collections/{self.collection_id}/query"
        )

        try:
            r = requests.post(url, json=payload, timeout=30)
            r.raise_for_status()
            raw = r.json()
        except Exception as e:
            return self._empty_result(str(e), adaptive)

        if not raw or "ids" not in raw:
            return self._empty_result("Invalid response", adaptive)

        if not adaptive:
            return raw

        # =========================
        # Adaptive filtering
        # =========================
        results = {k: [[] for _ in raw["ids"]] for k in ["ids", "documents", "metadatas", "distances"]}
        stats = []

        for qi in range(len(raw["ids"])):
            similarities = []

            for i, dist in enumerate(raw["distances"][qi]):
                # ChromaDB cosine distance ∈ [0, 2]
                # similarity = 1 - (distance / 2)
                sim = 1 - dist / 2

                if sim >= similarity_threshold and len(results["ids"][qi]) < max_results:
                    for k in results:
                        results[k][qi].append(raw[k][qi][i])
                    similarities.append(sim)

            stats.append({
                "query_index": qi,
                "retrieved": len(results["ids"][qi]),
                "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
            })

        results["retrieval_metadata"] = {
            "adaptive": True,
            "threshold": similarity_threshold,
            "candidate_pool": candidate_pool,
            "total_retrieved": sum(len(x) for x in results["ids"]),
            "stats": stats,
        }

        return results

    def _empty_result(self, error: str, adaptive: bool):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "retrieval_metadata": {
                "adaptive": adaptive,
                "error": error,
            },
        }

    # =========================
    # UTILS
    # =========================
    def count(self) -> int:
        try:
            url = (
                f"{self.base_url}/tenants/{self.tenant}/databases/"
                f"{self.database}/collections/{self.collection_id}/count"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return 0

    def delete(self, ids: List[str]):
        url = (
            f"{self.base_url}/tenants/{self.tenant}/databases/"
            f"{self.database}/collections/{self.collection_id}/delete"
        )
        requests.post(url, json={"ids": ids}, timeout=10)

    def get(self, ids=None, limit=None, where=None, include=None):
        payload = {}

        if ids is not None:
            payload["ids"] = ids
        if limit is not None:
            payload["limit"] = limit
        if where is not None:
            payload["where"] = where

        payload["include"] = include or ["documents", "metadatas"]

        url = (
            f"{self.base_url}/tenants/{self.tenant}/databases/"
            f"{self.database}/collections/{self.collection_id}/get"
        )
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
