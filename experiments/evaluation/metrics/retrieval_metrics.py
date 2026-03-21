import re
import numpy as np
from typing import List, Dict, Union
from collections import Counter
import math

# =======================
# CitationCounter (FIXED — regex-based, no external dependency)
# =======================
class CitationCounter:
    # Matches: [Nguồn: ...], [Source: ...], (Nguồn: ...), Nguồn: ...
    CITATION_PATTERN = re.compile(
        r'(?:\[Nguồn:[^\]]+\]|\[Source:[^\]]+\]|\(Nguồn:[^)]+\)|Nguồn:\s*[^,;\n]+)',
        re.IGNORECASE | re.UNICODE
    )

    def count_citations(self, text: str) -> int:
        return len(self.CITATION_PATTERN.findall(text or ''))

    def extract_citations(self, text: str):
        return self.CITATION_PATTERN.findall(text or '')

    def has_source_section(self, text: str):
        found = self.CITATION_PATTERN.findall(text or '')
        return (len(found) > 0, len(found))

    def get_citation_density(self, text: str) -> float:
        if not text:
            return 0.0
        words = len(re.findall(r'\b\w+\b', text))
        citations = self.count_citations(text)
        return round(citations / max(words, 1) * 1000, 4)

# =======================
# Embedding (OPTIONAL)
# =======================
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class MetricsCalculator:
    """
    Quantitative metrics for evaluating multi-agent debate quality
    """

    def __init__(self, embedding_model: str = None):
        self.citation_counter = CitationCounter()

        self.model = None
        if embedding_model and EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer(embedding_model)
                print(f" MetricsCalculator: Embedding model loaded ({embedding_model})")
            except Exception:
                self.model = None
                print(" MetricsCalculator: Failed to load embedding model, fallback to lexical")

    # =====================================================
    # METRIC 1: LENGTH
    # =====================================================
    def count_words(self, text: str) -> Dict[str, int]:
        if not text or not isinstance(text, str):
            return dict(word_count=0, char_count=0, sentence_count=0, avg_word_per_sentence=0)

        words = re.findall(r"\b\w+\b", text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

        return {
            "word_count": len(words),
            "char_count": len(re.sub(r"\s+", "", text)),
            "sentence_count": len(sentences),
            "avg_word_per_sentence": round(len(words) / max(len(sentences), 1), 2)
        }

    # =====================================================
    # METRIC 2: CITATION
    # =====================================================
    def count_citations(self, text: str) -> Dict:
        if not text:
            return dict(total_citations=0, citation_density=0, found_citations=[])

        return {
            "total_citations": self.citation_counter.count_citations(text),
            "citation_density": self.citation_counter.get_citation_density(text),
            "found_citations": self.citation_counter.extract_citations(text)[:5],
            "has_sources": self.citation_counter.has_source_section(text)[0]
        }

    # =====================================================
    # METRIC 3: DIVERSITY
    # =====================================================
    def diversity_score(self, texts: List[str], method: str = "lexical") -> Dict:
        if len(texts) < 2:
            return dict(diversity_score=0.0, method_used=method)

        texts = [t.strip() for t in texts if t.strip()]

        if method == "embedding" and self.model:
            return self._embedding_diversity(texts)

        return self._lexical_diversity(texts)

    def _lexical_diversity(self, texts: List[str]) -> Dict:
        sets = [set(re.findall(r"\b\w+\b", t.lower())) for t in texts]

        distances = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                distances.append(1 - inter / union if union else 0)

        return {
            "diversity_score": round(float(np.mean(distances)), 4),
            "method_used": "lexical",
            "explanation": "Lexical Jaccard distance across agent responses"
        }

    def _embedding_diversity(self, texts: List[str]) -> Dict:
        emb = self.model.encode(texts)
        sims = []

        for i in range(len(emb)):
            for j in range(i + 1, len(emb)):
                sims.append(
                    np.dot(emb[i], emb[j]) /
                    (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
                )

        return {
            "diversity_score": round(float(1 - np.mean(sims)), 4),
            "method_used": "embedding",
            "explanation": "Semantic diversity based on cosine distance"
        }

    # =====================================================
    # COMPARISON
    # =====================================================
    def compare_responses(self, baseline: str, agents: List[str]) -> Dict:
        base_len = self.count_words(baseline)["word_count"]
        agent_lens = [self.count_words(a)["word_count"] for a in agents]

        base_len = max(base_len, 1)

        return {
            "avg_agent_length": round(np.mean(agent_lens), 2),
            "length_change_pct": round((np.mean(agent_lens) - base_len) / base_len * 100, 2),
            "diversity": self.diversity_score(agents)
        }

    def compute(self, judgment: dict) -> dict:
        """
        FIX: Adapter method for Evaluator.evaluate().
        Receives judgment dict from LLMJudge, returns aggregated metrics.
        """
        text = judgment.get("explanation", "") or str(judgment)
        return {
            "word_count":     self.count_words(text),
            "citation_count": self.count_citations(text),
            "coherence":      judgment.get("coherence", 0),
            "factuality":     judgment.get("factuality", 0),
        }