"""
Quantitative Evaluation Metrics Module for Multi-Agent Debate System
Author: [Student Name]
Date: 2024-01-09
Project: Multi-Agent Debate System for Carbon Tax Policy Analysis - Vietnam Textile Industry
"""

import re
import numpy as np
from typing import List, Dict, Union
from collections import Counter
import math

# Embedding library (install: pip install sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Warning: sentence-transformers not installed. diversity_score will use fallback method.")


class MetricsCalculator:
    """Class for calculating evaluation metrics"""
    
    def __init__(self, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize calculator with embedding model
        
        Args:
            embedding_model: Model name from sentence-transformers
                           Default uses multilingual model supporting Vietnamese
        """
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer(embedding_model)
        else:
            self.model = None
    
    # METRIC 1: Response Length
    
    def count_words(self, text: str) -> Dict[str, int]:
        """
        Count words in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing statistics:
            - word_count: Total words
            - char_count: Total characters (excluding whitespace)
            - sentence_count: Number of sentences
            - avg_word_per_sentence: Average words per sentence
        """
        if not text or not isinstance(text, str):
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_per_sentence': 0.0
            }
        
        # Clean text
        text_clean = text.strip()
        
        # Count words (split by whitespace and special characters)
        words = re.findall(r'\b\w+\b', text_clean)
        word_count = len(words)
        
        # Count characters (excluding whitespace)
        char_count = len(re.sub(r'\s+', '', text_clean))
        
        # Count sentences (based on punctuation)
        sentences = re.split(r'[.!?]+', text_clean)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Calculate average
        avg_word_per_sentence = word_count / sentence_count if sentence_count > 0 else 0.0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_per_sentence': round(avg_word_per_sentence, 2)
        }
    
    # METRIC 2: Citation Count
    
    def count_citations(self, text: str) -> Dict[str, Union[int, List[str]]]:
        """
        Count citations in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing:
            - total_citations: Total number of citations
            - citation_patterns: Dict counting each pattern type
            - citation_density: Citations per 100 words
            - found_citations: List of found citations
        """
        if not text or not isinstance(text, str):
            return {
                'total_citations': 0,
                'citation_patterns': {},
                'citation_density': 0.0,
                'found_citations': []
            }
        
        # Common citation patterns (English and Vietnamese)
        citation_patterns = {
            'according_to_report': r'[Aa]ccording to (the )?report\s+[\w\s,]+|[Tt]heo báo cáo\s+[\w\s,]+',
            'according_to_decree': r'[Aa]ccording to Decree\s+[\w/\-]+|[Tt]heo Nghị định\s+\d+/\d+/[\w-]+',
            'according_to_law': r'[Aa]ccording to (the )?Law\s+[\w\s]+|[Tt]heo Luật\s+[\w\s]+',
            'data_shows': r'[Dd]ata shows?|[Dd]ữ liệu cho thấy',
            'research_by': r'[Rr]esearch by\s+[\w\s]+|[Nn]ghiên cứu của\s+[\w\s]+',
            'according_to_data': r'[Aa]ccording to (the )?data\s+[\w\s,]+|[Tt]heo số liệu\s+[\w\s,]+',
            'source_from': r'[Ss]ource from\s+[\w\s,]+|[Nn]guồn từ\s+[\w\s,]+',
            'reference_material': r'[Rr]eference material|[Tt]ài liệu tham khảo',
            'cbam_regulation': r'CBAM\s+(regulation|quy định)',
            'eu_requires': r'EU\s+(requires|yêu cầu)',
            'study_indicates': r'[Ss]tudy indicates|[Ss]tudy shows',
            'analysis_reveals': r'[Aa]nalysis reveals|[Aa]nalysis shows',
            'according_to_article': r'[Aa]ccording to Article\s+\d+',
            'vitas_report': r'VITAS\s+(report|báo cáo)',
            'ministry_data': r'[Mm]inistry of\s+[\w\s]+\s+(data|report)'
        }
        
        found_citations = []
        pattern_counts = {}
        
        for pattern_name, pattern in citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            pattern_counts[pattern_name] = len(matches)
            found_citations.extend(matches)
        
        total_citations = sum(pattern_counts.values())
        
        # Calculate citation density (citations per 100 words)
        word_count = self.count_words(text)['word_count']
        citation_density = (total_citations / word_count * 100) if word_count > 0 else 0.0
        
        return {
            'total_citations': total_citations,
            'citation_patterns': pattern_counts,
            'citation_density': round(citation_density, 2),
            'found_citations': found_citations[:10]  # Return first 10 citations only
        }
    
    # METRIC 3: Diversity Score
    
    def diversity_score(
        self, 
        texts: List[str], 
        method: str = "embedding"
    ) -> Dict[str, Union[float, str]]:
        """
        Measure diversity between responses
        
        Args:
            texts: List of texts to compare
            method: Calculation method ("embedding", "lexical", "ngram")
            
        Returns:
            Dict containing:
            - diversity_score: Score from 0-1 (higher = more diverse)
            - method_used: Method used
            - explanation: Explanation
            - details: Calculation details
        """
        if not texts or len(texts) < 2:
            return {
                'diversity_score': 0.0,
                'method_used': method,
                'explanation': 'Need at least 2 texts to calculate diversity',
                'details': {}
            }
        
        # Clean texts
        texts = [t.strip() for t in texts if t and t.strip()]
        
        if method == "embedding" and EMBEDDING_AVAILABLE and self.model:
            return self._diversity_embedding(texts)
        elif method == "lexical":
            return self._diversity_lexical(texts)
        elif method == "ngram":
            return self._diversity_ngram(texts)
        else:
            # Fallback to lexical if embedding not available
            return self._diversity_lexical(texts)
    
    def _diversity_embedding(self, texts: List[str]) -> Dict:
        """Calculate diversity based on cosine distance of embeddings"""
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Calculate cosine similarity matrix
        n = len(embeddings)
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        return {
            'diversity_score': round(float(diversity), 4),
            'method_used': 'embedding',
            'explanation': f'Based on semantic distance. Average similarity: {avg_similarity:.4f}',
            'details': {
                'avg_similarity': round(float(avg_similarity), 4),
                'num_comparisons': len(similarities),
                'min_similarity': round(float(min(similarities)), 4),
                'max_similarity': round(float(max(similarities)), 4)
            }
        }
    
    def _diversity_lexical(self, texts: List[str]) -> Dict:
        """Calculate diversity based on Jaccard distance of vocabulary"""
        # Create word sets for each text
        word_sets = []
        for text in texts:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            word_sets.append(words)
        
        # Calculate Jaccard distance for each pair
        n = len(word_sets)
        distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                jaccard_sim = intersection / union if union > 0 else 0
                distances.append(1 - jaccard_sim)  # Distance = 1 - similarity
        
        avg_distance = np.mean(distances)
        
        return {
            'diversity_score': round(float(avg_distance), 4),
            'method_used': 'lexical',
            'explanation': f'Based on vocabulary difference (Jaccard distance)',
            'details': {
                'avg_jaccard_distance': round(float(avg_distance), 4),
                'num_comparisons': len(distances),
                'unique_words_per_text': [len(ws) for ws in word_sets]
            }
        }
    
    def _diversity_ngram(self, texts: List[str], n: int = 2) -> Dict:
        """Calculate diversity based on n-gram overlap"""
        # Create n-grams for each text
        ngram_sets = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            ngrams = set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
            ngram_sets.append(ngrams)
        
        # Calculate overlap for each pair
        num_texts = len(ngram_sets)
        overlaps = []
        
        for i in range(num_texts):
            for j in range(i + 1, num_texts):
                intersection = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                overlap = intersection / union if union > 0 else 0
                overlaps.append(1 - overlap)
        
        avg_diversity = np.mean(overlaps)
        
        return {
            'diversity_score': round(float(avg_diversity), 4),
            'method_used': f'{n}-gram',
            'explanation': f'Based on {n}-gram phrase differences',
            'details': {
                'avg_diversity': round(float(avg_diversity), 4),
                'num_comparisons': len(overlaps),
                'ngram_counts': [len(ns) for ns in ngram_sets]
            }
        }
    
    # ADDITIONAL METRICS
    
    def calculate_all_metrics(self, text: str) -> Dict:
        """
        Calculate all metrics for a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing all metrics
        """
        return {
            'length_metrics': self.count_words(text),
            'citation_metrics': self.count_citations(text),
            'timestamp': self._get_timestamp()
        }
    
    def compare_responses(
        self, 
        baseline_response: str, 
        agent_responses: List[str]
    ) -> Dict:
        """
        Compare baseline response vs multi-agent system responses
        
        Args:
            baseline_response: Response from single ChatGPT
            agent_responses: List of responses from agents
            
        Returns:
            Dict containing comparison analysis
        """
        baseline_metrics = self.calculate_all_metrics(baseline_response)
        
        agent_metrics_list = [
            self.calculate_all_metrics(resp) for resp in agent_responses
        ]
        
        # Calculate diversity of agent system
        diversity = self.diversity_score(agent_responses)
        
        # Compare average length
        baseline_length = baseline_metrics['length_metrics']['word_count']
        agent_avg_length = np.mean([
            m['length_metrics']['word_count'] for m in agent_metrics_list
        ])
        
        # Compare citations
        baseline_citations = baseline_metrics['citation_metrics']['total_citations']
        agent_avg_citations = np.mean([
            m['citation_metrics']['total_citations'] for m in agent_metrics_list
        ])
        
        return {
            'baseline': baseline_metrics,
            'agents': {
                'individual_metrics': agent_metrics_list,
                'diversity': diversity,
                'avg_length': round(agent_avg_length, 2),
                'avg_citations': round(agent_avg_citations, 2)
            },
            'comparison': {
                'length_improvement': round(
                    (agent_avg_length - baseline_length) / baseline_length * 100, 2
                ),
                'citation_improvement': round(
                    (agent_avg_citations - baseline_citations) / baseline_citations * 100 
                    if baseline_citations > 0 else 0, 2
                ),
                'diversity_score': diversity['diversity_score']
            }
        }
    
    @staticmethod
    def _get_timestamp():
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# UTILITY FUNCTIONS

def print_metrics_report(metrics: Dict, title: str = "Metrics Report"):
    """Print metrics report in readable format"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")
    
    if 'length_metrics' in metrics:
        print("Response Length:")
        lm = metrics['length_metrics']
        print(f"  - Word count: {lm['word_count']}")
        print(f"  - Sentence count: {lm['sentence_count']}")
        print(f"  - Avg words/sentence: {lm['avg_word_per_sentence']}")
    
    if 'citation_metrics' in metrics:
        print("\nCitations:")
        cm = metrics['citation_metrics']
        print(f"  - Total citations: {cm['total_citations']}")
        print(f"  - Citation density: {cm['citation_density']} citations/100 words")
        if cm['found_citations']:
            print(f"  - Example: {cm['found_citations'][0]}")
    
    print(f"\n{'='*60}\n")


# MAIN: USAGE EXAMPLES

if __name__ == "__main__":
    # Initialize calculator
    calc = MetricsCalculator()
    
    # Example 1: Word count
    sample_text = """
    According to Decree No. 06/2022/ND-CP, textile manufacturing facilities with 
    emissions of 3,000 tons of CO2 equivalent per year or more must conduct 
    greenhouse gas inventories. Data shows that Vietnam's textile industry 
    contributes approximately 15% of total industrial emissions. Research by 
    the Ministry of Industry and Trade in 2023 indicates that CBAM compliance 
    costs will reduce profit margins of export enterprises by 5-8%.
    """
    
    print("EXAMPLE 1: Word and Character Count")
    length_result = calc.count_words(sample_text)
    print(length_result)
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Citation count
    print("EXAMPLE 2: Citation Count")
    citation_result = calc.count_citations(sample_text)
    print(f"Total citations: {citation_result['total_citations']}")
    print(f"Density: {citation_result['citation_density']}")
    print(f"Details: {citation_result['citation_patterns']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Diversity score
    print("EXAMPLE 3: Diversity Score")
    responses = [
        "Carbon tax policy will increase textile production costs. Enterprises need to invest in green technology.",
        "EU's CBAM has a strong impact on exports. Vietnam should negotiate to extend the implementation timeline.",
        "The textile industry must upgrade equipment to reduce emissions. Financial support from the government is necessary."
    ]
    
    diversity_result = calc.diversity_score(responses, method="lexical")
    print(f"Diversity score: {diversity_result['diversity_score']}")
    print(f"Method: {diversity_result['method_used']}")
    print(f"Explanation: {diversity_result['explanation']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Compare baseline vs agents
    print("EXAMPLE 4: Baseline vs Multi-Agent Comparison")
    baseline = "Carbon tax increases costs. Enterprises need to invest."
    agents = [
        "According to ILO research, carbon tax increases textile production costs by 10-15%.",
        "Data shows that 60% of Vietnamese textile enterprises are not ready for CBAM.",
        "Decree No. 06/2022/ND-CP stipulates the GHG inventory threshold is 3000 tons CO2e/year."
    ]
    
    comparison = calc.compare_responses(baseline, agents)
    print(f"Length improvement: {comparison['comparison']['length_improvement']}%")
    print(f"Citation improvement: {comparison['comparison']['citation_improvement']}%")
    print(f"Agent diversity score: {comparison['comparison']['diversity_score']}")