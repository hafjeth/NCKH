"""
Visualization Module for Multi-Agent Debate Evaluation
======================================================

Purpose:
- Compare baseline vs multi-agent debate performance
- Support quantitative analysis for academic reporting
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


class DebateVisualizer:
    """
    Visualization utilities for debate metrics and LLM evaluation
    """

    def __init__(self, style: str = "default"):
        plt.style.use(style)

    # ======================================================
    # MAIN COMPARISON FIGURE
    # ======================================================
    def plot_comparison(
        self,
        baseline_df: pd.DataFrame,
        multiagent_df: pd.DataFrame,
        title: Optional[str] = None
    ):
        """
        Compare baseline vs multi-agent debate results

        Expected columns:
        - word_count
        - total_citations
        - diversity_score (multi-agent only)
        - coherence
        - factuality
        """

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # --------------------------------------------------
        # 1. Word count comparison
        # --------------------------------------------------
        if "word_count" in baseline_df and "word_count" in multiagent_df:
            axes[0, 0].boxplot(
                [baseline_df["word_count"], multiagent_df["word_count"]],
                labels=["Baseline", "Multi-Agent"],
                showfliers=True
            )
            axes[0, 0].set_title("Response Length Comparison")
            axes[0, 0].set_ylabel("Word Count")
        else:
            axes[0, 0].set_visible(False)

        # --------------------------------------------------
        # 2. Average citation count
        # --------------------------------------------------
        if "total_citations" in baseline_df and "total_citations" in multiagent_df:
            axes[0, 1].bar(
                ["Baseline", "Multi-Agent"],
                [
                    baseline_df["total_citations"].mean(),
                    multiagent_df["total_citations"].mean(),
                ]
            )
            axes[0, 1].set_title("Average Number of Citations")
            axes[0, 1].set_ylabel("Citations")
        else:
            axes[0, 1].set_visible(False)

        # --------------------------------------------------
        # 3. Diversity score distribution (Multi-agent only)
        # --------------------------------------------------
        if "diversity_score" in multiagent_df:
            axes[1, 0].hist(
                multiagent_df["diversity_score"],
                bins=15,
                alpha=0.75
            )
            axes[1, 0].set_title("Argument Diversity Distribution")
            axes[1, 0].set_xlabel("Diversity Score")
            axes[1, 0].set_ylabel("Frequency")
        else:
            axes[1, 0].set_visible(False)

        # --------------------------------------------------
        # 4. Coherence vs Factuality (LLM Judge)
        # --------------------------------------------------
        if {"coherence", "factuality"}.issubset(multiagent_df.columns):
            axes[1, 1].scatter(
                multiagent_df["coherence"],
                multiagent_df["factuality"],
                alpha=0.7
            )
            axes[1, 1].set_title("Coherence vs Factuality (LLM Judge)")
            axes[1, 1].set_xlabel("Coherence Score")
            axes[1, 1].set_ylabel("Factuality Score")
        else:
            axes[1, 1].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()
        return fig

    # ======================================================
    # SAVE FIGURE
    # ======================================================
    def save_plot(self, fig, filename: str):
        """
        Save figure to file
        """
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[Visualizer] Saved plot to: {filename}")
