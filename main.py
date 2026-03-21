"""
Multi-Agent Debate System for Carbon Tax Policy Analysis
========================================================
Vietnam Textile Industry Case Study

Entry point for running the debate system.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

from src.core.debate_manager import DebateManager
from experiments.evaluation.metrics.retrieval_metrics import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Debate System')
    parser.add_argument('--topic',  type=str, required=True,          help='Debate topic')
    parser.add_argument('--rounds', type=int, default=3,              help='Number of rounds')
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    args = parser.parse_args()

    print(f"🚀 Starting debate on: {args.topic}")
    print(f"📊 Rounds: {args.rounds}\n")

    # ======================================================
    # RUN DEBATE
    # ======================================================
    manager = DebateManager()
    manager.setup_agents()

    expert_text, history = manager.run_debate(
        topic=args.topic,
        max_rounds=args.rounds
    )

    # ======================================================
    # CALCULATE METRICS
    # ======================================================
    calc = MetricsCalculator()

    all_responses  = [h["content"] for h in history]
    agent_responses = [h["content"] for h in history if h["agent"] != "Moderator"]

    metrics = calc.compare_responses(
        baseline=all_responses[0] if all_responses else "",
        agents=agent_responses[1:] if len(agent_responses) > 1 else []
    )

    # Citation & word count on expert synthesis
    citation_stats = calc.count_citations(expert_text)
    word_stats     = calc.count_words(expert_text)

    # ======================================================
    # BUILD RESULT DICT
    # ======================================================
    results = {
        "metadata": {
            "topic":      args.topic,
            "rounds":     args.rounds,
            "timestamp":  datetime.utcnow().isoformat(),
            "total_turns": len(history),
        },
        "debate_history": history,
        "expert_synthesis": expert_text,
        "metrics": {
            "diversity_score":   metrics.get("diversity", {}).get("diversity_score", 0),
            "avg_agent_length":  metrics.get("avg_agent_length", 0),
            "length_change_pct": metrics.get("length_change_pct", 0),
            "expert_word_count": word_stats.get("word_count", 0),
            "expert_citations":  citation_stats.get("total_citations", 0),
            "citation_density":  citation_stats.get("citation_density", 0),
        }
    }

    # ======================================================
    # SAVE TO FILE
    # ======================================================
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ======================================================
    # PRINT SUMMARY
    # ======================================================
    print(f"\n✅ Debate completed!")
    print(f"📊 Turns          : {len(history)}")
    print(f"📈 Diversity score: {results['metrics']['diversity_score']}")
    print(f"📝 Expert words   : {results['metrics']['expert_word_count']}")
    print(f"📎 Citations      : {results['metrics']['expert_citations']}")
    print(f"💾 Results saved  : {output_path.resolve()}")


if __name__ == "__main__":
    main()