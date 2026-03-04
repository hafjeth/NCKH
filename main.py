"""
Multi-Agent Debate System for Carbon Tax Policy Analysis
========================================================
Vietnam Textile Industry Case Study

Entry point for running the debate system.
"""

import argparse
from src.core.debate_manager import DebateManager
from src.evaluation.metrics import MetricsCalculator

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Multi-Agent Debate System')
    parser.add_argument('--topic', type=str, required=True, help='Debate topic')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds')
    parser.add_argument('--output', type=str, default='results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Initialize system
    print(f"🚀 Starting debate on: {args.topic}")
    print(f"📊 Rounds: {args.rounds}\n")
    
    manager = DebateManager()
    manager.setup_agents()
    
    # Run debate
    final_summary, history = manager.run_debate(
        topic=args.topic,
        max_rounds=args.rounds
    )
    
    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_all_metrics(final_summary)
    
    print(f"\n✅ Debate completed!")
    print(f"📊 Final summary: {len(final_summary)} characters")
    print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()