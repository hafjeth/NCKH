from typing import List, Dict, Any
from datetime import datetime


class Evaluator:
    """
    Orchestrates the evaluation process by coordinating judges and metrics.
    """

    def __init__(self, judge, metrics: List):
        """
        Args:
            judge: An instance of BaseJudge
            metrics (list): List of metric objects with a compute() method
        """
        self.judge = judge
        self.metrics = metrics

    def evaluate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation pipeline:
        1. Judge the input data
        2. Compute metrics based on judgment
        3. Aggregate results

        Args:
            input_data (dict): Output from debate or expert synthesis

        Returns:
            dict: Evaluation report
        """
        evaluation_time = datetime.utcnow().isoformat()

        # Step 1: Judge
        judgment = self.judge.judge(input_data)

        # Step 2: Metrics
        metric_results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            metric_results[metric_name] = metric.compute(judgment)

        # Step 3: Aggregate
        evaluation_report = {
            "judge": self.judge.name,
            "timestamp": evaluation_time,
            "metrics": metric_results,
            "raw_judgment": judgment
        }

        return evaluation_report
