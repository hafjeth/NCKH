"""
expert_consultation_flow.py
============================
FIX: from experiments.utils.logger → experiments.evaluation.utils.logger
"""

import json
import os
from datetime import datetime
from typing import Dict

from src.agents.expert.expert_council_agent import ExpertCouncilAgent
from experiments.evaluation.utils.logger import get_logger   # FIX: correct path


class ExpertConsultationFlow:
    """
    Conducts post-debate expert evaluation and synthesis.
    """

    def __init__(self, debate_artifact_path: str, output_dir: str):
        self.debate_artifact_path = debate_artifact_path
        self.output_dir           = output_dir
        self.logger               = get_logger("ExpertConsultationFlow")
        self.expert_council       = ExpertCouncilAgent()
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(self) -> Dict:
        self.logger.info("Starting expert consultation flow")

        debate_data = self._load_debate_artifact()

        environmental_assessment = self.expert_council.run_environmental_assessment(debate_data)
        economic_assessment      = self.expert_council.run_economic_assessment(debate_data)
        conflict_matrix          = self.expert_council.analyze_conflicts(
            environmental_assessment, economic_assessment
        )
        final_recommendation     = self.expert_council.synthesize_policy(
            environmental_assessment, economic_assessment, conflict_matrix
        )

        synthesis = {
            "timestamp":                datetime.utcnow().isoformat(),
            "environmental_assessment": environmental_assessment,
            "economic_assessment":      economic_assessment,
            "conflict_matrix":          conflict_matrix,
            "final_recommendation":     final_recommendation,
        }

        self._save_outputs(synthesis)
        self.logger.info("Expert consultation flow completed")
        return synthesis

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------
    def _load_debate_artifact(self) -> Dict:
        self.logger.info("Loading debate artifacts")
        with open(self.debate_artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_outputs(self, synthesis: Dict):
        self._save_json(synthesis["environmental_assessment"], "environmental_assessment.json")
        self._save_json(synthesis["economic_assessment"],      "economic_assessment.json")
        self._save_json(synthesis["conflict_matrix"],          "conflict_matrix.json")
        self._save_json(synthesis["final_recommendation"],     "final_recommendation.json")

    def _save_json(self, data, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)