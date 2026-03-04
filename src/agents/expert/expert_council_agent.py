from core.base_agent import BaseAgent
from agents.expert.roles.sustainability_expert import SustainabilityExpert
from agents.expert.roles.economic_expert import EconomicExpert

class ExpertCouncilAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Expert Council Agent",
            description="Evaluates debate outcomes using multiple expert perspectives"
        )
        self.roles = [
            SustainabilityExpert(),
            EconomicExpert()
        ]

    def evaluate_debate(self, debate_artifacts: dict) -> dict:
        """
        Evaluate structured debate output from DebateManager.
        """

        evaluations = {}

        for role in self.roles:
            evaluations[role.name] = role.evaluate(debate_artifacts)

        return {
            "evaluations": evaluations,
            "input_debate_summary": debate_artifacts.get("final_moderator_summary", "")
        }
