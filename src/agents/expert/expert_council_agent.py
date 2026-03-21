"""
ExpertCouncilAgent
==================
FIX: Added all methods required by expert_consultation_flow.py:
  - run_environmental_assessment()
  - run_economic_assessment()
  - analyze_conflicts()
  - synthesize_policy()
FIX: evaluate() in roles now receives str summary (not raw dict)
"""

from src.core.base_agent import BaseAgent
from src.agents.expert.roles.sustainability_expert import SustainabilityExpert
from src.agents.expert.roles.economic_expert import EconomicExpert


class ExpertCouncilAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="Expert Council Agent",
            role="You are an independent expert council for carbon tax policy analysis."
        )
        self.sustainability_expert = SustainabilityExpert()
        self.economic_expert       = EconomicExpert()

    # ======================================================
    # ORIGINAL METHOD (kept for debate_manager.py)
    # ======================================================
    def evaluate_debate(self, debate_artifacts: dict) -> dict:
        """Used by debate_manager.py — takes full debate dict"""
        summary = debate_artifacts.get("final_moderator_summary", str(debate_artifacts))
        return {
            "evaluations": {
                self.sustainability_expert.name: self.sustainability_expert.evaluate(summary),
                self.economic_expert.name:       self.economic_expert.evaluate(summary),
            },
            "input_debate_summary": summary
        }

    # ======================================================
    # NEW METHODS (required by expert_consultation_flow.py)
    # ======================================================
    def run_environmental_assessment(self, debate_data: dict) -> dict:
        """Sustainability expert evaluates the debate"""
        summary = self._extract_summary(debate_data)
        assessment = self.sustainability_expert.evaluate(summary)
        return {
            "expert": self.sustainability_expert.name,
            "assessment": assessment,
            "criteria": self.sustainability_expert.evaluation_criteria,
        }

    def run_economic_assessment(self, debate_data: dict) -> dict:
        """Economic expert evaluates the debate"""
        summary = self._extract_summary(debate_data)
        assessment = self.economic_expert.evaluate(summary)
        return {
            "expert": self.economic_expert.name,
            "assessment": assessment,
            "criteria": self.economic_expert.evaluation_criteria,
        }

    def analyze_conflicts(
        self,
        environmental_assessment: dict,
        economic_assessment: dict
    ) -> dict:
        """Identify trade-offs between environmental and economic assessments"""
        prompt = f"""
You are analyzing trade-offs between two expert assessments on carbon tax policy
for Vietnam's textile industry.

ENVIRONMENTAL ASSESSMENT:
{environmental_assessment.get('assessment', '')}

ECONOMIC ASSESSMENT:
{economic_assessment.get('assessment', '')}

Identify:
1. Key conflicts between environmental and economic goals
2. Areas of potential agreement
3. Critical trade-offs for policymakers

Output a structured analysis. Be concise and academic.
"""
        conflict_text = self.chat(prompt)
        return {
            "conflicts": conflict_text,
            "environmental_expert": environmental_assessment.get("expert"),
            "economic_expert":      economic_assessment.get("expert"),
        }

    def synthesize_policy(
        self,
        environmental_assessment: dict,
        economic_assessment: dict,
        conflict_matrix: dict
    ) -> dict:
        """Produce final policy recommendation synthesizing all assessments"""
        prompt = f"""
You are producing a final policy synthesis for carbon tax implementation
in Vietnam's textile industry.

ENVIRONMENTAL ASSESSMENT:
{environmental_assessment.get('assessment', '')}

ECONOMIC ASSESSMENT:
{economic_assessment.get('assessment', '')}

CONFLICT ANALYSIS:
{conflict_matrix.get('conflicts', '')}

Produce:
1. A balanced policy recommendation
2. Phased implementation suggestions
3. Key support mechanisms needed
4. Monitoring and evaluation metrics

Academic tone. 300–400 words.
"""
        recommendation_text = self.chat(prompt)
        return {
            "recommendation": recommendation_text,
            "based_on": [
                environmental_assessment.get("expert"),
                economic_assessment.get("expert"),
            ]
        }

    # ======================================================
    # HELPER
    # ======================================================
    def _extract_summary(self, debate_data: dict) -> str:
        """Extract a usable text summary from debate artifact dict"""
        # Try common keys
        for key in ("moderator_summary", "final_moderator_summary", "expert_synthesis"):
            if debate_data.get(key):
                val = debate_data[key]
                return val if isinstance(val, str) else str(val)

        # Fallback: build from debate_history
        history = debate_data.get("debate_history", [])
        if history:
            return "\n".join(
                f"{h.get('agent', 'Agent')}: {h.get('content', '')}"
                for h in history[-6:]  # last 6 turns
            )

        return str(debate_data)