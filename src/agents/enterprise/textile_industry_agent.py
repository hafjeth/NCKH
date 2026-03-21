"""
textile_industry_agent.py
=========================
FIX: Correct import path (src. prefix)
FIX: BaseAgent now requires name + role (not description)
"""
from src.core.base_agent import BaseAgent


class TextileIndustryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Enterprise Agent",
            role=(
                "You represent the Vietnam Textile and Apparel Association (VITAS), "
                "speaking for textile and garment enterprises including SMEs and exporters. "
                "Focus on economic impact, competitiveness, cost burden, and feasibility. "
                "Propose supportive mechanisms rather than outright rejection. Academic tone."
            )
        )

    def respond_to_policy(self, policy: str) -> str:
        """Legacy method — kept for backward compatibility"""
        return self.chat(
            f"Respond to this proposed policy from the enterprise perspective:\n{policy}"
        )