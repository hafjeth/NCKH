"""
carbon_policy_agent.py
======================
FIX: Correct import path (src. prefix)
FIX: BaseAgent now requires name + role (not description)
"""
from src.core.base_agent import BaseAgent


class CarbonPolicyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Government Agent",
            role=(
                "You are a senior policymaker representing the Ministry of Natural Resources "
                "and Environment (MONRE), Vietnam. You advocate for carbon tax implementation, "
                "CBAM compliance, and Vietnam's Net Zero 2050 commitment. "
                "Use legal frameworks and policy evidence in your arguments. Academic tone."
            )
        )

    def propose_policy(self, context: dict) -> str:
        """Legacy method — kept for backward compatibility"""
        return self.chat(
            "Propose a carbon tax policy for Vietnam's textile industry. "
            "Be specific about mechanisms and timelines."
        )