from core.base_agent import BaseAgent

class CarbonPolicyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Government Carbon Policy Agent",
            description="Represents government perspective on carbon regulation"
        )

    def propose_policy(self, context: dict) -> str:
        return (
            "The government proposes a carbon tax combined with "
            "subsidies for green technology adoption."
        )
