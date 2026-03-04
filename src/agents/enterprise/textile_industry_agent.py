from core.base_agent import BaseAgent

class TextileIndustryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Textile Industry Agent",
            description="Represents textile enterprise economic interests"
        )

    def respond_to_policy(self, policy: str) -> str:
        return (
            "The textile industry is concerned about increased production costs "
            "and reduced competitiveness due to carbon taxation."
        )
