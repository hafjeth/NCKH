class EconomicExpert:
    name = "Economic Expert"

    system_prompt = """
    You evaluate decisions based on economic efficiency,
    cost-benefit balance, and market competitiveness.
    """

    evaluation_criteria = [
        "cost_impact",
        "roi",
        "industry_competitiveness"
    ]

    def evaluate(self, proposal: str) -> str:
        return (
            "From an economic perspective, the proposal may increase costs "
            "for enterprises in the short term."
        )
