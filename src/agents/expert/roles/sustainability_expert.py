class SustainabilityExpert:
    name = "Sustainability Expert"

    system_prompt = """
    You evaluate decisions based on environmental sustainability,
    long-term ecological impact, and ESG compliance.
    """

    evaluation_criteria = [
        "carbon_emission_reduction",
        "resource_efficiency",
        "climate_alignment"
    ]

    def evaluate(self, proposal: str) -> str:
        return (
            "From a sustainability perspective, the proposal supports "
            "long-term emission reduction goals."
        )
