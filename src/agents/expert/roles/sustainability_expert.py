"""
sustainability_expert.py
========================
FIX: evaluate() now calls GPT-4o-mini instead of returning a hardcoded string.
"""

import os
from openai import OpenAI


class SustainabilityExpert:
    name = "Sustainability Expert"

    system_prompt = """You are a senior sustainability and environmental policy expert
specializing in carbon emissions, ESG compliance, and climate policy for
the Vietnamese textile and garment industry.

Evaluate the given debate summary or proposal strictly from an environmental
sustainability perspective. Be concise, evidence-based, and academic in tone.
Focus on: GHG reduction potential, resource efficiency, and alignment with
Net Zero 2050 and EU CBAM requirements.

Output: 3–5 sentences. No bullet points. No policy recommendations beyond
what is supported by the provided text."""

    evaluation_criteria = [
        "carbon_emission_reduction",
        "resource_efficiency",
        "climate_alignment"
    ]

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("[SustainabilityExpert] OPENAI_API_KEY not found in environment.")
        self._client = OpenAI(api_key=api_key)

    def evaluate(self, proposal: str) -> str:
        """
        Evaluate a debate summary or policy proposal from a sustainability lens.

        Args:
            proposal: Text summary of the debate or policy proposal to evaluate.

        Returns:
            str: Expert assessment (3–5 sentences, academic tone).
        """
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": f"Please evaluate the following:\n\n{proposal}"}
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[SustainabilityExpert Error] {str(e)}"