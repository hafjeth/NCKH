"""
economic_expert.py
==================
FIX: evaluate() now calls GPT-4o-mini instead of returning a hardcoded string.
"""

import os
from openai import OpenAI


class EconomicExpert:
    name = "Economic Expert"

    system_prompt = """You are a senior economic analyst specializing in trade competitiveness,
carbon pricing instruments, and industrial policy for the Vietnamese textile
and garment sector.

Evaluate the given debate summary or proposal strictly from an economic
perspective. Be concise, evidence-based, and academic in tone.
Focus on: cost-benefit balance, SME financial burden, export competitiveness,
and short- vs. long-term ROI of carbon tax compliance.

Output: 3–5 sentences. No bullet points. No policy recommendations beyond
what is supported by the provided text."""

    evaluation_criteria = [
        "cost_impact",
        "roi",
        "industry_competitiveness"
    ]

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("[EconomicExpert] OPENAI_API_KEY not found in environment.")
        self._client = OpenAI(api_key=api_key)

    def evaluate(self, proposal: str) -> str:
        """
        Evaluate a debate summary or policy proposal from an economic lens.

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
            return f"[EconomicExpert Error] {str(e)}"