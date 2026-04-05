"""
economic_expert.py
==================
FIXES SO VỚI BẢN CŨ:
  [FIX-1] Không còn hard-code OpenAI — dùng Config.API_PROVIDER để chọn provider
          Bản cũ: luôn dùng OpenAI(api_key=OPENAI_API_KEY) →
          nếu chỉ có ANTHROPIC_API_KEY thì raise ValueError → crash silent
  [FIX-2] Hỗ trợ cả OpenAI lẫn Anthropic — chọn tự động theo Config
"""

import os
from config.settings import Config


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
        # [FIX-1] Dùng Config thay vì hard-code OpenAI
        self._provider = Config.API_PROVIDER
        self._model    = Config.MODEL_NAME
        self._client   = self._build_client()

    def _build_client(self):
        """[FIX-2] Khởi tạo client theo provider đang active."""
        if self._provider == "openai":
            from openai import OpenAI
            api_key = Config.OPENAI_API_KEY
            if not api_key:
                raise ValueError("[EconomicExpert] OPENAI_API_KEY not found in environment.")
            return OpenAI(api_key=api_key)

        elif self._provider == "anthropic":
            import anthropic
            api_key = Config.ANTHROPIC_API_KEY
            if not api_key:
                raise ValueError("[EconomicExpert] ANTHROPIC_API_KEY not found in environment.")
            return anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(
                f"[EconomicExpert] Unsupported API_PROVIDER: '{self._provider}'. "
                f"Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
            )

    def evaluate(self, proposal: str) -> str:
        """
        Evaluate a debate summary or policy proposal from an economic lens.

        Args:
            proposal: Text summary of the debate or policy proposal to evaluate.

        Returns:
            str: Expert assessment (3–5 sentences, academic tone).
        """
        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": f"Please evaluate the following:\n\n{proposal}"}
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )
                return response.choices[0].message.content.strip()

            else:  # anthropic
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=400,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": f"Please evaluate the following:\n\n{proposal}"}
                    ],
                )
                return response.content[0].text.strip()

        except Exception as e:
            return f"[EconomicExpert Error] {type(e).__name__}: {str(e)}"