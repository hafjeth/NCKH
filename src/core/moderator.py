"""
ModeratorAgent
=====================================
Neutral moderator for multi-agent policy debate

FIX: Replaced client.responses.create() → client.chat.completions.create()
FIX: Replaced response.output_text     → response.choices[0].message.content
FIX: Import path for Config
"""

from typing import List, Tuple
from openai import OpenAI
from config.settings import Config   # giữ nguyên vì settings.py nằm ở config/


class ModeratorAgent:
    """
    Neutral Moderator Agent for academic policy debate
    """

    def __init__(self, name: str = "Moderator", max_rounds: int = 3):
        self.name = name
        self.max_rounds = max_rounds

        self.system_role = """
You are a NEUTRAL ACADEMIC MODERATOR for a policy debate on
carbon tax and CBAM in the Vietnamese textile industry.

STRICT RULES:
- Academic, neutral tone
- No personal opinions
- No policy recommendations
- No new factual information
- Only summarize and structure existing arguments
- Do NOT act as an expert or stakeholder
"""

        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found")

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"
        self.history = []

    # ======================================================
    # INTERNAL UTILITIES
    # ======================================================
    def _next_speaker(self, last_speaker: str) -> str:
        last = last_speaker.lower()
        if "government" in last or "chính phủ" in last:
            return "Business"
        if "business" in last or "doanh nghiệp" in last:
            return "Government"
        return "Government"

    # ======================================================
    # CORE MODERATION  (FIXED API CALL)
    # ======================================================
    def moderate(
        self,
        last_speaker: str,
        last_content: str,
        round_num: int,
        debate_history: List[str] = None
    ) -> Tuple[str, bool]:

        debate_history = debate_history or []
        recent_context = "\n".join(debate_history[-3:])
        is_final = round_num >= self.max_rounds

        if is_final:
            prompt = f"""
STAGE: FINAL ROUND ({round_num}/{self.max_rounds})

LAST SPEAKER: {last_speaker}

STATEMENT:
\"\"\"{last_content}\"\"\"

RECENT CONTEXT:
{recent_context if recent_context else "N/A"}

TASK:
1. Neutral summary of Government arguments
2. Neutral summary of Business arguments
3. Identify main points of agreement and disagreement

End strictly with:
=== END OF DEBATE ===

Length: 150–200 words. Academic, neutral tone.
"""
        else:
            next_speaker = self._next_speaker(last_speaker)
            prompt = f"""
STAGE: ROUND {round_num}/{self.max_rounds}

LAST SPEAKER: {last_speaker}

STATEMENT:
\"\"\"{last_content}\"\"\"

TASK:
1. Summarize the main argument (2–3 sentences)
2. Provide a neutral clarification (1 sentence)
3. Invite {next_speaker} to respond with ONE guiding question
   related to feasibility or impact of carbon tax / CBAM

Length: 80–120 words
End with: "Invite {next_speaker} to respond."
"""

        try:
            # FIX: Sử dụng đúng OpenAI Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_role},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.3,
                max_tokens=700
            )

            # FIX: Đọc response đúng cách
            text = response.choices[0].message.content.strip()
            should_end = "=== END OF DEBATE ===" in text

            self.history.append({
                "round": round_num,
                "last_speaker": last_speaker,
                "output": text
            })

            return text, should_end

        except Exception as e:
            return f"[Moderator Error] {str(e)}", False

    # ======================================================
    # FINAL SUMMARY  (FIXED API CALL)
    # ======================================================
    def summarize_debate(self, full_history: List[str]) -> str:

        history_text = "\n\n".join(full_history)

        prompt = f"""
Write an ACADEMIC SUMMARY of the following policy debate:

{history_text}

REQUIREMENTS:
- Summarize Government position
- Summarize Business position
- Identify agreements and disagreements
- Assess internal consistency of arguments
- NO policy recommendation
- Neutral academic tone
Length: 200–300 words
"""

        try:
            # FIX: Sử dụng đúng OpenAI Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_role},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.3,
                max_tokens=900
            )

            # FIX: Đọc response đúng cách
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"[Summary Error] {str(e)}"

    def get_history(self):
        return self.history