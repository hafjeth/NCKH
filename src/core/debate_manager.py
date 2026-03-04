"""
Debate Manager (Research-Grade Version)
======================================
Multi-agent academic debate controller for policy analysis

Agents:
- Government Agent (Policy authority)
- Business Agent (Industry representative)
- Expert Agent (Independent analytical synthesis ONLY)
- Moderator Agent (Neutral academic controller)
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# ======================================================
# PATH SETUP
# ======================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ======================================================
# IMPORTS
# ======================================================
from src.core.base_agent import BaseAgent
from src.core.moderator import ModeratorAgent
from src.knowledge.personas import AGENT_PERSONAS
from src.knowledge.retrieval import KnowledgeRetriever

logger = logging.getLogger(__name__)


class DebateManager:
    """
    Academic multi-agent debate orchestrator

    Debate pipeline:
    1. Government ↔ Business debate
    2. Expert analytical synthesis (NO advocacy)
    3. Moderator academic summary
    """

    def __init__(self):
        self.debate_history: List[dict] = []

        self.government_agent: Optional[BaseAgent] = None
        self.business_agent: Optional[BaseAgent] = None
        self.expert_agent: Optional[BaseAgent] = None
        self.moderator: Optional[ModeratorAgent] = None

        self.retriever = KnowledgeRetriever()

    # ======================================================
    # AGENT SETUP
    # ======================================================
    def setup_agents(self):
        """Initialize agents from personas"""

        def build_agent(key):
            p = AGENT_PERSONAS[key]
            return BaseAgent(
                name=p["agent_name"],
                role=self._build_role_prompt(p),
                retriever=self.retriever
            )

        self.government_agent = build_agent("government")
        self.business_agent = build_agent("business")
        self.expert_agent = build_agent("expert")

        self.moderator = ModeratorAgent(
            name="Moderator",
            max_rounds=3
        )

        logger.info("Agents initialized successfully")

    def _build_role_prompt(self, persona: dict) -> str:
        def fmt(items):
            return "\n".join(f"- {i}" for i in items)

        return f"""
{persona['role_description']}

REPRESENTED ENTITY:
{persona['represented_entity']}

CORE OBJECTIVES:
{fmt(persona['core_objectives'])}

POLICY PRIORITIES:
{fmt(persona['policy_priorities'])}

CONSTRAINTS:
{fmt(persona['constraints_and_challenges'])}

REASONING STYLE:
{persona['reasoning_style']}

{persona['response_guidelines']}
""".strip()

    # ======================================================
    # DEBATE PROMPT
    # ======================================================
    def build_debate_prompt(self, agent_name, topic, is_first_turn, recent_history):
        if is_first_turn:
            return f"""
DEBATE TOPIC:
{topic}

You are {agent_name}.
Present your initial position.

Requirements:
- 2–3 core arguments
- Evidence-based reasoning
- Academic tone
- 150–200 words
"""

        history_text = "\n".join(h["content"] for h in recent_history)

        return f"""
DEBATE TOPIC:
{topic}

RECENT DISCUSSION:
{history_text}

YOUR TASK ({agent_name}):
1. Briefly summarize previous arguments (1–2 sentences)
2. Respond or counter-argue from your perspective
3. Add new analytical insights if relevant

Requirements:
- Academic tone
- Evidence-based
- 150–200 words
"""

    # ======================================================
    # EXPERT SYNTHESIS (CORE FIX)
    # ======================================================
    def expert_synthesis(self, topic: str) -> str:
        """
        Independent analytical synthesis of debate
        (NO policy advocacy)
        """

        gov_views = [h["content"] for h in self.debate_history if h["agent"] == "Government"]
        biz_views = [h["content"] for h in self.debate_history if h["agent"] == "Business"]

        prompt = f"""
You are an INDEPENDENT POLICY ANALYST.

TASK:
1. Summarize the Government's position
2. Summarize the Business sector's position
3. Compare the two perspectives
4. Identify:
   - Key trade-offs
   - Areas of consensus
   - Core disagreements

IMPORTANT RULES:
- DO NOT propose new policies
- DO NOT take sides
- DO NOT introduce external arguments
- ONLY synthesize what was debated

DEBATE TOPIC:
{topic}

GOVERNMENT ARGUMENTS:
{" ".join(gov_views)}

BUSINESS ARGUMENTS:
{" ".join(biz_views)}

OUTPUT:
- Structured academic analysis
- Neutral tone
- 250–350 words
"""

        return self.expert_agent.chat(prompt)

    # ======================================================
    # RUN FULL DEBATE
    # ======================================================
    def run_debate(self, topic: str, max_rounds: int = 2) -> Tuple[str, List[dict]]:

        if not all([self.government_agent, self.business_agent, self.expert_agent, self.moderator]):
            self.setup_agents()

        logger.info(f"Starting debate on: {topic}")

        agents = [self.government_agent, self.business_agent]

        for round_num in range(1, max_rounds + 1):
            for idx, agent in enumerate(agents):
                prompt = self.build_debate_prompt(
                    agent.name,
                    topic,
                    is_first_turn=(round_num == 1 and idx == 0),
                    recent_history=self.debate_history[-4:]
                )

                response = agent.chat(prompt)
                self.debate_history.append({
                    "round": round_num,
                    "agent": agent.name,
                    "content": response
                })

            if round_num < max_rounds:
                last = self.debate_history[-1]
                mod_text, _ = self.moderator.moderate(
                    last_speaker=last["agent"],
                    last_content=last["content"],
                    round_num=round_num,
                    debate_history=[h["content"] for h in self.debate_history]
                )
                self.debate_history.append({
                    "round": round_num,
                    "agent": "Moderator",
                    "content": mod_text
                })

        # Expert synthesis
        expert_text = self.expert_synthesis(topic)
        self.debate_history.append({
            "round": "post",
            "agent": "Expert",
            "content": expert_text
        })

        # Moderator final summary
        final_summary = self.moderator.summarize_debate(
            [h["content"] for h in self.debate_history]
        )

        self.debate_history.append({
            "round": "final",
            "agent": "Moderator",
            "content": final_summary
        })

        logger.info("Debate pipeline completed")

        return expert_text, self.debate_history
