"""
DebateFlow
==========
FIX: from src.core.moderator import Moderator → ModeratorAgent
FIX: self.moderator.summarize()               → summarize_debate()
FIX: Added generate_argument() calls via agent.chat() since
     CarbonPolicyAgent / TextileIndustryAgent don't have generate_argument()
"""

import json
import os
from datetime import datetime
from typing import Dict, List

from src.agents.government.carbon_policy_agent import CarbonPolicyAgent
from src.agents.enterprise.textile_industry_agent import TextileIndustryAgent
from src.core.moderator import ModeratorAgent          # FIX: was Moderator
from experiments.evaluation.utils.logger import get_logger


class DebateFlow:
    """
    Multi-round stakeholder debate: Government ↔ Enterprise
    """

    def __init__(
        self,
        topic: str,
        knowledge_retriever,
        output_dir: str,
        num_rounds: int = 2,
    ):
        self.topic      = topic
        self.retriever  = knowledge_retriever
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        self.logger     = get_logger("DebateFlow")

        self.government_agent = CarbonPolicyAgent()
        self.enterprise_agent = TextileIndustryAgent()
        self.moderator        = ModeratorAgent(max_rounds=num_rounds)  # FIX

        self.debate_history: List[Dict] = []
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(self) -> Dict:
        self.logger.info("Starting debate flow")

        for round_idx in range(1, self.num_rounds + 1):
            self.logger.info(f"Running debate round {round_idx}")
            self._run_single_round(round_idx)

        summary = self._moderator_summary()

        artifacts = {
            "topic":             self.topic,
            "num_rounds":        self.num_rounds,
            "timestamp":         datetime.utcnow().isoformat(),
            "debate_history":    self.debate_history,
            "moderator_summary": summary,
        }

        self._save_artifacts(artifacts)
        self.logger.info("Debate flow completed")
        return artifacts

    # --------------------------------------------------
    # Internal
    # --------------------------------------------------
    def _build_prompt(self, agent_name: str, is_first: bool) -> str:
        """Build debate prompt for any agent"""
        if is_first:
            return (
                f"DEBATE TOPIC: {self.topic}\n\n"
                f"You are {agent_name}. Present your initial position.\n"
                f"Requirements: 2-3 core arguments, evidence-based, academic tone, 150-200 words."
            )

        history_text = "\n".join(
            f"{h['agent']}: {h['content']}"
            for h in self.debate_history[-4:]
        )
        return (
            f"DEBATE TOPIC: {self.topic}\n\n"
            f"RECENT DISCUSSION:\n{history_text}\n\n"
            f"You are {agent_name}. Respond with counter-arguments. "
            f"150-200 words, academic tone, evidence-based."
        )

    def _run_single_round(self, round_idx: int):
        is_first = (round_idx == 1)

        # Government
        gov_prompt    = self._build_prompt(self.government_agent.name, is_first and round_idx == 1)
        gov_response  = self.government_agent.chat(gov_prompt)   # FIX: use chat() directly
        gov_entry     = {"agent": self.government_agent.name, "round": round_idx, "content": gov_response}
        self.debate_history.append(gov_entry)

        # Enterprise
        ent_prompt    = self._build_prompt(self.enterprise_agent.name, False)
        ent_response  = self.enterprise_agent.chat(ent_prompt)   # FIX: use chat() directly
        ent_entry     = {"agent": self.enterprise_agent.name, "round": round_idx, "content": ent_response}
        self.debate_history.append(ent_entry)

        # Moderator guides between rounds (not on last round)
        if round_idx < self.num_rounds:
            mod_text, _ = self.moderator.moderate(
                last_speaker  = self.enterprise_agent.name,
                last_content  = ent_response,
                round_num     = round_idx,
                debate_history= [h["content"] for h in self.debate_history]
            )
            self.debate_history.append({
                "agent": "Moderator", "round": round_idx, "content": mod_text
            })

        self._save_round(
            {"round": round_idx, "arguments": [gov_entry, ent_entry]},
            round_idx
        )

    def _moderator_summary(self) -> str:
        # FIX: method is summarize_debate(), not summarize()
        return self.moderator.summarize_debate(
            [h["content"] for h in self.debate_history]
        )

    def _save_round(self, round_record: Dict, round_idx: int):
        round_dir = os.path.join(self.output_dir, "round_summaries")
        os.makedirs(round_dir, exist_ok=True)
        path = os.path.join(round_dir, f"round_{round_idx}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(round_record, f, ensure_ascii=False, indent=2)

    def _save_artifacts(self, artifacts: Dict):
        path = os.path.join(self.output_dir, "full_debate.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)