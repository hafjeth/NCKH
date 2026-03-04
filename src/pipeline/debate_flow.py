import json
import os
from datetime import datetime
from typing import Dict, List

from src.agents.government.carbon_policy_agent import CarbonPolicyAgent
from src.agents.enterprise.textile_industry_agent import TextileIndustryAgent
from src.core.moderator import Moderator
from experiments.utils.logger import get_logger

# ===============================
# Debate Flow (Gov ↔ Enterprise)
# ===============================

class DebateFlow:
    """
    Implements a multi-round stakeholder debate
    between Government and Enterprise agents.
    """

    def __init__(
        self,
        topic: str,
        knowledge_retriever,
        output_dir: str,
        num_rounds: int = 2,
    ):
        self.topic = topic
        self.knowledge_retriever = knowledge_retriever
        self.output_dir = output_dir
        self.num_rounds = num_rounds

        self.logger = get_logger("DebateFlow")

        # Agents
        self.government_agent = CarbonPolicyAgent()
        self.enterprise_agent = TextileIndustryAgent()
        self.moderator = Moderator()

        # Internal state
        self.debate_history: List[Dict] = []

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run(self) -> Dict:
        """
        Execute the full debate flow.
        """
        self.logger.info("Starting debate flow")
        context = self._retrieve_context()

        for round_idx in range(1, self.num_rounds + 1):
            self.logger.info(f"Running debate round {round_idx}")
            self._run_single_round(round_idx, context)

        summary = self._moderator_summary()

        artifacts = {
            "topic": self.topic,
            "num_rounds": self.num_rounds,
            "timestamp": datetime.utcnow().isoformat(),
            "debate_history": self.debate_history,
            "moderator_summary": summary,
        }

        self._save_artifacts(artifacts)
        self.logger.info("Debate flow completed")

        return artifacts

    # --------------------------------------------------
    # Internal methods
    # --------------------------------------------------
    def _retrieve_context(self) -> Dict:
        """
        Retrieve relevant knowledge for the debate topic.
        """
        self.logger.info("Retrieving contextual knowledge")
        context = self.knowledge_retriever.retrieve(self.topic)

        return context

    def _run_single_round(self, round_idx: int, context: Dict):
        """
        Run one debate round: Government → Enterprise
        """
        round_record = {
            "round": round_idx,
            "arguments": []
        }

        # --- Government argument ---
        gov_argument = self.government_agent.generate_argument(
            topic=self.topic,
            context=context,
            debate_history=self.debate_history
        )
        gov_argument["agent"] = "government"
        gov_argument["round"] = round_idx

        round_record["arguments"].append(gov_argument)
        self.debate_history.append(gov_argument)

        # --- Enterprise counter-argument ---
        ent_argument = self.enterprise_agent.generate_argument(
            topic=self.topic,
            context=context,
            debate_history=self.debate_history
        )
        ent_argument["agent"] = "enterprise"
        ent_argument["round"] = round_idx

        round_record["arguments"].append(ent_argument)
        self.debate_history.append(ent_argument)

        # Persist round-level artifact
        self._save_round(round_record, round_idx)

    def _moderator_summary(self) -> Dict:
        """
        Moderator synthesizes the full debate.
        """
        self.logger.info("Moderator generating debate summary")
        summary = self.moderator.summarize(self.debate_history)
        return summary

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------
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
