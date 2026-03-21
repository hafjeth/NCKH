"""
debate_manager.py
=================
ĐẶT FILE NÀY TẠI: src/core/debate_manager.py  (thay file cũ)

THAY ĐỔI SO VỚI BẢN CŨ:
  1. Import Tier3Synthesizer từ đúng path: src/agents/expert/tier3_synthesizer
  2. Import Config từ đúng path: config.settings
  3. DebateManager.__init__() khởi tạo self.tier3
  4. run_debate() reset self.debate_history mỗi câu hỏi (tránh dữ liệu chồng lấp)
  5. run_debate() trả về Tuple[str, List[dict], Dict] — thêm tier3_output
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.base_agent import BaseAgent
from src.core.moderator import ModeratorAgent
from src.core.personas import AGENT_PERSONAS
from src.knowledge.retrieval.retriever import KnowledgeRetriever
from src.agents.expert.tier3_synthesizer import Tier3Synthesizer   # NEW

logger = logging.getLogger(__name__)

CITATION_RULE = """
CITATION RULES (BẮT BUỘC):
- Mỗi lập luận tham chiếu văn bản pháp lý PHẢI có trích dẫn inline
- Format: [Nguồn: <tên văn bản>]
- Ví dụ: [Nguồn: Nghị định 06/2022/NĐ-CP], [Nguồn: Quyết định 888/QĐ-TTg]
- Tối thiểu 2 citations mỗi lượt phát biểu
"""


class DebateManager:

    def __init__(self):
        self.debate_history: List[dict] = []
        self.government_agent: Optional[BaseAgent] = None
        self.business_agent:   Optional[BaseAgent] = None
        self.expert_agent:     Optional[BaseAgent] = None
        self.moderator:        Optional[ModeratorAgent] = None
        self.retriever  = KnowledgeRetriever()
        self.tier3      = Tier3Synthesizer()               # NEW

    def setup_agents(self):
        def build_agent(key):
            p = AGENT_PERSONAS[key]
            return BaseAgent(
                name=p["agent_name"],
                role=self._build_role_prompt(p),
                retriever=self.retriever
            )
        self.government_agent = build_agent("government")
        self.business_agent   = build_agent("business")
        self.expert_agent     = build_agent("expert")
        self.moderator        = ModeratorAgent(name="Moderator", max_rounds=3)
        logger.info("Agents initialized successfully")

    def _build_role_prompt(self, persona: dict) -> str:
        def fmt(items):
            return "\n".join(f"- {i}" for i in items)
        return f"""
{persona['role_description']}
REPRESENTED ENTITY: {persona['represented_entity']}
CORE OBJECTIVES:\n{fmt(persona['core_objectives'])}
POLICY PRIORITIES:\n{fmt(persona['policy_priorities'])}
CONSTRAINTS:\n{fmt(persona['constraints_and_challenges'])}
REASONING STYLE: {persona['reasoning_style']}
{persona['response_guidelines']}
CITATION REQUIREMENT: Khi tham chiếu văn bản pháp lý PHẢI dùng [Nguồn: tên văn bản]. Tối thiểu 2 citations/lượt.
""".strip()

    # ── Round prompts (giữ nguyên từ bản gốc) ────────────────────────────────

    def _prompt_round1(self, agent_name: str, topic: str) -> str:
        if "Government" in agent_name:
            role_instruction = """Trình bày lập trường ban đầu của Chính phủ:
- Đưa ra 2–3 luận điểm chính dựa trên văn bản pháp luật cụ thể
- Nêu rõ cơ sở pháp lý và cam kết quốc tế
- Thể hiện lộ trình chính sách dài hạn"""
        else:
            role_instruction = """Phản hồi từ góc độ doanh nghiệp:
- Trình bày 2–3 quan ngại chính (chi phí, năng lực, cạnh tranh)
- Tập trung vào tác động thực tế với SME
- Đề xuất hướng hỗ trợ mong muốn từ Chính phủ"""
        return f"""VÒNG 1 — TRÌNH BÀY QUAN ĐIỂM BAN ĐẦU
CHỦ ĐỀ: {topic}
Bạn là {agent_name}.
{role_instruction}
Giọng văn học thuật | 150–200 từ
{CITATION_RULE}"""

    def _prompt_round2(self, agent_name: str, topic: str, recent_history: List[dict]) -> str:
        history_text = "\n\n".join(
            f"[{h['agent'].upper()} - Vòng {h['round']}]\n{h['content']}"
            for h in recent_history if h.get("agent") != "Moderator"
        )
        opponent  = "Doanh nghiệp" if "Government" in agent_name else "Chính phủ"
        your_role = "Chính phủ"    if "Government" in agent_name else "Doanh nghiệp"
        return f"""VÒNG 2 — PHẢN BIỆN CHÉO
CHỦ ĐỀ: {topic}
LỊCH SỬ VÒNG 1:\n{history_text}
Bạn là {agent_name} — đại diện {your_role}.
NHIỆM VỤ:
1. Tóm tắt lập luận của {opponent} (1–2 câu)
2. Chỉ ra 2 điểm yếu cụ thể trong lập luận đó
3. Dẫn chứng cụ thể để bác bỏ hoặc làm rõ
4. Củng cố lập trường với bằng chứng bổ sung
Giọng văn học thuật | 150–200 từ
{CITATION_RULE}"""

    def _prompt_round3(self, agent_name: str, topic: str, recent_history: List[dict]) -> str:
        history_text = "\n\n".join(
            f"[{h['agent'].upper()} - Vòng {h['round']}]\n{h['content']}"
            for h in recent_history[-6:] if h.get("agent") != "Moderator"
        )
        return f"""VÒNG 3 — BẢO VỆ LẬP LUẬN CUỐI CÙNG
CHỦ ĐỀ: {topic}
LỊCH SỬ TRANH LUẬN:\n{history_text}
Bạn là {agent_name}.
NHIỆM VỤ:
1. Thừa nhận ngắn gọn điểm hợp lý của đối phương (1 câu)
2. Bảo vệ lập trường cốt lõi với lý lẽ mạnh nhất
3. Đề xuất 1 điểm có thể đồng thuận giữa hai bên
4. Kết luận lập trường cuối cùng rõ ràng
Giọng văn học thuật | 150–200 từ
{CITATION_RULE}"""

    # ── Expert synthesis (giữ nguyên từ bản gốc) ─────────────────────────────

    def expert_synthesis(self, topic: str) -> str:
        gov_views = [h["content"] for h in self.debate_history if h["agent"] == "Government Agent"]
        biz_views = [h["content"] for h in self.debate_history if h["agent"] == "Enterprise Agent"]
        prompt = f"""Bạn là CHUYÊN GIA PHÂN TÍCH CHÍNH SÁCH ĐỘC LẬP.
NHIỆM VỤ — Phân tích toàn bộ cuộc tranh luận 3 vòng:
1. TÓM TẮT lập trường Chính phủ (qua 3 vòng)
2. TÓM TẮT lập trường Doanh nghiệp (qua 3 vòng)
3. SO SÁNH hai quan điểm
4. XÁC ĐỊNH: trade-offs | điểm đồng thuận | bất đồng cốt lõi | tiến triển qua 3 vòng
NGUYÊN TẮC: KHÔNG đề xuất chính sách mới | KHÔNG đứng về phía nào
CHỦ ĐỀ: {topic}
LẬP LUẬN CHÍNH PHỦ:\n{chr(10).join(f'Vòng {i+1}: {v}' for i,v in enumerate(gov_views))}
LẬP LUẬN DOANH NGHIỆP:\n{chr(10).join(f'Vòng {i+1}: {v}' for i,v in enumerate(biz_views))}
ĐẦU RA: 300–400 từ | Tối thiểu 3 trích dẫn [Nguồn: ...]"""
        return self.expert_agent.chat(prompt)

    # ── MAIN: run_debate ──────────────────────────────────────────────────────

    def run_debate(
        self,
        topic: str,
        max_rounds: int = 3,
    ) -> Tuple[str, List[dict], Dict]:
        """
        Chạy toàn bộ pipeline 3 tầng.

        Returns:
            expert_text    (str)   — Expert Council output (Tier 2)
            debate_history (list)  — Lịch sử toàn bộ Tier 2
            tier3_output   (dict)  — 5 module Policy Synthesis (Tier 3)  ← NEW
        """
        if not all([self.government_agent, self.business_agent,
                    self.expert_agent, self.moderator]):
            self.setup_agents()

        # Reset history mỗi lần chạy (tránh chồng dữ liệu giữa các câu hỏi)
        self.debate_history = []

        logger.info(f"[Tier2] Debate bắt đầu: {topic[:60]}...")
        agents = [self.government_agent, self.business_agent]

        # ── Tier 2: 3 vòng debate ────────────────────────────────────────────
        for round_num in range(1, max_rounds + 1):
            logger.info(f"  [Tier2] Round {round_num}/{max_rounds}")
            for agent in agents:
                if round_num == 1:
                    prompt = self._prompt_round1(agent.name, topic)
                elif round_num == 2:
                    prompt = self._prompt_round2(agent.name, topic, self.debate_history)
                else:
                    prompt = self._prompt_round3(agent.name, topic, self.debate_history)

                response = agent.chat(prompt)
                self.debate_history.append({
                    "round":   round_num,
                    "agent":   agent.name,
                    "content": response,
                })

            # Moderator dẫn dắt (không ở vòng cuối)
            if round_num < max_rounds:
                last = self.debate_history[-1]
                mod_text, _ = self.moderator.moderate(
                    last_speaker=last["agent"],
                    last_content=last["content"],
                    round_num=round_num,
                    debate_history=[h["content"] for h in self.debate_history],
                )
                self.debate_history.append({
                    "round":   round_num,
                    "agent":   "Moderator",
                    "content": mod_text,
                })

        # Expert Council phân tích độc lập
        expert_text = self.expert_synthesis(topic)
        self.debate_history.append({
            "round": "post", "agent": "Expert", "content": expert_text,
        })

        # Moderator tổng kết
        final_summary = self.moderator.summarize_debate(
            [h["content"] for h in self.debate_history]
        )
        self.debate_history.append({
            "round": "final", "agent": "Moderator", "content": final_summary,
        })

        logger.info(f"[Tier2] Hoàn thành — {len(self.debate_history)} turns")

        # ── Tier 3: Policy Synthesis ─────────────────────────────────────────
        logger.info("[Tier3] Bắt đầu Policy Synthesis...")
        tier3_output = self.tier3.synthesize(
            topic=topic,
            expert_text=expert_text,
            final_summary=final_summary,
            debate_history=self.debate_history,
        )
        logger.info("[Tier3] Hoàn thành")

        return expert_text, self.debate_history, tier3_output