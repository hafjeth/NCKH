"""
debate_manager.py
=================
ĐẶT FILE NÀY TẠI: src/core/debate_manager.py

FIXES SO VỚI BẢN TRƯỚC:
  [FIX-5a] CITATION_RULE đơn giản hóa — không còn hardcode danh sách nguồn cứng
           Tên nguồn hợp lệ lấy từ RAG (50+ tài liệu) thay vì whitelist cứng 5-6 nguồn
  [FIX-5b] expert_synthesis() thêm CITATION_RULE vào prompt
  [FIX-5c] _prompt_round3(): thêm anchor lập trường cho từng agent
  [FIX-10] CITATION_RULE_WITH_EXAMPLES: thêm ví dụ cụ thể để LLM dễ hiểu
  [FIX-11] expert_synthesis(): đưa CITATION_RULE lên ĐẦU prompt
  [FIX-12] _validate_citations(): post-processing kiểm tra citations hợp lệ
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.base_agent import BaseAgent
from src.core.moderator import ModeratorAgent
from src.core.personas import AGENT_PERSONAS
from src.knowledge.retrieval.retriever import KnowledgeRetriever
from src.agents.expert.tier3_synthesizer import Tier3Synthesizer

logger = logging.getLogger(__name__)

# [FIX-10] CITATION_RULE với ví dụ cụ thể
CITATION_RULE_WITH_EXAMPLES = """
CITATION RULES (BẮT BUỘC — VI PHẠM = PHẢN HỒI KHÔNG HỢP LỆ):

📌 ĐÚNG — Các dạng citation hợp lệ:
  • [Nguồn: Quyết định 232 QĐ-TTg]
  • [Nguồn: Nghị định 06/2022/NĐ-CP]
  • [Nguồn: Luật Bảo vệ Môi trường 2020]
  • [Nguồn: Báo cáo IFC World Bank]
  • [Nguồn: Nghị định 45/2022/NĐ-CP]

❌ SAI — Các dạng citation KHÔNG được phép (tuyệt đối tránh):
  • "theo thực tiễn quốc tế..."
  • "theo nghiên cứu cho thấy..."
  • "theo các chuyên gia..."
  • "theo thông tin từ..."
  • "theo báo cáo gần đây..."
  • [Nguồn: không rõ]
  • [Nguồn: theo một nghiên cứu]

📋 NGUYÊN TẮC CỐT LÕI:
  1. CHỈ cite nguồn có trong TÀI LIỆU THAM KHẢO được cung cấp trong prompt này
  2. Format bắt buộc: [Nguồn: <tên nguồn chính xác như trong TÀI LIỆU THAM KHẢO>]
  3. Trước khi cite: nội dung bạn trình bày phải XUẤT HIỆN TRONG ĐOẠN VĂN của tài liệu đó
  4. Tối thiểu 2 citations/lượt nếu tài liệu có nội dung liên quan
  5. Nếu không có tài liệu phù hợp → trình bày bình thường, KHÔNG cần cite

⚠️ QUY TẮC VÀNG:
  - CÓ nguồn hợp lệ trong TÀI LIỆU THAM KHẢO → cite [Nguồn: Tên cụ thể]
  - KHÔNG có nguồn hợp lệ → KHÔNG cite, KHÔNG dùng bất kỳ cụm giả nguồn nào
"""

# CITATION_RULE ngắn gọn cho các agent khác (giữ nguyên)
CITATION_RULE = """
CITATION RULES (BẮT BUỘC — VI PHẠM = PHẢN HỒI KHÔNG HỢP LỆ):

NGUYÊN TẮC CỐT LÕI:
- CHỈ cite nguồn có trong TÀI LIỆU THAM KHẢO được cung cấp trong prompt này
- Format bắt buộc: [Nguồn: <tên nguồn chính xác như trong TÀI LIỆU THAM KHẢO>]
- Trước khi cite [Nguồn: X]: kiểm tra nội dung bạn trình bày có XUẤT HIỆN
  TRONG ĐOẠN VĂN của TÀI LIỆU X không — nếu KHÔNG thì KHÔNG cite
- Tối thiểu 2 citations/lượt nếu tài liệu có nội dung liên quan
- Nếu không có tài liệu phù hợp: trình bày bình thường, không cần cite

NGHIÊM CẤM:
- Tự bịa tên nguồn không có trong TÀI LIỆU THAM KHẢO
- Gán nội dung của tài liệu này vào tên tài liệu khác
- Dùng bất kỳ cụm giả nguồn nào như 'theo thực tiễn quốc tế',
  'theo nghiên cứu quốc tế', 'theo các chuyên gia', v.v.
- Cite chung chung không rõ tên tài liệu cụ thể
"""


class DebateManager:

    def __init__(self):
        self.debate_history: List[dict] = []
        self.government_agent: Optional[BaseAgent] = None
        self.business_agent:   Optional[BaseAgent] = None
        self.expert_agent:     Optional[BaseAgent] = None
        self.moderator:        Optional[ModeratorAgent] = None
        self.retriever  = KnowledgeRetriever()
        self.tier3      = Tier3Synthesizer()

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
CITATION REQUIREMENT: Khi tham chiếu tài liệu PHẢI dùng [Nguồn: tên tài liệu
chính xác từ TÀI LIỆU THAM KHẢO được cung cấp]. Tối thiểu 2 citations/lượt.
""".strip()

    # ── Round prompts ─────────────────────────────────────────────────────────

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

        if "Government" in agent_name:
            your_role    = "Chính phủ"
            core_stance  = "chính sách, pháp lý, cam kết quốc tế, lợi ích quốc gia dài hạn"
            voice_anchor = (
                "Ngôn ngữ của bạn mang tính CHÍNH SÁCH — dùng các từ như "
                "'lộ trình', 'quy định', 'cam kết', 'thị trường carbon', 'NDC'."
            )
        else:
            your_role    = "Doanh nghiệp"
            core_stance  = "chi phí tuân thủ, năng lực SMEs, tác động kinh doanh thực tế"
            voice_anchor = (
                "Ngôn ngữ của bạn mang tính DOANH NGHIỆP — dùng các từ như "
                "'chi phí', 'lợi nhuận', 'năng lực', 'đầu tư', 'SMEs', 'cạnh tranh'."
            )

        return f"""VÒNG 3 — BẢO VỆ LẬP LUẬN CUỐI CÙNG
CHỦ ĐỀ: {topic}
LỊCH SỬ TRANH LUẬN:\n{history_text}
Bạn là {agent_name} — đại diện {your_role}.

NHIỆM VỤ:
1. Thừa nhận ngắn gọn điểm hợp lý của đối phương (1 câu) — nếu thực sự hợp lý
2. Bảo vệ lập trường cốt lõi của {your_role} về: {core_stance}
3. Đề xuất 1 điểm có thể đồng thuận giữa hai bên
4. Kết luận lập trường cuối cùng rõ ràng

⚠️ GIỮ BẢN SẮC: {voice_anchor}
KHÔNG dùng lại câu chữ hoặc cấu trúc câu của đối phương, dù quan điểm có thể gần nhau.
Đồng thuận thực chất là bình thường — nhưng phải thể hiện qua góc nhìn của {your_role}.

Giọng văn học thuật | 150–200 từ
{CITATION_RULE}"""

    # ── Expert synthesis ──────────────────────────────────────────────────────
    # [FIX-11] Đưa CITATION_RULE lên ĐẦU prompt
    # [FIX-12] Thêm post-processing validation

    def _extract_valid_sources_from_history(self) -> Set[str]:
        """
        [FIX-12] Trích xuất tất cả nguồn hợp lệ từ debate history.
        Dùng để validate citations của Expert.
        """
        valid_sources = set()
        pattern = r'\[Nguồn:([^\]]+)\]'
        
        for turn in self.debate_history:
            content = turn.get("content", "")
            matches = re.findall(pattern, content)
            for match in matches:
                # Clean tên nguồn
                source = match.strip()
                # Bỏ đuôi file nếu có
                source = re.sub(r'\.(txt|pdf|docx|doc)$', '', source, flags=re.IGNORECASE)
                valid_sources.add(source)
        
        return valid_sources

    def _validate_citations(self, response: str, valid_sources: Set[str]) -> str:
        """
        [FIX-12] Post-processing: kiểm tra và loại bỏ citations không hợp lệ.
        """
        # Tìm tất cả citations trong response
        citation_pattern = r'\[Nguồn:([^\]]+)\]'
        matches = re.findall(citation_pattern, response)
        
        invalid_citations = []
        for match in matches:
            source = match.strip()
            # Kiểm tra xem nguồn có trong danh sách hợp lệ không
            is_valid = False
            for valid in valid_sources:
                if source == valid or source in valid or valid in source:
                    is_valid = True
                    break
            if not is_valid:
                invalid_citations.append(f"[Nguồn:{match}]")
        
        # Loại bỏ citations không hợp lệ
        for invalid in invalid_citations:
            response = response.replace(invalid, "")
            logger.warning(f"⚠️ Removed invalid citation: {invalid}")
        
        # Dọn dẹp khoảng trắng thừa
        response = re.sub(r'\s+', ' ', response)
        response = response.strip()
        
        return response

    def expert_synthesis(self, topic: str) -> str:
        """
        [FIX-5b] Thêm CITATION_RULE vào prompt expert_synthesis.
        [FIX-11] Đưa CITATION_RULE lên ĐẦU prompt.
        [FIX-12] Thêm post-processing validation.
        """
        gov_views = [h["content"] for h in self.debate_history
                     if h["agent"] == "Government Agent"]
        biz_views = [h["content"] for h in self.debate_history
                     if h["agent"] == "Enterprise Agent"]
        
        # [FIX-11] Lấy valid sources từ history để validate sau
        valid_sources = self._extract_valid_sources_from_history()
        sources_list = "\n".join(f"  - {s}" for s in sorted(valid_sources)[:20]) if valid_sources else "  (chưa có citations nào trong debate)"

        # [FIX-11] Đặt CITATION_RULE_WITH_EXAMPLES lên ĐẦU prompt
        prompt = f"""{CITATION_RULE_WITH_EXAMPLES}

📚 CÁC NGUỒN HỢP LỆ ĐÃ XUẤT HIỆN TRONG TRANH LUẬN (CHỈ ĐƯỢC DÙNG CÁC NGUỒN NÀY):
{sources_list}

Bạn là CHUYÊN GIA PHÂN TÍCH CHÍNH SÁCH ĐỘC LẬP.

NHIỆM VỤ — Phân tích toàn bộ cuộc tranh luận 3 vòng:
1. TÓM TẮT lập trường Chính phủ (qua 3 vòng)
2. TÓM TẮT lập trường Doanh nghiệp (qua 3 vòng)
3. SO SÁNH hai quan điểm
4. XÁC ĐỊNH: trade-offs | điểm đồng thuận | bất đồng cốt lõi | tiến triển qua 3 vòng

NGUYÊN TẮC PHÂN TÍCH:
- KHÔNG đề xuất chính sách mới
- KHÔNG đứng về phía nào
- CHỈ phân tích những gì đã được lập luận trong debate
- KHI CITE: CHỈ dùng các nguồn trong danh sách 📚 bên trên

CHỦ ĐỀ: {topic}

LẬP LUẬN CHÍNH PHỦ:
{chr(10).join(f'Vòng {i+1}: {v}' for i, v in enumerate(gov_views))}

LẬP LUẬN DOANH NGHIỆP:
{chr(10).join(f'Vòng {i+1}: {v}' for i, v in enumerate(biz_views))}

ĐẦU RA: 300–400 từ
"""

        # Gọi LLM
        response = self.expert_agent.chat(prompt)
        
        # [FIX-12] Post-processing: validate citations
        if valid_sources:
            response = self._validate_citations(response, valid_sources)
        
        return response

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
            tier3_output   (dict)  — 5 module Policy Synthesis (Tier 3)
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