"""
tier3_synthesizer.py
====================
ĐẶT FILE NÀY TẠI: src/agents/expert/tier3_synthesizer.py

Tier 3: Policy Synthesis & Outputs
Nhận kết quả Tier 2 → tổng hợp 5 module có cấu trúc:
  1. policy_options    — Lựa chọn chính sách khả thi
  2. pros_cons         — Ưu/nhược điểm từng lựa chọn
  3. impact_analysis   — Phân tích tác động (kinh tế/môi trường/xã hội)
  4. evidence          — Bằng chứng & trích dẫn tổng hợp
  5. decision_support  — Hỗ trợ ra quyết định

Import trong debate_manager.py:
    from src.agents.expert.tier3_synthesizer import Tier3Synthesizer
"""

import re
import json
import logging
from typing import List, Dict, Optional

from config.settings import Config          # cấu trúc thực tế của project NCKH

logger = logging.getLogger(__name__)

_SYSTEM = """Bạn là chuyên gia phân tích chính sách học thuật cao cấp.
Nhiệm vụ: Tổng hợp kết quả tranh luận đa tác nhân thành báo cáo chính sách có cấu trúc.
Nguyên tắc:
- Trung lập, học thuật, có cơ sở bằng chứng
- Không bịa thêm thông tin ngoài cuộc tranh luận
- Phản ánh đầy đủ cả quan điểm Chính phủ lẫn Doanh nghiệp
- Trích dẫn inline [Nguồn: ...] khi có thể
- Trả lời CHỈ bằng JSON hợp lệ, không có markdown fence, không có text ngoài JSON"""


class Tier3Synthesizer:
    """
    Tier 3: Policy Synthesis & Outputs
    Tương thích với cả OpenAI và Anthropic (dựa vào Config.API_PROVIDER)
    """

    def __init__(self):
        self._client      = None
        self._provider    = Config.API_PROVIDER
        self._model       = Config.MODEL_NAME
        self._temperature = 0.2
        self._max_tokens  = 1500

    # ── Lazy client init (hỗ trợ cả OpenAI và Anthropic) ─────────────────────
    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self._provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"[Tier3] Unsupported provider: {self._provider}")
        return self._client

    # ── Entry point ───────────────────────────────────────────────────────────
    def synthesize(
        self,
        topic: str,
        expert_text: str,
        final_summary: str,
        debate_history: Optional[List[dict]] = None,
    ) -> Dict:
        """
        Chạy toàn bộ Tier 3 pipeline.

        Args:
            topic          : Câu hỏi / chủ đề tranh luận
            expert_text    : Expert Council output từ Tier 2
            final_summary  : Moderator final summary từ Tier 2
            debate_history : Toàn bộ lịch sử tranh luận

        Returns:
            dict với 5 module + metadata
        """
        logger.info("[Tier3] Bắt đầu Policy Synthesis...")
        ctx = self._build_context(expert_text, final_summary, debate_history)

        policy_options   = self._gen_policy_options(topic, ctx)
        pros_cons        = self._gen_pros_cons(topic, ctx, policy_options)
        impact_analysis  = self._gen_impact_analysis(topic, ctx)
        evidence         = self._gen_evidence(topic, ctx, debate_history)
        decision_support = self._gen_decision_support(
            topic, ctx, policy_options, impact_analysis
        )

        logger.info("[Tier3] Hoàn thành Policy Synthesis.")
        return {
            "tier":             3,
            "topic":            topic,
            "policy_options":   policy_options,
            "pros_cons":        pros_cons,
            "impact_analysis":  impact_analysis,
            "evidence":         evidence,
            "decision_support": decision_support,
        }

    # ── Module 1 ──────────────────────────────────────────────────────────────
    def _gen_policy_options(self, topic: str, ctx: str) -> Dict:
        prompt = f"""CHỦ ĐỀ: {topic}

NGỮ CẢNH TRANH LUẬN:
{ctx}

NHIỆM VỤ — MODULE 1: POLICY OPTIONS
Dựa trên cuộc tranh luận, xác định 3–4 lựa chọn chính sách khả thi.

Trả về JSON:
{{
  "options": [
    {{
      "id": "OPT-1",
      "title": "Tên ngắn gọn",
      "description": "Mô tả 2–3 câu",
      "stakeholder_alignment": "government|business|both",
      "feasibility": "high|medium|low",
      "citations": ["[Nguồn: ...]"]
    }}
  ],
  "summary": "Tóm tắt 1–2 câu về không gian lựa chọn chính sách"
}}"""
        return self._call(prompt, "options")

    # ── Module 2 ──────────────────────────────────────────────────────────────
    def _gen_pros_cons(self, topic: str, ctx: str, policy_options: Dict) -> Dict:
        opts_str = json.dumps(
            policy_options.get("options", []), ensure_ascii=False, indent=2
        )
        prompt = f"""CHỦ ĐỀ: {topic}

CÁC LỰA CHỌN CHÍNH SÁCH:
{opts_str}

NGỮ CẢNH: {ctx[:800]}

NHIỆM VỤ — MODULE 2: PROS & CONS
Phân tích ưu/nhược điểm từng lựa chọn từ góc độ Chính phủ và Doanh nghiệp.

Trả về JSON:
{{
  "analysis": [
    {{
      "option_id": "OPT-1",
      "pros": [{{"point": "...", "perspective": "government|business|both"}}],
      "cons": [{{"point": "...", "perspective": "government|business|both"}}]
    }}
  ]
}}"""
        return self._call(prompt, "analysis")

    # ── Module 3 ──────────────────────────────────────────────────────────────
    def _gen_impact_analysis(self, topic: str, ctx: str) -> Dict:
        prompt = f"""CHỦ ĐỀ: {topic}

NGỮ CẢNH TRANH LUẬN:
{ctx}

NHIỆM VỤ — MODULE 3: IMPACT ANALYSIS
Phân tích tác động theo 3 chiều dựa trên nội dung đã tranh luận.

Trả về JSON:
{{
  "economic": {{
    "short_term": "Tác động 1–3 năm",
    "long_term": "Tác động 5–10 năm",
    "sme_impact": "Tác động với SME dệt may",
    "competitiveness": "Tác động cạnh tranh xuất khẩu",
    "severity": "high|medium|low"
  }},
  "environmental": {{
    "ghg_reduction_potential": "Tiềm năng giảm phát thải",
    "mrv_readiness": "Năng lực MRV",
    "cbam_alignment": "Mức phù hợp với EU CBAM",
    "severity": "high|medium|low"
  }},
  "social": {{
    "employment": "Tác động việc làm",
    "equity": "Công bằng SME vs doanh nghiệp lớn",
    "severity": "high|medium|low"
  }},
  "overall_risk": "high|medium|low",
  "key_uncertainties": ["...", "..."]
}}"""
        return self._call(prompt, "economic")

    # ── Module 4 ──────────────────────────────────────────────────────────────
    def _gen_evidence(
        self,
        topic: str,
        ctx: str,
        debate_history: Optional[List[dict]],
    ) -> Dict:
        citations: List[str] = []
        if debate_history:
            for turn in debate_history:
                found = re.findall(r'\[Nguồn:[^\]]+\]', turn.get("content", ""))
                citations.extend(found)
        unique = list(dict.fromkeys(citations))

        prompt = f"""CHỦ ĐỀ: {topic}

NGỮ CẢNH TRANH LUẬN:
{ctx}

CITATIONS ĐÃ SỬ DỤNG:
{chr(10).join(unique) if unique else "(xem trong ngữ cảnh)"}

NHIỆM VỤ — MODULE 4: EVIDENCE
Tổng hợp bằng chứng và trích dẫn được dùng trong tranh luận.

Trả về JSON:
{{
  "legal_documents": [
    {{
      "name": "Tên văn bản",
      "relevance": "Liên quan đến điểm nào trong tranh luận",
      "used_by": "government|business|both|expert"
    }}
  ],
  "key_facts": [
    {{
      "fact": "Sự kiện / số liệu quan trọng",
      "source": "Nguồn",
      "contested": false
    }}
  ],
  "evidence_gaps": ["Khoảng trống bằng chứng 1", "Khoảng trống 2"],
  "citation_count": {len(unique)}
}}"""
        return self._call(prompt, "legal_documents")

    # ── Module 5 ──────────────────────────────────────────────────────────────
    def _gen_decision_support(
        self,
        topic: str,
        ctx: str,
        policy_options: Dict,
        impact_analysis: Dict,
    ) -> Dict:
        opts_str   = json.dumps(
            policy_options.get("options", []), ensure_ascii=False, indent=2
        )
        impact_str = json.dumps(impact_analysis, ensure_ascii=False, indent=2)

        prompt = f"""CHỦ ĐỀ: {topic}

CÁC LỰA CHỌN CHÍNH SÁCH:
{opts_str}

PHÂN TÍCH TÁC ĐỘNG:
{impact_str}

NGỮ CẢNH: {ctx[:600]}

NHIỆM VỤ — MODULE 5: DECISION SUPPORT
Hỗ trợ ra quyết định có căn cứ từ tranh luận.
QUAN TRỌNG: Không đưa ra khuyến nghị ngoài những gì đã tranh luận.

Trả về JSON:
{{
  "recommended_option": {{
    "option_id": "OPT-X",
    "rationale": "Lý do dựa trên trade-offs (2–3 câu)",
    "conditions": ["Điều kiện 1", "Điều kiện 2"]
  }},
  "consensus_points": ["Điểm đồng thuận 1", "Điểm đồng thuận 2"],
  "contested_points": ["Bất đồng 1", "Bất đồng 2"],
  "next_steps": [
    {{
      "action": "Hành động cụ thể",
      "responsible": "government|business|both",
      "timeline": "short|medium|long"
    }}
  ],
  "data_gaps_to_address": ["Khoảng trống cần thu thập"]
}}"""
        return self._call(prompt, "recommended_option")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_context(
        self,
        expert_text: str,
        final_summary: str,
        debate_history: Optional[List[dict]],
    ) -> str:
        parts = []
        if expert_text:
            parts.append(f"=== EXPERT COUNCIL ANALYSIS ===\n{expert_text[:2000]}")
        if final_summary:
            parts.append(f"=== MODERATOR FINAL SUMMARY ===\n{final_summary[:2000]}")
        if debate_history:
            key_turns = [
                h for h in debate_history
                if h.get("agent") in ("Government Agent", "Enterprise Agent")
            ]
            turns_text = "\n\n".join(
                f"[{h['agent']} - Round {h.get('round','?')}]\n{h['content'][:500]}"
                for h in key_turns[-6:]
            )
            if turns_text:
                parts.append(f"=== KEY DEBATE TURNS ===\n{turns_text}")
        return "\n\n".join(parts)

    def _call(self, prompt: str, fallback_key: str) -> Dict:
        try:
            client = self._get_client()

            if self._provider == "openai":
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()

            else:  # anthropic
                resp = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()

            return self._parse_json(raw, fallback_key)

        except Exception as e:
            logger.error(f"[Tier3] LLM call failed ({fallback_key}): {e}")
            return {fallback_key: [], "_error": str(e)}

    @staticmethod
    def _parse_json(raw: str, fallback_key: str) -> Dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            logger.warning(f"[Tier3] JSON parse failed ({fallback_key})")
            return {fallback_key: [], "_raw": raw[:300]}