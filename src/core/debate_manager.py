import logging
import time
import os
import sys
from typing import List
from src.core.agent_base import BaseAgent
from src.core.moderator import ModeratorAgent

# --- CẤU HÌNH LOGGER ---
logger = logging.getLogger(__name__)

# --- 1. KIỂM TRA MODULE PERSONAS (BẮT BUỘC) ---
try:
    from src.knowledge.personas import PersonaManager, PersonaType
except ImportError:
    logger.critical("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy module 'src.knowledge.personas'.")
    logger.critical("👉 Vui lòng kiểm tra lại file personas.py.")
    sys.exit(1)

# --- 2. KIỂM TRA MODULE RAG (BẮT BUỘC) ---
try:
    from src.knowledge.retrieval import RetrievalSystem
except ImportError:
    logger.critical("❌ LỖI NGHIÊM TRỌNG: Không tìm thấy module 'src.knowledge.retrieval'.")
    sys.exit(1)

class DebateManager:
    def __init__(self):
        self.debate_history: List[str] = []
        self.agents: List[BaseAgent] = []
        self.moderator = None
        self.retriever = None
        self.persona_manager = PersonaManager()

        # --- KHỞI TẠO RAG (CHẾ ĐỘ NGHIÊM NGẶT) ---
        try:
            # Tính toán đường dẫn đến data/chroma_db
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            db_path = os.path.join(project_root, "data", "chroma_db")
            
            logger.info(f"🔌 Đang kết nối RAG tại: {db_path}")
            
            # Kiểm tra xem thư mục DB có tồn tại không
            if not os.path.exists(db_path) or not os.listdir(db_path):
                raise FileNotFoundError(f"Thư mục Database trống hoặc không tồn tại: {db_path}")

            self.retriever = RetrievalSystem(
                chroma_db_dir=db_path,
                collection_name="knowledge_base",
                top_k=3
            )
            logger.info("✅ KẾT NỐI RAG THÀNH CÔNG.")
            
        except Exception as e:
            logger.critical("\n" + "="*50)
            logger.critical("⛔ KHÔNG THỂ KHỞI ĐỘNG HỆ THỐNG VÌ LỖI RAG!")
            logger.critical(f"Lỗi chi tiết: {e}")
            logger.critical("👉 HƯỚNG DẪN FIX: Hãy xóa thư mục 'data/chroma_db' và chạy lại 'python src/knowledge/ingestion.py'")
            logger.critical("="*50 + "\n")
            sys.exit(1) # Dừng chương trình ngay lập tức

    def _get_persona_prompt(self, p_type):
        """Lấy system prompt từ PersonaManager"""
        raw = self.persona_manager.get_system_prompt(p_type)
        return raw.replace("{context}", "").replace("{question}", "")

    def setup_agents(self):
        self.moderator = ModeratorAgent()

        # --- AGENT 1: CHÍNH PHỦ (Cần RAG) ---
        agent_gov = BaseAgent(
            name="DaiDien_BoTNMT",
            role=self._get_persona_prompt(PersonaType.GOVERNMENT),
            retriever=self.retriever # Bắt buộc có RAG
        )

        # --- AGENT 2: DOANH NGHIỆP (Thực tế) ---
        agent_biz = BaseAgent(
            name="HiepHoi_DetMay",
            role=self._get_persona_prompt(PersonaType.ENTERPRISE),
            retriever=None 
        )

        # --- AGENT 3: NGO / CHUYÊN GIA (Cần RAG) ---
        agent_ngo = BaseAgent(
            name="ChuyenGia_KinhTe",
            role=self._get_persona_prompt(PersonaType.NGO),
            retriever=self.retriever # Bắt buộc có RAG
        )

        self.agents = [agent_gov, agent_biz, agent_ngo]
        logger.info(f"✅ Đã thiết lập 3 Agents: Chính phủ, Doanh nghiệp, NGO.")

    def construct_prompt(self, current_agent_name: str, current_agent_role: str, topic: str) -> str:
        history_excerpt = "\n".join(self.debate_history[-3:])
        return (
            f"CHỦ ĐỀ: {topic}\n"
            f"LỊCH SỬ GẦN NHẤT:\n{history_excerpt}\n\n"
            f"VAI TRÒ: {current_agent_role}\n"
            f"NHIỆM VỤ: Phản biện ngắn gọn, tập trung vào số liệu và dẫn chứng."
        )

    def run_round(self, topic: str, max_rounds: int = 2):
        print(f"\n=== BẮT ĐẦU TỌA ĐÀM: {topic} ===\n")
        if not self.agents: self.setup_agents()

        # MC mở màn
        print("🎙️ [MC] Đang khai mạc...")
        mc_intro = self.moderator.chat(f"Chủ đề: '{topic}'. Giới thiệu ngắn 3 bên tham gia.")
        print(f"-> MC: {mc_intro}\n")
        self.debate_history.append(f"[MC]: {mc_intro}")
        time.sleep(5)

        should_continue = True
        round_count = 1
        
        while round_count <= max_rounds and should_continue:
            print(f"--- VÒNG {round_count} ---")
            
            for i, agent in enumerate(self.agents):
                # 1. Agent phát biểu
                prompt = self.construct_prompt(agent.name, agent.role, topic)
                print(f"🤔 [{agent.name}] đang suy nghĩ...")
                
                response = agent.chat(prompt)
                
                print(f"🗣️ {agent.name}: {response}\n")
                self.debate_history.append(f"[{agent.name}]: {response}")
                
                # --- QUAN TRỌNG: CHỜ 20S ĐỂ KHÔNG BỊ KHÓA API ---
                print("⏳ Đang nghỉ 20s để hồi phục API Gemini...")
                time.sleep(20)

                # 2. MC điều phối
                next_idx = (i + 1) % len(self.agents)
                next_name = self.agents[next_idx].name
                is_last_turn = (round_count == max_rounds) and (i == len(self.agents) - 1)
                
                print(f"🎙️ [MC] Đang điều phối...")
                mc_resp = self.moderator.moderate(
                    last_speaker=agent.name,
                    last_message=response,
                    next_speaker=next_name,
                    current_round=max_rounds + 1 if is_last_turn else round_count,
                    max_rounds=max_rounds
                )
                
                print(f"-> MC: {mc_resp}\n")
                self.debate_history.append(f"[MC]: {mc_resp}")

                if "KẾT THÚC" in mc_resp.upper():
                    should_continue = False
                    break
                
                time.sleep(5)
            
            round_count += 1
            
        print("\n=== KẾT THÚC ===")
        return self.debate_history