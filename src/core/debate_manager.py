import logging
import os
from typing import List
from src.core.agent_base import BaseAgent

try:
    from src.knowledge.retrieval import RetrievalSystem
    RAG_AVAILABLE = True
except ImportError:
    print("[WARNING] Could not import RetrievalSystem. Running in NO-RAG mode.")
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)

class DebateManager:
    def __init__(self):
        self.debate_history: List[str] = []
        self.agents: List[BaseAgent] = []
        self.retriever = None

        if RAG_AVAILABLE:
            try:
                logger.info("Initializing RAG System...")
                
                # Calculate absolute path for database
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                db_path = os.path.join(project_root, "data", "chroma_db")
                
                self.retriever = RetrievalSystem(
                    chroma_db_dir=db_path,
                    collection_name="knowledge_base",
                    top_k=3
                )
                logger.info("RAG System ready.")
            except Exception as e:
                logger.error(f"Error initializing RAG: {e}")

    def setup_agents(self):
        # Agent 1: Expert (Uses RAG)
        agent_expert = BaseAgent(
            name="ChuyenGia_PhapLy",
            role="Bạn là chuyên gia pháp lý môi trường. Hãy tranh luận dựa trên văn bản luật, nghị định 06, 45 và dữ liệu CBAM. Luôn trích dẫn cụ thể.",
            retriever=self.retriever
        )

        # Agent 2: Business (No RAG)
        agent_business = BaseAgent(
            name="DoanhNghiep_DetMay",
            role="Bạn là chủ doanh nghiệp dệt may. Bạn lo lắng về chi phí kiểm kê và thuế suất cao. Hãy phản biện gay gắt về tính khả thi.",
            retriever=None
        )

        self.agents = [agent_expert, agent_business]
        logger.info(f"Agents setup complete. Total: {len(self.agents)}")

    def construct_prompt(self, current_agent_name: str, current_agent_role: str, topic: str) -> str:
        if not self.debate_history:
            return (
                f"CHỦ ĐỀ TRANH LUẬN: {topic}\n"
                f"NHIỆM VỤ: Bạn là người mở đầu. Hãy trình bày quan điểm cốt lõi dựa trên vai trò: {current_agent_role}.\n"
                f"YÊU CẦU: Đưa ra ít nhất 2 luận điểm chính kèm số liệu hoặc dẫn chứng."
            )
        
        history_str = "\n".join(self.debate_history)
        
        prompt = (
            f"CHỦ ĐỀ GỐC: {topic}\n\n"
            f"--- DIỄN BIẾN TRANH LUẬN ---\n"
            f"{history_str}\n"
            f"-----------------------------\n\n"
            f"Đến lượt bạn: {current_agent_name}\n"
            f"Vai trò: {current_agent_role}\n\n"
            f"NHIỆM VỤ PHẢN BIỆN:\n"
            f"1. TÓM TẮT: Đối phương vừa nói gì?\n"
            f"2. PHẢN BÁC: Tìm điểm yếu trong lập luận đó (chi phí, tính pháp lý, thực tế).\n"
            f"3. BẢO VỆ: Đưa ra luận điểm của phe mình.\n\n"
            f"QUY TẮC:\n"
            f"- KHÔNG dùng từ sáo rỗng.\n"
            f"- Tranh luận trực diện, gay gắt nhưng logic."
        )
        return prompt

    def run_round(self, topic: str, max_rounds: int = 1):
        print(f"\n=== BẮT ĐẦU TRANH LUẬN: {topic} ===\n")
        
        if not self.agents:
            self.setup_agents()

        for round_num in range(1, max_rounds + 1):
            print(f"--- VÒNG {round_num} ---")
            
            for agent in self.agents:
                prompt = self.construct_prompt(agent.name, agent.role, topic)
                response = agent.chat(prompt)
                
                formatted_response = f"[{agent.name}]: {response}"
                self.debate_history.append(formatted_response)
                
                print(f"-> {agent.name} đã trả lời.") 
                print(f"{formatted_response}\n")
                
        print("\n=== KẾT THÚC TRANH LUẬN ===")
        return self.debate_history