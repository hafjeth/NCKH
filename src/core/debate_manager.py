from typing import List
from src.core.agent_base import BaseAgent

class DebateManager:
    def __init__(self, agents: List[BaseAgent]):
        """
        Quản lý vòng tranh luận.
        """
        self.agents = agents
        self.debate_history: List[str] = [] 

    def construct_prompt(self, current_agent_name: str, current_agent_role: str, topic: str) -> str:
        """
        [UPDATE] Thêm tham số current_agent_role để nhắc lại vai trò.
        Tạo Prompt ép buộc Agent phải tư duy phản biện.
        """
        # Nếu chưa ai nói gì (lượt đầu tiên)
        if not self.debate_history:
            return (
                f"CHỦ ĐỀ TRANH LUẬN: {topic}\n"
                f"NHIỆM VỤ: Là người mở đầu, hãy trình bày quan điểm cốt lõi của bạn dựa trên vai trò được giao.\n"
                f"Hãy đưa ra ít nhất 2 luận điểm chính kèm dẫn chứng (nếu có)."
            )
        
        # Gộp lịch sử
        history_str = "\n".join(self.debate_history)
        
        # [UPDATE] Prompt này được thiết kế theo kỹ thuật "Chain-of-Thought" (CoT)
        # để bắt Agent suy nghĩ trước khi trả lời, tránh việc đồng ý bừa bãi.
        prompt = (
            f"CHỦ ĐỀ GỐC: {topic}\n\n"
            f"--- DIỄN BIẾN TRANH LUẬN ---\n"
            f"{history_str}\n"
            f"-----------------------------\n\n"
            f"Đến lượt bạn: {current_agent_name}\n"
            f"Vai trò của bạn: {current_agent_role}\n\n"
            f"NHIỆM VỤ CỦA BẠN (Tư duy phản biện):\n"
            f"1. TÓM TẮT: Người trước vừa nói gì?\n"
            f"2. PHẢN BIỆN: Tìm ra điểm yếu/bất hợp lý trong đó. Tấn công vào lập luận của họ.\n"
            f"3. BẢO VỆ: Đưa ra dẫn chứng để bảo vệ lợi ích phe mình.\n\n"
            f"QUY TẮC CẤM (Bắt buộc tuân thủ):\n"
            f"- KHÔNG được bắt đầu bằng: 'Kính thưa', 'Cảm ơn', 'Tôi hoàn toàn đồng ý', 'Rất hoan nghênh'.\n"
            f"- KHÔNG được khen đối thủ.\n"
            f"- Hãy dùng giọng văn tranh luận trực diện, đi thẳng vào vấn đề tiền bạc và lợi ích."
        )
        return prompt

    def run_round(self, topic: str, max_rounds: int = 1):
        """
        Chạy vòng lặp tranh luận.
        """
        print(f"\n=== BẮT ĐẦU TRANH LUẬN: {topic} ===\n")
        
        for round_num in range(1, max_rounds + 1):
            print(f"--- VÒNG {round_num} ---")
            
            for agent in self.agents:
                # [UPDATE] Truyền thêm agent.role vào hàm construct_prompt
                # Để đảm bảo Agent không bị "quên vai" sau khi đọc lịch sử dài
                prompt = self.construct_prompt(agent.name, agent.role, topic)
                
                # Gọi Agent
                response = agent.chat(prompt)
                
                formatted_response = f"[{agent.name}]: {response}"
                self.debate_history.append(formatted_response)
                
                print(f"-> {agent.name} đã trả lời.") 
                print(f"{formatted_response}\n")
                
        print("\n=== KẾT THÚC TRANH LUẬN ===")
        return self.debate_history