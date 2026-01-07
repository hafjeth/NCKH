import google.generativeai as genai
from src.core.config import Config

# Cấu hình API Key
genai.configure(api_key=Config.API_KEY)

# --- ĐỊNH NGHĨA TOOL (HÀM GIẢ) ---
def dummy_search_tool(query: str):
    """
    Hàm tìm kiếm thông tin trong cơ sở dữ liệu RAG.
    Sử dụng hàm này khi cần tra cứu các thông tin cụ thể về luật, nghị định, hoặc số liệu dệt may.
    
    Args:
        query: Câu hỏi hoặc từ khóa cần tìm kiếm.
    """
    print(f"\n[SYSTEM LOG] >> Agent đang gọi Tool 'dummy_search_tool' với query: '{query}'")
    
    # Đây là dữ liệu giả định (Mock Data) để test
    return (
        f"KẾT QUẢ TÌM KIẾM CHO: '{query}'\n"
        f"Nguồn: Cơ sở dữ liệu nội bộ.\n"
        f"Nội dung: Theo Nghị định 06/2022/NĐ-CP, doanh nghiệp dệt may bắt buộc phải kiểm kê khí nhà kính "
        f"nếu nằm trong danh mục quy định. Mức phạt vi phạm có thể lên đến 100 triệu đồng. "
        f"Châu Âu (EU) sẽ áp dụng CBAM đầy đủ từ năm 2026."
    )

class BaseAgent:
    def __init__(self, name: str, role: str, model_name: str = Config.MODEL_NAME):
        self.name = name
        self.role = role
        self.model_name = model_name
        
        # --- CẤU HÌNH TOOL ---
        # Đưa hàm Python vào danh sách tools
        # Sau này M2 làm xong hàm thật thì mình import vào đây thay thế
        self.tools = [dummy_search_tool]

        # Khởi tạo model có gắn Tool
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.role,
            tools=self.tools, # <-- Gắn tool vào đây
            generation_config=genai.types.GenerationConfig(
                temperature=Config.TEMPERATURE,
                max_output_tokens=Config.MAX_TOKENS
            )
        )
        
        # Khởi tạo chat session với chế độ TỰ ĐỘNG GỌI HÀM
        self.chat_session = self.model.start_chat(
            history=[],
            enable_automatic_function_calling=True 
        )

    def chat(self, user_input: str) -> str:
        """Gửi tin nhắn và nhận phản hồi."""
        try:
            print(f"[{self.name}] đang suy nghĩ...") 
            response = self.chat_session.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Lỗi kết nối Gemini: {str(e)}"

    def get_history(self):
        """Lấy lại lịch sử chat"""
        formatted_history = []
        for msg in self.chat_session.history:
            role = "user" if msg.role == "user" else "assistant"
            # Xử lý trường hợp tin nhắn là Function Call (không có text)
            parts = msg.parts[0]
            content = "Function Call/Response"
            if hasattr(parts, "text"):
                content = parts.text
            
            formatted_history.append({"role": role, "content": content})
        return formatted_history