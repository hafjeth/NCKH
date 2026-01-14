import logging
import time
import google.generativeai as genai
from google.api_core import exceptions
from src.core.config import Config

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, role: str, retriever=None):
        self.name = name
        self.role = role
        self.retriever = retriever
        
        if not Config.API_KEY:
            raise ValueError("API Key chưa được cấu hình trong src.core.config")
        
        genai.configure(api_key=Config.API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, user_input: str) -> str:
        logger.info(f"[{self.name}] Processing input...")
        
        context_str = ""
        
        # --- RAG Logic ---
        if self.retriever:
            try:
                # Gọi hàm retrieve
                results = self.retriever.retrieve(query=user_input, top_k=3)
                if results:
                    docs_text = [r['content'] for r in results]
                    context_str = "\n".join([f"- {d}" for d in docs_text])
                    logger.info(f"[RAG] ✅ Tìm thấy {len(results)} tài liệu dẫn chứng.")
                else:
                    logger.warning(f"[RAG] ⚠️ Không tìm thấy tài liệu nào khớp trong DB.")
            except Exception as e:
                # Vì yêu cầu nghiêm ngặt, nếu RAG lỗi trong lúc chạy, ta ghi log lỗi rõ ràng
                logger.error(f"[RAG Error] Lỗi khi truy xuất dữ liệu: {e}")

        full_prompt = self._build_prompt(user_input, context_str)
        
        # --- RETRY LOGIC CHO GEMINI API ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.chat_session.send_message(full_prompt)
                return response.text
            
            except Exception as e:
                # Kiểm tra nếu là lỗi 429 (Resource Exhausted)
                error_msg = str(e)
                wait_time = 60 # Mặc định chờ 60s
                
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    logger.warning(f"⚠️ Hết Quota (429). Đang chờ {wait_time}s để thử lại... (Lần {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Lỗi API khác: {e}")
                    # Nếu lỗi khác thì chờ ít hơn
                    time.sleep(10)
        
        return "Xin lỗi, hệ thống đang quá tải và không thể phản hồi sau nhiều lần thử."

    def _build_prompt(self, user_input: str, context_str: str) -> str:
        prompt = f"Role: {self.role}\n"
        if context_str:
            prompt += f"\n--- THÔNG TIN TRA CỨU TỪ TÀI LIỆU (BẮT BUỘC DẪN CHỨNG) ---\n{context_str}\n-----------------------------------\n"
        prompt += f"\nUser Input: {user_input}\nAnswer:"
        return prompt