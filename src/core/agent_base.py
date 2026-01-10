import logging
import google.generativeai as genai
from src.core.config import Config

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, role: str, retriever=None):
        self.name = name
        self.role = role
        self.retriever = retriever
        
        if not Config.API_KEY:
            raise ValueError("API Key not configured in src.core.config")
        
        genai.configure(api_key=Config.API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, user_input: str) -> str:
        logger.info(f"[{self.name}] Processing input...")
        
        context_str = ""
        
        # RAG Logic
        if self.retriever:
            try:
                logger.info(f"[RAG] Agent {self.name} querying knowledge base...")
                # Calling retrieval.py's retrieve method
                results = self.retriever.retrieve(query=user_input, top_k=3)
                
                if results:
                    # Extract content from result dictionaries
                    docs_text = [r['content'] for r in results]
                    context_str = "\n".join([f"- {d}" for d in docs_text])
                    logger.info(f"[RAG] Found {len(results)} relevant documents.")
                else:
                    logger.warning("[RAG] No documents found.")
            except Exception as e:
                logger.error(f"[RAG Error] {e}")

        full_prompt = self._build_prompt(user_input, context_str)
        
        try:
            response = self.chat_session.send_message(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"[LLM Error] {e}")
            return "Xin lỗi, tôi đang gặp sự cố kết nối."

    def _build_prompt(self, user_input: str, context_str: str) -> str:
        prompt = f"Role: {self.role}\n"
        
        if context_str:
            prompt += f"\n--- RETRIEVED INFORMATION (MUST USE FOR EVIDENCE) ---\n{context_str}\n------------------------------------------------\n"
        
        prompt += f"\nUser Input: {user_input}\nAnswer:"
        return prompt

    def get_history(self):
        return [
            {"role": msg.role, "parts": [p.text for p in msg.parts]} 
            for msg in self.chat_session.history
        ]