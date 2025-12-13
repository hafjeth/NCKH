import google.generativeai as genai
from src.core.config import Config 

genai.configure(api_key=Config.API_KEY)

class BaseAgent:
    def __init__(self, name: str, role: str, model_name: str = Config.MODEL_NAME):
        self.name = name
        self.role = role
        self.model_name = model_name
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.role,
            generation_config=genai.types.GenerationConfig(
                temperature=Config.TEMPERATURE,
                max_output_tokens=Config.MAX_TOKENS
            )
        )
        
        self.chat_session = self.model.start_chat(history=[])

    def chat(self, user_input: str) -> str:
        try:
            print(f"[{self.name}] đang suy nghĩ...") 
            response = self.chat_session.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Lỗi kết nối Gemini: {str(e)}"

    def get_history(self):
        formatted_history = []
        for msg in self.chat_session.history:
            role = "user" if msg.role == "user" else "assistant"
            content = msg.parts[0].text 
            formatted_history.append({"role": role, "content": content})
        return formatted_history