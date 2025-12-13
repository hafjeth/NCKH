from src.core.agent_base import BaseAgent

# 1. Định nghĩa vai trò 
role_description = """
Bạn là một chuyên gia về Dệt may Việt Nam. 
Bạn chỉ trả lời các câu hỏi liên quan đến vải vóc, xuất khẩu và CBAM.
Nếu hỏi chuyện khác, hãy từ chối khéo.
"""

# 2. Khởi tạo Agent
bot = BaseAgent(name="TextileExpert", role=role_description)

# 3. Chat thử
print("--- TEST BASE AGENT (GEMINI) ---")
while True:
    user_in = input("\nBạn: ")
    if user_in.lower() == "exit":
        break
    
    reply = bot.chat(user_in)
    print(f"{bot.name}: {reply}")