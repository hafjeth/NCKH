import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 1. Cấu hình Model 
    API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"

    # 2. Cấu hình sinh văn bản
    TEMPERATURE = 0.7       # 0.0: Chính xác, robot; 1.0: Sáng tạo, bay bổng
    MAX_TOKENS = 1000       # Độ dài tối đa câu trả lời

    # 3. Cấu hình RAG 
    CHUNK_SIZE = 1000       # Kích thước cắt nhỏ văn bản
    CHUNK_OVERLAP = 200     # Độ chồng lấn
    VECTOR_DB_PATH = "data/chroma_db" # Đường dẫn lưu DB

    # 4. Cấu hình hệ thống
    DEBUG_MODE = True       # In log ra màn hình để sửa lỗi