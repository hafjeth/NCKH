from src.core.agent_base import BaseAgent

class ModeratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MC_DieuPhoi",
            role="Bạn là Điều phối viên (MC) của một buổi tọa đàm chính sách. Nhiệm vụ của bạn là tóm tắt ngắn gọn ý kiến của người vừa phát biểu và mời người tiếp theo. Giữ thái độ khách quan, chuyên nghiệp.",
            retriever=None
        )

    def moderate(self, last_speaker: str, last_message: str, next_speaker: str, current_round: int, max_rounds: int) -> str:
        if current_round > max_rounds:
            prompt = (
                f"Tình huống: Các bên đã hoàn thành {max_rounds} vòng tranh luận. "
                f"Người phát biểu cuối cùng: {last_speaker}.\n"
                f"Nội dung: '{last_message[:300]}...'\n\n"
                "NHIỆM VỤ:\n"
                "1. Tóm tắt ngắn gọn mâu thuẫn chính giữa các bên.\n"
                "2. Cảm ơn các bên đã tham gia.\n"
                "3. BẮT BUỘC kết thúc câu nói bằng cụm từ: 'KẾT THÚC TRANH LUẬN'."
            )
        else:
            prompt = (
                f"Tình huống: Đang ở Vòng {current_round}/{max_rounds}.\n"
                f"Người vừa phát biểu: {last_speaker}.\n"
                f"Nội dung tóm tắt: '{last_message[:300]}...'\n"
                f"Người tiếp theo sẽ phát biểu: {next_speaker}.\n\n"
                "NHIỆM VỤ:\n"
                "1. Tóm tắt cực ngắn (1 câu) luận điểm của người vừa nói.\n"
                "2. Mời người tiếp theo ({next_speaker}) đưa ra ý kiến phản hồi."
            )

        return self.chat(prompt)