import os
import re
import unicodedata
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RAW_DIR = "data/raw_pdfs"
OUT_DIR = "data/processed_text"


# ==========================
# 1) ĐỌC TEXT BẰNG PYMUPDF
# ==========================
def extract_text_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text
    except Exception:
        return ""


# ==========================
# 2) OCR BẰNG PYMUPDF
# ==========================
def ocr_with_pymupdf(pdf_path):
    """Render từng trang PDF thành ảnh rồi OCR."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print("❌ Không mở được PDF:", e)
        return ""

    text = ""
    for i, page in enumerate(doc):
        print(f"OCR trang {i+1}...")
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text += pytesseract.image_to_string(img, lang="vie") + "\n"

    doc.close()
    return text


# ==========================
# 3) LÀM SẠCH TEXT - HOÀN HẢO
# ==========================
def clean_text(raw_text):
    """Làm sạch text triệt để - không còn thông tin thừa."""
    text = unicodedata.normalize("NFC", raw_text)

    # ========== BƯỚC 1: XÓA METADATA JOURNAL/PAPER ==========
    # Xóa toàn bộ thông tin journal, ISSN, DOI
    patterns_journal = [
        r"RESEARCH\s+(?:REVIEW|PAPER).*?\n",
        r"(?:International\s+)?Journal\s+of[^\n]+",
        r"e-ISSN:?[\s\d\-\|]+(?:Vol|No|March|pp)[\s\d\-\|\.]+",
        r"ISSN:?[\s\d\-]+",
        r"DOI:?[\s\d\./]+",
        r"Double[- ]Blind\s+Peer\s+Reviewed[^\n]*",
        r"https?://[^\s]+",
        r"Original\s+Article\s*\d*",
        r"Article\s+Publication[^\n]*",
    ]
    for p in patterns_journal:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ========== BƯỚC 2: XÓA NGÀY THÁNG VÀ PUBLISHED INFO ==========
    patterns_dates = [
        r"Received\s+\d+[a-z]*\s+[A-Za-z]+\s+\d{4}[^\n]*",
        r"Revised\s+\d+[a-z]*\s+[A-Za-z]+\s+\d{4}[^\n]*",
        r"Accepted\s+\d+[a-z]*\s+[A-Za-z]+\s+\d{4}[^\n]*",
        r"Published\s+(?:Online:?\s*)?\d+[^\n]*",
        r"©\s*\d{4}[^\n]+(?:Author|Publisher)[^\n]*",
        r".{0,30}?,?\s*ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}[^\n]*",
    ]
    for p in patterns_dates:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ========== BƯỚC 3: XÓA HEADER/FOOTER VĂN BẢN CHÍNH THỨC ==========
    patterns_official = [
        r"CÔNG\s+BÁO[^\n]*",
        r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM[^\n]*",
        r"Độc\s+lập\s+-\s+Tự\s+do\s+-\s+Hạnh\s+phúc[^\n]*",
        r"Số:\s*\d+[^\n]*",
        r"V\/v:[^\n]*",
        r"(?:Nơi nhận|THỦ TƯỚNG|BỘ TRƯỞNG|CHỦ TỊCH|GIÁM ĐỐC|TỔNG GIÁM ĐỐC)[^\n]*",
    ]
    for p in patterns_official:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ========== BƯỚC 4: XÓA SỐ TRANG ==========
    patterns_page = [
        r"(?:Trang|Page)\s+\d+\s*[/of]+\s*\d+",
        r"pp\.\s*\d+[-–]\d+",
        r"^\s*\d+\s*$",  # Dòng chỉ có số
        r"^-\s*\d+\s*-$",
    ]
    for p in patterns_page:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ========== BƯỚC 5: XÓA ABSTRACT, KEYWORDS ==========
    text = re.sub(r"^Abstract:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^Keywords?:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)

    # ========== BƯỚC 6: XÓA VĂN BẢN PHÁP LUẬT LẶP LẠI ==========
    patterns_law = [
        r"Căn\s+cứ\s+Luật[^;.]+[;.]",
        r"Căn\s+cứ\s+Nghị\s+định[^;.]+[;.]",
        r"Căn\s+cứ\s+Quyết\s+định[^;.]+[;.]",
        r"Căn\s+cứ\s+Thông\s+tư[^;.]+[;.]",
        r"Căn\s+cứ\s+Nghị\s+quyết[^;.]+[;.]",
        r"Theo\s+đề\s+nghị\s+của[^;.]+[;.]",
    ]
    for p in patterns_law:
        matches = re.findall(p, text, flags=re.IGNORECASE)
        if len(matches) > 1:
            text = re.sub(p, "", text, flags=re.IGNORECASE)

    # ========== BƯỚC 7: XÓA BẢNG BIỂU VỠ ==========
    text = re.sub(r"\.{5,}", " ", text)
    text = re.sub(r"-{4,}", " ", text)
    text = re.sub(r"_{4,}", " ", text)
    text = re.sub(r"={4,}", " ", text)
    text = re.sub(r"—+", " ", text)
    text = re.sub(r"\s{3,}", " ", text)

    # ========== BƯỚC 8: XỬ LÝ TỪNG DÒNG - LỌC THÔNG MINH ==========
    lines = text.split("\n")
    cleaned_lines = []
    prev_line = ""

    for line in lines:
        line = line.strip()

        # Bỏ dòng trống
        if not line:
            continue

        # Bỏ dòng quá ngắn (< 3 ký tự)
        if len(line) < 3:
            continue

        # Bỏ dòng chỉ có số và ký tự đặc biệt
        if re.match(r"^[\d\s,\.\-\|]+$", line):
            continue

        # Bỏ dòng trùng lặp liên tiếp
        if line == prev_line:
            continue

        # Bỏ dòng có email (cả dạng @ và [at])
        if re.search(r"[a-zA-Z0-9._%+-]+[@\[at\]][a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", line, re.IGNORECASE):
            continue

        # Bỏ dòng tên tác giả (có *, số hoặc & ở trong)
        # VD: "*1Thi Ngoc Do, 2Thi Ngoc Huyen Nguyen, 3Thu Huong Tran & 4Thi Thuy Tien Tong"
        if re.search(r"[\*\d]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+[\s\d,&]+", line):
            continue

        # Bỏ dòng có "School of" hoặc tên tổ chức
        if re.search(r"(?:School|University|Institute|Department|College|Faculty|Center|Ministry)\s+of", line, re.IGNORECASE):
            continue

        # Bỏ dòng có địa chỉ cụ thể: số nhà + tên đường
        if re.search(r"\d{1,4}\s+[A-Z][a-z]+.*?(?:Street|Road|Phong|District|Ward|Avenue|Boulevard)", line, re.IGNORECASE):
            continue

        # Bỏ dòng kết thúc bằng địa danh Việt Nam
        if re.search(r",\s*(?:Vietnam|Viet\s+Nam|Hanoi|Ha\s+Noi|Ho\s+Chi\s+Minh|HCMC|Saigon)\s*$", line, re.IGNORECASE):
            continue

        # Bỏ dòng có "Correspondence" (thông tin liên hệ tác giả)
        if re.search(r"(?:Author'?s?\s+)?Correspondence", line, re.IGNORECASE):
            continue

        # Bỏ dòng có thông tin license
        if re.search(r"(?:CC\s+BY|open\s+access|license)", line, re.IGNORECASE):
            continue

        # Bỏ dòng chỉ chứa chữ in hoa ngắn (< 20 ký tự)
        if line.isupper() and len(line) < 20:
            continue

        cleaned_lines.append(line)
        prev_line = line

    return "\n".join(cleaned_lines)


# ==========================
# 4) CHUẨN HÓA TÊN FILE - CHÍNH XÁC 100%
# ==========================
def normalize_filename(filename):
    """
    Bỏ dấu tiếng Việt CHÍNH XÁC, không khoảng trắng.
    VD: 'Nghị định 06.pdf' -> 'NghiDinh06.txt'
    """
    # Bỏ đuôi .pdf
    name = filename.replace(".pdf", "").replace(".PDF", "")

    # Bảng chuyển đổi tiếng Việt CHÍNH XÁC
    vietnamese_map = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd',
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        'Đ': 'D',
    }

    # Chuyển từng ký tự
    result = []
    for char in name:
        if char in vietnamese_map:
            result.append(vietnamese_map[char])
        else:
            result.append(char)
    name = ''.join(result)

    # Xóa ký tự đặc biệt, giữ chữ cái và số
    name = re.sub(r'[^\w\s-]', '', name)

    # Thay khoảng trắng bằng underscore
    name = re.sub(r'\s+', '_', name.strip())

    # Xóa underscore thừa
    name = re.sub(r'_+', '_', name).strip('_')

    # Viết hoa chữ cái đầu mỗi từ (PascalCase)
    parts = name.split('_')
    name = ''.join(p.capitalize() for p in parts if p)

    return name + ".txt"


# ==========================
# 5) XỬ LÝ TOÀN BỘ PDF
# ==========================
def process_all_pdfs():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for filename in os.listdir(RAW_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, filename)
            print(f"\n==============================")
            print(f"Đang xử lý: {filename}")

            # Bước 1: thử đọc text
            text = extract_text_pymupdf(pdf_path)

            if len(text.strip()) > 50:
                print("→ PDF dạng text, không cần OCR.")
            else:
                print("→ PDF dạng ảnh hoặc PDF lỗi → dùng OCR PyMuPDF...")
                text = ocr_with_pymupdf(pdf_path)

                if len(text.strip()) == 0:
                    print(f"❌ Bỏ qua file vì không đọc được: {filename}")
                    continue

            # Làm sạch
            clean_txt = clean_text(text)

            # Chuẩn hóa tên file
            out_name = normalize_filename(filename)
            out_path = os.path.join(OUT_DIR, out_name)

            # Lưu file
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(clean_txt)

            print(f"→ Đã lưu: {out_name}")


# ==========================
# 6) MAIN
# ==========================
if __name__ == "__main__":
    print("Pipeline bắt đầu chạy...")
    process_all_pdfs()