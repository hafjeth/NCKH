"""
business_chunker.py — FIXED v8
==============================
Sửa so với bản FIXED v7:
  [FIX-17] clean_text(): bổ sung strip dòng "© Shutterstock/..." và các
           dạng © không kèm năm (© 20xx đã có, nhưng "© Shutterstock" thì
           không có năm). Cũng strip dòng chỉ là số trang dạng "2 3" hoặc
           "6 7 6 7" mà PDF extract lẫn vào giữa nội dung.
  [FIX-18] is_toc_chunk(): bổ sung pattern TOC tiếng Việt ngắn dạng
           "Tóm tắt 02", "Giới thiệu 08" — từ khóa mục lục + số trang
           1-2 chữ số, không cần prefix dài. Hạ ngưỡng xuống 30% vì TOC
           tiếng Việt thường có nhiều dòng ngắn hơn.
  [FIX-19] is_abbreviation_chunk(): filter mới phát hiện bảng chữ viết
           tắt (Abbreviation/Acronym list) — nhiều dòng dạng "ABBR Mô tả
           đầy đủ" với tỷ lệ cao. Không có giá trị RAG vì chỉ định nghĩa
           từ viết tắt, không chứa luận điểm hay dữ liệu.
"""

import json
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ================= CONFIG =================
NORMALIZED_DIR   = Path("data/processed/normalized")
OUTPUT_DIR       = Path("data/processed/chunks/business_paragraphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE       = 800   # chars per chunk (target)
CHUNK_OVERLAP    = 150   # overlap tối đa khi tìm ranh giới câu
MIN_CHUNK_LENGTH = 150   # bỏ qua chunks quá ngắn


# ================= CLEAN =================

def clean_text(text: str) -> str:
    """
    Loại bỏ các artifact phổ biến từ PDF extract trước khi chunk.

    1. Header journal: "L. T. N. Anh et al. / VNU Journal..., Vol. 41, No. 1 (2025) 55-67"
    2. Cặp số trang đứng riêng trên dòng: "56 57"
    3. Số trang đơn lẻ đứng riêng trên dòng
    4. [FIX-7] Số trang dạng "text 56" ở cuối dòng nội dung (PDF 2-cột)
    5. [FIX-8] Dòng boilerplate inline: "© 20xx Company", tên sản phẩm dạng
       header/footer lặp lại (vd: "© 2024 Deloitte The Netherlands CBAM Compliance Manager 2")
    6. Nhiều dòng trống liên tiếp → giữ tối đa 1 dòng trống
    7. Trim khoảng trắng cuối mỗi dòng
    """
    # 1. Xóa header journal: "Xxx et al. / ..." đến hết dòng
    text = re.sub(
        r'[A-Z][a-zA-ZÀ-ỹ\.\s,]+et al\.\s*/[^\n]+\n?',
        ' ',
        text
    )

    # 2. Xóa cặp số trang dạng "56 57" đứng riêng trên dòng
    text = re.sub(r'\n\s*\d{1,3}\s+\d{1,3}\s*\n', '\n', text)

    # 3. Xóa số trang đơn lẻ đứng riêng trên dòng (không có ký tự chữ xung quanh)
    text = re.sub(r'(?<!\w)\n\s*\d{1,3}\s*\n(?!\w)', '\n', text)

    # 4. [FIX-7] Xóa số trang bị PDF extract dính vào cuối dòng nội dung
    text = re.sub(r'(?<=[a-zA-ZÀ-ỹ,])\s{2,}\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # 5. [FIX-8 + FIX-17] Xóa dòng boilerplate inline lẫn vào nội dung:
    #    a) Dòng bắt đầu bằng "© ..." — bất kỳ dạng © nào (có hoặc không có năm)
    #       vd: "© 2024 Deloitte", "© Shutterstock / ANAID Studio"
    text = re.sub(r'^©[^\n]*\n?', '', text, flags=re.MULTILINE)
    #    b) Dòng toàn chữ hoa dạng "C A R B O N B O R D E R ..." (spaced acronym header)
    text = re.sub(r'^(?:[A-Z]\s){4,}[A-Z][^\n]*\n?', '', text, flags=re.MULTILINE)
    #    c) Dòng chứa tên contact: "Partner/Manager/Consultant" + email @domain
    text = re.sub(
        r'^[^\n]{0,80}(?:Partner|Manager|Consultant|Director)[^\n]{0,80}@\S+\n?',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    #    d) [FIX-17] Dòng chỉ gồm số trang lặp lại dạng "2 3 2 3" hoặc "6 7 6 7"
    #       — PDF extract dính số trang đầu/cuối trang vào giữa nội dung
    text = re.sub(r'^\s*(?:\d{1,3}\s+){2,}\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    # 6. [FIX-14] Xóa sidebar/widget text từ web scrape
    #    a) CTA navigation thường thấy trên trang web báo/tạp chí
    text = re.sub(
        r'^(?:FIND BUSINESS SUPPORT|Subscribe(?:d)?|Read more|Newsletter|'
        r'Sign up|Related articles?|More articles?|You may also like|'
        r'Tags?:|Share this|Follow us|Advertisement|Sponsored)[^\n]*\n?',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    #    b) Dòng tiếp ngay sau "FIND BUSINESS SUPPORT" thường là mô tả CTA ngắn —
    #       chỉ strip nếu dòng NGẮN (< 80 chars) và không có dấu chấm kết thúc câu
    #       để tránh xóa câu nội dung hợp lệ bắt đầu bằng những từ tương tự.
    text = re.sub(
        r'^(?:Identify|Discover|Learn|Navigate|Optimize|Find|Access|Explore)'
        r' (?:the |your |our |an )?(?:Optimal|Best|Right|Key|Business|'
        r'Investment|Strategy|Support|Resources?)[^\n.!?]{0,60}\n',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    #    c) Dòng chứa brand/CTA dạng "How [Company] can help..."
    text = re.sub(
        r'^[^\n]{0,60}(?:Dezan Shira|Asia Briefing|Vietnam Briefing)[^\n]{0,80}'
        r'(?:help|assist|support|advise|provide)[^\n]*\n?',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    #    d) Dòng rác UI: tỷ lệ ký tự lạ (không phải chữ/số/khoảng trắng) > 40%
    #       ví dụ: "e eye FDI", "k4 lj e", "oste Vietnam Briefing tte"
    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if len(stripped) > 0:
            non_alnum_space = sum(
                1 for c in stripped
                if not (c.isalnum() or c.isspace() or c in '.,;:!?-\'\"()[]{}/@#%&*+=<>_\\|~`^')
            )
            junk_ratio = non_alnum_space / len(stripped)
            # Dòng ngắn + ký tự lạ nhiều → rác UI
            if len(stripped) <= 30 and junk_ratio > 0.25:
                continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # 7. Xóa nhiều dòng trống liên tiếp → giữ tối đa 1 dòng trống
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 8. Trim khoảng trắng cuối mỗi dòng
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ================= FILTERS =================

def is_heading(text: str) -> bool:
    """
    Heuristic: coi là heading nếu toàn chữ hoa hoặc rất ngắn không có dấu chấm.
    Chỉ skip chunk CỰC NGẮN là heading, không skip chunk dài.
    """
    text = text.strip()
    if len(text) > 200:
        return False
    if len(text) < 60 and "." not in text:
        return True
    if text.isupper() and len(text) < 100:
        return True
    return False


def is_toc_chunk(text: str) -> bool:
    """
    [FIX-4 + FIX-18] Phát hiện chunk là mục lục (Table of Contents).

    Bắt 4 dạng TOC:
    - Dạng cũ (v3): dòng kết thúc bằng số trang 2-3 chữ số "2. Giới thiệu 08"
    - Dạng dotdot (v4): dấu chấm lửng + số trang "1.0 Introduction...1"
    - [FIX-18a] Dạng tiếng Việt ngắn: từ khóa + số trang "Tóm tắt 02", "Kết luận 82"
    - [FIX-18b] Dạng inline: TOC bị extract thành 1 dòng dài không có newline,
      nhiều cặp "chữ + số_trang" liên tiếp "Tóm tắt 02 1.1. Bối cảnh 02 ..."

    Nếu > 30% dòng có pattern OR là TOC inline → skip.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False

    # [FIX-18b] TOC inline (1 dòng dài): đếm cặp "chữ + số trang" trên tổng token
    if len(lines) <= 3:  # ít dòng → có thể là TOC bị collapse thành 1 dòng
        full_text = text.strip()
        inline_toc_pat = re.compile(r'[^\d]\s\d{2,3}(?:\s|$)')
        inline_matches = len(inline_toc_pat.findall(full_text))
        tokens = full_text.split()
        if len(tokens) > 5 and inline_matches / max(len(tokens), 1) > 0.12:
            return True

    # Pattern cũ: kết thúc bằng số trang 2-3 chữ số (có prefix đủ dài)
    toc_pattern = re.compile(r'\w[\w\s,\(\)]{2,50}\s+\d{2,3}\s*$')
    # Dấu chấm lửng + số trang
    dotdot_pattern = re.compile(r'\.{3,}\s*\d{1,3}\s*$')
    # [FIX-18a] TOC tiếng Việt ngắn: cụm chữ ngắn + số trang 1-3 chữ số ở cuối
    viet_toc = re.compile(r'^[\w\sÀ-ỹ\.()]{2,50}\s+\d{1,3}\s*$')

    toc_count = sum(
        1 for line in lines
        if toc_pattern.search(line)
        or dotdot_pattern.search(line)
        or (len(line) <= 60 and viet_toc.match(line))
    )
    return toc_count / len(lines) > 0.30


def is_references_chunk(text: str) -> bool:
    """
    [FIX-12] Phát hiện chunk là danh sách tài liệu tham khảo.

    Bắt 2 dạng đánh số:
    - Dạng cũ: "[1] Author, Title..."  (ngoặc vuông)
    - Dạng mới: "1. Author, Title..."  (số + dấu chấm, phổ biến ở bài hội nghị)

    Ngoài ra còn tính URL và DOI.
    Nếu tổng tín hiệu / số dòng > 30% → skip.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False

    bracket_ref  = re.compile(r'^\[\d+\]\s+[A-Z]')                   # [1] Author
    numbered_ref = re.compile(r'^\d{1,3}\.\s+[A-Z][a-zA-ZÀ-ỹ\s,]+') # 1. Author
    url_pattern  = re.compile(r'https?://')
    doi_pattern  = re.compile(r'doi\.org|DOI:')

    bracket_count  = sum(1 for l in lines if bracket_ref.match(l))
    numbered_count = sum(1 for l in lines if numbered_ref.match(l))
    url_count      = sum(1 for l in lines if url_pattern.search(l))
    doi_count      = sum(1 for l in lines if doi_pattern.search(l))

    ratio = (bracket_count + numbered_count + url_count + doi_count) / len(lines)
    return ratio > 0.30


def is_metadata_chunk(text: str) -> bool:
    """
    Phát hiện chunk là metadata bài báo (tên tác giả, địa chỉ, email, ngày nhận).
    Thường xuất hiện ở đầu file bài báo khoa học.
    Nếu có từ 3 dấu hiệu metadata trở lên trong chunk ngắn → skip.
    """
    text_lower = text.lower()
    metadata_signals = [
        'received', 'revised', 'accepted', 'corresponding author',
        'e-mail address', 'email address', '@',
        'nhận ngày', 'chỉnh sửa ngày', 'chấp nhận',
        'tác giả liên hệ', 'địa chỉ email',
    ]
    signal_count = sum(1 for s in metadata_signals if s in text_lower)
    return signal_count >= 3 and len(text) < 600


def is_boilerplate_chunk(text: str) -> bool:
    """
    [FIX-5 + FIX-9] Phát hiện chunk là boilerplate: bản quyền, địa chỉ tổ chức,
    thông tin xuất bản, hoặc footer liên hệ (contact info).

    FIX-9: bổ sung signal contact info dạng Deloitte/KPMG two-pager:
    - Dòng có chức danh (Partner, Senior Manager...) kèm email @domain
    - Nhiều email @domain trong chunk ngắn

    Nếu có từ 2 tín hiệu trở lên → skip.
    """
    text_lower = text.lower()
    boilerplate_signals = [
        # Bản quyền / giấy phép
        'creative commons', 'all rights reserved', 'licensed under',
        'copyright ©', '© 20',
        # Thông tin xuất bản
        'published by', 'first published', 'printed in',
        # Địa chỉ / liên hệ tổ chức quốc tế
        'suite ', 'tel:', 'website:', 'x-twitter:',
        'head office', 'winnipeg', 'geneva', 'ottawa', 'washington, d.c',
        'iisd.org', 'ifc.org', 'worldbank.org',
        # Thông tin tổ chức
        'registered in', 'incorporated in', 'charity number',
        # [FIX-16] "About Us" boilerplate từ trang web báo/tư vấn
        'about us', 'one of five regional', 'maintains offices',
        'subscribe to', 'our newsletter', 'sign up for',
        'vietnam briefing is', 'asia briefing is', 'dezan shira is',
        'this article was', 'this content is provided for', 'visit our website',
    ]
    signal_count = sum(1 for s in boilerplate_signals if s in text_lower)

    # [FIX-9] Contact info: đếm số email @domain trong chunk
    email_count = len(re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text))
    if email_count >= 2:
        signal_count += 2  # 2+ email trong 1 chunk → rất có thể là footer liên hệ

    # [FIX-9] Chức danh kèm email — dấu hiệu contact block
    contact_title = re.compile(
        r'(?:Partner|Senior Manager|Manager|Consultant|Director|Advisor)\b',
        re.IGNORECASE,
    )
    if contact_title.search(text) and email_count >= 1:
        signal_count += 2

    return signal_count >= 2


def is_chart_data_chunk(text: str) -> bool:
    """
    [FIX-6] Phát hiện chunk là dữ liệu thô của biểu đồ hoặc bảng số liệu.
    Thường là số liệu trục tọa độ, legend, hoặc bảng với nhiều số liên tiếp.

    Dấu hiệu: > 35% token trong chunk là số (kể cả số âm, phần trăm,
    số có dấu phẩy, giá trị trong ngoặc vuông như [-1,7;-0,0]).

    Nếu vượt ngưỡng → skip vì không có giá trị ngữ nghĩa cho RAG.
    """
    tokens = text.split()
    if len(tokens) < 5:
        return False

    number_pattern = re.compile(r'^-?\d[\d\.,;\[\]%\-]*$')
    number_count = sum(1 for t in tokens if number_pattern.match(t))

    return number_count / len(tokens) > 0.35


def is_inline_table_chunk(text: str) -> bool:
    """
    [FIX-13] Phát hiện chunk là bảng dữ liệu inline — dạng phổ biến trong
    báo cáo khi bảng được extract thành text phẳng:

    Ví dụ (chunk 023 bioconf):
      "Metric Vietnam Indonesia Export carbon intensity 0.7 kg CO₂ / US$ ..."
      "MRV/ETS status Pilot ETS ... No mandatory ETS yet ..."

    Dấu hiệu nhận biết:
    1. Chunk bắt đầu bằng từ khóa header bảng phổ biến
    2. Dòng chứa 2+ cụm (chữ + số/đơn vị) xen kẽ nhau — cấu trúc "Label Val1 Val2"
    3. Mật độ cao các đơn vị đo lường (%, USD, tCO₂, GW, billion, million...)

    Nếu có từ 2 tín hiệu trở lên → skip.
    """
    text_stripped = text.strip()
    signal_count = 0

    # 1. Bắt đầu bằng header bảng phổ biến
    table_headers = re.compile(
        r'^(?:Metric|Indicator|Category|Country|Parameter|Variable|Item|'
        r'Criteria|Factor|Aspect|Measure|Component|Sector|Source)\b',
        re.IGNORECASE,
    )
    if table_headers.match(text_stripped):
        signal_count += 1

    # 2. Nhiều đơn vị đo lường trong chunk ngắn
    unit_pattern = re.compile(
        r'\b(?:kg|tCO[₂2]|CO[₂2]|GW|MW|kW|USD|US\$|€|billion|million|'
        r'trillion|percent|tonne|ton|Mt|Gt|MtCO[₂2]|kgCO[₂2]|'
        r'est\.|approx\.)\b',
        re.IGNORECASE,
    )
    unit_count = len(unit_pattern.findall(text))
    tokens = text.split()
    if len(tokens) > 0 and unit_count / max(len(tokens), 1) > 0.05:
        signal_count += 1

    # 3. Nhiều dòng ngắn (< 80 chars) xen kẽ nhau — cấu trúc bảng nhiều cột
    lines = [l.strip() for l in text_stripped.split('\n') if l.strip()]
    if len(lines) >= 3:
        short_lines = sum(1 for l in lines if len(l) < 80)
        if short_lines / len(lines) > 0.6:
            signal_count += 1

    # 4. Chunk chứa pattern "Label Value1 Value2" — 2 giá trị số/đơn vị cạnh nhau
    col_pattern = re.compile(
        r'[A-Za-zÀ-ỹ\s]{3,30}\s+'         # Label (chữ)
        r'[\d\.,%\$€]+\s*(?:kg|GW|MW|USD|US\$|billion|million|tCO[₂2]|est\.)?\s+'
        r'[\d\.,%\$€]+',                    # Value2
        re.IGNORECASE,
    )
    if col_pattern.search(text):
        signal_count += 1

    return signal_count >= 2


def is_abbreviation_chunk(text: str) -> bool:
    """
    [FIX-19] Phát hiện chunk là bảng chữ viết tắt (Abbreviations / Acronyms).

    Đặc trưng:
    - Nhiều dòng dạng "ABBR Mô tả đầy đủ" — cụm chữ viết hoa ngắn (2-8 ký tự)
      theo sau là mô tả
    - Ví dụ: "BATs Các kỹ thuật tốt nhất hiện có"
             "CPTPP Hiệp định Đối tác Toàn diện..."
             "GDP Tổng sản phẩm quốc nội"
    - Hoặc bắt đầu bằng tiêu đề "DANH MỤC VIẾT TẮT" / "ABBREVIATIONS"

    Không có giá trị RAG vì chỉ định nghĩa ký hiệu, không chứa luận điểm.
    """
    text_stripped = text.strip()

    # Nhanh: bắt tiêu đề bảng viết tắt
    if re.search(
        r'(?:DANH\s+MỤC\s+VIẾT\s+TẮT|LIST\s+OF\s+ABBREVIATIONS?|ACRONYMS?)',
        text_stripped, re.IGNORECASE
    ):
        return True

    lines = [l.strip() for l in text_stripped.split('\n') if l.strip()]
    if len(lines) < 4:
        return False

    # Pattern: dòng bắt đầu bằng 2-8 ký tự CHỮ HOA (viết tắt) rồi khoảng trắng + mô tả
    abbr_line = re.compile(r'^[A-ZÀÂÉÊÙÛỸ&/\-]{2,10}\s{1,4}\S')
    abbr_count = sum(1 for l in lines if abbr_line.match(l))

    return abbr_count / len(lines) > 0.50


def is_infographic_chunk(text: str) -> bool:
    """
    [FIX-15] Phát hiện chunk là infographic bị extract sai từ PDF/web.

    Dấu hiệu:
    - Ký tự đặc biệt lạ (¬, Ö, ›, »...) xen lẫn text — OCR sai
    - Pattern "text..." rồi ký tự lạ rồi text (label biểu đồ bị cắt)
    - Hỗn hợp chữ + ký tự symbol xen kẽ dày đặc

    Ví dụ bắt được:
      "Export turnover...¬ 9 Growth rate...ae » _Ö@ Industry...k=› | I I iN"
    """
    if len(text) < 50:
        return False

    # Ký tự tiếng Việt hợp lệ (range chặt, loại Ö U+00D6 ra ngoài)
    VIET_RANGES = [
        (0x00C0, 0x00C3), (0x00C8, 0x00CA), (0x00CC, 0x00CD),
        (0x00D2, 0x00D5), (0x00D9, 0x00DA), (0x00DD, 0x00DD),
        (0x00E0, 0x00E3), (0x00E8, 0x00EA), (0x00EC, 0x00ED),
        (0x00F2, 0x00F5), (0x00F9, 0x00FA), (0x00FD, 0x00FD),
        (0x0102, 0x0103), (0x0110, 0x0111), (0x01A0, 0x01B0),
        (0x1EA0, 0x1EF9), (0x2080, 0x2099),
    ]
    COMMON_SYM = set('°€£¥©®™×÷±²³')

    def is_ok(c):
        if c in COMMON_SYM:
            return True
        o = ord(c)
        return any(lo <= o <= hi for lo, hi in VIET_RANGES)

    special = [c for c in text if ord(c) > 127 and not is_ok(c)]
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars > 0 and len(special) >= 2 and len(special) / total_chars > 0.02:
        return True

    # Pattern: "text..." + ký tự lạ hoặc chữ + symbol + chữ xen kẽ
    garbled = re.compile(
        r'\.\.\.[^a-zA-ZÀ-ỹ\s]{1,5}[a-zA-ZÀ-ỹ@]{1,4}'
        r'|[a-zA-ZÀ-ỹ]{1,4}[=›»¬|□■→←↑↓▲▼◆●○_@#]{1,3}[a-zA-ZÀ-ỹ\d]{1,4}'
    )
    if len(garbled.findall(text)) >= 2:
        return True

    return False


def find_word_boundary(text: str, pos: int, search_range: int = 50) -> int:
    """
    Tìm ranh giới từ hoàn chỉnh gần vị trí `pos`.
    Tìm khoảng trắng hoặc newline gần nhất TRƯỚC pos trong phạm vi search_range.
    Đảm bảo chunk tiếp theo không bắt đầu bằng nửa từ bị cụt.
    """
    min_pos = max(0, pos - search_range)
    for sep in ['\n', ' ']:
        idx = text.rfind(sep, min_pos, pos)
        if idx != -1:
            return idx + 1  # bắt đầu SAU khoảng trắng/newline
    return pos  # fallback: không tìm được → giữ nguyên


def find_sentence_boundary_back(text: str, end: int, min_pos: int) -> int:
    """
    Tìm ranh giới câu hoàn chỉnh gần nhất TRƯỚC vị trí `end`.
    Ưu tiên: '.\n' > '. ' > '\n' > ranh giới từ.
    Sau khi tìm được ranh giới câu, áp thêm find_word_boundary để đảm bảo
    không bắt đầu giữa từ.
    """
    for sep in [".\n", ". ", "\n"]:
        idx = text.rfind(sep, min_pos, end)
        if idx != -1:
            candidate = idx + len(sep)
            return find_word_boundary(text, candidate, search_range=30)
    return find_word_boundary(text, end, search_range=50)


def deduplicate_overlap(prev_chunk: str, curr_chunk: str) -> str:
    """
    [FIX-11] Loại bỏ phần overlap lặp lại ở đầu curr_chunk.

    Thuật toán sliding window:
    - Thử các đoạn có độ dài 40–200 chars từ đầu curr_chunk
    - Kiểm tra xem đoạn đó có xuất hiện trong 300 chars cuối của prev_chunk không
    - Lấy đoạn trùng DÀI NHẤT tìm được → strip toàn bộ phần đó khỏi curr_chunk
    - Sau khi strip, trim đến ranh giới từ/câu hoàn chỉnh

    Bắt được overlap nhiều câu (không chỉ 1 câu như v5), ví dụ:
      prev kết thúc: "...CBAM standards. Vietnam shows proactive alignment..."
      curr bắt đầu:  "CBAM standards. Vietnam shows proactive alignment. At the..."
      → strip "CBAM standards. Vietnam shows proactive alignment. " khỏi curr
    """
    curr_stripped = curr_chunk.strip()
    prev_tail = prev_chunk[-300:]  # chỉ so sánh với 300 chars cuối chunk trước

    best_len = 0  # độ dài đoạn trùng dài nhất tìm được

    # Thử các window từ dài → ngắn để lấy match dài nhất
    for win_size in range(200, 39, -10):
        candidate = curr_stripped[:win_size]
        if candidate in prev_tail:
            best_len = win_size
            break  # đã tìm được match dài nhất ở window này

    if best_len == 0:
        return curr_chunk  # không có overlap → giữ nguyên

    # Strip phần trùng, sau đó tìm ranh giới câu/từ sạch để bắt đầu
    remainder = curr_stripped[best_len:]

    # Trim đến ranh giới câu: tìm chữ hoa đầu câu hoặc khoảng trắng
    for sep in ['. ', '.\n', '\n', ' ']:
        idx = remainder.find(sep)
        if idx != -1 and idx < 50:
            remainder = remainder[idx + len(sep):]
            break

    remainder = remainder.strip()
    return remainder if len(remainder) >= MIN_CHUNK_LENGTH else curr_chunk


def split_into_chunks(text: str) -> list[str]:
    """
    Split text thành chunks ~CHUNK_SIZE chars với overlap tại ranh giới câu/từ.

    Pipeline:
    1. clean_text() để loại artifact PDF trước khi split
    2. Sliding window CHUNK_SIZE chars
    3. Tìm điểm cắt tốt nhất (ranh giới câu)
    4. Tìm điểm bắt đầu chunk tiếp theo tại ranh giới từ hoàn chỉnh
    5. [FIX-10] deduplicate_overlap() loại bỏ câu lặp giữa 2 chunk liên tiếp
    """
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= CHUNK_SIZE:
        return [text] if len(text) >= MIN_CHUNK_LENGTH else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end >= len(text):
            chunk = text[start:].strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            break

        # Bước 1: tìm điểm cắt tốt nhất gần CHUNK_SIZE
        cut = end
        search_from = start + CHUNK_SIZE // 2

        for sep in ["\n\n", ".\n", ". ", "\n"]:
            idx = text.rfind(sep, search_from, end)
            if idx != -1:
                cut = idx + len(sep)
                break

        # Lưu chunk hiện tại (với dedup overlap so với chunk trước)
        chunk = text[start:cut].strip()
        if len(chunk) >= MIN_CHUNK_LENGTH and not is_heading(chunk):
            if chunks:  # [FIX-10] dedup câu lặp với chunk trước
                chunk = deduplicate_overlap(chunks[-1], chunk)
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)

        # Bước 2: tìm điểm bắt đầu chunk tiếp theo tại ranh giới câu + từ
        overlap_min = max(start + 1, cut - CHUNK_OVERLAP * 2)
        overlap_max = max(start + 1, cut - CHUNK_OVERLAP // 2)

        next_start = find_sentence_boundary_back(text, overlap_max, overlap_min)

        # Safety: luôn tiến về phía trước
        if next_start <= start or next_start >= cut:
            next_start = cut

        start = next_start

    return chunks


# ================= MAIN =================

def main():
    # Lọc file business — bỏ qua văn bản pháp lý
    legal_patterns = [
        "luật", "nghị định", "thông tư", "quyết định",
        "luat", "nghi dinh", "thong tu", "quyet dinh",
    ]

    business_files = []
    for txt_file in NORMALIZED_DIR.glob("*.txt"):
        name_lower = txt_file.name.lower()
        if not any(pat in name_lower for pat in legal_patterns):
            business_files.append(txt_file)

    logging.info(f"Tìm thấy {len(business_files)} file BUSINESS trong {NORMALIZED_DIR}")

    total_chunks  = 0
    total_chars   = 0
    total_skipped = 0
    skipped_files = 0

    # Đếm chi tiết từng loại skip
    skip_counts = {"TOC": 0, "REF": 0, "META": 0, "BOILERPLATE": 0, "CONTACT": 0, "CHART": 0, "TABLE": 0, "INFOGRAPHIC": 0, "ABBREV": 0}

    for txt_file in sorted(business_files):
        text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            logging.warning(f"  ⚠️  Bỏ qua (rỗng): {txt_file.name}")
            skipped_files += 1
            continue

        raw_chunks = split_into_chunks(text)

        if not raw_chunks:
            logging.warning(f"  ⚠️  Không có chunks: {txt_file.name}")
            skipped_files += 1
            continue

        # Lọc các loại chunk không có giá trị ngữ nghĩa
        para_chunks  = []
        file_skipped = 0
        file_skip_counts = {"TOC": 0, "REF": 0, "META": 0, "BOILERPLATE": 0, "CONTACT": 0, "CHART": 0, "TABLE": 0, "INFOGRAPHIC": 0, "ABBREV": 0}

        for chunk_text in raw_chunks:
            skip_reason = None

            if is_toc_chunk(chunk_text):
                skip_reason = "TOC"
            elif is_references_chunk(chunk_text):
                skip_reason = "REF"
            elif is_metadata_chunk(chunk_text):
                skip_reason = "META"
            elif is_boilerplate_chunk(chunk_text):
                skip_reason = "BOILERPLATE"
            elif is_chart_data_chunk(chunk_text):
                skip_reason = "CHART"
            elif is_inline_table_chunk(chunk_text):        # [FIX-13]
                skip_reason = "TABLE"
            elif is_infographic_chunk(chunk_text):         # [FIX-15]
                skip_reason = "INFOGRAPHIC"
            elif is_abbreviation_chunk(chunk_text):        # [FIX-19]
                skip_reason = "ABBREV"

            if skip_reason:
                file_skipped += 1
                file_skip_counts[skip_reason] += 1
                skip_counts[skip_reason] += 1
                continue

            para_chunks.append({
                "para_id":  f"{txt_file.stem}_{len(para_chunks)+1:03d}",
                "text":     chunk_text,
                "position": len(para_chunks) + 1,
                "char_len": len(chunk_text),
            })

        total_skipped += file_skipped

        if not para_chunks:
            logging.warning(f"  ⚠️  Không còn chunk sau khi lọc: {txt_file.name}")
            skipped_files += 1
            continue

        output = {
            "source_file":     txt_file.name,
            "source_type":     "BUSINESS",
            "paragraphs":      para_chunks,
            "paragraph_count": len(para_chunks),
        }

        out_path = OUTPUT_DIR / f"{txt_file.stem}_paragraphs.json"
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        file_chars = sum(len(c["text"]) for c in para_chunks)
        total_chunks += len(para_chunks)
        total_chars  += file_chars
        avg_chars     = file_chars // len(para_chunks)

        # Log chi tiết skip theo từng loại
        if file_skipped:
            skip_detail = " | ".join(
                f"{k}:{v}" for k, v in file_skip_counts.items() if v > 0
            )
            skip_note = f" | bỏ qua: {file_skipped} ({skip_detail})"
        else:
            skip_note = ""

        logging.info(
            f"  ✅ {txt_file.name[:50]}: "
            f"{len(para_chunks)} chunks | avg {avg_chars} chars/chunk{skip_note}"
        )

    logging.info(f"\n{'='*60}")
    logging.info(f"DONE")
    logging.info(f"  Files xử lý        : {len(business_files) - skipped_files}")
    logging.info(f"  Files bỏ qua       : {skipped_files}")
    logging.info(f"  Tổng chunks giữ lại: {total_chunks}")
    logging.info(f"  Chunks lọc bỏ      : {total_skipped}")
    for reason, count in skip_counts.items():
        if count > 0:
            logging.info(f"    - {reason:<12}: {count}")
    logging.info(f"  Tổng chars         : {total_chars:,}")
    avg_global = total_chars // total_chunks if total_chunks else 0
    logging.info(f"  Avg chars/chunk    : {avg_global}")


if __name__ == "__main__":
    main()