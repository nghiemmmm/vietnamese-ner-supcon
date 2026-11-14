"""
Hệ thống hằng số cho tác vụ PAP_NER (Thủ tục Hành chính Việt Nam)
Bao gồm 11 nhãn BIO và quy tắc ánh xạ phân cấp sang 5 loại thực thể (Entity Types).
"""

# Danh sách 11 nhãn BIO chuẩn
LABEL2ID_PAP_NER = {
    'O': 0,
    'B-CQ': 1,
    'I-CQ': 2,
    'B-ĐT': 3,
    'I-ĐT': 4,
    'B-VBPL': 5,
    'I-VBPL': 6,
    'B-NG': 7,
    'I-NG': 8,
    'B-SL': 9,
    'I-SL': 10
}

ID2LABEL_PAP_NER = {v: k for k, v in LABEL2ID_PAP_NER.items()}

# Ánh xạ 5 loại thực thể cấp cao (Entity-level Types):
# CQ: Cơ quan (ID 1, 2)
# ĐT: Đối tượng thực hiện (ID 3, 4)
# VBPL: Văn bản pháp luật (ID 5, 6)
# NG: Ngày tháng / Thời hạn (ID 7, 8)
# SL: Số lượng / Lệ phí / Hồ sơ (ID 9, 10)
ENTITY_TYPES = ['CQ', 'ĐT', 'VBPL', 'NG', 'SL']

# Từ điển ánh xạ từ BIO ID sang Entity Type ID (1..5)
# Lưu ý: Nhãn 'O' (ID 0) có Entity Type ID = 0 (Non-entity)
BIO2ENTITY_TYPE = {
    0: 0,   # O
    1: 1,   # B-CQ -> CQ
    2: 1,   # I-CQ -> CQ
    3: 2,   # B-ĐT -> ĐT
    4: 2,   # I-ĐT -> ĐT
    5: 3,   # B-VBPL -> VBPL
    6: 3,   # I-VBPL -> VBPL
    7: 4,   # B-NG -> NG
    8: 4,   # I-NG -> NG
    9: 5,   # B-SL -> SL
    10: 5   # I-SL -> SL
}

LABEL_MAPPING = {
    'pap_ner': {
        'label2id': LABEL2ID_PAP_NER,
        'id2label': ID2LABEL_PAP_NER,
        'bio2type': BIO2ENTITY_TYPE,
        'entity_types': ENTITY_TYPES
    }
}

