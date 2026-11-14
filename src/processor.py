from typing import Union, List
from tqdm import tqdm
import os
import torch
import pandas as pd
import numpy as np


class NerFeatures(object):
    """Lưu trữ tensors đặc trưng của một câu sau khi gióng hàng subword."""
    def __init__(self, input_ids, token_type_ids=None, attention_mask=None,
                 valid_ids=None, labels=None, label_masks=None, **kwargs):
        self.input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        self.token_type_ids = (torch.as_tensor(token_type_ids, dtype=torch.long)
                               if token_type_ids is not None
                               else torch.zeros_like(self.input_ids))
        self.attention_mask = (torch.as_tensor(attention_mask, dtype=torch.long)
                               if attention_mask is not None
                               else torch.ones_like(self.input_ids))
        self.valid_ids = torch.as_tensor(valid_ids, dtype=torch.long) if valid_ids is not None else None
        self.labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        self.label_masks = torch.as_tensor(label_masks, dtype=torch.long) if label_masks is not None else None


def convert_word_segment_examples_features(data_path: Union[str, os.PathLike],
                                           tokenizer,
                                           label2id: dict,
                                           header_names: List[str] = None,
                                           max_seq_len: int = 256) -> List[NerFeatures]:
    """
    Xử lý file dữ liệu định dạng CoNLL thành các features của PhoBERT với cơ chế gióng hàng valid_ids.
    valid_ids: Trỏ tới subword đầu tiên của mỗi từ gốc (bỏ qua subword đuôi @@).
    """
    if header_names is None:
        header_names = ['token', 'ner']

    features = []
    tokens = []
    tag_ids = []

    # Đọc file dữ liệu CoNLL (phân cách bằng khoảng trắng hoặc tab)
    data = pd.read_csv(data_path,
                       sep=r'\s+',
                       encoding='utf-8',
                       skip_blank_lines=False,
                       names=header_names)
    data = data.ffill()

    label_list = list(label2id.keys())

    for row_idx, row in enumerate(data.itertuples(index=False)):
        token_val = getattr(row, header_names[0], None)
        ner_val = getattr(row, header_names[1], None)

        if pd.notna(token_val) and str(token_val).strip() != '':
            tokens.append(str(token_val).strip().replace(' ', '_'))
            ner_tag = str(ner_val).strip() if pd.notna(ner_val) else 'O'
            tag_id = label2id.get(ner_tag, label2id.get('O', 0))
            tag_ids.append(tag_id)
            if row_idx != len(data) - 1:
                continue

        seq_len = len(tokens)
        if seq_len == 0:
            continue

        sentence = ' '.join(tokens)
        encoding = tokenizer(sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_seq_len)
        subwords = tokenizer.tokenize(sentence)

        valid_ids = np.zeros(len(encoding['input_ids']), dtype=int)
        label_masks = np.zeros(len(encoding['input_ids']), dtype=int)

        i = 1
        for idx, subword in enumerate(subwords[:max_seq_len - 2]):
            if idx != 0 and subwords[idx - 1].endswith("@@"):
                continue
            valid_ids[i - 1] = idx + 1
            i += 1

        if max_seq_len >= seq_len:
            label_masks[:seq_len] = 1
            tag_ids.extend([0] * (max_seq_len - seq_len))
        else:
            tag_ids = tag_ids[:max_seq_len]
            label_masks[:-2] = 1
            tag_ids[-2:] = 0

        items = {key: val for key, val in encoding.items()}
        if 'token_type_ids' not in items:
            items['token_type_ids'] = [0] * max_seq_len

        items['labels'] = tag_ids
        items['valid_ids'] = valid_ids
        items['label_masks'] = label_masks

        features.append(NerFeatures(**items))

        tokens = []
        tag_ids = []

    return features

