import os
import glob
from typing import List, Union
import torch
from torch.utils.data import Dataset, DataLoader

from .processor import NerFeatures, convert_word_segment_examples_features
from .constant import LABEL_MAPPING


class PapNerDataset(Dataset):
    """PyTorch Dataset chứa các NerFeatures cho mô hình PhoBERT."""
    def __init__(self, features: List[NerFeatures]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            'input_ids': feat.input_ids,
            'token_type_ids': feat.token_type_ids,
            'attention_mask': feat.attention_mask,
            'valid_ids': feat.valid_ids,
            'labels': feat.labels,
            'label_masks': feat.label_masks
        }


def find_data_file(data_dir: str, split: str) -> str:
    """Tự động tìm kiếm file dữ liệu theo split ('train', 'dev', 'test') trên mọi môi trường."""
    candidates = [
        os.path.join(data_dir, f"{split}.txt"),
        os.path.join(data_dir, f"{split}_data.txt"),
        os.path.join(data_dir, f"{split}.csv"),
        os.path.join("./Dataset", f"{split}.txt"),
        os.path.join("./Dataset", f"{split}_data.txt"),
        os.path.join("../Dataset", f"{split}.txt"),
        os.path.join("../Dataset", f"{split}_data.txt")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Tìm kiếm fallback trong drive hoặc thư mục lân cận
    matches = glob.glob(f"**/{split}*.txt", recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Không thể tìm thấy file dữ liệu cho split='{split}' trong thư mục {data_dir}!")


def get_data_loader(data_dir: str,
                    split: str,
                    tokenizer,
                    batch_size: int,
                    max_seq_len: int = 256,
                    task: str = 'pap_ner',
                    shuffle: bool = False,
                    num_workers: int = 2) -> DataLoader:
    """Xây dựng PyTorch DataLoader cho một split dữ liệu."""
    file_path = find_data_file(data_dir, split)
    label2id = LABEL_MAPPING[task]['label2id']

    features = convert_word_segment_examples_features(
        data_path=file_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_seq_len=max_seq_len
    )

    dataset = PapNerDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader

