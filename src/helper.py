import os
import random
import logging
import datetime
from contextlib import nullcontext
from typing import List

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def init_logger(log_file: str = None) -> logging.Logger:
    """Khởi tạo logger đồng thời ghi ra màn hình console và file log."""
    logger = logging.getLogger("baseline_hsupcon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (nếu có)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int = 42):
    """Cố định seed để tái lập hoàn toàn kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_amp_context(use_amp: bool, device: torch.device):
    """Context manager an toàn cho Mixed Precision FP16 tương thích mọi phiên bản PyTorch."""
    if not use_amp or not str(device).startswith('cuda'):
        return nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()


def _get_grad_scaler(use_amp: bool, device: torch.device):
    """GradScaler an toàn cho Mixed Precision FP16."""
    if not use_amp or not str(device).startswith('cuda'):
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda')
    return torch.cuda.amp.GradScaler()


def get_total_model_parameters(model: torch.nn.Module):
    """Đếm tổng số tham số và số tham số có thể cập nhật (trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def safe_torch_load(path: str, map_location='cpu'):
    """Tải trọng số PyTorch an toàn tương thích PyTorch 2.6+."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def plot_confusion_matrix(golds: List[int], preds: List[int], classes: List[str], output_dir: str,
                          title: str = 'Normalized Confusion Matrix', normalize: bool = True):
    """Vẽ và lưu Confusion Matrix 11x11 ra file PNG."""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(golds, preds, labels=list(range(len(classes))))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = format(cm[i, j], fmt) if cm[i, j] > 0 else '0'
            plt.text(j, i, val,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

    plt.ylabel('True BIO Label', fontsize=12)
    plt.xlabel('Predicted BIO Label', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

