import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .base import NerOutput
from .projection import ProjectionHead
from .losses import HierarchicalTokenSupConLoss


class PhoBertSupConPretrainer(nn.Module):
    """
    GIAI ĐOẠN 1: Pre-training PhoBERT bằng Hierarchical Token-level Contrastive Loss.
    
    Cấu trúc:
      Input Tokens -> PhoBERT Backbone -> valid_ids alignment -> ProjectionHead -> H-SupCon Loss
    """
    def __init__(self, model_name_or_path: str = 'vinai/phobert-base',
                 proj_dim: int = 128,
                 temperature: float = 0.07,
                 alpha: float = 0.3,
                 o_label_id: int = 0):
        super(PhoBertSupConPretrainer, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.projection_head = ProjectionHead(input_dim=self.config.hidden_size, hidden_dim=256, output_dim=proj_dim)
        self.supcon_loss = HierarchicalTokenSupConLoss(temperature=temperature, alpha=alpha)
        self.o_label_id = o_label_id

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_masks=None, **kwargs):
        # 1. Biểu diễn ngữ cảnh từ PhoBERT
        outputs = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        seq_output = outputs[0]  # [B, max_len, 768]

        # 2. Gióng hàng subword để lấy biểu diễn cấp từ (word-level representations)
        batch_size, max_len, feat_dim = seq_output.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_output.device).unsqueeze(1)
        word_representations = seq_output[range_vector, valid_ids]  # [B, max_words, 768]

        # 3. Lọc riêng các token thực thể (loại bỏ nhãn 'O' và padding)
        valid_entity_mask = (label_masks != 0) & (labels != self.o_label_id)
        if not valid_entity_mask.any():
            zero_loss = seq_output.sum() * 0.0
            return NerOutput(loss=zero_loss, loss_supcon=zero_loss)

        entity_words = word_representations[valid_entity_mask]  # [N_entity, 768]
        entity_labels = labels[valid_entity_mask]                # [N_entity]

        # 4. Chiếu qua Projection Head và tính H-SupCon Loss
        entity_projections = self.projection_head(entity_words)  # [N_entity, 128]
        loss = self.supcon_loss(entity_projections, entity_labels)

        return NerOutput(loss=loss, loss_supcon=loss)


class PhoBertLinearProbe(nn.Module):
    """
    GIAI ĐOẠN 2: Linear Probe trên Backbone ĐÓNG BĂNG 100% (KHÔNG DÙNG CRF).
    
    Cấu trúc:
      Input Tokens -> PhoBERT (requires_grad = False) -> valid_ids alignment -> nn.Linear(768, num_labels) -> Cross-Entropy
      Suy luận: argmax(logits, dim=-1)
    """
    def __init__(self, model_name_or_path: str = 'vinai/phobert-base',
                 num_labels: int = 11,
                 freeze_backbone: bool = True):
        super(PhoBertLinearProbe, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.num_labels = num_labels
        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # ĐÓNG BĂNG HOÀN TOÀN BACKBONE THEO CHUẨN GOOGLE LINEAR EVALUATION
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, valid_ids=None, label_masks=None, **kwargs):
        # 1. Trích xuất đặc trưng với backbone đã đóng băng
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.roberta.parameters())):
            outputs = self.roberta(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            seq_output = outputs[0]

        # 2. Gióng hàng subword
        batch_size, max_len, feat_dim = seq_output.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_output.device).unsqueeze(1)
        word_representations = seq_output[range_vector, valid_ids]

        # 3. Phân loại tuyến tính (Linear Classifier)
        word_representations = self.dropout(word_representations)
        logits = self.classifier(word_representations)  # [B, max_words, num_labels]

        # 4. Dự đoán nhãn độc lập từng token (KHÔNG CÓ CRF)
        preds = torch.argmax(logits, dim=-1)  # [B, max_words]

        # Tách danh sách nhãn dự đoán theo từng câu dựa trên mask
        tags = []
        for b in range(batch_size):
            active_len = label_masks[b].sum().item()
            tags.append(preds[b, :active_len].cpu().tolist())

        # 5. Tính hàm mất mát Cross-Entropy (chỉ tính trên các vị trí token hợp lệ)
        loss = None
        if labels is not None:
            active_loss = label_masks.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels).mean()

        return NerOutput(loss=loss, logits=logits, tags=tags, loss_ce=loss)

