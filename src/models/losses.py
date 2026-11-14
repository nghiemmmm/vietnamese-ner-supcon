import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalTokenSupConLoss(nn.Module):
    """
    Hàm mất mát Tương phản Phân cấp ở cấp độ Token (Hierarchical Token-level SupCon Loss).
    
    Quy tắc trọng số 3 mức (3-tier weighting):
      1. Strong Positives (Cùng chính xác nhãn BIO, ví dụ B-CQ với B-CQ): Trọng số w = 1.0
      2. Weak Positives   (Cùng loại thực thể nhưng khác BIO, ví dụ B-CQ với I-CQ): Trọng số w = alpha (0.3)
      3. Negatives        (Khác hoàn toàn loại thực thể): Trọng số w = 0.0 (đẩy dạt ở mẫu số)
      
    Hàm này hỗ trợ hoàn toàn cơ chế chống tràn số và NaN trong môi trường Mixed Precision FP16.
    """
    def __init__(self, temperature: float = 0.07, alpha: float = 0.3):
        super(HierarchicalTokenSupConLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N_entity, proj_dim] (Tensor vector chiếu 128-dim của các token thực thể)
            labels:   [N_entity] (1D tensor chứa nhãn BIO của token)
        """
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        # 1. Ép kiểu sang float32 để đảm bảo độ ổn định số học tối đa trong môi trường AMP
        features = features.float()
        features = F.normalize(features, p=2, dim=1)

        # 2. Ma trận tương đồng Cosine chuẩn hóa theo nhiệt độ tau
        similarity = torch.matmul(features, features.T) / self.temperature
        N = features.size(0)

        # 3. Mặt nạ loại trừ chính nó (self-mask)
        self_mask = torch.eye(N, device=features.device, dtype=torch.bool)

        # 4. Tính toán ma trận trọng số phân cấp (Weight Mask)
        labels_col = labels.contiguous().view(-1, 1)
        same_bio = torch.eq(labels_col, labels_col.T)

        if self.alpha == 0.0:
            # Thu về Flat SupCon nhị phân thuần túy
            weight_mask = (same_bio & (~self_mask)).float()
        else:
            # Ánh xạ BIO ID sang Entity Type ID:
            # 1,2 (CQ) -> 1; 3,4 (ĐT) -> 2; 5,6 (VBPL) -> 3; 7,8 (NG) -> 4; 9,10 (SL) -> 5
            entity_types = (labels_col + 1) // 2
            same_type = torch.eq(entity_types, entity_types.T)

            strong_pos = same_bio.float()
            weak_pos = (same_type & (~same_bio)).float() * self.alpha
            weight_mask = (strong_pos + weak_pos) * (~self_mask).float()

        # 5. Khử đường chéo mẫu số bằng -1e9 (tránh 0.0 * (-inf) = NaN)
        similarity = similarity.masked_fill(self_mask, -1e9)

        # 6. Ổn định số học: trừ max trên từng hàng
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # 7. Tính xác suất log trên toàn bộ ứng viên mẫu số
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # 8. Gán đường chéo bằng 0.0
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        # 9. Lọc các anchor hợp lệ có ít nhất 1 liên kết dương (weight > 0)
        weight_count = weight_mask.sum(dim=1)
        valid_anchor = weight_count > 0

        if not valid_anchor.any():
            return features.new_tensor(0.0)

        # 10. Kỳ vọng log-likelihood có trọng số trên các cặp dương (Weighted Expectation)
        mean_log_prob_pos = (weight_mask * log_prob).sum(dim=1) / weight_count.clamp_min(1e-12)

        # 11. Hàm mất mát H-SupCon scalar
        loss = -mean_log_prob_pos[valid_anchor].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return features.new_tensor(0.0)
        return loss

