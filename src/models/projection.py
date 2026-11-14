import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection Head cho Supervised Contrastive Learning (CUA-MNER design).
    Chiếu biểu diễn ngữ cảnh 768 chiều sang không gian compact 128 chiều có L2-norm.
    Đầu chiếu này hấp thụ các biến dạng phi tuyến trong Stage 1 và bị vứt bỏ hoàn toàn ở Stage 2.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, input_dim]
        z = self.mlp(x)
        return F.normalize(z, p=2, dim=-1)

