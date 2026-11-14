import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage Linear Evaluation Protocol: Hierarchical SupCon + Linear Probe (No CRF) for PAP_NER"
    )

    # 1. Dữ liệu và Mô hình nền tảng
    parser.add_argument('--task', type=str, default='pap_ner', help="Tên tác vụ (mặc định: pap_ner)")
    parser.add_argument('--data_dir', type=str, default='./Dataset', help="Thư mục chứa dữ liệu train.txt, dev.txt, test.txt")
    parser.add_argument('--model_name_or_path', type=str, default='vinai/phobert-base',
                        help="Tên pretrained checkpoint (vinai/phobert-base hoặc trungbb8/distilphobert)")
    parser.add_argument('--output_dir', type=str, default='outputs/baseline_hsupcon',
                        help="Thư mục lưu mô hình và logs kết quả")
    parser.add_argument('--max_seq_length', type=int, default=256, help="Độ dài chuỗi tối đa")
    parser.add_argument('--seed', type=int, default=42, help="Random seed tái lập kết quả")
    parser.add_argument('--fp16', action='store_true', default=True, help="Bật Mixed Precision FP16 (tăng tốc độ, tiết kiệm VRAM)")
    parser.add_argument('--num_workers', type=int, default=2, help="Số worker nạp dữ liệu")

    # 2. Siêu tham số GIAI ĐOẠN 1: Hierarchical SupCon Pre-training
    parser.add_argument('--skip_stage1', action='store_true', default=False,
                        help="Bỏ qua Stage 1 nếu đã có sẵn checkpoint pre-trained encoder")
    parser.add_argument('--pretrained_encoder_path', type=str, default=None,
                        help="Đường dẫn file trọng số encoder đã pre-train (nếu bỏ qua Stage 1)")
    parser.add_argument('--stage1_epochs', type=int, default=15, help="Số epoch tối đa cho Stage 1")
    parser.add_argument('--stage1_batch_size', type=int, default=64, help="Batch size hiệu dụng cho Stage 1")
    parser.add_argument('--stage1_lr', type=float, default=2e-5, help="Learning rate của PhoBERT trong Stage 1")
    parser.add_argument('--stage1_proj_lr', type=float, default=1e-3, help="Learning rate của Projection Head")
    parser.add_argument('--supcon_temp', type=float, default=0.07, help="Nhiệt độ tau trong hàm tương phản SupCon")
    parser.add_argument('--supcon_alpha', type=float, default=0.3,
                        help="Trọng số cầu nối mềm alpha (Weak Positive) giữa B và I của cùng thực thể")
    parser.add_argument('--proj_dim', type=int, default=128, help="Số chiều không gian chiếu tương phản L2-normalized")
    parser.add_argument('--stage1_early_stop', type=int, default=5,
                        help="Dừng sớm sau N epochs nếu Dev SupCon Loss không cải thiện")

    # 3. Siêu tham số GIAI ĐOẠN 2: Linear Probe (Backbone FROZEN, NO CRF)
    parser.add_argument('--stage2_epochs', type=int, default=20, help="Số epoch cho Stage 2 (Linear Probe)")
    parser.add_argument('--stage2_batch_size', type=int, default=64, help="Batch size cho Stage 2")
    parser.add_argument('--stage2_lr', type=float, default=1e-3, help="Learning rate cho tầng Linear Classifier")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Hệ số Weight decay L2 regularization")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help="Tỷ lệ warmup bước học")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Ngưỡng Gradient Clipping")
    parser.add_argument('--stage2_early_stop', type=int, default=5,
                        help="Dừng sớm sau N epochs nếu Dev Strict F1 không cải thiện")

    args = parser.parse_args()
    return args

