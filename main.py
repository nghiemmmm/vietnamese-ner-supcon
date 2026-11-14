import os
import sys
import torch
from transformers import AutoTokenizer

from src.arguments import parse_args
from src.helper import init_logger, set_random_seed
from src.dataset import get_data_loader
from src.trainer import train_stage1_hsupcon, train_stage2_linear_probe


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, 'training.log')
    logger = init_logger(log_file)

    logger.info("=" * 70)
    logger.info("  PAP_NER: TWO-STAGE H-SUPCON + LINEAR PROBE (NO CRF)")
    logger.info(f"  Task: {args.task} | Model: {args.model_name_or_path} | Seed: {args.seed}")
    logger.info(f"  SupCon Temp: {args.supcon_temp} | Alpha: {args.supcon_alpha}")
    logger.info(f"  Stage 1 Epochs: {args.stage1_epochs} | Stage 2 Epochs: {args.stage2_epochs}")
    logger.info("=" * 70)

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Thiết bị tính toán: {device}")

    # 1. Khởi tạo Tokenizer
    logger.info(f"  Khởi tạo Tokenizer từ: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # 2. Xây dựng DataLoaders cho Stage 1 và Stage 2
    logger.info("  Đang nạp và gióng hàng dữ liệu CoNLL PAP_NER...")
    train_loader_s1 = get_data_loader(
        data_dir=args.data_dir, split='train', tokenizer=tokenizer,
        batch_size=args.stage1_batch_size, max_seq_len=args.max_seq_length,
        task=args.task, shuffle=True, num_workers=args.num_workers
    )
    dev_loader_s1 = get_data_loader(
        data_dir=args.data_dir, split='dev', tokenizer=tokenizer,
        batch_size=args.stage1_batch_size, max_seq_len=args.max_seq_length,
        task=args.task, shuffle=False, num_workers=args.num_workers
    )

    train_loader_s2 = get_data_loader(
        data_dir=args.data_dir, split='train', tokenizer=tokenizer,
        batch_size=args.stage2_batch_size, max_seq_len=args.max_seq_length,
        task=args.task, shuffle=True, num_workers=args.num_workers
    )
    dev_loader_s2 = get_data_loader(
        data_dir=args.data_dir, split='dev', tokenizer=tokenizer,
        batch_size=args.stage2_batch_size, max_seq_len=args.max_seq_length,
        task=args.task, shuffle=False, num_workers=args.num_workers
    )
    test_loader = get_data_loader(
        data_dir=args.data_dir, split='test', tokenizer=tokenizer,
        batch_size=args.stage2_batch_size, max_seq_len=args.max_seq_length,
        task=args.task, shuffle=False, num_workers=args.num_workers
    )

    logger.info(f"  Số mẫu - Train: {len(train_loader_s1.dataset)} | Dev: {len(dev_loader_s1.dataset)} | Test: {len(test_loader.dataset)}")

    # 3. GIAI ĐOẠN 1: H-SupCon Pre-training
    if args.skip_stage1:
        logger.info(f"  Bỏ qua Stage 1 theo yêu cầu cờ --skip_stage1.")
        encoder_weights = args.pretrained_encoder_path
        if not encoder_weights or not os.path.exists(encoder_weights):
            default_path = os.path.join(args.output_dir, 'best_encoder.pt')
            if os.path.exists(default_path):
                encoder_weights = default_path
    else:
        encoder_weights = train_stage1_hsupcon(
            args=args,
            train_loader=train_loader_s1,
            dev_loader=dev_loader_s1,
            device=device
        )

    # 4. GIAI ĐOẠN 2: Linear Probe Evaluation
    train_stage2_linear_probe(
        args=args,
        train_loader=train_loader_s2,
        dev_loader=dev_loader_s2,
        test_loader=test_loader,
        device=device,
        encoder_weights_path=encoder_weights
    )

    logger.info("=" * 70)
    logger.info("  HOÀN THÀNH TOÀN BỘ QUY TRÌNH H-SUPCON LINEAR EVALUATION PROTOCOL!")
    logger.info(f"  Toàn bộ kết quả và logs đã được lưu tại: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

