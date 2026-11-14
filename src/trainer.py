import os
import time
import math
import logging
import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

from .constant import LABEL_MAPPING, LABEL2ID_PAP_NER
from .helper import _get_amp_context, _get_grad_scaler, plot_confusion_matrix, safe_torch_load
from .conlleval import evaluate
from .models import PhoBertSupConPretrainer, PhoBertLinearProbe

LOGGER = logging.getLogger("baseline_hsupcon")


def train_stage1_hsupcon(args, train_loader: DataLoader, dev_loader: DataLoader, device: torch.device):
    """
    GIAI ĐOẠN 1: Huấn luyện Biểu diễn tương phản phân cấp (Hierarchical SupCon) trên PhoBERT.
    - Hàm mất mát: HierarchicalTokenSupConLoss (alpha = args.supcon_alpha).
    - Tiêu chí dừng sớm và lưu checkpoint: Dev H-SupCon Loss thấp nhất.
    """
    LOGGER.info("=" * 70)
    LOGGER.info("  BẮT ĐẦU GIAI ĐOẠN 1: HIERARCHICAL SUPCON PRE-TRAINING (NO LABELS, NO CRF)")
    LOGGER.info(f"  Mô hình: {args.model_name_or_path} | Alpha: {args.supcon_alpha} | Temp: {args.supcon_temp}")
    LOGGER.info(f"  Batch size: {args.stage1_batch_size} | Epochs: {args.stage1_epochs} | Dev: {device}")
    LOGGER.info("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Khởi tạo mô hình Pretrainer
    model = PhoBertSupConPretrainer(
        model_name_or_path=args.model_name_or_path,
        proj_dim=args.proj_dim,
        temperature=args.supcon_temp,
        alpha=args.supcon_alpha,
        o_label_id=LABEL2ID_PAP_NER['O']
    ).to(device)

    # 2. Phân chia nhóm tham số với các Learning Rates khác nhau
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad], 'lr': args.stage1_lr},
        {'params': [p for n, p in model.projection_head.named_parameters() if p.requires_grad], 'lr': args.stage1_proj_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.stage1_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = args.fp16 and str(device).startswith('cuda')
    scaler = _get_grad_scaler(use_amp, device)

    # 3. Huấn luyện qua các Epochs
    best_dev_loss = float('inf')
    early_stop_count = 0
    best_model_path = os.path.join(args.output_dir, 'best_encoder.pt')

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with _get_amp_context(use_amp, device):
                outputs = model(**batch)
                loss = outputs.loss

            if loss.item() > 0:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

            scheduler.step()
            loss_val = loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                train_loss += loss_val
                train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # Validation H-SupCon loss
        model.eval()
        dev_loss = 0.0
        dev_steps = 0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with _get_amp_context(use_amp, device):
                    outputs = model(**batch)
                    d_loss = outputs.loss
                d_val = d_loss.item()
                if not math.isnan(d_val) and not math.isinf(d_val):
                    dev_loss += d_val
                    dev_steps += 1

        avg_dev_loss = dev_loss / max(dev_steps, 1)
        epoch_dur = datetime.timedelta(seconds=int(time.time() - epoch_start))

        LOGGER.info(f"[STAGE 1 - Epoch {epoch:02d}/{args.stage1_epochs:02d}] "
                    f"Train Loss: {avg_train_loss:.4f} | Dev Loss: {avg_dev_loss:.4f} | Time: {epoch_dur}")

        # Kiểm tra điều kiện lưu checkpoint
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            early_stop_count = 0
            torch.save(model.roberta.state_dict(), best_model_path)
            LOGGER.info(f"  ⭐ Đã lưu best encoder checkpoint tại: {best_model_path} (Dev Loss: {best_dev_loss:.4f})")
        else:
            early_stop_count += 1
            if early_stop_count >= args.stage1_early_stop:
                LOGGER.info(f"  🛑 Kích hoạt Early Stopping ở Stage 1 sau {epoch} epochs không cải thiện.")
                break

    LOGGER.info(f"  Hoàn tất Giai đoạn 1! Best Dev H-SupCon Loss: {best_dev_loss:.4f}")
    return best_model_path


def train_stage2_linear_probe(args, train_loader: DataLoader, dev_loader: DataLoader,
                              test_loader: DataLoader, device: torch.device, encoder_weights_path: str):
    """
    GIAI ĐOẠN 2: Huấn luyện Đầu dò Tuyến tính (Linear Probe).
    - Backbone PhoBERT ĐÓNG BĂNG 100% (requires_grad = False).
    - Phân loại bằng nn.Linear(768, 11) + Cross-Entropy Loss (KHÔNG CÓ CRF).
    - Đánh giá và dừng sớm dựa trên Validation Strict Entity-level F1 (CoNLL-2003 metric).
    """
    LOGGER.info("=" * 70)
    LOGGER.info("  BẮT ĐẦU GIAI ĐOẠN 2: LINEAR PROBE EVALUATION (BACKBONE FROZEN, NO CRF)")
    LOGGER.info(f"  Nạp encoder weights từ: {encoder_weights_path}")
    LOGGER.info(f"  Số nhãn: {len(LABEL2ID_PAP_NER)} | Batch size: {args.stage2_batch_size} | Epochs: {args.stage2_epochs}")
    LOGGER.info("=" * 70)

    # 1. Khởi tạo mô hình Linear Probe
    model = PhoBertLinearProbe(
        model_name_or_path=args.model_name_or_path,
        num_labels=len(LABEL2ID_PAP_NER),
        freeze_backbone=True
    ).to(device)

    # 2. Nạp trọng số Backbone đã được pre-train từ Stage 1
    if encoder_weights_path and os.path.exists(encoder_weights_path):
        state_dict = safe_torch_load(encoder_weights_path, map_location=device)
        model.roberta.load_state_dict(state_dict)
        LOGGER.info("  ✅ Đã nạp thành công trọng số PhoBERT từ Stage 1!")
    else:
        LOGGER.warning(f"  ⚠️ Không tìm thấy file {encoder_weights_path}, sử dụng checkpoint gốc của HuggingFace.")

    # 3. Chỉ tối ưu hóa duy nhất các tham số của tầng Linear Classifier
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    LOGGER.info(f"  Số lượng tham số huấn luyện được ở Stage 2: {num_trainable:,} (Tầng Linear 768x11)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.stage2_lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.stage2_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = args.fp16 and str(device).startswith('cuda')
    scaler = _get_grad_scaler(use_amp, device)

    best_dev_f1 = 0.0
    early_stop_count = 0
    best_model_path = os.path.join(args.output_dir, 'best_linear_probe.pt')

    # 4. Huấn luyện qua các Epochs
    for epoch in range(1, args.stage2_epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with _get_amp_context(use_amp, device):
                outputs = model(**batch)
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        # Đánh giá trên tập Dev bằng CoNLL-2003 Strict Entity-level F1
        dev_prec, dev_rec, dev_f1 = evaluate_linear_probe(
            model=model,
            data_loader=dev_loader,
            device=device,
            use_amp=use_amp,
            is_test=False
        )

        epoch_dur = datetime.timedelta(seconds=int(time.time() - epoch_start))
        LOGGER.info(f"[STAGE 2 - Epoch {epoch:02d}/{args.stage2_epochs:02d}] "
                    f"Train Loss: {avg_train_loss:.4f} | Dev Strict P: {dev_prec:.4f} | R: {dev_rec:.4f} | F1: {dev_f1:.4f} | Time: {epoch_dur}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            early_stop_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'best_dev_f1': best_dev_f1,
                'args': vars(args)
            }, best_model_path)
            LOGGER.info(f"  ⭐ Đã lưu best Linear Probe checkpoint tại: {best_model_path} (Dev Strict F1: {best_dev_f1:.4f})")
        else:
            early_stop_count += 1
            if early_stop_count >= args.stage2_early_stop:
                LOGGER.info(f"  🛑 Kích hoạt Early Stopping ở Stage 2 sau {epoch} epochs không cải thiện.")
                break

    # 5. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
    LOGGER.info("=" * 70)
    LOGGER.info("  TIẾN HÀNH ĐÁNH GIÁ TRÊN TẬP KIỂM TRA (TEST EVALUATION)")
    LOGGER.info("=" * 70)

    # Nạp lại checkpoint tốt nhất trên tập Dev
    if os.path.exists(best_model_path):
        ckpt = safe_torch_load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        LOGGER.info(f"  Đã nạp checkpoint tốt nhất tại epoch {ckpt.get('epoch', '?')} (Dev Strict F1: {ckpt.get('best_dev_f1', 0):.4f})")

    test_prec, test_rec, test_f1 = evaluate_linear_probe(
        model=model,
        data_loader=test_loader,
        device=device,
        use_amp=use_amp,
        is_test=True,
        output_dir=args.output_dir
    )

    LOGGER.info("=" * 70)
    LOGGER.info(f"  KẾT QUẢ CUỐI CÙNG (TEST SET - H-SUPCON + LINEAR PROBE, NO CRF):")
    LOGGER.info(f"   Strict Precision: {test_prec:.4f}")
    LOGGER.info(f"   Strict Recall:    {test_rec:.4f}")
    LOGGER.info(f"   Strict Micro-F1:  {test_f1:.4f}")
    LOGGER.info("=" * 70)
    return test_prec, test_rec, test_f1


def evaluate_linear_probe(model: nn.Module, data_loader: DataLoader, device: torch.device,
                          use_amp: bool = False, is_test: bool = False, output_dir: str = None) -> Tuple[float, float, float]:
    """Hàm đánh giá Linear Probe trên tập Dev hoặc Test."""
    model.eval()
    eval_golds = []
    eval_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with _get_amp_context(use_amp, device):
                outputs = model(**batch)

            labels = batch['labels']
            active_mask = batch['label_masks'] == 1
            batch_tags = outputs.tags

            for b in range(len(batch_tags)):
                length = active_mask[b].sum().item()
                gold_seq = labels[b, :length].cpu().tolist()
                pred_seq = batch_tags[b]
                eval_golds.extend(gold_seq)
                eval_preds.extend(pred_seq)

    id2label = LABEL_MAPPING['pap_ner']['id2label']
    gold_str = [id2label[tag] for tag in eval_golds]
    pred_str = [id2label[tag] for tag in eval_preds]

    # Tính CoNLL Strict Entity-level F1
    prec, rec, f1 = evaluate(gold_str, pred_str, verbose=is_test)

    # Nếu là tập Test, in báo cáo chi tiết cấp Token và vẽ ma trận nhầm lẫn
    if is_test:
        report = classification_report(
            eval_golds, eval_preds,
            target_names=list(LABEL2ID_PAP_NER.keys()),
            digits=4,
            zero_division=0
        )
        LOGGER.info("\n" + report)
        plot_confusion_matrix(
            golds=eval_golds,
            preds=eval_preds,
            classes=list(LABEL2ID_PAP_NER.keys()),
            output_dir=output_dir,
            title='Normalized Confusion Matrix (H-SupCon + Linear Probe, No CRF)',
            normalize=True
        )

    return prec, rec, f1

