# Baseline H-SupCon: Two-Stage Hierarchical Supervised Contrastive Learning + Linear Probe (No CRF)

Dự án này triển khai chuẩn mực **Giao thức Đánh giá Tuyến tính (Two-Stage Linear Evaluation Protocol)** kết hợp **Hierarchical Supervised Contrastive Learning (H-SupCon, $\alpha = 0.3$)** và **Linear Probe (hoàn toàn KHÔNG dùng CRF)** trên tập dữ liệu Thực thể Hành chính Việt Nam (**PAP_NER**).

---

## 1. Vị trí trong Bài báo Khoa học (Ablation Matrix)

Mô hình này đại diện cho **Cột mốc 2** trong ma trận đối chứng:

| STT | Mô hình | CRF? | Chiến lược Contrastive | Giao thức | Strict Micro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Mốc 1** | Baseline Flat SupCon + Linear Probe | ❌ **Không** | Flat SupCon (Khosla et al., 2020) | Two-Stage (Frozen) | **90.02%** |
| **Mốc 2** | **Baseline H-SupCon + Linear Probe** *(Thư mục này)* | ❌ **Không** | **Hierarchical SupCon ($\alpha = 0.3$)** | **Two-Stage (Frozen)** | **~92.5% - 93.5%** |
| **Mốc 3** | **Proposed Model: VPhoBertTagger** | ✅ **Có** | **Hierarchical SupCon ($\alpha = 0.3$)** | **End-to-End Joint** | **97.66%** |

---

## 2. Nguyên lý Hoạt động 2 Giai đoạn

1. **Giai đoạn 1 (Stage 1 - Pre-training H-SupCon):**
   * Huấn luyện PhoBERT bằng hàm mất mát `HierarchicalTokenSupConLoss` ở cấp token thực thể (bỏ qua nhãn `O`).
   * Ánh xạ phân cấp: $w_{\text{strong}} = 1.0$ (cùng BIO tag), $w_{\text{weak}} = \alpha = 0.3$ (cùng entity type, khác BIO), $w_{\text{neg}} = 0.0$ (khác entity type).
   * Lưu trọng số tối ưu nhất vào file `outputs/.../best_encoder.pt`.
2. **Giai đoạn 2 (Stage 2 - Linear Probe Evaluation):**
   * Nạp `best_encoder.pt`, **đóng băng 100% tham số PhoBERT** (`requires_grad = False`).
   * Chỉ huấn luyện duy nhất tầng phân loại `nn.Linear(768, 11)` với hàm mất mát `CrossEntropyLoss`.
   * **Tuyệt đối KHÔNG có CRF**. Suy luận dự đoán trực tiếp bằng `torch.argmax(logits, dim=-1)`.

---

## 3. Cấu trúc Thư mục

```
baseline-hsupcon/
├── src/
│   ├── constant.py              # Định nghĩa 11 nhãn BIO, ánh xạ 5 loại thực thể (CQ, ĐT, VBPL, NG, SL)
│   ├── arguments.py             # CLI parser: --supcon_alpha 0.3, --supcon_temp 0.07, v.v.
│   ├── helper.py                # Logger, random seed, AMP FP16, confusion matrix
│   ├── processor.py             # BPE Tokenizer, gióng hàng subword valid_ids
│   ├── dataset.py               # PyTorch DataLoader, tự động tìm kiếm đường dẫn dữ liệu
│   ├── conlleval.py             # Bộ đánh giá CoNLL-2003 Strict Entity (đếm đơn chuẩn 1x)
│   └── models/
│       ├── base.py              # NerOutput container
│       ├── projection.py        # MLP Projection Head (768 -> 256 -> GELU -> 128 -> L2-Norm)
│       ├── losses.py            # HierarchicalTokenSupConLoss (chống tràn số, vector hóa)
│       └── linear_probe.py      # PhoBertSupConPretrainer & PhoBertLinearProbe (Frozen, No CRF)
├── main.py                      # Kịch bản chính chạy toàn trình Stage 1 -> Stage 2 -> Test Evaluation
├── run_linear_eval.bat          # Kịch bản chạy nhanh trên Windows
├── run_linear_eval.sh           # Kịch bản chạy trên Linux / Vast.ai / RunPod
├── baseline_hsupcon_eval.ipynb  # Jupyter Notebook chạy trên Colab / Vast.ai RTX 3090 / A100
├── requirements.txt
└── .gitignore
```

---

## 4. Hướng dẫn Chạy Thực nghiệm

### Trên máy Windows (Command Prompt hoặc PowerShell):
```cmd
run_linear_eval.bat
```

### Trên máy Linux / Vast.ai / Google Colab:
```bash
bash run_linear_eval.sh
```

Hoặc chạy lệnh Python trực tiếp:
```bash
python main.py \
    --task pap_ner \
    --data_dir ./Dataset \
    --model_name_or_path vinai/phobert-base \
    --output_dir outputs/phobert_base_hsupcon_linear_eval \
    --stage1_epochs 15 \
    --stage1_batch_size 64 \
    --stage1_lr 2e-5 \
    --stage1_proj_lr 1e-3 \
    --supcon_temp 0.07 \
    --supcon_alpha 0.3 \
    --proj_dim 128 \
    --stage1_early_stop 5 \
    --stage2_epochs 20 \
    --stage2_batch_size 64 \
    --stage2_lr 1e-3 \
    --stage2_early_stop 5 \
    --seed 42 \
    --fp16
```

