@echo off
chcp 65001 >nul
echo ======================================================================
echo   CHẠY THỰC NGHIỆM: TWO-STAGE H-SUPCON + LINEAR PROBE (NO CRF)
echo   Backbone: PhoBERT-base | Alpha: 0.3 | Seed: 42
echo ======================================================================

python main.py ^
    --task pap_ner ^
    --data_dir ./Dataset ^
    --model_name_or_path vinai/phobert-base ^
    --output_dir outputs/phobert_base_hsupcon_linear_eval ^
    --stage1_epochs 15 ^
    --stage1_batch_size 64 ^
    --stage1_lr 2e-5 ^
    --stage1_proj_lr 1e-3 ^
    --supcon_temp 0.07 ^
    --supcon_alpha 0.3 ^
    --proj_dim 128 ^
    --stage1_early_stop 5 ^
    --stage2_epochs 20 ^
    --stage2_batch_size 64 ^
    --stage2_lr 1e-3 ^
    --stage2_early_stop 5 ^
    --seed 42 ^
    --fp16

echo ======================================================================
echo   HOÀN TẤT THỰC NGHIỆM! KIỂM TRA LOG TẠI:
echo   outputs/phobert_base_hsupcon_linear_eval/training.log
echo ======================================================================
pause

