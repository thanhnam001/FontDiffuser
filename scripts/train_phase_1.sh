CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --nnodes=1 train3.py \
    --seed=123 \
    --experience_name="FontDiffuser_training_phase_1" \
    --data_root="/data/ocr/namvt17/WordStylist/data" \
    --output_dir="outputs/v4" \
    --report_to="tensorboard" \
    --resolution=96 \
    --style_image_size=96 \
    --content_image_size=96 \
    --content_encoder_downsample_size=3 \
    --channel_attn=True \
    --content_start_channel=64 \
    --style_start_channel=64 \
    --train_batch_size=10 \
    --perceptual_coefficient=0.01 \
    --offset_coefficient=0.5 \
    --max_train_steps=440000 \
    --ckpt_interval=20000 \
    --gradient_accumulation_steps=1 \
    --log_interval=50 \
    --learning_rate=1e-4 \
    --lr_scheduler="linear" \
    --lr_warmup_steps=10000 \
    --drop_prob=0.1 \
    --mixed_precision="no"
    