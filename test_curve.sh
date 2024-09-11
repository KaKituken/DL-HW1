python train.py \
    --batch_size 256 \
    --run_name test_run/test_bs256_lr0.003_wd0 \
    --epoch 5000 \
    --train_steps 20000 \
    --test_steps 500 \
    --save_steps 5000 \
    --log_steps 50 \
    --warmup_steps 100 \
    --save_dir ./save \
    --lr 0.003 \
    --weight_decay 0 \
    --test_curve

python train.py \
    --batch_size 256 \
    --run_name test_run/test_bs256_lr0.003_wd1e-4 \
    --epoch 5000 \
    --train_steps 20000 \
    --test_steps 500 \
    --save_steps 5000 \
    --log_steps 50 \
    --warmup_steps 100 \
    --save_dir ./save \
    --lr 0.003 \
    --weight_decay 1e-4 \
    --test_curve

python train.py \
    --batch_size 256 \
    --run_name test_run/test_bs256_lr0.003_wd0_drop0.3 \
    --epoch 5000 \
    --train_steps 20000 \
    --test_steps 500 \
    --save_steps 5000 \
    --log_steps 50 \
    --warmup_steps 100 \
    --save_dir ./save \
    --lr 0.003 \
    --use_drop_out \
    --drop_out_prob 0.3 \
    --test_curve

python train.py \
    --batch_size 256 \
    --run_name test_run/test_bs256_lr0.003_wd0_bn \
    --epoch 5000 \
    --train_steps 20000 \
    --test_steps 500 \
    --save_steps 5000 \
    --log_steps 50 \
    --warmup_steps 100 \
    --save_dir ./save \
    --lr 0.003 \
    --use_batch_norm \
    --test_curve