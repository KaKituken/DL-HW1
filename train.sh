#!/bin/bash

# 定义超参数的范围
batch_sizes=(32 64 128 256)
learning_rates=(0.01 0.003 0.001)
weight_decays=(1e-4 3e-4 1e-3 3e-3)

# 遍历每一个超参数组合
for bs in "${batch_sizes[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for wd in "${weight_decays[@]}"; do
            # 定义run_name
            run_name="run_bs${bs}_lr${lr}_wd${wd}"

            # 打印当前组合
            echo "Running with batch_size=${bs}, lr=${lr}, weight_decay=${wd}"

            # 执行Python训练脚本
            python train.py \
                --batch_size ${bs} \
                --run_name run/${run_name} \
                --epoch 50 \
                --train_steps 20000 \
                --test_steps 1000 \
                --save_steps 5000 \
                --log_steps 50 \
                --warmup_steps 100 \
                --save_dir ./save \
                --lr ${lr} \
                --weight_decay ${wd}

        done
    done
done
