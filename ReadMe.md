# ReadMe
This is the implemenation of Customized LeNet5 for CS5787 Deep Learning @ Cornell Tech. Exercises 1.

## Set up
```sh
pip install -r requirements.txt
```
If using GPU, please install torch and torchvision according to your CUDA version.

## Train

Use `train.py` to train a model and set hyper-parameters manually.
We support setting whether to use batch norm, weight decay, drop out and the random seed.

Specify gpu device using `--gpu`. Distributed training not supported yet.

```sh
python train.py \
    --batch_size bs \		# batch_size
    --run_name run/your_name \	# dir to store tensorboard log
    --epoch 5000 \			# max epoch
    --train_steps 20000 \	# max steps
    --test_steps 500 \		# test interval
    --save_steps 5000 \		# model save interval
    --log_steps 50 \		# log interval
    --warmup_steps 100 \	# lr warm-up
    --save_dir ./save \		# model save dir
    --lr 0.003 \			# learning rate
    --weight_decay 0	\	# weight decay
    --gpu 0 \				# gpu device
    --use_batch_norm \		# whether to use batch norm
    --use_drop_out \		# whether to use dropout
    --drop_out_prob 0.5	\	# dropout probability
    --seed 42 \             # random seed
    --test_curve			# whether to run on test set
```
## Visualize training
The training log will be saved in `--run_name`. Use tensorboard to view the training curves.
```sh
tensorboard --logdir /your/run/name
```

## Grid Search
We provide the script to perform grid search on `bs`, `lr` and `weight decay`. Simply run 
```sh
./grid_search.sh
```
and the curves will be saved in `run/`. Use tensorboard to visualize.

## Test

We provide two approaches to test the model. 
- If you want to load a pretrained model checkpoint and evaluate it on fashionMNIST test set, run
    ```sh
    python test.py \
        --dataset ./data\
        --gpu 0\
        --checkpoint /path/to/checkpoint
    ```
- If you want to draw the curve on testing set, run the following
    ```sh
    ./test_curve.sh
    tensorboard --logdir test_run
    ```
    Then you will get the exactly the same results as in the report.


 
