#!/bin/bash

gpu=0
dataset_dir='./data'

python train_supernet/train.py \
    --gpu $gpu \
    --T 8 --init_tau 2.0 --v_threshold 1.0 --neuron PLIF \
    --epochs 600 \
    --dataset_dir $dataset_dir \
    --dataset_name CIFAR10 \
    --save uniform_sampling \
    --search_space AutoSNN_16 \
    --seed 2022

