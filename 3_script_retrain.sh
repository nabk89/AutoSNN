#!/bin/bash

gpu=0
ataset_dir='./data'

dataset=CIFAR10
## static datasets: CIFAR10, CIFAR100, SVHN, Tiny-ImageNet-200
## neuromorphic datasets: CIFAR10DVS, DVS128Gesture

## Various architectures searched by AutoSNN are included in search_arch/arch.py 
arch=AutoSNN

## Macro architecture types that we used in this study are included in space.py (see MACRO_SEARCH_SPACE)
macro_type=AutoSNN_16

python retrain/train.py \
    --gpu $gpu \
    --T 8 --init_tau 2.0 --v_threshold 1.0 --neuron PLIF \
    --epochs 2 \
    --dataset_dir $dataset_dir \
    --dataset_name CIFAR10 \
    --save $arch \
    --arch $arch \
    --macro_type $macro_type \
    --seed 2022

