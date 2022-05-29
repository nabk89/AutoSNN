#!/bin/bash

gpu=0
dataset_dir='./data'

#search_mode=random ## WS + random search (in Table 5)
search_mode=evolution ## AutoSNN

## This super-network will be generated after executing 1_script_train_supernet.sh
trained_supernet='macro_search_result/uniform_sampling/AutoSNN_16_CIFAR10_SNN_Adam_1ep_2022/checkpoint.pth.tar'

python search_arch/search.py \
    --gpu $gpu \
    --T 8 --init_tau 2.0 --v_threshold 1.0 --neuron PLIF \
    --dataset_dir $dataset_dir \
    --dataset_name CIFAR10 \
    --supernet $trained_supernet \
    --seed 2022 \
    --search_space AutoSNN_16 \
    --search_algo $search_mode \
    --fitness ACC_pow_spikes \
    --fitness_lambda -0.08 \
    --avg_num_spikes 110000
    
